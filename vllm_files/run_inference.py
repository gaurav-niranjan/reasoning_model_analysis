"""
run_inference.py — Resumable Qwen3-VL-Thinking inference for SB-Bench

Two image modes controlled by --inpainted flag:
  --inpainted    → loads INPAINTED_DIR/<id>/inpainted_bg.png
  --no-inpainted → loads original image from dataset's file_name field

Checkpoint/resume: if the job dies, re-run the exact same command.

Usage:
    python run_inference.py \
        --server http://127.0.0.1:34521 \
        --inpainted-dir /path/to/inpainted \
        --id-to-index /path/to/id_to_index.json \
        --inpainted \
        --output results/sbbench_inpainted
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
import fcntl
from pathlib import Path

import aiohttp
import pandas as pd
from PIL import Image
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ================================================================
#  Prompt & image helpers
# ================================================================

def build_prompt(example: dict) -> str:
    """Build the MCQ prompt from SB-Bench fields."""
    return (
        f"Context: {example['context']}\n"
        f"Question: {example['question']}\n\n"
        f"A) {example['ans0']}\n"
        f"B) {example['ans1']}\n"
        f"C) {example['ans2']}\n\n"
        "Think and answer with only A, B, or C.\n"
        "Answer:"
    )


def downsample_image(img: Image.Image, short_side: int = 1536) -> Image.Image:
    """Downsample so the short side is at most `short_side` pixels."""
    W, H = img.size
    if min(W, H) <= short_side:
        return img
    scale = short_side / min(W, H)
    new_W = int(W * scale)
    new_H = int(H * scale)
    return img.resize((new_W, new_H), Image.Resampling.LANCZOS)


def pil_to_base64_url(img: Image.Image) -> str:
    """Convert a PIL image to a base64 data URL for the vLLM API."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ================================================================
#  Checkpoint manager
# ================================================================

class CheckpointManager:
    """Append-only JSONL checkpoint with resume support."""

    def __init__(self, checkpoint_path: str):
        self.path = Path(checkpoint_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.completed_ids: set[str] = set()
        self._file = None
        self._lock = asyncio.Lock()
        self._load_existing()

    def _load_existing(self):
        if not self.path.exists():
            log.info(f"No checkpoint at {self.path} — starting fresh")
            return

        count = 0
        errors = 0
        with open(self.path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("status") == "ok":
                        self.completed_ids.add(str(obj["id"]))
                        count += 1
                except json.JSONDecodeError:
                    errors += 1

        log.info(f"Checkpoint: {count} completed ({errors} corrupt lines skipped)")

    def is_done(self, sample_id: str) -> bool:
        return str(sample_id) in self.completed_ids

    def filter_remaining(self, samples: list[dict]) -> list[dict]:
        remaining = [s for s in samples if not self.is_done(str(s["id"]))]
        skipped = len(samples) - len(remaining)
        if skipped > 0:
            log.info(f"Resuming: {skipped} done, {len(remaining)} remaining")
        return remaining

    def open(self):
        self._file = open(self.path, "a", buffering=1)

    def close(self):
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None

    async def write_result(self, result: dict):
        if self._file is None:
            raise RuntimeError("Checkpoint not opened")
        line = json.dumps(result, ensure_ascii=False)
        async with self._lock:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
            try:
                self._file.write(line + "\n")
                self._file.flush()
                os.fsync(self._file.fileno())
            finally:
                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        if result.get("status") == "ok":
            self.completed_ids.add(str(result["id"]))

    def to_parquet(self, parquet_path: str) -> pd.DataFrame | None:
        rows = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not rows:
            log.warning("No results to convert")
            return None
        seen = {}
        for row in rows:
            seen[str(row["id"])] = row
        df = pd.DataFrame(list(seen.values()))
        Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        ok = (df["status"] == "ok").sum()
        log.info(f"Wrote {parquet_path}: {len(df)} rows ({ok} ok)")
        return df


# ================================================================
#  Dataset loading
# ================================================================

def load_sbbench(
    id_to_index_path: str,
    use_inpainted: bool,
    inpainted_dir: str | None,
    original_images_dir: str | None = None,
    limit: int | None = None,
    class_cat: int | None = None,
) -> list[dict]:
    """
    Load SB-Bench, filter to the inpainted subset, prepare samples.
    """
    from datasets import load_dataset

    log.info("Loading ucf-crcv/SB-Bench split='real' ...")
    ds = load_dataset("ucf-crcv/SB-Bench", split="real")

    if use_inpainted:
        log.info(f"Using inpainted images from {inpainted_dir}")
        with open(id_to_index_path, "r") as f:
            id_to_index = json.load(f)

        inpainted_path = Path(inpainted_dir)
        ids_to_process = [d.stem for d in inpainted_path.iterdir() if d.is_dir()]
        log.info(f"Found {len(ids_to_process)} IDs with inpainted images")

        ds_indexes = [id_to_index[id_] for id_ in ids_to_process if id_ in id_to_index]
        ds = ds.select(ds_indexes)
        log.info(f"Dataset after filtering: {len(ds)} samples")

    else:
        log.info("Using original images from dataset")
        if class_cat is not None:
            log.info(f"Filtering to class category {class_cat} ...")
            ds = ds.filter(lambda x: x['category'] == class_cat)
            log.info(f"Dataset after filtering: {len(ds)} samples")
        
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    samples = []
    skipped = 0

    for row in ds:
        sample_id = str(row["id"])

        if use_inpainted:
            img_path = inpainted_path / sample_id / "inpainted_bg.png"
        else:
            img = row["file_name"]
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            img_path = None  # no file on disk


        if img_path is not None and not Path(img_path).exists():
            log.warning(f"Image not found for {sample_id}: {img_path}")
            skipped += 1
            continue

        correct_idx = row.get("label", row.get("answer", ""))
        gt_map = {0: "A", 1: "B", 2: "C", "0": "A", "1": "B", "2": "C"}
        ground_truth = gt_map.get(correct_idx, str(correct_idx))

        samples.append({
            "id": sample_id,
            "image_path": str(img_path) if img_path else None,
            "pil_image": img if img_path is None else None,
            "inpainted": use_inpainted,
            "question": row["question"],
            "context": row.get("context", ""),
            "ans0": row["ans0"],
            "ans1": row["ans1"],
            "ans2": row["ans2"],
            "ground_truth": ground_truth,
        })

    if skipped:
        log.warning(f"Skipped {skipped} samples (missing images)")
    log.info(f"Prepared {len(samples)} samples (inpainted={use_inpainted})")
    return samples


# ================================================================
#  Inference
# ================================================================

def _make_error_result(sample: dict, error: str, latency: float) -> dict:
    return {
        "id": sample["id"],
        "inpainted": sample["inpainted"],
        "question": sample["question"],
        "context": sample.get("context", ""),
        "ans0": sample["ans0"],
        "ans1": sample["ans1"],
        "ans2": sample["ans2"],
        "ground_truth": sample["ground_truth"],
        "status": "error",
        "error": error,
        "thinking": "",
        "answer": "",
        "latency_s": round(latency, 3),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "thinking_word_count": 0,
        "answer_word_count": 0,
    }


async def query_one(
    session: aiohttp.ClientSession,
    sample: dict,
    server_url: str,
    model_name: str,
    max_tokens: int,
    timeout: aiohttp.ClientTimeout,
    ckpt: CheckpointManager,
) -> dict:
    """Load image, downsample, build prompt, send to vLLM, checkpoint."""
    url = f"{server_url}/v1/chat/completions"

    try:
        if sample.get("pil_image") is not None:
            img = sample["pil_image"]
        else:
            img = Image.open(sample["image_path"]).convert("RGB")
        img = downsample_image(img)
        image_b64_url = pil_to_base64_url(img)
    except Exception as e:
        result = _make_error_result(sample, f"Image load failed: {e}", 0)
        await ckpt.write_result(result)
        return result

    prompt = build_prompt(sample)

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_b64_url},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
    }

    t0 = time.perf_counter()

    try:
        async with session.post(url, json=payload, timeout=timeout) as resp:
            latency = time.perf_counter() - t0
            if resp.status != 200:
                error_text = await resp.text()
                result = _make_error_result(
                    sample, f"HTTP {resp.status}: {error_text[:500]}", latency
                )
                await ckpt.write_result(result)
                return result
            api_result = await resp.json()

    except asyncio.TimeoutError:
        result = _make_error_result(sample, "Timeout", time.perf_counter() - t0)
        result["status"] = "timeout"
        await ckpt.write_result(result)
        return result
    except Exception as e:
        result = _make_error_result(sample, str(e)[:500], time.perf_counter() - t0)
        await ckpt.write_result(result)
        return result

    choice = api_result["choices"][0]["message"]
    usage = api_result.get("usage", {})
    thinking = choice.get("reasoning", "") or ""
    answer = choice.get("content", "") or ""

    result = {
        "id": sample["id"],
        "inpainted": sample["inpainted"],
        "question": sample["question"],
        "context": sample.get("context", ""),
        "ans0": sample["ans0"],
        "ans1": sample["ans1"],
        "ans2": sample["ans2"],
        "ground_truth": sample["ground_truth"],
        "status": "ok",
        "error": "",
        "thinking": thinking,
        "answer": answer,
        "latency_s": round(latency, 3),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "thinking_word_count": len(thinking.split()) if thinking else 0,
        "answer_word_count": len(answer.split()) if answer else 0,
    }
    await ckpt.write_result(result)
    return result


async def run_batch(
    samples: list[dict],
    server_url: str,
    model_name: str,
    max_tokens: int,
    concurrency: int,
    timeout_s: int,
    ckpt: CheckpointManager,
) -> list[dict]:
    if not samples:
        return []

    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    connector = aiohttp.TCPConnector(limit=concurrency + 10)

    async def bounded_query(session, sample):
        async with semaphore:
            return await query_one(
                session, sample, server_url, model_name,
                max_tokens, timeout, ckpt,
            )

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [bounded_query(session, s) for s in samples]
        results = await tqdm_asyncio.gather(*tasks, desc="Inference")

    return results


async def retry_failures(
    all_samples: list[dict],
    server_url: str,
    model_name: str,
    max_tokens: int,
    concurrency: int,
    timeout_s: int,
    ckpt: CheckpointManager,
    max_retries: int = 2,
):
    sample_lookup = {str(s["id"]): s for s in all_samples}

    for attempt in range(1, max_retries + 1):
        latest_status = {}
        if ckpt.path.exists():
            with open(ckpt.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        latest_status[str(obj["id"])] = obj.get("status", "error")
                    except json.JSONDecodeError:
                        continue

        failed_ids = [
            sid for sid, status in latest_status.items()
            if status != "ok" and sid in sample_lookup
        ]

        if not failed_ids:
            log.info("All samples succeeded.")
            return

        log.info(f"Retry {attempt}/{max_retries}: {len(failed_ids)} failed samples")
        retry_samples = [sample_lookup[fid] for fid in failed_ids]
        await run_batch(
            retry_samples, server_url, model_name,
            max_tokens, concurrency, timeout_s, ckpt,
        )


# ================================================================
#  CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Resumable Qwen3-VL-Thinking inference on SB-Bench"
    )

    parser.add_argument("--server", type=str, required=True)
    parser.add_argument("--model-name", type=str,
                        default="Qwen/Qwen3-VL-8B-Thinking")

    parser.add_argument("--inpainted-dir", type=str, default=None)
    parser.add_argument("--id-to-index", type=str, required=True)
    parser.add_argument("--original-images-dir", type=str, default=None)

    parser.add_argument("--inpainted", dest="use_inpainted",
                        action="store_true", default=True)
    parser.add_argument("--no-inpainted", dest="use_inpainted",
                        action="store_false")
    parser.add_argument("--class-cat", type=int, default=None)

    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retries", type=int, default=2)

    args = parser.parse_args()
    server_url = args.server.rstrip("/")

    samples = load_sbbench(
        id_to_index_path=args.id_to_index,
        inpainted_dir=args.inpainted_dir,
        use_inpainted=args.use_inpainted,
        original_images_dir=args.original_images_dir,
        limit=args.limit,
        class_cat=args.class_cat,
    )

    if not samples:
        log.error("No samples loaded. Check paths.")
        sys.exit(1)

    ids = [s["id"] for s in samples]
    if len(ids) != len(set(ids)):
        from collections import Counter
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        log.error(f"Duplicate IDs: {dupes[:10]}")
        sys.exit(1)

    checkpoint_path = f"{args.output}.checkpoint.jsonl"
    parquet_path = f"{args.output}.parquet"

    ckpt = CheckpointManager(checkpoint_path)
    remaining = ckpt.filter_remaining(samples)

    if not remaining:
        log.info("All samples already done. Converting to Parquet ...")
        df = ckpt.to_parquet(parquet_path)
        if df is not None:
            ok = (df["status"] == "ok").sum()
            log.info(f"Final: {ok}/{len(df)} succeeded")
        return

    total = len(samples)
    done = total - len(remaining)
    log.info(f"Total: {total} | Done: {done} | Remaining: {len(remaining)}")
    log.info(f"  Server:      {server_url}")
    log.info(f"  Inpainted:   {args.use_inpainted}")
    log.info(f"  Concurrency: {args.concurrency}")
    log.info(f"  Checkpoint:  {checkpoint_path}")

    ckpt.open()
    try:
        asyncio.run(run_batch(
            remaining, server_url, args.model_name,
            args.max_tokens, args.concurrency, args.timeout, ckpt,
        ))

        if args.retries > 0:
            asyncio.run(retry_failures(
                samples, server_url, args.model_name,
                args.max_tokens, args.concurrency,
                args.timeout, ckpt, args.retries,
            ))
    except KeyboardInterrupt:
        log.info("Interrupted. Progress saved. Re-run to resume.")
    finally:
        ckpt.close()

    log.info("Converting checkpoint → Parquet ...")
    df = ckpt.to_parquet(parquet_path)
    if df is not None:
        ok_df = df[df["status"] == "ok"]
        failed = (df["status"] != "ok").sum()
        log.info(f"Done: {len(ok_df)} succeeded, {failed} failed")
        if len(ok_df) > 0:
            log.info(f"  Avg latency:        {ok_df['latency_s'].mean():.1f}s")
            log.info(f"  Avg thinking words: {ok_df['thinking_word_count'].mean():.0f}")
            log.info(f"  Avg completion tok: {ok_df['completion_tokens'].mean():.0f}")


if __name__ == "__main__":
    main()