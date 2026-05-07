"""
run_judge.py — Classify thinking traces for background signals using LLM-as-judge

Reads the parquet output from run_inference.py and sends each thinking trace
to a text-only vLLM server (e.g. Gemma-3-27B-IT) for binary classification.

Checkpoint/resume: if the job dies, re-run the exact same command.

Usage:
    python run_judge.py \
        --server http://127.0.0.1:34521 \
        --input results/sbbench_inpainted_8B.parquet \
        --output results/judge_kitchen_bg \
        --signal "the image background being a kitchen or kitchen-like setting"
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import fcntl
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ================================================================
#  Judge prompt
# ================================================================

#DEFAULT_SIGNAL = "the image background being a kitchen or kitchen-like setting"
#DEFAULT_SIGNAL = "the image background being a cluttered, messy, or disorganized room"
DEFAULT_SIGNAL = "the image background being a laboratory or science lab setting"

JUDGE_SYSTEM_PROMPT = """\
You are a precise text classifier. You read thinking traces from a vision-language \
model and determine whether a specific signal is present. You respond with ONLY a \
JSON object, nothing else. No markdown, no explanation."""

def build_judge_prompt(thinking_trace: str, signal_description: str) -> str:
    return (
        f"Below is a thinking trace from a vision-language model that was analyzing an image.\n\n"
        f"<trace>\n{thinking_trace}\n</trace>\n\n"
        f"Does this thinking trace discuss or mention: {signal_description}?\n\n"
        f"Consider any direct or indirect references — e.g. mentioning microscopes, "
        f"beakers, flasks, colorful liquids or chemicals, a whiteboard with equations "
        f"or formulas, lab countertops, scientific equipment or instruments, "
        f"or describing the setting as a laboratory, science room, or research environment.\n\n"
        f'Respond with ONLY this JSON:\n'
        f'{{"signal_present": true/false, "evidence": "brief quote or description of what you found, or empty string if not found"}}'
    )

'''
return (
        f"Below is a thinking trace from a vision-language model that was analyzing an image.\n\n"
        f"<trace>\n{thinking_trace}\n</trace>\n\n"
        f"Does this thinking trace discuss or mention: {signal_description}?\n\n"
        f"Consider any direct or indirect references — e.g. mentioning countertops, "
        f"cabinets, stove, cooking area, sink, kitchen appliances, or describing the "
        f"scene as a kitchen. Also consider if the model notes the background or setting "
        f"in a way that implies a kitchen.\n\n"
        f'Respond with ONLY this JSON:\n'
        f'{{"signal_present": true/false, "evidence": "brief quote or description of what you found, or empty string if not found"}}'
'''


# ================================================================
#  Checkpoint manager (reused from run_inference.py)
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
            for line in f:
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
#  Dataset loading — reads parquet from inference step
# ================================================================

def load_traces(
    parquet_path: str,
    limit: int | None = None,
    min_thinking_words: int = 5,
) -> list[dict]:
    """
    Load thinking traces from the inference parquet output.

    Filters to status='ok' rows that have a non-empty thinking trace.
    """
    log.info(f"Loading traces from {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    log.info(f"  Total rows: {len(df)}")

    # only judge successful inferences with actual thinking
    df = df[df["status"] == "ok"].copy()
    df = df[df["thinking"].str.strip().str.len() > 0]
    df = df[df["thinking_word_count"] >= min_thinking_words]
    log.info(f"  After filtering (ok + has thinking): {len(df)}")

    if limit:
        df = df.head(limit)

    samples = []
    for _, row in df.iterrows():
        samples.append({
            "id": str(row["id"]),
            "thinking": row["thinking"],
            "answer": row.get("answer", ""),
            "question": row.get("question", ""),
            "ground_truth": row.get("ground_truth", ""),
            "inpainted": row.get("inpainted", None),
            "thinking_word_count": row.get("thinking_word_count", 0),
        })

    log.info(f"  Prepared {len(samples)} samples for judging")
    return samples


# ================================================================
#  Judge inference
# ================================================================

def _make_error_result(sample: dict, error: str, latency: float) -> dict:
    return {
        "id": sample["id"],
        "inpainted": sample.get("inpainted"),
        "question": sample.get("question", ""),
        "ground_truth": sample.get("ground_truth", ""),
        "status": "error",
        "error": error,
        "signal_present": None,
        "evidence": "",
        "judge_raw": "",
        "latency_s": round(latency, 3),
    }


def _parse_judge_response(raw_text: str) -> dict:
    """
    Try to parse the judge's JSON response. Be lenient.

    Returns {"signal_present": bool|None, "evidence": str, "parse_ok": bool}
    """
    text = raw_text.strip()

    # strip markdown fences if model wraps in ```json ... ```
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        obj = json.loads(text)
        signal = obj.get("signal_present")
        # handle string "true"/"false"
        if isinstance(signal, str):
            signal = signal.lower().strip() == "true"
        evidence = str(obj.get("evidence", ""))
        return {"signal_present": bool(signal), "evidence": evidence, "parse_ok": True}
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Fallback: look for YES/NO or true/false in raw text
    lower = text.lower()
    if "true" in lower or '"signal_present": true' in lower:
        return {"signal_present": True, "evidence": "", "parse_ok": False}
    if "false" in lower or '"signal_present": false' in lower:
        return {"signal_present": False, "evidence": "", "parse_ok": False}

    return {"signal_present": None, "evidence": "", "parse_ok": False}


async def judge_one(
    session: aiohttp.ClientSession,
    sample: dict,
    server_url: str,
    model_name: str,
    signal_description: str,
    max_tokens: int,
    timeout: aiohttp.ClientTimeout,
    ckpt: CheckpointManager,
) -> dict:
    """Send one thinking trace to the judge model and classify it."""
    url = f"{server_url}/v1/chat/completions"

    prompt = build_judge_prompt(sample["thinking"], signal_description)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,  # deterministic for classification
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
    raw_text = choice.get("content", "") or ""
    usage = api_result.get("usage", {})

    parsed = _parse_judge_response(raw_text)

    result = {
        "id": sample["id"],
        "inpainted": sample.get("inpainted"),
        "question": sample.get("question", ""),
        "ground_truth": sample.get("ground_truth", ""),
        "thinking_word_count": sample.get("thinking_word_count", 0),
        "status": "ok",
        "error": "",
        "signal_present": parsed["signal_present"],
        "evidence": parsed["evidence"],
        "parse_ok": parsed["parse_ok"],
        "judge_raw": raw_text[:2000],  # truncate just in case
        "latency_s": round(latency, 3),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }
    await ckpt.write_result(result)
    return result


async def run_batch(
    samples: list[dict],
    server_url: str,
    model_name: str,
    signal_description: str,
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
            return await judge_one(
                session, sample, server_url, model_name,
                signal_description, max_tokens, timeout, ckpt,
            )

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [bounded_query(session, s) for s in samples]
        results = await tqdm_asyncio.gather(*tasks, desc="Judging")

    return results


async def retry_failures(
    all_samples: list[dict],
    server_url: str,
    model_name: str,
    signal_description: str,
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
            signal_description, max_tokens, concurrency, timeout_s, ckpt,
        )


# ================================================================
#  CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-judge: classify thinking traces for specific signals"
    )

    # Server
    parser.add_argument("--server", type=str, required=True,
                        help="vLLM server URL, e.g. http://127.0.0.1:34521")
    parser.add_argument("--model-name", type=str,
                        default="google/gemma-3-27b-it",
                        help="Model name as registered on the vLLM server")

    # Input / output
    parser.add_argument("--input", type=str, required=True,
                        help="Path to inference parquet (from run_inference.py)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output prefix (creates .parquet and .checkpoint.jsonl)")

    # Signal description — the thing you're looking for
    parser.add_argument("--signal", type=str, default=DEFAULT_SIGNAL,
                        help="Natural language description of the signal to detect")

    # Tuning
    parser.add_argument("--concurrency", type=int, default=64,
                        help="Concurrent requests (text-only is lighter, can go higher)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens for judge response (only needs ~50)")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only judge first N traces (for testing)")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--min-thinking-words", type=int, default=5,
                        help="Skip traces shorter than this many words")

    args = parser.parse_args()
    server_url = args.server.rstrip("/")

    # Load traces
    samples = load_traces(
        parquet_path=args.input,
        limit=args.limit,
        min_thinking_words=args.min_thinking_words,
    )

    if not samples:
        log.error("No samples loaded. Check the input parquet.")
        sys.exit(1)

    # Setup checkpoint
    checkpoint_path = f"{args.output}.checkpoint.jsonl"
    parquet_path = f"{args.output}.parquet"

    ckpt = CheckpointManager(checkpoint_path)
    remaining = ckpt.filter_remaining(samples)

    if not remaining:
        log.info("All samples already judged. Converting to Parquet ...")
        df = ckpt.to_parquet(parquet_path)
        if df is not None:
            _print_summary(df)
        return

    total = len(samples)
    done = total - len(remaining)
    log.info(f"Total: {total} | Done: {done} | Remaining: {len(remaining)}")
    log.info(f"  Server:      {server_url}")
    log.info(f"  Model:       {args.model_name}")
    log.info(f"  Signal:      {args.signal}")
    log.info(f"  Concurrency: {args.concurrency}")
    log.info(f"  Checkpoint:  {checkpoint_path}")

    ckpt.open()
    try:
        asyncio.run(run_batch(
            remaining, server_url, args.model_name,
            args.signal, args.max_tokens, args.concurrency,
            args.timeout, ckpt,
        ))

        if args.retries > 0:
            asyncio.run(retry_failures(
                samples, server_url, args.model_name,
                args.signal, args.max_tokens, args.concurrency,
                args.timeout, ckpt, args.retries,
            ))
    except KeyboardInterrupt:
        log.info("Interrupted. Progress saved. Re-run to resume.")
    finally:
        ckpt.close()

    log.info("Converting checkpoint → Parquet ...")
    df = ckpt.to_parquet(parquet_path)
    if df is not None:
        _print_summary(df)


def _print_summary(df: pd.DataFrame):
    """Print a summary of the judge results."""
    ok_df = df[df["status"] == "ok"]
    failed = (df["status"] != "ok").sum()

    log.info(f"Done: {len(ok_df)} judged, {failed} failed")

    if len(ok_df) == 0:
        return

    # Signal stats
    signal_true = ok_df["signal_present"].sum()
    signal_false = (ok_df["signal_present"] == False).sum()
    signal_none = ok_df["signal_present"].isna().sum()
    parse_ok = ok_df["parse_ok"].sum() if "parse_ok" in ok_df.columns else "N/A"

    log.info(f"  Signal present:  {signal_true} ({100*signal_true/len(ok_df):.1f}%)")
    log.info(f"  Signal absent:   {signal_false} ({100*signal_false/len(ok_df):.1f}%)")
    log.info(f"  Unparseable:     {signal_none}")
    log.info(f"  Clean JSON parse: {parse_ok}/{len(ok_df)}")
    log.info(f"  Avg latency:     {ok_df['latency_s'].mean():.2f}s")

    # Breakdown by inpainted vs original if both are present
    if "inpainted" in ok_df.columns and ok_df["inpainted"].nunique() > 1:
        for label, group in ok_df.groupby("inpainted"):
            n_signal = group["signal_present"].sum()
            log.info(
                f"  inpainted={label}: {n_signal}/{len(group)} "
                f"({100*n_signal/len(group):.1f}%) mention signal"
            )


if __name__ == "__main__":
    main()