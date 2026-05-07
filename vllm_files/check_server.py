"""
check_server.py — Verify a vLLM server is running and test with a single image.

Usage:
    python check_server.py --server http://127.0.0.1:12345
"""

import argparse
import sys
import time

import requests


def check_health(server_url: str) -> bool:
    try:
        resp = requests.get(f"{server_url}/health", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def check_models(server_url: str) -> str | None:
    try:
        resp = requests.get(f"{server_url}/v1/models", timeout=5)
        data = resp.json()
        models = [m["id"] for m in data.get("data", [])]
        return models[0] if models else None
    except Exception:
        return None


def smoke_test(server_url: str, model: str, image_url: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "What do you see in this image? Describe it briefly."},
                ],
            }
        ],
        "max_tokens": 4096,
    }

    t0 = time.perf_counter()
    resp = requests.post(f"{server_url}/v1/chat/completions", json=payload, timeout=300)
    latency = time.perf_counter() - t0

    if resp.status_code != 200:
        print(f"ERROR: Server returned {resp.status_code}")
        print(resp.text[:1000])
        sys.exit(1)

    return resp.json(), latency


def main():
    parser = argparse.ArgumentParser(description="Check vLLM server health")
    parser.add_argument("--server", type=str, required=True)
    parser.add_argument(
        "--image", type=str,
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    )
    args = parser.parse_args()
    server_url = args.server.rstrip("/")

    print(f"[1/3] Health check: {server_url} ...")
    if check_health(server_url):
        print("      OK")
    else:
        print("      FAIL — server not responding")
        sys.exit(1)

    print("[2/3] Checking models ...")
    model = check_models(server_url)
    if model:
        print(f"      OK — {model}")
    else:
        print("      FAIL — no models loaded")
        sys.exit(1)

    print("[3/3] Smoke test ...")
    result, latency = smoke_test(server_url, model, args.image)

    choice = result["choices"][0]["message"]
    thinking = choice.get("reasoning", "") or ""
    answer = choice.get("content", "") or ""
    usage = result.get("usage", {})

    print(f"      OK — {latency:.1f}s")
    print(f"      Prompt tokens:     {usage.get('prompt_tokens', '?')}")
    print(f"      Completion tokens: {usage.get('completion_tokens', '?')}")
    print()
    print("=" * 60)
    print("THINKING TRACE (first 800 chars)")
    print("=" * 60)
    print(thinking[:800] if thinking else "(empty — check --reasoning-parser)")
    print()
    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)
    print()
    print("Ready for batch inference.")


if __name__ == "__main__":
    main()