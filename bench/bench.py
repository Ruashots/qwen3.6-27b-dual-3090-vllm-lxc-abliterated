#!/usr/bin/env python3
"""
vLLM concurrency benchmark — fires N identical structured-JSON requests in parallel,
reports per-stream and aggregate t/s. Same script used in the base-model recipe.

Configure via env vars:
  VLLM_URL    — base URL of the OpenAI-compatible endpoint
                (default: http://127.0.0.1:8000)
  VLLM_MODEL  — model id (default: shawnw3i/Huihui-Qwen3.6-27B-abliterated-AWQ-MTP)

Example:
  VLLM_URL=http://192.168.1.64:8000 python3 bench.py shawnw3i_uncapped results.json
"""
import json
import os
import sys
import time
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000").rstrip("/")
API = f"{BASE_URL}/v1/chat/completions"
MODEL = os.environ.get("VLLM_MODEL", "shawnw3i/Huihui-Qwen3.6-27B-abliterated-AWQ-MTP")

PAYLOAD = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": (
            "Generate a JSON array of exactly 12 fictional employees. "
            "Each employee must have: id (integer), name (string), department "
            "(string), salary (integer), hire_date (YYYY-MM-DD string), and "
            "skills (array of 3 strings). Output ONLY the JSON array, no prose."
        )}
    ],
    "temperature": 0.7,
    "max_tokens": 1024,
    "chat_template_kwargs": {"enable_thinking": False},
}


def fire_one(idx):
    body = json.dumps(PAYLOAD).encode()
    req = Request(API, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    dt = time.perf_counter() - t0
    ct = data.get("usage", {}).get("completion_tokens", 0)
    pt = data.get("usage", {}).get("prompt_tokens", 0)
    msg = data["choices"][0]["message"]
    content = msg.get("content") or msg.get("reasoning_content") or ""
    return {
        "idx": idx,
        "elapsed_s": dt,
        "completion_tokens": ct,
        "prompt_tokens": pt,
        "tps": ct / dt if dt > 0 else 0,
        "preview": content[:140].replace("\n", " "),
    }


def run_concurrent(n):
    print(f"\n=== concurrency = {n} ===")
    t_wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(fire_one, i) for i in range(n)]
        results = [f.result() for f in as_completed(futures)]
    t_wall = time.perf_counter() - t_wall_start
    total_ct = sum(r["completion_tokens"] for r in results)
    agg_tps = total_ct / t_wall
    per_stream_avg = sum(r["tps"] for r in results) / len(results)
    print(f"wall: {t_wall:.2f}s   total completion tokens: {total_ct}")
    print(f"AGGREGATE: {agg_tps:.1f} t/s   AVG per-stream: {per_stream_avg:.1f} t/s")
    for r in sorted(results, key=lambda x: x["idx"]):
        print(
            f"  [{r['idx']}] {r['elapsed_s']:5.2f}s  "
            f"{r['completion_tokens']:4d}tok  {r['tps']:5.1f} t/s  "
            f"prev={r['preview']!r}"
        )
    return {
        "n": n,
        "wall": t_wall,
        "total_ct": total_ct,
        "agg_tps": agg_tps,
        "per_stream_avg": per_stream_avg,
        "results": results,
    }


def warmup():
    print("warmup (1 request to prime caches)...", flush=True)
    r = fire_one(-1)
    print(f"  warmup: {r['elapsed_s']:.2f}s  {r['completion_tokens']}tok  {r['tps']:.1f} t/s")


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "unlabeled"
    out_path = sys.argv[2] if len(sys.argv) > 2 else f"bench_{label}.json"
    print(f"=== bench label={label}  api={API}  model={MODEL} ===")
    warmup()
    runs = []
    for n in (1, 2, 4):
        runs.append(run_concurrent(n))
        time.sleep(2)
    with open(out_path, "w") as f:
        json.dump({"label": label, "runs": runs, "api": API, "model": MODEL}, f, indent=2)
    print(f"\n=== summary ({label}) ===")
    for r in runs:
        print(
            f"  c={r['n']:<2d}  agg={r['agg_tps']:6.1f} t/s   "
            f"per-stream-avg={r['per_stream_avg']:5.1f} t/s   wall={r['wall']:.2f}s"
        )
    print(f"\nwrote: {out_path}")
