#!/usr/bin/env python3
"""P0 streaming smoke test against a local OpenAI-compatible vLLM server.

Timings are environment evidence only. They are not benchmark results.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_PROMPT = "Reply with exactly this sentence: P0 smoke ok."
DEFAULT_MAX_TOKENS = 32
DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "p0" / "smoke.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P0 OpenAI-compatible streaming smoke test")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def iter_sse_data(raw: bytes):
    text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        return
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                yield line[5:].strip()


def main() -> int:
    args = parse_args()
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": True,
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        args.url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    t0 = time.perf_counter()
    request_send_s = 0.0
    first_content_s = None
    completion_s = None
    chunks: list[str] = []
    chunk_count = 0
    success = False
    error = None
    reported_model = None

    print(f"POST {args.url}", flush=True)
    print(f"model={args.model} stream=true max_tokens={args.max_tokens}", flush=True)
    print("--- stream ---", flush=True)

    try:
        request_send_s = time.perf_counter() - t0
        with urllib.request.urlopen(request, timeout=180) as response:
            while True:
                raw_line = response.readline()
                if not raw_line:
                    break
                for data in iter_sse_data(raw_line):
                    if data == "[DONE]":
                        completion_s = time.perf_counter() - t0
                        continue
                    event = json.loads(data)
                    if reported_model is None:
                        reported_model = event.get("model")
                    delta = event.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        if first_content_s is None:
                            first_content_s = time.perf_counter() - t0
                        chunks.append(content)
                        chunk_count += 1
                        print(content, end="", flush=True)
        if completion_s is None:
            completion_s = time.perf_counter() - t0
        success = chunk_count > 0
        print(flush=True)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        completion_s = time.perf_counter() - t0
        error = f"{type(exc).__name__}: {exc}"
        print(f"\nERROR {error}", flush=True)

    e2e = completion_s if completion_s is not None else None
    record = {
        "model": reported_model or args.model,
        "backend": "vllm",
        "backend_version": "pending",
        "streaming": True,
        "success": success,
        "ttft_seconds": first_content_s,
        "e2e_seconds": e2e,
        "request_send_seconds": request_send_s,
        "first_streamed_content_seconds": first_content_s,
        "completion_seconds": completion_s,
        "streamed_chunk_count": chunk_count,
        "streamed_text": "".join(chunks),
        "error": error,
    }

    try:
        record["backend_version"] = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        record["backend_version"] = "unknown: vllm package metadata not found"

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print("--- summary ---", flush=True)
    print(f"success={success}", flush=True)
    print(f"model={record['model']}", flush=True)
    print(f"chunks={chunk_count}", flush=True)
    print(f"request_send_s={request_send_s:.4f}", flush=True)
    print(
        f"first_content_s={first_content_s:.4f}" if first_content_s is not None else "first_content_s=not observed",
        flush=True,
    )
    print(f"completion_s={e2e:.4f}" if e2e is not None else "completion_s=not observed", flush=True)
    print(f"artifact={args.artifact}", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
