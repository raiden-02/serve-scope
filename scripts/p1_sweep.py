#!/usr/bin/env python3
"""P1 open-loop saturation sweep against a local vLLM server.

This is a measurement harness. It is not a scheduler and it is not a benchmark
framework for later workload classes.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.metadata
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from servescope.client import build_chat_payload, stream_chat_request
from servescope.metrics import (
    METRIC_KV,
    METRIC_RUNNING,
    METRIC_WAITING,
    aggregate_repeats,
    first_slo_violation,
    parse_prometheus_gauges,
    summarize_repeat,
)
from servescope.workload import scheduled_arrivals, select_prompt


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def git_state(repo: Path) -> dict:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=repo, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(run(["git", "status", "--porcelain"])),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def nvidia_smi_snapshot() -> dict | None:
    try:
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        name, util, used, total = [part.strip() for part in text.split(",")]
        return {
            "name": name,
            "utilization_gpu_pct": float(util),
            "memory_used_mib": float(used),
            "memory_total_mib": float(total),
        }
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def nvidia_smi_processes() -> str:
    try:
        return subprocess.check_output(["nvidia-smi"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nvidia-smi unavailable"


async def fetch_runtime_gauges(client: httpx.AsyncClient, url: str) -> dict[str, float]:
    try:
        response = await client.get(url, timeout=2.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return {}
    return parse_prometheus_gauges(
        response.text, {METRIC_RUNNING, METRIC_WAITING, METRIC_KV}
    )


async def wait_until_idle(client: httpx.AsyncClient, config: dict) -> bool:
    metrics_url = config["base_url"].rstrip("/") + config["metrics_path"]
    needed = int(config["drain_idle_needed"])
    timeout_s = float(config["drain_timeout_s"])
    poll_s = float(config["drain_poll_s"])
    idle = 0
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        gauges = await fetch_runtime_gauges(client, metrics_url)
        running = gauges.get(METRIC_RUNNING)
        waiting = gauges.get(METRIC_WAITING)
        if running == 0.0 and waiting == 0.0:
            idle += 1
            if idle >= needed:
                return True
        else:
            idle = 0
        await asyncio.sleep(poll_s)
    return False


async def warmup(client: httpx.AsyncClient, config: dict) -> None:
    url = config["base_url"].rstrip("/") + config["chat_path"]
    for i in range(int(config["warmup_requests"])):
        prompt_id, prompt = select_prompt(int(config["seed"]), i)
        payload = build_chat_payload(
            model=config["model"],
            prompt=prompt,
            temperature=config["temperature"],
            min_tokens=config["min_tokens"],
            max_completion_tokens=config["max_completion_tokens"],
        )
        record = await stream_chat_request(
            client,
            url=url,
            payload=payload,
            request_id=f"warmup-{i}",
            prompt_id=prompt_id,
            workload_class=config["workload_class"],
            scheduled_s=time.perf_counter(),
            timeout_s=float(config["request_timeout_s"]),
            clock=time.perf_counter,
        )
        print(f"warmup {i} status={record['status']} ttft={record['client_ttft_s']}", flush=True)


async def sample_runtime(
    client: httpx.AsyncClient,
    url: str,
    interval_s: float,
    stop: asyncio.Event,
    out: list[dict],
    t0: float,
) -> None:
    while not stop.is_set():
        gauges = await fetch_runtime_gauges(client, url)
        out.append(
            {
                "t_s": time.perf_counter() - t0,
                "num_requests_running": gauges.get(METRIC_RUNNING),
                "num_requests_waiting": gauges.get(METRIC_WAITING),
                "kv_cache_usage_perc": gauges.get(METRIC_KV),
            }
        )
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except TimeoutError:
            continue


async def sample_gpu(interval_s: float, stop: asyncio.Event, out: list[dict], t0: float) -> None:
    while not stop.is_set():
        snap = await asyncio.to_thread(nvidia_smi_snapshot)
        if snap is not None:
            out.append({"t_s": time.perf_counter() - t0, **snap})
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except TimeoutError:
            continue


async def run_open_loop(
    client: httpx.AsyncClient,
    config: dict,
    *,
    suite_id: str,
    repeat_id: str,
    offered_rps: float,
    n_requests: int,
    clock,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    url = config["base_url"].rstrip("/") + config["chat_path"]
    metrics_url = config["base_url"].rstrip("/") + config["metrics_path"]
    max_inflight = int(config["max_inflight"])
    t0 = clock()
    targets = scheduled_arrivals(n_requests, offered_rps, t0)
    records: list[dict] = []
    runtime_rows: list[dict] = []
    gpu_rows: list[dict] = []
    stop = asyncio.Event()
    runtime_task = asyncio.create_task(
        sample_runtime(client, metrics_url, float(config["runtime_sample_interval_s"]), stop, runtime_rows, t0)
    )
    gpu_task = asyncio.create_task(
        sample_gpu(float(config["gpu_sample_interval_s"]), stop, gpu_rows, t0)
    )
    pending: set[asyncio.Task] = set()
    aborted = False
    abort_reason = None

    async def launch(i: int, scheduled_s: float) -> None:
        prompt_id, prompt = select_prompt(int(config["seed"]), i)
        payload = build_chat_payload(
            model=config["model"],
            prompt=prompt,
            temperature=config["temperature"],
            min_tokens=config["min_tokens"],
            max_completion_tokens=config["max_completion_tokens"],
        )
        record = await stream_chat_request(
            client,
            url=url,
            payload=payload,
            request_id=f"{repeat_id}-r{i:05d}",
            prompt_id=prompt_id,
            workload_class=config["workload_class"],
            scheduled_s=scheduled_s,
            timeout_s=float(config["request_timeout_s"]),
            clock=clock,
        )
        record.update(
            {
                "suite_id": suite_id,
                "repeat_id": repeat_id,
                "offered_rps": offered_rps,
                "request_index": i,
            }
        )
        records.append(record)

    for i, scheduled_s in enumerate(targets):
        now = clock()
        if now < scheduled_s:
            await asyncio.sleep(scheduled_s - now)
        if len(pending) >= max_inflight:
            aborted = True
            abort_reason = f"in-flight reached {max_inflight}; aborting rather than closing the loop"
            break
        task = asyncio.create_task(launch(i, scheduled_s))
        pending.add(task)
        task.add_done_callback(pending.discard)

    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    stop.set()
    await asyncio.gather(runtime_task, gpu_task)
    duration_s = clock() - t0
    records.sort(key=lambda row: row.get("request_index", 0))
    meta = {"aborted": aborted, "abort_reason": abort_reason, "duration_s": duration_s}
    return records, runtime_rows, gpu_rows, meta


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_ttft(path: Path, aggregates: list[dict], slo_s: float) -> None:
    xs = [row["offered_rps"] for row in aggregates]
    p50 = [row.get("headline_p50_ttft_s_median") for row in aggregates]
    p95 = [row.get("headline_p95_ttft_s_median") for row in aggregates]
    p95_min = [row.get("headline_p95_ttft_s_min") for row in aggregates]
    p95_max = [row.get("headline_p95_ttft_s_max") for row in aggregates]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(xs, p50, marker="o", label="p50 client TTFT")
    ax.plot(xs, p95, marker="o", label="p95 client TTFT (median repeat)")
    if all(v is not None for v in p95_min + p95_max):
        ax.fill_between(xs, p95_min, p95_max, color="tab:orange", alpha=0.2, label="p95 repeat range")
    ax.axhline(slo_s, color="red", linestyle="--", label=f"SLO p95 TTFT = {slo_s:.0f} s")
    ax.set_xlabel("offered requests / sec")
    ax.set_ylabel("client TTFT (seconds)")
    ax.set_title("Client TTFT vs offered load")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_throughput(path: Path, aggregates: list[dict]) -> None:
    xs = [row["offered_rps"] for row in aggregates]
    tokens = [row.get("headline_output_token_tps_median") for row in aggregates]
    reqs = [row.get("headline_completed_rps_median") for row in aggregates]
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.plot(xs, tokens, marker="o", color="tab:blue", label="output tokens / sec")
    ax1.set_xlabel("offered requests / sec")
    ax1.set_ylabel("output tokens / sec")
    ax2 = ax1.twinx()
    ax2.plot(xs, reqs, marker="s", color="tab:green", label="completed requests / sec")
    ax2.set_ylabel("completed requests / sec")
    ax1.set_title("Throughput vs offered load")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="best")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def package_versions() -> dict[str, str]:
    versions = {}
    for name in ("vllm", "torch", "httpx", "numpy", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


async def run_suite(config: dict, rates: list[float], repeats: int, duration_s: float, out_dir: Path, mode: str) -> dict:
    suite_id = out_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    limits = httpx.Limits(
        max_connections=int(config["client_max_connections"]),
        max_keepalive_connections=int(config["client_max_connections"]),
    )
    timeout = httpx.Timeout(float(config["request_timeout_s"]))
    start = datetime.now(timezone.utc).isoformat()
    pre_gpu = nvidia_smi_snapshot()
    pre_smi = nvidia_smi_processes()
    (out_dir / "nvidia_smi_before.txt").write_text(pre_smi, encoding="utf-8")
    all_repeat_summaries: list[dict] = []
    all_runtime: list[dict] = []
    all_gpu: list[dict] = []
    aggregates: list[dict] = []

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        print("warmup", flush=True)
        await warmup(client, config)
        idle = await wait_until_idle(client, config)
        print(f"post-warmup idle={idle}", flush=True)
        await asyncio.sleep(float(config["inter_run_idle_s"]))

        for rate in rates:
            rate_summaries: list[dict] = []
            for repeat in range(1, repeats + 1):
                repeat_id = f"{mode}-rps{rate:g}-rep{repeat}"
                print(f"=== {repeat_id} n~{max(8, int(rate * duration_s))} ===", flush=True)
                idle = await wait_until_idle(client, config)
                if not idle:
                    print("WARNING: queue did not return to idle before run", flush=True)
                n_requests = max(8, int(round(rate * duration_s)))
                records, runtime_rows, gpu_rows, meta = await run_open_loop(
                    client,
                    config,
                    suite_id=suite_id,
                    repeat_id=repeat_id,
                    offered_rps=rate,
                    n_requests=n_requests,
                    clock=time.perf_counter,
                )
                write_jsonl(out_dir / "requests.jsonl", records)
                for row in runtime_rows:
                    row.update({"repeat_id": repeat_id, "offered_rps": rate})
                for row in gpu_rows:
                    row.update({"repeat_id": repeat_id, "offered_rps": rate})
                all_runtime.extend(runtime_rows)
                all_gpu.extend(gpu_rows)
                waiting_vals = [row["num_requests_waiting"] for row in runtime_rows if row.get("num_requests_waiting") is not None]
                kv_vals = [row["kv_cache_usage_perc"] for row in runtime_rows if row.get("kv_cache_usage_perc") is not None]
                summary = summarize_repeat(records, offered_rps=rate, duration_s=meta["duration_s"])
                summary.update(
                    {
                        "suite_id": suite_id,
                        "repeat_id": repeat_id,
                        "aborted": meta["aborted"],
                        "abort_reason": meta["abort_reason"],
                        "max_waiting_requests": max(waiting_vals) if waiting_vals else None,
                        "typical_waiting_requests": (
                            sorted(waiting_vals)[len(waiting_vals) // 2] if waiting_vals else None
                        ),
                        "max_kv_cache_usage_perc": max(kv_vals) if kv_vals else None,
                    }
                )
                if meta["aborted"]:
                    summary["client_limited"] = True
                rate_summaries.append(summary)
                all_repeat_summaries.append(summary)
                print(
                    f"{repeat_id} completed={summary['completed_count']}/{summary['request_count']} "
                    f"ttft_p95={summary['client_ttft_p95_s']} client_limited={summary['client_limited']}",
                    flush=True,
                )
                idle = await wait_until_idle(client, config)
                print(f"drain idle={idle}", flush=True)
                await asyncio.sleep(float(config["inter_run_idle_s"]))
            aggregates.append(aggregate_repeats(rate_summaries))

    write_csv(out_dir / "repeat_summary.csv", all_repeat_summaries)
    write_csv(out_dir / "aggregate_summary.csv", aggregates)
    write_csv(out_dir / "runtime_metrics.csv", all_runtime)
    write_csv(out_dir / "gpu_metrics.csv", all_gpu)
    plot_ttft(out_dir / "ttft_vs_load.png", aggregates, float(config["slo_p95_ttft_seconds"]))
    plot_throughput(out_dir / "throughput_vs_load.png", aggregates)
    versions = package_versions()
    crossing = first_slo_violation(aggregates, float(config["slo_p95_ttft_seconds"]))
    end = datetime.now(timezone.utc).isoformat()
    manifest = {
        "suite_id": suite_id,
        "mode": mode,
        "git": git_state(ROOT),
        "model": config["model"],
        "backend": "vllm",
        "backend_version": versions.get("vllm"),
        "torch_version": versions.get("torch"),
        "torch_cuda_build": "13.0",
        "gpu": "NVIDIA GeForce RTX 4080 SUPER",
        "server_command": (
            "vllm serve Qwen/Qwen3-1.7B --gpu-memory-utilization 0.85 "
            "--enforce-eager --enable-per-request-metrics"
        ),
        "server_env": {
            "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "HF_HOME": "/home/mayur/.cache/huggingface",
        },
        "notes": [
            "WSLg/Xwayland shares the GPU with the vLLM process.",
            "vLLM is running with --enforce-eager. That is a known WSL limitation, not a tuned serving mode.",
            "P1 timings are measured on a controlled 64-token interactive workload. They are not general chat SLOs.",
            "prefix caching stays at the vLLM default (enabled).",
        ],
        "slo_p95_ttft_seconds": config["slo_p95_ttft_seconds"],
        "percentile_method": config["percentile_method"],
        "start_utc": start,
        "end_utc": end,
        "offered_rates_rps": rates,
        "repeats": repeats,
        "duration_s": duration_s,
        "pre_run_gpu": pre_gpu,
        "first_tested_slo_violation_rps": crossing,
        "aggregates": aggregates,
        "package_versions": versions,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "result.json").write_text(
        json.dumps(
            {
                "first_tested_slo_violation_rps": crossing,
                "slo": "p95 client TTFT < 1.0 s",
                "aggregates": aggregates,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1 open-loop saturation sweep")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "p1_short_chat.json")
    parser.add_argument("--mode", choices=("smoke", "pilot", "final"), required=True)
    parser.add_argument("--rates", default=None, help="comma-separated offered RPS list")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "smoke":
        rates = [2.0]
        repeats = 1
        duration_s = 8.0
    elif args.mode == "pilot":
        rates = [float(x) for x in config["pilot_rates_rps"]]
        repeats = 1
        duration_s = float(config["pilot_duration_s"])
    else:
        rates = [float(x) for x in args.rates.split(",")] if args.rates else [2.0, 4.0, 8.0, 16.0]
        repeats = int(config["repeats"])
        duration_s = float(config["final_duration_s"])
    if args.rates and args.mode != "final":
        rates = [float(x) for x in args.rates.split(",")]
    stamp = now_utc()
    out = args.out or (ROOT / "artifacts" / "p1" / f"{args.mode}-{stamp}")
    print(f"writing {out}", flush=True)
    asyncio.run(run_suite(config, rates, repeats, duration_s, out, args.mode))
    print(f"done {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
