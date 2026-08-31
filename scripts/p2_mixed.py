#!/usr/bin/env python3
"""Mixed-workload interference harness.

Interactive traffic stays at a healthy offered load. A timed background burst
is injected onto the same default vLLM scheduler.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.metadata
import json
import resource
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
    parse_prometheus_gauges,
    percentile,
)
from servescope.p2_metrics import (
    CLASS_BACKGROUND,
    CLASS_INTERACTIVE,
    PHASE_BURST,
    PHASE_PRE,
    PHASE_RECOVERY,
    PHASES,
    absolute_ttft_increase,
    attach_phase,
    background_goodput,
    bin_interactive_ttft,
    class_schedule_summary,
    filter_class,
    filter_phase,
    interference_ratio,
    last_completion_rel_s,
    mark_class_validity,
    runtime_phase_summary,
    slo_flags,
    summarize_phase_latencies,
    sustained_waiting_queue,
)
from servescope.workload import select_background_prompt, select_prompt, windowed_arrivals


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
    return parse_prometheus_gauges(response.text, {METRIC_RUNNING, METRIC_WAITING, METRIC_KV})


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
        idle = 0 if not (running == 0.0 and waiting == 0.0) else idle
        await asyncio.sleep(poll_s)
    return False


async def warmup(client: httpx.AsyncClient, config: dict) -> None:
    url = config["base_url"].rstrip("/") + config["chat_path"]
    interactive = config["interactive"]
    for i in range(int(config["warmup_requests"])):
        prompt_id, prompt = select_prompt(int(config["seed"]), i)
        payload = build_chat_payload(
            model=config["model"],
            prompt=prompt,
            temperature=interactive["temperature"],
            min_tokens=interactive["min_tokens"],
            max_completion_tokens=interactive["max_completion_tokens"],
        )
        record = await stream_chat_request(
            client,
            url=url,
            payload=payload,
            request_id=f"warmup-{i}",
            prompt_id=prompt_id,
            workload_class=CLASS_INTERACTIVE,
            scheduled_s=time.perf_counter(),
            timeout_s=float(config["request_timeout_s"]),
            clock=time.perf_counter,
        )
        print(f"warmup {i} status={record['status']} ttft={record['client_ttft_s']}", flush=True)


def make_client(config: dict, class_cfg: dict) -> httpx.AsyncClient:
    connections = int(class_cfg["client_max_connections"])
    limits = httpx.Limits(max_connections=connections, max_keepalive_connections=connections)
    timeout = httpx.Timeout(
        connect=30.0,
        read=float(config["request_timeout_s"]),
        write=30.0,
        pool=2.0,
    )
    return httpx.AsyncClient(limits=limits, timeout=timeout)


async def sample_runtime(client, url, interval_s, stop, out, t0):
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


async def sample_gpu(interval_s, stop, out, t0):
    while not stop.is_set():
        snap = await asyncio.to_thread(nvidia_smi_snapshot)
        if snap is not None:
            out.append({"t_s": time.perf_counter() - t0, **snap})
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except TimeoutError:
            continue


class Inflight:
    def __init__(self, limit: int):
        self.limit = limit
        self.count = 0
        self.peak = 0

    def try_acquire(self) -> bool:
        if self.count >= self.limit:
            return False
        self.count += 1
        self.peak = max(self.peak, self.count)
        return True

    def release(self) -> None:
        self.count = max(0, self.count - 1)


async def offer_class(
    client: httpx.AsyncClient,
    config: dict,
    class_cfg: dict,
    *,
    suite_id: str,
    repeat_id: str,
    scenario: str,
    t0: float,
    targets: list[float],
    clock,
    records: list[dict],
    inflight: Inflight,
    prompt_fn,
    seed: int,
    request_priority: int | None = None,
    scheduler_policy: str = "fcfs",
    send_priority_field: bool = False,
) -> dict:
    url = config["base_url"].rstrip("/") + config["chat_path"]
    workload_class = class_cfg["workload_class"]
    pending: set[asyncio.Task] = set()
    aborted = False
    abort_reason = None
    pre_end = float(config["pre_burst_end_s"])
    burst_end = float(config["burst_end_s"])

    async def launch(i: int, scheduled_s: float) -> None:
        prompt_id, prompt = prompt_fn(seed, i)
        payload = build_chat_payload(
            model=config["model"],
            prompt=prompt,
            temperature=class_cfg["temperature"],
            min_tokens=class_cfg["min_tokens"],
            max_completion_tokens=class_cfg["max_completion_tokens"],
            priority=request_priority if send_priority_field else None,
        )
        try:
            record = await stream_chat_request(
                client,
                url=url,
                payload=payload,
                request_id=f"{repeat_id}-{workload_class}-r{i:05d}",
                prompt_id=prompt_id,
                workload_class=workload_class,
                scheduled_s=scheduled_s,
                timeout_s=float(config["request_timeout_s"]),
                clock=clock,
            )
        finally:
            inflight.release()
        attach_phase(record, t0, pre_end_s=pre_end, burst_end_s=burst_end)
        record.update(
            {
                "suite_id": suite_id,
                "repeat_id": repeat_id,
                "scenario": scenario,
                "offered_rps": class_cfg.get("offered_rps"),
                "request_index": i,
                "scheduler_policy": scheduler_policy,
                "request_priority": 0 if request_priority is None else int(request_priority),
            }
        )
        records.append(record)

    for i, scheduled_s in enumerate(targets):
        now = clock()
        if now < scheduled_s:
            await asyncio.sleep(scheduled_s - now)
        if not inflight.try_acquire():
            aborted = True
            abort_reason = f"{workload_class} in-flight reached {inflight.limit}; aborting rather than closing the loop"
            break
        task = asyncio.create_task(launch(i, scheduled_s))
        pending.add(task)
        task.add_done_callback(pending.discard)

    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    return {"aborted": aborted, "abort_reason": abort_reason, "peak_inflight": inflight.peak}


async def run_scenario(
    interactive_client: httpx.AsyncClient,
    background_client: httpx.AsyncClient,
    telemetry_client: httpx.AsyncClient,
    config: dict,
    *,
    suite_id: str,
    repeat_id: str,
    scenario: str,
    background_rps: float | None,
    clock,
    scheduler_policy: str = "fcfs",
    class_priorities: dict[str, int] | None = None,
    send_priority_field: bool = False,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    interactive_cfg = dict(config["interactive"])
    background_cfg = dict(config["background"])
    duration_s = float(config["scenario_duration_s"])
    pre_end = float(config["pre_burst_end_s"])
    burst_end = float(config["burst_end_s"])
    t0 = clock()
    interactive_cfg["offered_rps"] = float(interactive_cfg["offered_rps"])
    interactive_targets = windowed_arrivals(interactive_cfg["offered_rps"], t0, 0.0, duration_s)
    background_targets: list[float] = []
    if scenario == "mixed" and background_rps:
        background_cfg["offered_rps"] = float(background_rps)
        background_targets = windowed_arrivals(float(background_rps), t0, pre_end, burst_end)
    else:
        background_cfg["offered_rps"] = 0.0

    records: list[dict] = []
    runtime_rows: list[dict] = []
    gpu_rows: list[dict] = []
    stop = asyncio.Event()
    metrics_url = config["base_url"].rstrip("/") + config["metrics_path"]
    runtime_task = asyncio.create_task(
        sample_runtime(
            telemetry_client,
            metrics_url,
            float(config["runtime_sample_interval_s"]),
            stop,
            runtime_rows,
            t0,
        )
    )
    gpu_task = asyncio.create_task(
        sample_gpu(float(config["gpu_sample_interval_s"]), stop, gpu_rows, t0)
    )
    interactive_inflight = Inflight(int(interactive_cfg["max_inflight"]))
    background_inflight = Inflight(int(background_cfg["max_inflight"]))
    interactive_seed = int(config["seed"])
    background_seed = int(config["seed"]) + 1_000_000

    interactive_offer = asyncio.create_task(
        offer_class(
            interactive_client,
            config,
            interactive_cfg,
            suite_id=suite_id,
            repeat_id=repeat_id,
            scenario=scenario,
            t0=t0,
            targets=interactive_targets,
            clock=clock,
            records=records,
            inflight=interactive_inflight,
            prompt_fn=select_prompt,
            seed=interactive_seed,
            request_priority=(class_priorities or {}).get(CLASS_INTERACTIVE, 0),
            scheduler_policy=scheduler_policy,
            send_priority_field=send_priority_field,
        )
    )
    if background_targets:
        background_offer = asyncio.create_task(
            offer_class(
                background_client,
                config,
                background_cfg,
                suite_id=suite_id,
                repeat_id=repeat_id,
                scenario=scenario,
                t0=t0,
                targets=background_targets,
                clock=clock,
                records=records,
                inflight=background_inflight,
                prompt_fn=select_background_prompt,
                seed=background_seed,
                request_priority=(class_priorities or {}).get(CLASS_BACKGROUND, 0),
                scheduler_policy=scheduler_policy,
                send_priority_field=send_priority_field,
            )
        )
        interactive_meta, background_meta = await asyncio.gather(interactive_offer, background_offer)
    else:
        interactive_meta = await interactive_offer
        background_meta = {"aborted": False, "abort_reason": None, "peak_inflight": 0}

    stop.set()
    await asyncio.gather(runtime_task, gpu_task)
    duration_wall = clock() - t0
    records.sort(key=lambda row: (row.get("workload_class") or "", row.get("request_index") or 0))
    bg_rows = filter_class(records, CLASS_BACKGROUND)
    meta = {
        "t0": t0,
        "aborted": bool(interactive_meta["aborted"] or background_meta["aborted"]),
        "abort_reason": interactive_meta["abort_reason"] or background_meta["abort_reason"],
        "duration_s": duration_wall,
        "peak_inflight_interactive": interactive_meta["peak_inflight"],
        "peak_inflight_background": background_meta["peak_inflight"],
        "last_background_completion_s": last_completion_rel_s(bg_rows, t0),
        "background_offered_rps": background_cfg.get("offered_rps") or 0.0,
        "interactive_offered_count": len(interactive_targets),
        "background_offered_count": len(background_targets),
    }
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


def package_versions() -> dict[str, str]:
    versions = {}
    for name in ("vllm", "torch", "httpx", "numpy", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def summarize_scenario(
    records: list[dict],
    runtime_rows: list[dict],
    gpu_rows: list[dict],
    meta: dict,
    config: dict,
    *,
    suite_id: str,
    repeat_id: str,
    scenario: str,
    background_rps: float | None,
) -> dict:
    slo_s = float(config["slo_p95_ttft_seconds"])
    duration_s = float(config["scenario_duration_s"])
    pre_end = float(config["pre_burst_end_s"])
    burst_end = float(config["burst_end_s"])
    interactive = filter_class(records, CLASS_INTERACTIVE)
    background = filter_class(records, CLASS_BACKGROUND)
    interactive_summary = class_schedule_summary(interactive, float(config["interactive"]["offered_rps"]), meta["duration_s"])
    interactive_summary.update(
        {
            "aborted": meta["aborted"] and (meta.get("abort_reason") or "").startswith(CLASS_INTERACTIVE),
            "abort_reason": meta["abort_reason"] if (meta.get("abort_reason") or "").startswith(CLASS_INTERACTIVE) else None,
            "peak_inflight": meta["peak_inflight_interactive"],
        }
    )
    if meta["aborted"] and (meta.get("abort_reason") or "").startswith(CLASS_INTERACTIVE):
        interactive_summary["aborted"] = True
        interactive_summary["abort_reason"] = meta["abort_reason"]
    mark_class_validity(interactive_summary, max_inflight=int(config["interactive"]["max_inflight"]))

    bg_offered = float(background_rps or 0.0)
    background_summary = class_schedule_summary(background, bg_offered if bg_offered else 1.0, meta["duration_s"])
    background_summary.update(
        {
            "aborted": meta["aborted"] and (meta.get("abort_reason") or "").startswith(CLASS_BACKGROUND),
            "abort_reason": meta["abort_reason"] if (meta.get("abort_reason") or "").startswith(CLASS_BACKGROUND) else None,
            "peak_inflight": meta["peak_inflight_background"],
            "offered_rps": bg_offered,
        }
    )
    if scenario == "mixed" and bg_offered > 0:
        mark_class_validity(background_summary, max_inflight=int(config["background"]["max_inflight"]))
    else:
        background_summary["valid_offered_load"] = len(background) == 0
        background_summary["invalid_reason"] = None if len(background) == 0 else "control_had_background_requests"
        background_summary["client_limited"] = False

    if meta["aborted"]:
        if (meta.get("abort_reason") or "").startswith(CLASS_INTERACTIVE):
            interactive_summary["valid_offered_load"] = False
            interactive_summary["invalid_reason"] = meta["abort_reason"]
        if (meta.get("abort_reason") or "").startswith(CLASS_BACKGROUND):
            background_summary["valid_offered_load"] = False
            background_summary["invalid_reason"] = meta["abort_reason"]

    phase_rows = []
    interactive_phases = {}
    for phase in PHASES:
        phase_records = filter_phase(interactive, phase)
        phase_summary = summarize_phase_latencies(phase_records, slo_s=slo_s)
        phase_summary.update(
            {
                "suite_id": suite_id,
                "repeat_id": repeat_id,
                "scenario": scenario,
                "workload_class": CLASS_INTERACTIVE,
                "phase": phase,
                "background_offered_rps": bg_offered,
            }
        )
        interactive_phases[phase] = phase_summary
        phase_rows.append(phase_summary)

    runtime_phases = runtime_phase_summary(
        runtime_rows,
        pre_end_s=pre_end,
        burst_end_s=burst_end,
        scenario_end_s=duration_s,
    )
    waiting_vals = [row["num_requests_waiting"] for row in runtime_rows if row.get("num_requests_waiting") is not None]
    kv_vals = [row["kv_cache_usage_perc"] for row in runtime_rows if row.get("kv_cache_usage_perc") is not None]
    gpu_util = [row["utilization_gpu_pct"] for row in gpu_rows if row.get("utilization_gpu_pct") is not None]
    vram = [row["memory_used_mib"] for row in gpu_rows if row.get("memory_used_mib") is not None]
    bg_goodput = background_goodput(background, meta["t0"])
    flags = slo_flags(interactive_phases)
    valid = bool(interactive_summary.get("valid_offered_load"))
    if scenario == "mixed":
        valid = valid and bool(background_summary.get("valid_offered_load"))
    else:
        valid = valid and len(background) == 0

    summary = {
        "suite_id": suite_id,
        "repeat_id": repeat_id,
        "scenario": scenario,
        "background_offered_rps": bg_offered,
        "valid_offered_load": valid,
        "invalid_reason": None
        if valid
        else ", ".join(
            reason
            for reason in (
                interactive_summary.get("invalid_reason"),
                background_summary.get("invalid_reason") if scenario == "mixed" else None,
                "control_had_background_requests" if scenario == "control" and background else None,
            )
            if reason
        )
        or "invalid",
        "interactive_actual_dispatch_rps": interactive_summary.get("actual_dispatch_rps"),
        "interactive_median_dispatch_lag_s": interactive_summary.get("median_dispatch_lag_s"),
        "interactive_peak_inflight": meta["peak_inflight_interactive"],
        "interactive_client_limited": interactive_summary.get("client_limited"),
        "background_actual_dispatch_rps": background_summary.get("actual_dispatch_rps") if background else None,
        "background_median_dispatch_lag_s": background_summary.get("median_dispatch_lag_s") if background else None,
        "background_peak_inflight": meta["peak_inflight_background"],
        "background_client_limited": background_summary.get("client_limited") if background else False,
        "pooltimeout_count": (interactive_summary.get("status_counts") or {}).get("client_capacity", 0)
        + (background_summary.get("status_counts") or {}).get("client_capacity", 0),
        "aborted": meta["aborted"],
        "abort_reason": meta["abort_reason"],
        "interactive_p95_pre_s": interactive_phases[PHASE_PRE]["client_ttft_p95_s"],
        "interactive_p95_burst_s": interactive_phases[PHASE_BURST]["client_ttft_p95_s"],
        "interactive_p95_recovery_s": interactive_phases[PHASE_RECOVERY]["client_ttft_p95_s"],
        "interactive_p50_pre_s": interactive_phases[PHASE_PRE]["client_ttft_p50_s"],
        "interactive_p50_burst_s": interactive_phases[PHASE_BURST]["client_ttft_p50_s"],
        "interactive_p50_recovery_s": interactive_phases[PHASE_RECOVERY]["client_ttft_p50_s"],
        "max_waiting_requests": max(waiting_vals) if waiting_vals else None,
        "max_waiting_burst": runtime_phases[PHASE_BURST]["max_waiting_requests"],
        "max_running_requests": max(
            (row["num_requests_running"] for row in runtime_rows if row.get("num_requests_running") is not None),
            default=None,
        ),
        "max_kv_cache_usage_perc": max(kv_vals) if kv_vals else None,
        "max_gpu_util_pct": max(gpu_util) if gpu_util else None,
        "max_vram_mib": max(vram) if vram else None,
        "last_background_completion_s": meta["last_background_completion_s"],
        "background_completed_count": bg_goodput["completed_count"],
        "background_offered_count": bg_goodput["offered_count"],
        "background_output_token_goodput_tps": bg_goodput["wall_clock_output_token_tps"],
        "background_request_goodput_rps": bg_goodput["wall_clock_completed_request_tps"],
        "background_prompt_tokens_p50": bg_goodput["prompt_tokens_p50"],
        "background_prompt_tokens_min": bg_goodput["prompt_tokens_min"],
        "background_prompt_tokens_max": bg_goodput["prompt_tokens_max"],
        "background_output_tokens_sum": bg_goodput["output_tokens_sum"],
        "sustained_burst_queue": sustained_waiting_queue(runtime_phases[PHASE_BURST]),
        **flags,
        "interactive_status_counts": interactive_summary.get("status_counts"),
        "background_status_counts": background_summary.get("status_counts"),
        "configured_interactive_connections": config["interactive"]["client_max_connections"],
        "configured_background_connections": config["background"]["client_max_connections"],
        "configured_interactive_max_inflight": config["interactive"]["max_inflight"],
        "configured_background_max_inflight": config["background"]["max_inflight"],
    }
    return {
        "repeat": summary,
        "phase_rows": phase_rows,
        "runtime_phases": runtime_phases,
        "background_goodput": bg_goodput,
        "interactive_summary": interactive_summary,
        "background_summary": background_summary,
    }


def plot_ttft_timeline(path: Path, records: list[dict], config: dict, title: str) -> None:
    bins = bin_interactive_ttft(records)
    pre_end = float(config["pre_burst_end_s"])
    burst_end = float(config["burst_end_s"])
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    if bins:
        xs = [row["t_s"] + 0.5 for row in bins]
        ax.plot(xs, [row["p50"] for row in bins], marker=".", label="interactive p50 TTFT")
        ax.plot(xs, [row["p95"] for row in bins], marker=".", label="interactive p95 TTFT")
    ax.axvspan(pre_end, burst_end, color="tab:orange", alpha=0.18, label="background injection 15-30 s")
    ax.set_xlabel("seconds from scenario start")
    ax.set_ylabel("client TTFT (seconds)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_queue_timeline(path: Path, runtime_rows: list[dict], config: dict, title: str) -> None:
    pre_end = float(config["pre_burst_end_s"])
    burst_end = float(config["burst_end_s"])
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    xs = [row["t_s"] for row in runtime_rows]
    ax.plot(xs, [row.get("num_requests_waiting") for row in runtime_rows], label="vLLM waiting")
    ax.plot(xs, [row.get("num_requests_running") for row in runtime_rows], label="vLLM running")
    ax.axvspan(pre_end, burst_end, color="tab:orange", alpha=0.18, label="background injection 15-30 s")
    ax.set_xlabel("seconds from scenario start")
    ax.set_ylabel("requests")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_control_vs_mixed(path: Path, control_repeats: list[dict], mixed_repeats: list[dict]) -> None:
    labels = ["pre", "burst", "recovery"]
    keys = ["interactive_p95_pre_s", "interactive_p95_burst_s", "interactive_p95_recovery_s"]

    def median_of(rows: list[dict], key: str) -> float | None:
        values = [row[key] for row in rows if row.get(key) is not None]
        return percentile(values, 50) if values else None

    control_y = [median_of(control_repeats, key) for key in keys]
    mixed_y = [median_of(mixed_repeats, key) for key in keys]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    xs = range(len(labels))
    width = 0.35
    ax.bar([x - width / 2 for x in xs], [v if v is not None else 0 for v in control_y], width, label="control p95")
    ax.bar([x + width / 2 for x in xs], [v if v is not None else 0 for v in mixed_y], width, label="mixed p95")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylabel("interactive p95 client TTFT (seconds)")
    ax.set_title("Control vs mixed interactive p95 TTFT")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def attach_interference(mixed: dict, control: dict | None) -> dict:
    if control is None:
        mixed["control_aligned_p95_s"] = None
        mixed["interference_ratio"] = None
        mixed["absolute_ttft_increase_s"] = None
        return mixed
    mixed["control_aligned_p95_s"] = control.get("interactive_p95_burst_s")
    mixed["interference_ratio"] = interference_ratio(
        mixed.get("interactive_p95_burst_s"),
        control.get("interactive_p95_burst_s"),
    )
    mixed["absolute_ttft_increase_s"] = absolute_ttft_increase(
        mixed.get("interactive_p95_burst_s"),
        control.get("interactive_p95_burst_s"),
    )
    return mixed


def aggregate_group(repeats: list[dict], key: str) -> dict[str, float | None]:
    values = [row[key] for row in repeats if row.get(key) is not None]
    return {
        "n": len(values),
        "median": percentile(values, 50) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


async def run_one_attempt(
    interactive_client,
    background_client,
    telemetry_client,
    config,
    out_dir: Path,
    *,
    suite_id: str,
    repeat_id: str,
    scenario: str,
    background_rps: float | None,
    scheduler_policy: str = "fcfs",
    class_priorities: dict[str, int] | None = None,
    send_priority_field: bool = False,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    idle = await wait_until_idle(telemetry_client, config)
    if not idle:
        print("WARNING: queue did not return to idle before run", flush=True)
    print(f"=== {repeat_id} scenario={scenario} bg_rps={background_rps} ===", flush=True)
    records, runtime_rows, gpu_rows, meta = await run_scenario(
        interactive_client,
        background_client,
        telemetry_client,
        config,
        suite_id=suite_id,
        repeat_id=repeat_id,
        scenario=scenario,
        background_rps=background_rps,
        clock=time.perf_counter,
        scheduler_policy=scheduler_policy,
        class_priorities=class_priorities,
        send_priority_field=send_priority_field,
    )
    write_jsonl(out_dir / "requests.jsonl", records)
    for row in runtime_rows:
        row.update({"repeat_id": repeat_id, "scenario": scenario, "background_offered_rps": background_rps})
    for row in gpu_rows:
        row.update({"repeat_id": repeat_id, "scenario": scenario, "background_offered_rps": background_rps})
    packed = summarize_scenario(
        records,
        runtime_rows,
        gpu_rows,
        meta,
        config,
        suite_id=suite_id,
        repeat_id=repeat_id,
        scenario=scenario,
        background_rps=background_rps,
    )
    summary = packed["repeat"]
    print(
        f"{repeat_id} valid={summary['valid_offered_load']} "
        f"ix_rps={summary['interactive_actual_dispatch_rps']} "
        f"ix_lag={summary['interactive_median_dispatch_lag_s']} "
        f"bg_rps={summary['background_actual_dispatch_rps']} "
        f"p95_pre={summary['interactive_p95_pre_s']} "
        f"p95_burst={summary['interactive_p95_burst_s']} "
        f"p95_rec={summary['interactive_p95_recovery_s']} "
        f"wait={summary['max_waiting_requests']} "
        f"reason={summary['invalid_reason']}",
        flush=True,
    )
    idle = await wait_until_idle(telemetry_client, config)
    print(f"drain idle={idle} last_bg={summary['last_background_completion_s']}", flush=True)
    await asyncio.sleep(float(config["inter_run_idle_s"]))
    return summary, packed["phase_rows"], runtime_rows, gpu_rows


async def collect_valid_repeats(
    interactive_client,
    background_client,
    telemetry_client,
    config,
    out_dir: Path,
    *,
    suite_id: str,
    scenario: str,
    background_rps: float | None,
    needed: int,
    max_attempts: int,
    label: str,
    scheduler_policy: str = "fcfs",
    class_priorities: dict[str, int] | None = None,
    send_priority_field: bool = False,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    repeats = []
    phases = []
    runtime = []
    gpu = []
    valid = 0
    attempt = 0
    while attempt < max_attempts and valid < needed:
        attempt += 1
        repeat_id = f"{label}-rep{attempt}"
        summary, phase_rows, runtime_rows, gpu_rows = await run_one_attempt(
            interactive_client,
            background_client,
            telemetry_client,
            config,
            out_dir,
            suite_id=suite_id,
            repeat_id=repeat_id,
            scenario=scenario,
            background_rps=background_rps,
            scheduler_policy=scheduler_policy,
            class_priorities=class_priorities,
            send_priority_field=send_priority_field,
        )
        repeats.append(summary)
        phases.extend(phase_rows)
        runtime.extend(runtime_rows)
        gpu.extend(gpu_rows)
        if summary["valid_offered_load"]:
            valid += 1
    return repeats, phases, runtime, gpu


def write_suite_outputs(
    out_dir: Path,
    config: dict,
    *,
    mode: str,
    start: str,
    control_repeats: list[dict],
    mixed_repeats: list[dict],
    phase_rows: list[dict],
    all_runtime: list[dict],
    all_gpu: list[dict],
    selected_rps: float | None,
    selection_reason: str,
    rates_tested: list[float],
    conclusion: str,
    representative_mixed_records: list[dict] | None = None,
    representative_mixed_runtime: list[dict] | None = None,
) -> dict:
    write_csv(out_dir / "repeat_summary.csv", control_repeats + mixed_repeats)
    write_csv(out_dir / "phase_summary.csv", phase_rows)
    write_csv(out_dir / "runtime_metrics.csv", all_runtime)
    write_csv(out_dir / "gpu_metrics.csv", all_gpu)
    valid_control = [row for row in control_repeats if row.get("valid_offered_load")]
    valid_mixed = [row for row in mixed_repeats if row.get("valid_offered_load")]
    paired = []
    for idx, mixed in enumerate(valid_mixed):
        control = valid_control[idx] if idx < len(valid_control) else (valid_control[-1] if valid_control else None)
        paired.append(attach_interference(dict(mixed), control))
    write_csv(out_dir / "aggregate_summary.csv", paired)
    if representative_mixed_records:
        plot_ttft_timeline(
            out_dir / "interactive_ttft_timeline.png",
            representative_mixed_records,
            config,
            "Interactive TTFT during mixed scenario",
        )
    if representative_mixed_runtime:
        plot_queue_timeline(
            out_dir / "queue_timeline.png",
            representative_mixed_runtime,
            config,
            "vLLM running and waiting during mixed scenario",
        )
    if valid_control and valid_mixed:
        plot_control_vs_mixed(out_dir / "control_vs_mixed.png", valid_control, valid_mixed)
    versions = package_versions()
    nofile_soft, nofile_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    end = datetime.now(timezone.utc).isoformat()
    aggregate = {
        "valid_control_repeats": len(valid_control),
        "valid_mixed_repeats": len(valid_mixed),
        "control_aligned_p95": aggregate_group(valid_control, "interactive_p95_burst_s"),
        "mixed_pre_p95": aggregate_group(valid_mixed, "interactive_p95_pre_s"),
        "mixed_burst_p95": aggregate_group(valid_mixed, "interactive_p95_burst_s"),
        "mixed_recovery_p95": aggregate_group(valid_mixed, "interactive_p95_recovery_s"),
        "interference_ratio": aggregate_group(paired, "interference_ratio"),
        "absolute_ttft_increase_s": aggregate_group(paired, "absolute_ttft_increase_s"),
        "max_waiting_requests": aggregate_group(valid_mixed, "max_waiting_requests"),
        "background_completed": aggregate_group(valid_mixed, "background_completed_count"),
        "background_output_token_goodput_tps": aggregate_group(valid_mixed, "background_output_token_goodput_tps"),
    }
    manifest = {
        "suite_id": out_dir.name,
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
            "Interactive traffic is the P1 short 64-token workload at a fixed 64 RPS.",
            "Background traffic is a synthetic longer document-summarization job. It is not production data.",
            "Prefix caching stays at the vLLM default (enabled).",
            "No request priority, admission control, or ServeScope intervention is used.",
            "The server stays on default scheduling behavior.",
            (
                "Separate HTTP clients: interactive "
                f"{config['interactive']['client_max_connections']} connections / "
                f"{config['interactive']['max_inflight']} in-flight, background "
                f"{config['background']['client_max_connections']} connections / "
                f"{config['background']['max_inflight']} in-flight. "
                f"ulimit -n soft={nofile_soft}."
            ),
        ],
        "slo_p95_ttft_seconds": config["slo_p95_ttft_seconds"],
        "percentile_method": config["percentile_method"],
        "start_utc": start,
        "end_utc": end,
        "scenario_timing": {
            "duration_s": config["scenario_duration_s"],
            "pre_burst_end_s": config["pre_burst_end_s"],
            "burst_end_s": config["burst_end_s"],
            "interactive_rps": config["interactive"]["offered_rps"],
        },
        "selected_background_rps": selected_rps,
        "selection_reason": selection_reason,
        "background_rates_tested_rps": rates_tested,
        "class_configs": {
            "interactive": config["interactive"],
            "background": config["background"],
        },
        "ulimit_n_soft": nofile_soft,
        "ulimit_n_hard": nofile_hard,
        "aggregates": aggregate,
        "conclusion": conclusion,
        "package_versions": versions,
    }
    result = {
        "question": "Can background AI work damage interactive responsiveness even when the interactive workload is healthy by itself?",
        "selected_background_rps": selected_rps,
        "selection_reason": selection_reason,
        "conclusion": conclusion,
        "aggregates": aggregate,
        "control_repeats": control_repeats,
        "mixed_repeats": paired,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def conclude(valid_mixed: list[dict], selected_rps: float | None) -> str:
    if selected_rps is None:
        return "no measurable interference at tested safe background rates"
    if not valid_mixed:
        return "invalid/client-limited scenario"
    ratios = [row.get("interference_ratio") for row in valid_mixed if row.get("interference_ratio") is not None]
    increases = [row.get("absolute_ttft_increase_s") for row in valid_mixed if row.get("absolute_ttft_increase_s") is not None]
    queues = [row.get("max_waiting_requests") or 0 for row in valid_mixed]
    if not ratios or not increases:
        return "invalid/client-limited scenario"
    median_ratio = percentile(ratios, 50)
    median_increase = percentile(increases, 50)
    if median_increase is None:
        return "invalid/client-limited scenario"
    if median_increase < 0.010 and max(queues) <= 0:
        return "no measurable interference at tested safe background rates"
    if median_ratio is not None and median_ratio >= 2.0 and median_increase >= 0.020:
        return "clear reproducible interference"
    return "weak interference"


async def run_pilot(config: dict, out_dir: Path) -> dict:
    suite_id = out_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    start = datetime.now(timezone.utc).isoformat()
    (out_dir / "nvidia_smi_before.txt").write_text(nvidia_smi_processes(), encoding="utf-8")
    rates = [float(x) for x in config["background"]["pilot_rates_rps"]]
    extra = float(config["background"]["extra_probe_rps"])
    all_repeats: list[dict] = []
    all_phases: list[dict] = []
    all_runtime: list[dict] = []
    all_gpu: list[dict] = []
    selected = None
    reason = "no tested rate produced a sustained waiting queue with valid client schedules"
    representative_records: list[dict] = []
    representative_runtime: list[dict] = []

    async with (
        make_client(config, config["interactive"]) as interactive_client,
        make_client(config, config["background"]) as background_client,
        make_client(config, config["interactive"]) as telemetry_client,
    ):
        print("warmup", flush=True)
        await warmup(interactive_client, config)
        print(f"post-warmup idle={await wait_until_idle(telemetry_client, config)}", flush=True)
        await asyncio.sleep(float(config["inter_run_idle_s"]))

        async def probe(rate: float) -> dict:
            nonlocal representative_records, representative_runtime
            summary, phase_rows, runtime_rows, gpu_rows = await run_one_attempt(
                interactive_client,
                background_client,
                telemetry_client,
                config,
                out_dir,
                suite_id=suite_id,
                repeat_id=f"pilot-bg{rate:g}-rep1",
                scenario="mixed",
                background_rps=rate,
            )
            all_repeats.append(summary)
            all_phases.extend(phase_rows)
            all_runtime.extend(runtime_rows)
            all_gpu.extend(gpu_rows)
            return summary

        for rate in rates:
            summary = await probe(rate)
            if summary["valid_offered_load"] and summary["sustained_burst_queue"] and selected is None:
                selected = rate
                reason = (
                    f"lowest tested background offered rate with a sustained waiting queue "
                    f"and valid interactive+background schedules ({rate:g} RPS)"
                )
        if selected is None:
            print(f"probing extra background rate {extra:g}", flush=True)
            rates = [*rates, extra]
            summary = await probe(extra)
            if summary["valid_offered_load"] and summary["sustained_burst_queue"]:
                selected = extra
                reason = (
                    f"16 RPS produced no sustained queue. Extra probe {extra:g} RPS did, "
                    "and both client schedules stayed valid."
                )

    write_suite_outputs(
        out_dir,
        config,
        mode="pilot",
        start=start,
        control_repeats=[],
        mixed_repeats=all_repeats,
        phase_rows=all_phases,
        all_runtime=all_runtime,
        all_gpu=all_gpu,
        selected_rps=selected,
        selection_reason=reason,
        rates_tested=rates,
        conclusion="pilot only. Final rate selection is recorded in this directory.",
    )
    (out_dir / "selected_background_rps.json").write_text(
        json.dumps({"selected_background_rps": selected, "selection_reason": reason, "rates_tested": rates}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"pilot selected_background_rps={selected} reason={reason}", flush=True)
    return {"selected_background_rps": selected, "selection_reason": reason, "rates_tested": rates}


async def run_final(config: dict, out_dir: Path, selected_rps: float | None, selection_reason: str, rates_tested: list[float]) -> dict:
    suite_id = out_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    start = datetime.now(timezone.utc).isoformat()
    (out_dir / "nvidia_smi_before.txt").write_text(nvidia_smi_processes(), encoding="utf-8")
    needed = int(config["min_valid_repeats"])
    max_attempts = int(config["max_repeat_attempts"])
    control_repeats: list[dict] = []
    mixed_repeats: list[dict] = []
    phases: list[dict] = []
    runtime: list[dict] = []
    gpu: list[dict] = []
    last_valid_mixed_records: list[dict] = []
    last_valid_mixed_runtime: list[dict] = []

    async with (
        make_client(config, config["interactive"]) as interactive_client,
        make_client(config, config["background"]) as background_client,
        make_client(config, config["interactive"]) as telemetry_client,
    ):
        print("warmup", flush=True)
        await warmup(interactive_client, config)
        print(f"post-warmup idle={await wait_until_idle(telemetry_client, config)}", flush=True)
        await asyncio.sleep(float(config["inter_run_idle_s"]))

        control_repeats, c_phases, c_runtime, c_gpu = await collect_valid_repeats(
            interactive_client,
            background_client,
            telemetry_client,
            config,
            out_dir,
            suite_id=suite_id,
            scenario="control",
            background_rps=None,
            needed=needed,
            max_attempts=max_attempts,
            label="final-control",
        )
        phases.extend(c_phases)
        runtime.extend(c_runtime)
        gpu.extend(c_gpu)

        if selected_rps is None:
            conclusion = "no measurable interference at tested safe background rates"
            write_suite_outputs(
                out_dir,
                config,
                mode="final",
                start=start,
                control_repeats=control_repeats,
                mixed_repeats=[],
                phase_rows=phases,
                all_runtime=runtime,
                all_gpu=gpu,
                selected_rps=None,
                selection_reason=selection_reason,
                rates_tested=rates_tested,
                conclusion=conclusion,
            )
            print(conclusion, flush=True)
            return {"conclusion": conclusion}

        mixed_repeats, m_phases, m_runtime, m_gpu = await collect_valid_repeats(
            interactive_client,
            background_client,
            telemetry_client,
            config,
            out_dir,
            suite_id=suite_id,
            scenario="mixed",
            background_rps=selected_rps,
            needed=needed,
            max_attempts=max_attempts,
            label=f"final-mixed-bg{selected_rps:g}",
        )
        phases.extend(m_phases)
        runtime.extend(m_runtime)
        gpu.extend(m_gpu)

    # Rebuild plots from the last mixed repeat in requests.jsonl if present.
    last_valid = next((row for row in reversed(mixed_repeats) if row.get("valid_offered_load")), None)
    if last_valid:
        last_id = last_valid["repeat_id"]
        last_valid_mixed_runtime = [row for row in runtime if row.get("repeat_id") == last_id]
        req_path = out_dir / "requests.jsonl"
        if req_path.exists():
            last_valid_mixed_records = []
            for line in req_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("repeat_id") == last_id:
                    last_valid_mixed_records.append(row)

    valid_control = [row for row in control_repeats if row.get("valid_offered_load")]
    valid_mixed = [row for row in mixed_repeats if row.get("valid_offered_load")]
    paired = []
    for idx, mixed in enumerate(valid_mixed):
        control = valid_control[idx] if idx < len(valid_control) else (valid_control[-1] if valid_control else None)
        paired.append(attach_interference(dict(mixed), control))
    conclusion = conclude(paired, selected_rps)
    write_suite_outputs(
        out_dir,
        config,
        mode="final",
        start=start,
        control_repeats=control_repeats,
        mixed_repeats=mixed_repeats,
        phase_rows=phases,
        all_runtime=runtime,
        all_gpu=gpu,
        selected_rps=selected_rps,
        selection_reason=selection_reason,
        rates_tested=rates_tested,
        conclusion=conclusion,
        representative_mixed_records=last_valid_mixed_records,
        representative_mixed_runtime=last_valid_mixed_runtime,
    )
    print(f"final conclusion={conclusion}", flush=True)
    return {"conclusion": conclusion}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2 mixed-workload interference")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "p2_mixed.json")
    parser.add_argument("--mode", choices=("pilot", "final"), required=True)
    parser.add_argument("--selected-rps", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--pilot-dir", type=Path, default=None, help="reuse a prior pilot selection")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    stamp = now_utc()
    out = args.out or (ROOT / "artifacts" / "p2" / f"{args.mode}-{stamp}")
    print(f"writing {out}", flush=True)
    if args.mode == "pilot":
        asyncio.run(run_pilot(config, out))
    else:
        selected = args.selected_rps
        reason = "provided on the command line"
        rates = [float(x) for x in config["background"]["pilot_rates_rps"]]
        if args.pilot_dir:
            selection = json.loads((args.pilot_dir / "selected_background_rps.json").read_text(encoding="utf-8"))
            selected = selection.get("selected_background_rps")
            reason = selection.get("selection_reason") or reason
            rates = selection.get("rates_tested") or rates
        asyncio.run(run_final(config, out, selected, reason, rates))
    print(f"done {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
