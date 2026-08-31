#!/usr/bin/env python3
"""External background admission against a native-priority vLLM server.

Native vLLM priority stays on. ServeScope only gates how many background
jobs are submitted.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import p2_mixed as p2  # noqa: E402

from servescope.backpressure import (
    P4_HASHED_SOURCE_FILES,
    AimdController,
    PendingJob,
    PendingQueue,
    annotate_gated_background,
    annotate_native_background,
    can_admit,
    controller_stats,
    offered_span_goodput,
    summarize_ingress,
)
from servescope.client import build_chat_payload, stream_chat_request
from servescope.metrics import METRIC_KV, METRIC_RUNNING, METRIC_WAITING, percentile
from servescope.p2_metrics import CLASS_BACKGROUND, CLASS_INTERACTIVE, attach_phase, filter_class
from servescope.p3_metrics import (
    POLICY_PRIORITY,
    aggregate_numeric,
    request_priority_for_class,
    should_send_priority_field,
    source_hashes,
    summarize_post_background,
)
from servescope.workload import select_background_prompt, select_prompt, windowed_arrivals

git_state = p2.git_state
load_config = p2.load_config
make_client = p2.make_client
nvidia_smi_processes = p2.nvidia_smi_processes
now_utc = p2.now_utc
package_versions = p2.package_versions
run_scenario = p2.run_scenario
summarize_scenario = p2.summarize_scenario
wait_until_idle = p2.wait_until_idle
warmup = p2.warmup
write_csv = p2.write_csv
write_jsonl = p2.write_jsonl

SERVER_COMMAND = (
    "vllm serve Qwen/Qwen3-1.7B --gpu-memory-utilization 0.85 "
    "--enforce-eager --enable-per-request-metrics --scheduling-policy priority"
)


def class_priorities() -> dict[str, int]:
    return {
        CLASS_INTERACTIVE: request_priority_for_class(CLASS_INTERACTIVE, POLICY_PRIORITY),
        CLASS_BACKGROUND: request_priority_for_class(CLASS_BACKGROUND, POLICY_PRIORITY),
    }


def make_controller(config: dict) -> AimdController:
    ctl = config["controller"]
    return AimdController(
        initial=int(ctl["initial_background_limit"]),
        minimum=int(ctl["minimum_background_limit"]),
        maximum=int(ctl["maximum_background_limit"]),
        increase_after_zero_samples=int(ctl["increase_after_zero_samples"]),
    )


def waiting_stats(runtime_rows: list[dict]) -> dict[str, float | None]:
    vals = [row["num_requests_waiting"] for row in runtime_rows if row.get("num_requests_waiting") is not None]
    return {
        "waiting_p50": percentile(vals, 50) if vals else None,
        "waiting_p95": percentile(vals, 95) if vals else None,
        "max_waiting_requests": max(vals) if vals else None,
    }


def enrich_p4(
    packed: dict,
    records: list[dict],
    meta: dict,
    config: dict,
    runtime_rows: list[dict],
    *,
    gated: bool,
    controller_rows: list[dict],
) -> dict:
    summary = packed["repeat"]
    slo_s = float(config["slo_p95_ttft_seconds"])
    bg = filter_class(records, CLASS_BACKGROUND)
    goodput = offered_span_goodput(bg, meta["t0"])
    last_bg = goodput["last_completion_rel_s"]
    post = summarize_post_background(records, last_bg, slo_s=slo_s)
    wait = waiting_stats(runtime_rows)
    phases = {row["phase"]: row for row in packed["phase_rows"]}
    ingress = summarize_ingress(bg, float(meta.get("background_offered_rps") or 0.0)) if bg else None
    ctl = controller_stats(controller_rows)
    summary.update(
        {
            "scheduler_policy": POLICY_PRIORITY,
            "admission_gated": gated,
            "slo_met_in_30_60_window": summary.get("slo_recovered_afterward"),
            "interactive_p99_pre_s": (phases.get("pre_burst") or {}).get("client_ttft_p99_s"),
            "interactive_p99_burst_s": (phases.get("burst_injection") or {}).get("client_ttft_p99_s"),
            "interactive_p99_recovery_s": (phases.get("recovery") or {}).get("client_ttft_p99_s"),
            "post_background_count": post["count"],
            "post_background_ttft_p50_s": post["client_ttft_p50_s"],
            "post_background_ttft_p95_s": post["client_ttft_p95_s"],
            "post_background_ttft_p99_s": post["client_ttft_p99_s"],
            "post_background_slo_met": post["slo_met"],
            "waiting_p50": wait["waiting_p50"],
            "waiting_p95": wait["waiting_p95"],
            "background_admitted_count": goodput["admitted_count"],
            "background_completed_count": goodput["completed_count"],
            "background_offered_count": goodput["offered_count"],
            "background_output_token_goodput_tps": goodput["wall_clock_output_token_tps"],
            "background_request_goodput_rps": goodput["wall_clock_completed_request_tps"],
            "background_total_e2e_p50_s": goodput["background_total_e2e_p50_s"],
            "background_total_e2e_p95_s": goodput["background_total_e2e_p95_s"],
            "background_total_e2e_p99_s": goodput["background_total_e2e_p99_s"],
            "background_e2e_p50_s": goodput["background_total_e2e_p50_s"],
            "background_e2e_p95_s": goodput["background_total_e2e_p95_s"],
            "background_e2e_p99_s": goodput["background_total_e2e_p99_s"],
            "admission_delay_p50_s": goodput["admission_delay_p50_s"],
            "admission_delay_p95_s": goodput["admission_delay_p95_s"],
            "admission_delay_p99_s": goodput["admission_delay_p99_s"],
            "max_local_pending_depth": meta.get("max_local_pending_depth") or 0,
            "last_background_completion_s": last_bg,
            **ctl,
        }
    )
    if gated and ingress is not None:
        summary["background_actual_ingress_rps"] = ingress["actual_ingress_rps"]
        summary["background_median_ingress_lag_s"] = ingress["median_ingress_lag_s"]
        ix_valid = bool(packed["interactive_summary"].get("valid_offered_load"))
        valid = ix_valid and ingress["valid_offered_load"] and not meta.get("aborted")
        drop_reason = None
        if goodput["completed_count"] < goodput["offered_count"]:
            valid = False
            drop_reason = "background_jobs_did_not_all_complete"
        summary["valid_offered_load"] = valid
        summary["invalid_reason"] = (
            None
            if valid
            else ", ".join(
                reason
                for reason in (
                    packed["interactive_summary"].get("invalid_reason"),
                    ingress.get("invalid_reason"),
                    meta.get("abort_reason"),
                    drop_reason,
                )
                if reason
            )
            or "invalid"
        )
    return summary


async def sample_runtime_and_control(
    client,
    url: str,
    interval_s: float,
    stop: asyncio.Event,
    runtime_rows: list[dict],
    controller_rows: list[dict],
    t0: float,
    controller: AimdController | None,
    counts: dict,
    pending: PendingQueue | None,
) -> None:
    while not stop.is_set():
        gauges = await p2.fetch_runtime_gauges(client, url)
        waiting = gauges.get(METRIC_WAITING)
        action = controller.observe(waiting) if controller is not None else "hold"
        now_rel = time.perf_counter() - t0
        runtime_rows.append(
            {
                "t_s": now_rel,
                "num_requests_running": gauges.get(METRIC_RUNNING),
                "num_requests_waiting": waiting,
                "kv_cache_usage_perc": gauges.get(METRIC_KV),
            }
        )
        controller_rows.append(
            {
                "t_s": now_rel,
                "background_limit": controller.limit if controller is not None else None,
                "background_inflight": counts.get("inflight", 0),
                "local_pending_depth": len(pending) if pending is not None else 0,
                "vllm_running": gauges.get(METRIC_RUNNING),
                "vllm_waiting": waiting,
                "controller_action": action,
            }
        )
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except TimeoutError:
            continue


async def offer_background_ingress(
    *,
    t0: float,
    targets: list[float],
    clock,
    pending: PendingQueue,
    seed: int,
) -> dict:
    aborted = False
    abort_reason = None
    for i, scheduled_s in enumerate(targets):
        now = clock()
        if now < scheduled_s:
            await asyncio.sleep(scheduled_s - now)
        enqueue_s = clock()
        prompt_id, prompt = select_background_prompt(seed, i)
        try:
            pending.enqueue(
                PendingJob(
                    index=i,
                    offered_arrival_s=scheduled_s,
                    ingress_enqueue_s=enqueue_s,
                    prompt_id=prompt_id,
                    prompt=prompt,
                )
            )
        except RuntimeError as exc:
            aborted = True
            abort_reason = str(exc)
            break
    return {"aborted": aborted, "abort_reason": abort_reason, "t0": t0}


async def admit_background(
    client,
    config: dict,
    *,
    suite_id: str,
    repeat_id: str,
    scenario: str,
    t0: float,
    clock,
    records: list[dict],
    pending: PendingQueue,
    controller: AimdController,
    counts: dict,
    safety: p2.Inflight,
    ingress_done: asyncio.Event,
    request_priority: int,
) -> dict:
    url = config["base_url"].rstrip("/") + config["chat_path"]
    class_cfg = config["background"]
    pre_end = float(config["pre_burst_end_s"])
    burst_end = float(config["burst_end_s"])
    tasks: set[asyncio.Task] = set()
    aborted = False
    abort_reason = None
    deadline = clock() + float(config["drain_timeout_s"])

    async def submit(job: PendingJob) -> None:
        payload = build_chat_payload(
            model=config["model"],
            prompt=job.prompt,
            temperature=class_cfg["temperature"],
            min_tokens=class_cfg["min_tokens"],
            max_completion_tokens=class_cfg["max_completion_tokens"],
            priority=request_priority,
        )
        try:
            record = await stream_chat_request(
                client,
                url=url,
                payload=payload,
                request_id=f"{repeat_id}-background-r{job.index:05d}",
                prompt_id=job.prompt_id,
                workload_class=CLASS_BACKGROUND,
                scheduled_s=job.offered_arrival_s,
                timeout_s=float(config["request_timeout_s"]),
                clock=clock,
            )
        finally:
            safety.release()
            counts["inflight"] = max(0, counts["inflight"] - 1)
            counts["completed"] += 1
        attach_phase(record, t0, pre_end_s=pre_end, burst_end_s=burst_end)
        annotate_gated_background(
            record,
            offered_arrival_s=job.offered_arrival_s,
            ingress_enqueue_s=job.ingress_enqueue_s,
        )
        record.update(
            {
                "suite_id": suite_id,
                "repeat_id": repeat_id,
                "scenario": scenario,
                "offered_rps": class_cfg.get("offered_rps"),
                "request_index": job.index,
                "scheduler_policy": POLICY_PRIORITY,
                "request_priority": request_priority,
            }
        )
        records.append(record)

    while clock() < deadline:
        while can_admit(counts["inflight"], controller.limit) and pending:
            if not safety.try_acquire():
                aborted = True
                abort_reason = (
                    f"background in-flight reached {safety.limit}; aborting rather than closing the loop"
                )
                break
            job = pending.dequeue()
            if job is None:
                safety.release()
                break
            counts["inflight"] += 1
            counts["admitted"] += 1
            task = asyncio.create_task(submit(job))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        if aborted:
            break
        if ingress_done.is_set() and not pending and counts["inflight"] == 0:
            break
        await asyncio.sleep(0.01)
    else:
        aborted = True
        abort_reason = "background admission exceeded drain_timeout_s"
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    return {
        "aborted": aborted,
        "abort_reason": abort_reason,
        "peak_inflight": safety.peak,
        "admitted": counts["admitted"],
        "completed": counts["completed"],
    }


async def run_gated_scenario(
    interactive_client,
    background_client,
    telemetry_client,
    config: dict,
    *,
    suite_id: str,
    repeat_id: str,
    background_rps: float,
    clock,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    interactive_cfg = dict(config["interactive"])
    background_cfg = dict(config["background"])
    duration_s = float(config["scenario_duration_s"])
    pre_end = float(config["pre_burst_end_s"])
    burst_end = float(config["burst_end_s"])
    t0 = clock()
    interactive_cfg["offered_rps"] = float(interactive_cfg["offered_rps"])
    interactive_targets = windowed_arrivals(interactive_cfg["offered_rps"], t0, 0.0, duration_s)
    background_cfg["offered_rps"] = float(background_rps)
    background_targets = windowed_arrivals(float(background_rps), t0, pre_end, burst_end)

    records: list[dict] = []
    runtime_rows: list[dict] = []
    gpu_rows: list[dict] = []
    controller_rows: list[dict] = []
    stop = asyncio.Event()
    ingress_done = asyncio.Event()
    pending = PendingQueue(int(config["controller"]["pending_queue_capacity"]))
    controller = make_controller(config)
    counts = {"inflight": 0, "admitted": 0, "completed": 0}
    safety = p2.Inflight(int(background_cfg["max_inflight"]))
    metrics_url = config["base_url"].rstrip("/") + config["metrics_path"]
    priorities = class_priorities()

    runtime_task = asyncio.create_task(
        sample_runtime_and_control(
            telemetry_client,
            metrics_url,
            float(config["controller"]["sample_interval_s"]),
            stop,
            runtime_rows,
            controller_rows,
            t0,
            controller,
            counts,
            pending,
        )
    )
    gpu_task = asyncio.create_task(p2.sample_gpu(float(config["gpu_sample_interval_s"]), stop, gpu_rows, t0))
    interactive_inflight = p2.Inflight(int(interactive_cfg["max_inflight"]))

    interactive_offer = asyncio.create_task(
        p2.offer_class(
            interactive_client,
            config,
            interactive_cfg,
            suite_id=suite_id,
            repeat_id=repeat_id,
            scenario="mixed",
            t0=t0,
            targets=interactive_targets,
            clock=clock,
            records=records,
            inflight=interactive_inflight,
            prompt_fn=select_prompt,
            seed=int(config["seed"]),
            request_priority=priorities[CLASS_INTERACTIVE],
            scheduler_policy=POLICY_PRIORITY,
            send_priority_field=True,
        )
    )
    ingress_offer = asyncio.create_task(
        offer_background_ingress(
            t0=t0,
            targets=background_targets,
            clock=clock,
            pending=pending,
            seed=int(config["seed"]) + 1_000_000,
        )
    )
    admit_task = asyncio.create_task(
        admit_background(
            background_client,
            config,
            suite_id=suite_id,
            repeat_id=repeat_id,
            scenario="mixed",
            t0=t0,
            clock=clock,
            records=records,
            pending=pending,
            controller=controller,
            counts=counts,
            safety=safety,
            ingress_done=ingress_done,
            request_priority=priorities[CLASS_BACKGROUND],
        )
    )

    interactive_meta, ingress_meta = await asyncio.gather(interactive_offer, ingress_offer)
    ingress_done.set()
    background_meta = await admit_task
    stop.set()
    await asyncio.gather(runtime_task, gpu_task)

    duration_wall = clock() - t0
    records.sort(key=lambda row: (row.get("workload_class") or "", row.get("request_index") or 0))
    bg_rows = filter_class(records, CLASS_BACKGROUND)
    meta = {
        "t0": t0,
        "aborted": bool(interactive_meta["aborted"] or ingress_meta["aborted"] or background_meta["aborted"]),
        "abort_reason": interactive_meta["abort_reason"]
        or ingress_meta["abort_reason"]
        or background_meta["abort_reason"],
        "duration_s": duration_wall,
        "peak_inflight_interactive": interactive_meta["peak_inflight"],
        "peak_inflight_background": background_meta["peak_inflight"],
        "last_background_completion_s": offered_span_goodput(bg_rows, t0)["last_completion_rel_s"],
        "background_offered_rps": background_cfg.get("offered_rps") or 0.0,
        "interactive_offered_count": len(interactive_targets),
        "background_offered_count": len(background_targets),
        "max_local_pending_depth": pending.max_depth,
    }
    return records, runtime_rows, gpu_rows, controller_rows, meta


def annotate_native_records(records: list[dict]) -> None:
    for row in records:
        if row.get("workload_class") == CLASS_BACKGROUND:
            annotate_native_background(row)


def write_manifest(out_dir: Path, config: dict, extra: dict) -> dict:
    nofile_soft, nofile_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    versions = package_versions()
    manifest = {
        "suite_id": out_dir.name,
        "git": git_state(ROOT),
        "source_hashes": source_hashes(ROOT, P4_HASHED_SOURCE_FILES),
        "model": config["model"],
        "backend": "vllm",
        "backend_version": versions.get("vllm"),
        "scheduler_policy": POLICY_PRIORITY,
        "server_command": SERVER_COMMAND,
        "server_env": {
            "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "HF_HOME": "/home/mayur/.cache/huggingface",
        },
        "priority_mechanism": "OpenAI chat request-body field `priority`",
        "class_priorities": class_priorities(),
        "controller": config.get("controller"),
        "class_configs": {
            "interactive": config["interactive"],
            "background": config["background"],
            "telemetry": config.get("telemetry"),
        },
        "ulimit_n_soft": nofile_soft,
        "ulimit_n_hard": nofile_hard,
        "notes": [
            "P4 keeps native --scheduling-policy priority.",
            "ServeScope only admits background jobs. Interactive is never gated.",
            "This is external backpressure, not a vLLM scheduler.",
        ],
        "package_versions": versions,
        **extra,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def tag_rows(rows: list[dict], repeat_id: str, mode: str, background_rps: float | None) -> None:
    for row in rows:
        row.update({"repeat_id": repeat_id, "mode": mode, "background_offered_rps": background_rps})


async def run_one(
    interactive_client,
    background_client,
    telemetry_client,
    config: dict,
    out_dir: Path,
    *,
    suite_id: str,
    repeat_id: str,
    mode: str,
    background_rps: float,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict], list[dict]]:
    idle = await wait_until_idle(telemetry_client, config)
    if not idle:
        print("WARNING: queue did not return to idle before run", flush=True)
    print(f"=== {repeat_id} mode={mode} ===", flush=True)
    gated = mode == "backpressure"
    if gated:
        records, runtime_rows, gpu_rows, controller_rows, meta = await run_gated_scenario(
            interactive_client,
            background_client,
            telemetry_client,
            config,
            suite_id=suite_id,
            repeat_id=repeat_id,
            background_rps=background_rps,
            clock=time.perf_counter,
        )
    else:
        records, runtime_rows, gpu_rows, meta = await run_scenario(
            interactive_client,
            background_client,
            telemetry_client,
            config,
            suite_id=suite_id,
            repeat_id=repeat_id,
            scenario="mixed",
            background_rps=background_rps,
            clock=time.perf_counter,
            scheduler_policy=POLICY_PRIORITY,
            class_priorities=class_priorities(),
            send_priority_field=True,
        )
        annotate_native_records(records)
        controller_rows = []
        meta["max_local_pending_depth"] = 0
    write_jsonl(out_dir / "requests.jsonl", records)
    tag_rows(runtime_rows, repeat_id, mode, background_rps)
    tag_rows(gpu_rows, repeat_id, mode, background_rps)
    tag_rows(controller_rows, repeat_id, mode, background_rps)
    packed = summarize_scenario(
        records,
        runtime_rows,
        gpu_rows,
        meta,
        config,
        suite_id=suite_id,
        repeat_id=repeat_id,
        scenario="mixed",
        background_rps=background_rps,
    )
    summary = enrich_p4(
        packed,
        records,
        meta,
        config,
        runtime_rows,
        gated=gated,
        controller_rows=controller_rows,
    )
    print(
        f"{repeat_id} valid={summary['valid_offered_load']} "
        f"ix_rps={summary['interactive_actual_dispatch_rps']} "
        f"bg_in={summary.get('background_actual_ingress_rps') or summary.get('background_actual_dispatch_rps')} "
        f"p95_burst={summary['interactive_p95_burst_s']} "
        f"wait={summary['max_waiting_requests']} "
        f"pending={summary.get('max_local_pending_depth')} "
        f"last_bg={summary['last_background_completion_s']} "
        f"reason={summary['invalid_reason']}",
        flush=True,
    )
    idle = await wait_until_idle(telemetry_client, config)
    print(f"drain idle={idle}", flush=True)
    await asyncio.sleep(float(config["inter_run_idle_s"]))
    return summary, packed["phase_rows"], runtime_rows, gpu_rows, controller_rows, records


async def collect_repeats(
    interactive_client,
    background_client,
    telemetry_client,
    config: dict,
    out_dir: Path,
    *,
    mode: str,
    needed: int,
    max_attempts: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    bg_rps = float(config["background"]["offered_rps"])
    repeats, phases, runtime, gpu, controller = [], [], [], [], []
    valid = 0
    attempt = 0
    while attempt < max_attempts and valid < needed:
        attempt += 1
        summary, phase_rows, runtime_rows, gpu_rows, controller_rows, _ = await run_one(
            interactive_client,
            background_client,
            telemetry_client,
            config,
            out_dir,
            suite_id=out_dir.name,
            repeat_id=f"{mode}-rep{attempt}",
            mode=mode,
            background_rps=bg_rps,
        )
        repeats.append(summary)
        phases.extend(phase_rows)
        runtime.extend(runtime_rows)
        gpu.extend(gpu_rows)
        controller.extend(controller_rows)
        if summary["valid_offered_load"]:
            valid += 1
    return repeats, phases, runtime, gpu, controller


async def run_suite(config: dict, out_dir: Path, mode: str, needed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (out_dir / "nvidia_smi_before.txt").write_text(nvidia_smi_processes(), encoding="utf-8")
    start = datetime.now(timezone.utc).isoformat()
    tel_cfg = {**config["interactive"], **(config.get("telemetry") or {})}
    max_attempts = 1 if needed == 1 else int(config["max_repeat_attempts"])
    async with (
        make_client(config, config["interactive"]) as interactive_client,
        make_client(config, config["background"]) as background_client,
        make_client(config, tel_cfg) as telemetry_client,
    ):
        print("warmup", flush=True)
        await warmup(interactive_client, config)
        print(f"post-warmup idle={await wait_until_idle(telemetry_client, config)}", flush=True)
        repeats, phases, runtime, gpu, controller = await collect_repeats(
            interactive_client,
            background_client,
            telemetry_client,
            config,
            out_dir,
            mode=mode,
            needed=needed,
            max_attempts=max_attempts,
        )
    write_csv(out_dir / "repeat_summary.csv", repeats)
    write_csv(out_dir / "phase_summary.csv", phases)
    write_csv(out_dir / "runtime_metrics.csv", runtime)
    write_csv(out_dir / "gpu_metrics.csv", gpu)
    write_csv(out_dir / "controller_metrics.csv", controller)
    result = {
        "mode": mode,
        "scheduler_policy": POLICY_PRIORITY,
        "valid_mixed": sum(1 for row in repeats if row.get("valid_offered_load")),
        "repeats": repeats,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_manifest(
        out_dir,
        config,
        {
            "mode": mode,
            "admission_gated": mode == "backpressure",
            "start_utc": start,
            "end_utc": datetime.now(timezone.utc).isoformat(),
            "result": result,
        },
    )
    return result


def _last_repeat_rows(path: Path, filename: str) -> list[dict]:
    csv_path = path / filename
    if not csv_path.exists():
        return []
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        return []
    last_id = rows[-1]["repeat_id"]
    return [row for row in rows if row.get("repeat_id") == last_id]


def plot_interactive_protection(path: Path, native: list[dict], gated: list[dict]) -> None:
    labels = ["pre", "burst", "30-60 backlog", "post-background"]
    keys = [
        "interactive_p95_pre_s",
        "interactive_p95_burst_s",
        "interactive_p95_recovery_s",
        "post_background_ttft_p95_s",
    ]
    native_y = [percentile([row[k] for row in native if row.get(k) is not None], 50) or 0 for k in keys]
    gated_y = [percentile([row[k] for row in gated if row.get(k) is not None], 50) or 0 for k in keys]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - 0.18 for i in x], native_y, width=0.36, label="native priority")
    ax.bar([i + 0.18 for i in x], gated_y, width=0.36, label="priority + backpressure")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("interactive p95 TTFT (s)")
    ax.set_title("Interactive p95: native priority vs ServeScope backpressure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_queue_movement(path: Path, controller_rows: list[dict]) -> None:
    t = [float(row["t_s"]) for row in controller_rows if row.get("t_s") not in (None, "")]
    waiting = [
        float(row["vllm_waiting"]) if row.get("vllm_waiting") not in (None, "") else 0.0 for row in controller_rows
    ]
    pending = [
        float(row["local_pending_depth"]) if row.get("local_pending_depth") not in (None, "") else 0.0
        for row in controller_rows
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, waiting, label="vLLM waiting")
    ax.plot(t, pending, label="ServeScope pending")
    ax.axvspan(15, 30, color="0.85", label="background ingress 15-30 s")
    ax.set_xlabel("scenario time (s)")
    ax.set_ylabel("queue depth")
    ax.set_title("Where the waiting moved")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_controller_window(path: Path, controller_rows: list[dict]) -> None:
    t = [float(row["t_s"]) for row in controller_rows if row.get("t_s") not in (None, "")]
    limit = [
        float(row["background_limit"]) if row.get("background_limit") not in (None, "") else 0.0
        for row in controller_rows
    ]
    inflight = [
        float(row["background_inflight"]) if row.get("background_inflight") not in (None, "") else 0.0
        for row in controller_rows
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, limit, label="background concurrency limit")
    ax.plot(t, inflight, label="background in-flight")
    ax.axvspan(15, 30, color="0.85", label="background ingress 15-30 s")
    ax.set_xlabel("scenario time (s)")
    ax.set_ylabel("requests")
    ax.set_title("AIMD background window")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def conclude(native: list[dict], gated: list[dict]) -> str:
    if len(native) < 3 or len(gated) < 3:
        return "controller is unstable or benchmark-invalid"
    if any((row.get("background_completed_count") or 0) < 240 for row in native + gated):
        return "controller is unstable or benchmark-invalid"
    n_burst = percentile([row["interactive_p95_burst_s"] for row in native], 50)
    g_burst = percentile([row["interactive_p95_burst_s"] for row in gated], 50)
    n_wait = percentile([row["max_waiting_requests"] for row in native], 50)
    g_wait = percentile([row["max_waiting_requests"] for row in gated], 50)
    n_e2e = percentile([row["background_total_e2e_p95_s"] for row in native], 50)
    g_e2e = percentile([row["background_total_e2e_p95_s"] for row in gated], 50)
    g_last = percentile([row["last_background_completion_s"] for row in gated], 50)
    if n_burst is None or g_burst is None:
        return "controller is unstable or benchmark-invalid"
    improved = g_burst < n_burst * 0.90
    queue_down = g_wait is not None and n_wait is not None and g_wait < n_wait * 0.75
    severe = (g_e2e is not None and n_e2e is not None and g_e2e > 2.5 * n_e2e) or (
        g_last is not None and g_last > 90
    )
    if improved and queue_down and severe:
        return "backpressure improves latency but background cost is severe"
    if improved and queue_down:
        return "backpressure materially improves interactive latency and reduces runtime backlog with a measurable background cost"
    return "native priority is already the better trade-off"


def write_comparison(out_dir: Path, config: dict, native_dir: Path, gated_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    native = [row for row in json.loads((native_dir / "result.json").read_text())["repeats"] if row.get("valid_offered_load")]
    gated = [row for row in json.loads((gated_dir / "result.json").read_text())["repeats"] if row.get("valid_offered_load")]
    plot_interactive_protection(out_dir / "interactive_protection.png", native, gated)
    plot_queue_movement(out_dir / "queue_movement.png", _last_repeat_rows(gated_dir, "controller_metrics.csv"))
    plot_controller_window(out_dir / "controller_window.png", _last_repeat_rows(gated_dir, "controller_metrics.csv"))
    conclusion = conclude(native, gated)
    result = {
        "conclusion": conclusion,
        "native_burst_p95": aggregate_numeric(native, "interactive_p95_burst_s"),
        "gated_burst_p95": aggregate_numeric(gated, "interactive_p95_burst_s"),
        "native_waiting": aggregate_numeric(native, "max_waiting_requests"),
        "gated_waiting": aggregate_numeric(gated, "max_waiting_requests"),
        "native_bg_e2e_p95": aggregate_numeric(native, "background_total_e2e_p95_s"),
        "gated_bg_e2e_p95": aggregate_numeric(gated, "background_total_e2e_p95_s"),
        "native_bg_tok_tps": aggregate_numeric(native, "background_output_token_goodput_tps"),
        "gated_bg_tok_tps": aggregate_numeric(gated, "background_output_token_goodput_tps"),
        "native_last_bg": aggregate_numeric(native, "last_background_completion_s"),
        "gated_last_bg": aggregate_numeric(gated, "last_background_completion_s"),
        "gated_max_pending": aggregate_numeric(gated, "max_local_pending_depth"),
        "source_hashes": source_hashes(ROOT, P4_HASHED_SOURCE_FILES),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_manifest(
        out_dir,
        config,
        {
            "mode": "comparison",
            "native_dir": str(native_dir),
            "backpressure_dir": str(gated_dir),
            "conclusion": conclusion,
            "aggregates": result,
        },
    )
    print(f"comparison conclusion={conclusion}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P4 ServeScope background backpressure")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "p4_backpressure.json")
    parser.add_argument("--mode", choices=("pilot", "native", "backpressure", "compare"), required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--native-dir", type=Path, default=None)
    parser.add_argument("--backpressure-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    stamp = now_utc()
    if args.mode == "compare":
        if not args.native_dir or not args.backpressure_dir:
            raise SystemExit("--native-dir and --backpressure-dir are required for compare")
        out = args.out or (ROOT / "artifacts" / "p4" / f"comparison-{stamp}")
        print(f"writing {out}", flush=True)
        write_comparison(out, config, args.native_dir, args.backpressure_dir)
    elif args.mode == "pilot":
        out = args.out or (ROOT / "artifacts" / "p4" / f"pilot-{stamp}")
        print(f"writing {out}", flush=True)
        asyncio.run(run_suite(config, out, "backpressure", needed=1))
    else:
        out = args.out or (ROOT / "artifacts" / "p4" / f"{args.mode}-{stamp}")
        print(f"writing {out}", flush=True)
        asyncio.run(run_suite(config, out, args.mode, needed=int(config["min_valid_repeats"])))
    print(f"done {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
