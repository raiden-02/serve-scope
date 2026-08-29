#!/usr/bin/env python3
"""P3 native vLLM priority baseline.

Compares the same P2 mixed workload under FCFS and --scheduling-policy priority.
This script does not implement a ServeScope scheduler, admission, or throttle.
"""

from __future__ import annotations

import argparse
import asyncio
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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import p2_mixed as p2  # noqa: E402

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
from servescope.client import build_chat_payload, stream_chat_request
from servescope.p2_metrics import CLASS_BACKGROUND, CLASS_INTERACTIVE, filter_class
from servescope.p3_metrics import (
    POLICY_FCFS,
    POLICY_PRIORITY,
    aggregate_numeric,
    pair_policies,
    request_priority_for_class,
    should_send_priority_field,
    source_hashes,
    summarize_post_background,
)
from servescope.workload import select_prompt


def class_priorities_for(policy: str) -> dict[str, int]:
    return {
        CLASS_INTERACTIVE: request_priority_for_class(CLASS_INTERACTIVE, policy),
        CLASS_BACKGROUND: request_priority_for_class(CLASS_BACKGROUND, policy),
    }


def make_telemetry_client(config: dict) -> httpx.AsyncClient:
    tel = config.get("telemetry") or {"client_max_connections": 8}
    return make_client(config, {**config["interactive"], **tel})


def enrich_repeat(summary: dict, records: list[dict], packed: dict, policy: str, config: dict) -> dict:
    slo_s = float(config["slo_p95_ttft_seconds"])
    last_bg = summary.get("last_background_completion_s")
    post = summarize_post_background(records, last_bg, slo_s=slo_s)
    bg = packed.get("background_goodput") or {}
    summary["scheduler_policy"] = policy
    summary["slo_met_in_30_60_window"] = summary.get("slo_recovered_afterward")
    summary["post_background_count"] = post["count"]
    summary["post_background_ttft_p50_s"] = post["client_ttft_p50_s"]
    summary["post_background_ttft_p95_s"] = post["client_ttft_p95_s"]
    summary["post_background_ttft_p99_s"] = post["client_ttft_p99_s"]
    summary["post_background_slo_met"] = post["slo_met"]
    summary["background_e2e_p50_s"] = bg.get("client_e2e_p50_s")
    summary["background_e2e_p95_s"] = bg.get("client_e2e_p95_s")
    summary["background_e2e_p99_s"] = bg.get("client_e2e_p99_s")
    return summary


async def run_policy_attempt(
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
    policy: str,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    idle = await wait_until_idle(telemetry_client, config)
    if not idle:
        print("WARNING: queue did not return to idle before run", flush=True)
    print(f"=== {repeat_id} policy={policy} scenario={scenario} ===", flush=True)
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
        scheduler_policy=policy,
        class_priorities=class_priorities_for(policy),
        send_priority_field=should_send_priority_field(policy),
    )
    write_jsonl(out_dir / "requests.jsonl", records)
    for row in runtime_rows:
        row.update(
            {
                "repeat_id": repeat_id,
                "scenario": scenario,
                "scheduler_policy": policy,
                "background_offered_rps": background_rps,
            }
        )
    for row in gpu_rows:
        row.update(
            {
                "repeat_id": repeat_id,
                "scenario": scenario,
                "scheduler_policy": policy,
                "background_offered_rps": background_rps,
            }
        )
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
    summary = enrich_repeat(packed["repeat"], records, packed, policy, config)
    print(
        f"{repeat_id} valid={summary['valid_offered_load']} "
        f"ix_rps={summary['interactive_actual_dispatch_rps']} "
        f"bg_rps={summary['background_actual_dispatch_rps']} "
        f"p95_burst={summary['interactive_p95_burst_s']} "
        f"wait={summary['max_waiting_requests']} "
        f"post_p95={summary['post_background_ttft_p95_s']} "
        f"reason={summary['invalid_reason']}",
        flush=True,
    )
    idle = await wait_until_idle(telemetry_client, config)
    print(f"drain idle={idle} last_bg={summary['last_background_completion_s']}", flush=True)
    await asyncio.sleep(float(config["inter_run_idle_s"]))
    return summary, packed["phase_rows"], runtime_rows, gpu_rows, records


async def collect_policy_repeats(
    interactive_client,
    background_client,
    telemetry_client,
    config,
    out_dir: Path,
    *,
    suite_id: str,
    scenario: str,
    background_rps: float | None,
    policy: str,
    needed: int,
    max_attempts: int,
    label: str,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    repeats, phases, runtime, gpu, last_records = [], [], [], [], []
    valid = 0
    attempt = 0
    while attempt < max_attempts and valid < needed:
        attempt += 1
        summary, phase_rows, runtime_rows, gpu_rows, records = await run_policy_attempt(
            interactive_client,
            background_client,
            telemetry_client,
            config,
            out_dir,
            suite_id=suite_id,
            repeat_id=f"{label}-rep{attempt}",
            scenario=scenario,
            background_rps=background_rps,
            policy=policy,
        )
        repeats.append(summary)
        phases.extend(phase_rows)
        runtime.extend(runtime_rows)
        gpu.extend(gpu_rows)
        if summary["valid_offered_load"]:
            valid += 1
            last_records = records
    return repeats, phases, runtime, gpu, last_records


def write_manifest(out_dir: Path, config: dict, extra: dict) -> dict:
    nofile_soft, nofile_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    versions = package_versions()
    policy = extra.get("scheduler_policy")
    if policy == POLICY_PRIORITY:
        server_command = (
            "vllm serve Qwen/Qwen3-1.7B --gpu-memory-utilization 0.85 "
            "--enforce-eager --enable-per-request-metrics --scheduling-policy priority"
        )
    else:
        server_command = (
            "vllm serve Qwen/Qwen3-1.7B --gpu-memory-utilization 0.85 "
            "--enforce-eager --enable-per-request-metrics"
        )
    manifest = {
        "suite_id": out_dir.name,
        "git": git_state(ROOT),
        "source_hashes": source_hashes(ROOT),
        "model": config["model"],
        "backend": "vllm",
        "backend_version": versions.get("vllm"),
        "scheduler_policy": policy,
        "server_command": server_command,
        "server_env": {
            "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "HF_HOME": "/home/mayur/.cache/huggingface",
        },
        "priority_mechanism": "OpenAI chat request-body field `priority`",
        "priority_semantics": "lower integer is handled earlier; ties by arrival time then request id",
        "class_priorities": extra.get("class_priorities"),
        "class_configs": {
            "interactive": config["interactive"],
            "background": config["background"],
            "telemetry": config.get("telemetry"),
        },
        "ulimit_n_soft": nofile_soft,
        "ulimit_n_hard": nofile_hard,
        "notes": [
            "WSLg/Xwayland shares the GPU with the vLLM process.",
            "vLLM is running with --enforce-eager. That is a known WSL limitation.",
            "P3 isolates native --scheduling-policy only. No ServeScope scheduler.",
            "Interactive and background keep separate HTTP pools. Telemetry is a third small client.",
        ],
        "package_versions": versions,
        **extra,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def plot_fcfs_vs_priority_ttft(path: Path, fcfs: list[dict], priority: list[dict]) -> None:
    labels = ["pre", "burst", "30-60 window", "post-background"]
    keys = [
        "interactive_p95_pre_s",
        "interactive_p95_burst_s",
        "interactive_p95_recovery_s",
        "post_background_ttft_p95_s",
    ]

    def med(rows, key):
        values = [row[key] for row in rows if row.get(key) is not None]
        from servescope.metrics import percentile

        return percentile(values, 50) if values else 0.0

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = range(len(labels))
    width = 0.35
    ax.bar([x - width / 2 for x in xs], [med(fcfs, k) for k in keys], width, label="FCFS p95")
    ax.bar([x + width / 2 for x in xs], [med(priority, k) for k in keys], width, label="native priority p95")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylabel("interactive p95 client TTFT (seconds)")
    ax.set_title("FCFS vs native priority interactive p95")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_queue(path: Path, fcfs_runtime: list[dict], pri_runtime: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(
        [row["t_s"] for row in fcfs_runtime],
        [row.get("num_requests_waiting") for row in fcfs_runtime],
        label="FCFS waiting",
    )
    ax.plot(
        [row["t_s"] for row in pri_runtime],
        [row.get("num_requests_waiting") for row in pri_runtime],
        label="priority waiting",
    )
    ax.axvspan(15, 30, color="tab:orange", alpha=0.15, label="background injection 15-30 s")
    ax.set_xlabel("seconds from scenario start")
    ax.set_ylabel("vLLM waiting requests")
    ax.set_title("Waiting queue: FCFS vs native priority")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_tradeoff(path: Path, fcfs: list[dict], priority: list[dict]) -> None:
    from servescope.metrics import percentile

    def med(rows, key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return percentile(values, 50) if values else 0.0

    fig, ax1 = plt.subplots(figsize=(6.8, 4.2))
    labels = ["FCFS", "native priority"]
    ttft = [med(fcfs, "interactive_p95_burst_s"), med(priority, "interactive_p95_burst_s")]
    goodput = [
        med(fcfs, "background_output_token_goodput_tps"),
        med(priority, "background_output_token_goodput_tps"),
    ]
    ax1.bar([0, 1], ttft, color="tab:blue", width=0.4)
    ax1.set_ylabel("interactive burst p95 TTFT (s)")
    ax2 = ax1.twinx()
    ax2.plot([0, 1], goodput, color="tab:orange", marker="o", label="background tok/s")
    ax2.set_ylabel("background output-token goodput (tok/s)")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(labels)
    ax1.set_title("Interactive protection vs background goodput")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


async def smoke(config: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = config["base_url"].rstrip("/") + config["chat_path"]
    prompt_id, prompt = select_prompt(int(config["seed"]), 0)
    results = []
    async with make_client(config, config["interactive"]) as client:
        default_payload = build_chat_payload(
            model=config["model"],
            prompt=prompt,
            temperature=0,
            min_tokens=8,
            max_completion_tokens=8,
        )
        default = await stream_chat_request(
            client,
            url=url,
            payload=default_payload,
            request_id="smoke-default",
            prompt_id=prompt_id,
            workload_class=CLASS_INTERACTIVE,
            scheduled_s=time.perf_counter(),
            timeout_s=60.0,
            clock=time.perf_counter,
        )
        results.append({"case": "fcfs_default_no_priority_field", **{k: default.get(k) for k in ("status", "http_status", "error")}})
        nonzero = build_chat_payload(
            model=config["model"],
            prompt=prompt,
            temperature=0,
            min_tokens=8,
            max_completion_tokens=8,
            priority=1,
        )
        rejected = await stream_chat_request(
            client,
            url=url,
            payload=nonzero,
            request_id="smoke-priority-1-on-fcfs",
            prompt_id=prompt_id,
            workload_class=CLASS_BACKGROUND,
            scheduled_s=time.perf_counter(),
            timeout_s=60.0,
            clock=time.perf_counter,
        )
        results.append(
            {
                "case": "fcfs_priority_1_should_reject",
                "status": rejected.get("status"),
                "http_status": rejected.get("http_status"),
                "error": rejected.get("error"),
            }
        )
    payload = {
        "vllm_version": package_versions().get("vllm"),
        "scheduling_cli": "--scheduling-policy {fcfs,priority}",
        "priority_field": "chat request body `priority`",
        "header_also_exists": "X-Vllm-Priority",
        "chosen_mechanism": "request-body priority",
        "semantics": "lower integer is handled earlier",
        "results": results,
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_manifest(
        out_dir,
        config,
        {
            "mode": "smoke",
            "scheduler_policy": POLICY_FCFS,
            "class_priorities": {"interactive": 0, "background": 0},
            "start_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps(payload, indent=2), flush=True)
    return payload


async def run_policy_suite(config: dict, out_dir: Path, policy: str, *, include_control: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (out_dir / "nvidia_smi_before.txt").write_text(nvidia_smi_processes(), encoding="utf-8")
    start = datetime.now(timezone.utc).isoformat()
    bg_rps = float(config["background"]["offered_rps"])
    needed = int(config["min_valid_repeats"])
    max_attempts = int(config["max_repeat_attempts"])
    tel_cfg = {**config["interactive"], **(config.get("telemetry") or {})}

    async with (
        make_client(config, config["interactive"]) as interactive_client,
        make_client(config, config["background"]) as background_client,
        make_client(config, tel_cfg) as telemetry_client,
    ):
        print("warmup", flush=True)
        await warmup(interactive_client, config)
        print(f"post-warmup idle={await wait_until_idle(telemetry_client, config)}", flush=True)
        control_repeats: list[dict] = []
        if include_control:
            c_reps, _, _, _, _ = await collect_policy_repeats(
                interactive_client,
                background_client,
                telemetry_client,
                config,
                out_dir,
                suite_id=out_dir.name,
                scenario="control",
                background_rps=None,
                policy=policy,
                needed=1,
                max_attempts=1,
                label=f"{policy}-control",
            )
            control_repeats = c_reps
        mixed, phases, runtime, gpu, last_records = await collect_policy_repeats(
            interactive_client,
            background_client,
            telemetry_client,
            config,
            out_dir,
            suite_id=out_dir.name,
            scenario="mixed",
            background_rps=bg_rps,
            policy=policy,
            needed=needed,
            max_attempts=max_attempts,
            label=f"{policy}-mixed",
        )
    write_csv(out_dir / "repeat_summary.csv", control_repeats + mixed)
    write_csv(out_dir / "phase_summary.csv", phases)
    write_csv(out_dir / "runtime_metrics.csv", runtime)
    write_csv(out_dir / "gpu_metrics.csv", gpu)
    result = {
        "scheduler_policy": policy,
        "valid_mixed": sum(1 for row in mixed if row.get("valid_offered_load")),
        "repeats": mixed,
        "control_repeats": control_repeats,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_manifest(
        out_dir,
        config,
        {
            "mode": policy,
            "scheduler_policy": policy,
            "class_priorities": class_priorities_for(policy),
            "send_priority_field": should_send_priority_field(policy),
            "start_utc": start,
            "end_utc": datetime.now(timezone.utc).isoformat(),
            "result": result,
        },
    )
    return result


def write_comparison(out_dir: Path, config: dict, fcfs_dir: Path, pri_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fcfs = json.loads((fcfs_dir / "result.json").read_text(encoding="utf-8"))
    pri = json.loads((pri_dir / "result.json").read_text(encoding="utf-8"))
    fcfs_valid = [row for row in fcfs["repeats"] if row.get("valid_offered_load")]
    pri_valid = [row for row in pri["repeats"] if row.get("valid_offered_load")]
    paired = pair_policies(fcfs_valid, pri_valid)
    write_csv(out_dir / "aggregate_summary.csv", paired)
    plot_fcfs_vs_priority_ttft(out_dir / "fcfs_vs_priority_ttft.png", fcfs_valid, pri_valid)
    # representative last valid runtime series
    import csv as csvlib

    def last_runtime(path: Path, policy: str) -> list[dict]:
        rows = list(csvlib.DictReader((path / "runtime_metrics.csv").open(encoding="utf-8")))
        mixed = [row for row in rows if "mixed" in (row.get("repeat_id") or "")]
        if not mixed:
            return []
        last_id = mixed[-1]["repeat_id"]
        out = []
        for row in mixed:
            if row["repeat_id"] != last_id:
                continue
            waiting = row.get("num_requests_waiting")
            out.append(
                {
                    "t_s": float(row["t_s"]) if row.get("t_s") else 0.0,
                    "num_requests_waiting": float(waiting) if waiting not in (None, "") else None,
                }
            )
        return out

    plot_queue(out_dir / "fcfs_vs_priority_queue.png", last_runtime(fcfs_dir, "fcfs"), last_runtime(pri_dir, "priority"))
    plot_tradeoff(out_dir / "tradeoff.png", fcfs_valid, pri_valid)
    agg = {
        "fcfs_burst_p95": aggregate_numeric(fcfs_valid, "interactive_p95_burst_s"),
        "priority_burst_p95": aggregate_numeric(pri_valid, "interactive_p95_burst_s"),
        "absolute_reduction_s": aggregate_numeric(paired, "absolute_reduction_s"),
        "priority_protection_ratio": aggregate_numeric(paired, "priority_protection_ratio"),
        "fcfs_waiting": aggregate_numeric(fcfs_valid, "max_waiting_requests"),
        "priority_waiting": aggregate_numeric(pri_valid, "max_waiting_requests"),
        "fcfs_background_tok_tps": aggregate_numeric(fcfs_valid, "background_output_token_goodput_tps"),
        "priority_background_tok_tps": aggregate_numeric(pri_valid, "background_output_token_goodput_tps"),
        "fcfs_background_e2e_p95": aggregate_numeric(fcfs_valid, "background_e2e_p95_s"),
        "priority_background_e2e_p95": aggregate_numeric(pri_valid, "background_e2e_p95_s"),
        "fcfs_last_bg": aggregate_numeric(fcfs_valid, "last_background_completion_s"),
        "priority_last_bg": aggregate_numeric(pri_valid, "last_background_completion_s"),
        "fcfs_post_bg_p95": aggregate_numeric(fcfs_valid, "post_background_ttft_p95_s"),
        "priority_post_bg_p95": aggregate_numeric(pri_valid, "post_background_ttft_p95_s"),
        "pairs": paired,
    }
    conclusion = conclude(agg)
    result = {"conclusion": conclusion, "aggregates": agg, "source_hashes": source_hashes(ROOT)}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_manifest(
        out_dir,
        config,
        {
            "mode": "comparison",
            "scheduler_policy": "fcfs_vs_priority",
            "class_priorities": class_priorities_for(POLICY_PRIORITY),
            "fcfs_dir": str(fcfs_dir),
            "priority_dir": str(pri_dir),
            "conclusion": conclusion,
            "aggregates": agg,
        },
    )
    print(f"comparison conclusion={conclusion}", flush=True)
    return result


def conclude(agg: dict) -> str:
    pri = (agg.get("priority_burst_p95") or {}).get("median")
    fcfs = (agg.get("fcfs_burst_p95") or {}).get("median")
    if pri is None or fcfs is None:
        return "native priority result invalid/inconclusive"
    ratio = pri / fcfs if fcfs > 0 else None
    if pri < 1.0 and (ratio is not None and ratio <= 0.25):
        return "native priority substantially mitigated the mixed-workload interference"
    if pri < 1.0:
        return "native priority substantially mitigated the mixed-workload interference"
    if ratio is not None and ratio <= 0.50 and pri >= 1.0:
        return "native priority helps but leaves meaningful interference"
    if ratio is not None and ratio > 0.80:
        return "native priority barely helps"
    if pri < fcfs:
        return "native priority helps but leaves meaningful interference"
    return "native priority barely helps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P3 native vLLM priority baseline")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "p3_native_priority.json")
    parser.add_argument("--mode", choices=("smoke", "fcfs", "priority", "compare"), required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fcfs-dir", type=Path, default=None)
    parser.add_argument("--priority-dir", type=Path, default=None)
    parser.add_argument("--include-control", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    stamp = now_utc()
    if args.mode == "smoke":
        out = args.out or (ROOT / "artifacts" / "p3" / f"smoke-{stamp}")
        print(f"writing {out}", flush=True)
        asyncio.run(smoke(config, out))
    elif args.mode == "compare":
        if not args.fcfs_dir or not args.priority_dir:
            raise SystemExit("--fcfs-dir and --priority-dir are required for compare")
        out = args.out or (ROOT / "artifacts" / "p3" / f"comparison-{stamp}")
        print(f"writing {out}", flush=True)
        write_comparison(out, config, args.fcfs_dir, args.priority_dir)
    else:
        out = args.out or (ROOT / "artifacts" / "p3" / f"{args.mode}-{stamp}")
        print(f"writing {out}", flush=True)
        include_control = args.include_control or args.mode == POLICY_PRIORITY
        asyncio.run(run_policy_suite(config, out, args.mode, include_control=include_control))
    print(f"done {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
