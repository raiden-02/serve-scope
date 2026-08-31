"""Mixed-workload phases and class-specific summaries. Request-clock math stays in metrics.py."""

from __future__ import annotations

from typing import Any

from servescope.metrics import (
    completed_records,
    mark_repeat_validity,
    percentile,
    summarize_latencies,
    summarize_repeat,
)

PHASE_PRE = "pre_burst"
PHASE_BURST = "burst_injection"
PHASE_RECOVERY = "recovery"
PHASES = (PHASE_PRE, PHASE_BURST, PHASE_RECOVERY)
CLASS_INTERACTIVE = "interactive"
CLASS_BACKGROUND = "background"


def classify_phase(
    t_rel_s: float,
    *,
    pre_end_s: float = 15.0,
    burst_end_s: float = 30.0,
) -> str:
    """Map a time relative to scenario start onto a mixed-workload phase.

    Boundaries: [0, pre_end) pre, [pre_end, burst_end) burst, [burst_end, inf) recovery.
    Uses the request's scheduled or attempt time, never completion.
    """
    if t_rel_s < pre_end_s:
        return PHASE_PRE
    if t_rel_s < burst_end_s:
        return PHASE_BURST
    return PHASE_RECOVERY


def request_phase_time_s(row: dict[str, Any], t0: float) -> float | None:
    stamp = row.get("request_attempt_s")
    if stamp is None:
        stamp = row.get("scheduled_arrival_s")
    if stamp is None:
        return None
    return float(stamp) - float(t0)


def attach_phase(
    row: dict[str, Any],
    t0: float,
    *,
    pre_end_s: float = 15.0,
    burst_end_s: float = 30.0,
) -> dict[str, Any]:
    t_rel = request_phase_time_s(row, t0)
    row["phase"] = classify_phase(t_rel, pre_end_s=pre_end_s, burst_end_s=burst_end_s) if t_rel is not None else None
    row["t_rel_s"] = t_rel
    return row


def interference_ratio(mixed_burst_p95: float | None, control_aligned_p95: float | None) -> float | None:
    """Descriptive mixed/control p95 ratio. Not a universal slowdown factor."""
    if mixed_burst_p95 is None or control_aligned_p95 is None:
        return None
    if control_aligned_p95 <= 0:
        return None
    return float(mixed_burst_p95) / float(control_aligned_p95)


def absolute_ttft_increase(mixed_p95: float | None, control_p95: float | None) -> float | None:
    if mixed_p95 is None or control_p95 is None:
        return None
    return float(mixed_p95) - float(control_p95)


def filter_class(records: list[dict[str, Any]], workload_class: str) -> list[dict[str, Any]]:
    return [row for row in records if row.get("workload_class") == workload_class]


def filter_phase(records: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    return [row for row in records if row.get("phase") == phase]


def summarize_phase_latencies(records: list[dict[str, Any]], *, slo_s: float = 1.0) -> dict[str, Any]:
    completed = completed_records(records)
    errors = [row for row in records if row.get("status") not in {"success", "length"}]
    ttfts = [row["client_ttft_s"] for row in completed if row.get("client_ttft_s") is not None]
    e2es = [row["client_e2e_s"] for row in completed if row.get("client_e2e_s") is not None]
    ttft = summarize_latencies(ttfts)
    e2e = summarize_latencies(e2es)
    p95 = ttft["p95"]
    return {
        "count": len(records),
        "completed_count": len(completed),
        "error_count": len(errors),
        "error_rate": (len(errors) / len(records)) if records else None,
        "client_ttft_p50_s": ttft["p50"],
        "client_ttft_p95_s": p95,
        "client_ttft_p99_s": ttft["p99"],
        "client_ttft_n": ttft["n"],
        "client_e2e_p50_s": e2e["p50"],
        "client_e2e_p95_s": e2e["p95"],
        "client_e2e_p99_s": e2e["p99"],
        "slo_met": p95 is not None and p95 < slo_s,
    }


def class_schedule_summary(records: list[dict[str, Any]], offered_rps: float, duration_s: float) -> dict[str, Any]:
    """Reuse summarize_repeat / mark_repeat_validity for one workload class."""
    summary = summarize_repeat(records, offered_rps=offered_rps, duration_s=duration_s)
    return summary


def mark_class_validity(
    summary: dict[str, Any],
    *,
    max_inflight: int | None = None,
) -> dict[str, Any]:
    return mark_repeat_validity(summary, max_inflight=max_inflight)


def last_completion_rel_s(records: list[dict[str, Any]], t0: float) -> float | None:
    stamps = [row.get("completion_s") for row in records if row.get("completion_s") is not None]
    if not stamps:
        return None
    return float(max(stamps)) - float(t0)


def first_attempt_rel_s(records: list[dict[str, Any]], t0: float) -> float | None:
    stamps = [
        row.get("request_attempt_s") if row.get("request_attempt_s") is not None else row.get("scheduled_arrival_s")
        for row in records
    ]
    stamps = [value for value in stamps if value is not None]
    if not stamps:
        return None
    return float(min(stamps)) - float(t0)


def background_goodput(records: list[dict[str, Any]], t0: float) -> dict[str, Any]:
    completed = completed_records(records)
    output_tokens = sum(int(row["completion_tokens"]) for row in completed if row.get("completion_tokens") is not None)
    prompt_tokens = [int(row["prompt_tokens"]) for row in completed if row.get("prompt_tokens") is not None]
    output_token_list = [int(row["completion_tokens"]) for row in completed if row.get("completion_tokens") is not None]
    first_rel = first_attempt_rel_s(records, t0)
    last_rel = last_completion_rel_s(records, t0)
    span = None
    if first_rel is not None and last_rel is not None and last_rel > first_rel:
        span = last_rel - first_rel
    e2es = [row["client_e2e_s"] for row in completed if row.get("client_e2e_s") is not None]
    e2e = summarize_latencies(e2es)
    return {
        "offered_count": len(records),
        "completed_count": len(completed),
        "error_count": len(records) - len(completed),
        "prompt_tokens_p50": percentile(prompt_tokens, 50) if prompt_tokens else None,
        "prompt_tokens_min": min(prompt_tokens) if prompt_tokens else None,
        "prompt_tokens_max": max(prompt_tokens) if prompt_tokens else None,
        "output_tokens_p50": percentile(output_token_list, 50) if output_token_list else None,
        "output_tokens_sum": output_tokens,
        "wall_clock_completed_request_tps": (len(completed) / span) if span else None,
        "wall_clock_output_token_tps": (output_tokens / span) if span else None,
        "busy_span_s": span,
        "first_attempt_rel_s": first_rel,
        "last_completion_rel_s": last_rel,
        "client_e2e_p50_s": e2e["p50"],
        "client_e2e_p95_s": e2e["p95"],
        "client_e2e_p99_s": e2e["p99"],
    }


def runtime_phase_summary(
    runtime_rows: list[dict[str, Any]],
    *,
    pre_end_s: float = 15.0,
    burst_end_s: float = 30.0,
    scenario_end_s: float = 60.0,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    windows = {
        PHASE_PRE: (0.0, pre_end_s),
        PHASE_BURST: (pre_end_s, burst_end_s),
        PHASE_RECOVERY: (burst_end_s, scenario_end_s),
    }
    for phase, (start, end) in windows.items():
        samples = [row for row in runtime_rows if start <= (row.get("t_s") or -1) < end]
        waiting = [row["num_requests_waiting"] for row in samples if row.get("num_requests_waiting") is not None]
        running = [row["num_requests_running"] for row in samples if row.get("num_requests_running") is not None]
        kv = [row["kv_cache_usage_perc"] for row in samples if row.get("kv_cache_usage_perc") is not None]
        out[phase] = {
            "sample_count": len(samples),
            "max_waiting_requests": max(waiting) if waiting else None,
            "p50_waiting_requests": percentile(waiting, 50) if waiting else None,
            "p95_waiting_requests": percentile(waiting, 95) if waiting else None,
            "max_running_requests": max(running) if running else None,
            "p50_running_requests": percentile(running, 50) if running else None,
            "max_kv_cache_usage_perc": max(kv) if kv else None,
            "waiting_positive_samples": sum(1 for value in waiting if value > 0),
        }
    return out


def sustained_waiting_queue(phase_runtime: dict[str, Any], *, min_positive_samples: int = 3) -> bool:
    """True when the burst window shows a waiting queue that is not a single blip."""
    max_waiting = phase_runtime.get("max_waiting_requests") or 0
    positive = phase_runtime.get("waiting_positive_samples") or 0
    return max_waiting >= 1 and positive >= min_positive_samples


def slo_flags(phase_summaries: dict[str, dict[str, Any]]) -> dict[str, bool | None]:
    return {
        "slo_met_before_burst": phase_summaries.get(PHASE_PRE, {}).get("slo_met"),
        "slo_met_during_burst": phase_summaries.get(PHASE_BURST, {}).get("slo_met"),
        "slo_recovered_afterward": phase_summaries.get(PHASE_RECOVERY, {}).get("slo_met"),
    }


def bin_interactive_ttft(records: list[dict[str, Any]], *, bin_s: float = 1.0) -> list[dict[str, Any]]:
    """One-second bins of interactive TTFT. Empty bins are omitted."""
    buckets: dict[int, list[float]] = {}
    for row in completed_records(records):
        if row.get("workload_class") != CLASS_INTERACTIVE:
            continue
        t_rel = row.get("t_rel_s")
        ttft = row.get("client_ttft_s")
        if t_rel is None or ttft is None:
            continue
        idx = int(t_rel // bin_s)
        buckets.setdefault(idx, []).append(float(ttft))
    rows = []
    for idx in sorted(buckets):
        values = buckets[idx]
        rows.append(
            {
                "t_s": idx * bin_s,
                "n": len(values),
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
            }
        )
    return rows
