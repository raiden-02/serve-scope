"""Measurement helpers: SSE parse, status, percentiles, summaries.

Percentiles use NumPy ``method='linear'`` (R type 7). That choice is fixed
for this project so aggregates stay comparable across runs.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

COMPLETED_STATUSES = frozenset({"success", "length"})
TERMINAL_FINISH_REASONS = frozenset({"length", "stop", "eos_token"})
MIN_VALID_REPEATS_FOR_CLEAN = 3

# Prometheus names confirmed on the local vLLM 0.28.0 /metrics endpoint.
METRIC_RUNNING = "vllm:num_requests_running"
METRIC_WAITING = "vllm:num_requests_waiting"
METRIC_KV = "vllm:kv_cache_usage_perc"


def sse_data_payloads(raw: str | bytes) -> list[str]:
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = raw
    payloads: list[str] = []
    for line in text.splitlines():
        if line.startswith("data:"):
            payloads.append(line[5:].lstrip())
    return payloads


def is_nonempty_generated_content(delta: dict[str, Any] | None) -> bool:
    """True only for nonempty generated text.

    Role-only, empty, and reasoning-only deltas do not count as first content.
    """
    if not isinstance(delta, dict):
        return False
    content = delta.get("content")
    return isinstance(content, str) and content != ""


def extract_finish_reason(event: dict[str, Any]) -> str | None:
    choices = event.get("choices") or []
    if not choices:
        return None
    reason = choices[0].get("finish_reason")
    return reason if isinstance(reason, str) and reason else None


def extract_usage(event: dict[str, Any]) -> dict[str, int | None]:
    usage = event.get("usage") or {}
    return {
        "prompt_tokens": _int_or_none(usage.get("prompt_tokens")),
        "completion_tokens": _int_or_none(usage.get("completion_tokens")),
        "total_tokens": _int_or_none(usage.get("total_tokens")),
    }


def extract_backend_metrics(event: dict[str, Any]) -> dict[str, float | None]:
    metrics = event.get("metrics")
    if not isinstance(metrics, dict):
        return {
            "backend_queue_time_ms": None,
            "backend_time_to_first_token_ms": None,
            "backend_generation_time_ms": None,
            "backend_mean_itl_ms": None,
            "backend_tokens_per_second": None,
        }
    return {
        "backend_queue_time_ms": _float_or_none(metrics.get("queue_time_ms")),
        "backend_time_to_first_token_ms": _float_or_none(metrics.get("time_to_first_token_ms")),
        "backend_generation_time_ms": _float_or_none(metrics.get("generation_time_ms")),
        "backend_mean_itl_ms": _float_or_none(metrics.get("mean_itl_ms")),
        "backend_tokens_per_second": _float_or_none(metrics.get("tokens_per_second")),
    }


def classify_status(
    *,
    http_status: int | None,
    finish_reason: str | None,
    got_content: bool,
    stream_done: bool,
    timed_out: bool,
    cancelled: bool,
    stream_error: bool,
    error_message: str | None,
    client_capacity: bool = False,
) -> str:
    if cancelled:
        return "cancelled"
    if client_capacity:
        return "client_capacity"
    if timed_out:
        return "timeout"
    if stream_error:
        return "stream_error"
    if http_status is not None and http_status >= 400:
        return "http_error"
    if http_status == 200 and finish_reason == "length" and got_content and stream_done:
        return "length"
    if http_status == 200 and finish_reason in {"stop", "eos_token"} and got_content and stream_done:
        return "success"
    if http_status == 200 and finish_reason not in TERMINAL_FINISH_REASONS:
        return "stream_error"
    if error_message:
        return "other_error"
    return "other_error"


def percentile(values: list[float], q: float) -> float | None:
    """q is 0-100. Empty input returns None. Uses NumPy linear interpolation."""
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, q, method="linear"))


def parse_prometheus_gauges(text: str, names: set[str]) -> dict[str, float]:
    found: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric_name = line.split("{", 1)[0].split(" ", 1)[0]
        if metric_name not in names:
            continue
        try:
            value = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        found[metric_name] = value
    return found


def summarize_latencies(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "mean": float(np.mean(values)) if values else None,
        "max": float(max(values)) if values else None,
    }


def derive_request_timings(
    *,
    scheduled_arrival_s: float,
    request_attempt_s: float,
    response_headers_s: float | None,
    first_content_s: float | None,
    completion_s: float | None,
) -> dict[str, float | None]:
    """Pure timing math. Header delay is diagnostic and stays inside client TTFT."""
    headers_latency = None
    if response_headers_s is not None:
        headers_latency = response_headers_s - request_attempt_s
    return {
        "scheduled_arrival_s": scheduled_arrival_s,
        "request_attempt_s": request_attempt_s,
        "response_headers_s": response_headers_s,
        "first_content_s": first_content_s,
        "completion_s": completion_s,
        "dispatch_lag_s": request_attempt_s - scheduled_arrival_s,
        "response_headers_latency_s": headers_latency,
        "client_ttft_s": (first_content_s - request_attempt_s) if first_content_s is not None else None,
        "client_e2e_s": (completion_s - request_attempt_s) if completion_s is not None else None,
        "actual_dispatch_s": request_attempt_s,
    }


def completed_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in records if row.get("status") in COMPLETED_STATUSES]


def summarize_repeat(records: list[dict[str, Any]], offered_rps: float, duration_s: float) -> dict[str, Any]:
    completed = completed_records(records)
    errors = [row for row in records if row.get("status") not in COMPLETED_STATUSES]
    ttfts = [row["client_ttft_s"] for row in completed if row.get("client_ttft_s") is not None]
    e2es = [row["client_e2e_s"] for row in completed if row.get("client_e2e_s") is not None]
    dispatch_lags = [row["dispatch_lag_s"] for row in records if row.get("dispatch_lag_s") is not None]
    header_latencies = [
        row["response_headers_latency_s"]
        for row in records
        if row.get("response_headers_latency_s") is not None
    ]
    output_tokens = sum(int(row["completion_tokens"]) for row in completed if row.get("completion_tokens") is not None)
    attempts = [
        row["request_attempt_s"] if row.get("request_attempt_s") is not None else row.get("actual_dispatch_s")
        for row in records
    ]
    attempts = [value for value in attempts if value is not None]
    actual_rps = None
    if len(attempts) >= 2:
        span = max(attempts) - min(attempts)
        actual_rps = (len(attempts) - 1) / span if span > 0 else None
    client_limited = False
    if actual_rps is not None and offered_rps > 0:
        client_limited = actual_rps < 0.90 * offered_rps
    if dispatch_lags:
        median_lag = percentile(dispatch_lags, 50) or 0.0
        if median_lag > 0.050:
            client_limited = True
    queue_ms = [
        row["backend_queue_time_ms"]
        for row in completed
        if row.get("backend_queue_time_ms") is not None
    ]
    backend_ttft_ms = [
        row["backend_time_to_first_token_ms"]
        for row in completed
        if row.get("backend_time_to_first_token_ms") is not None
    ]
    ttft_summary = summarize_latencies(ttfts)
    e2e_summary = summarize_latencies(e2es)
    queue_summary = summarize_latencies(queue_ms)
    backend_ttft_summary = summarize_latencies(backend_ttft_ms)
    header_summary = summarize_latencies(header_latencies)
    return {
        "offered_rps": offered_rps,
        "actual_dispatch_rps": actual_rps,
        "duration_s": duration_s,
        "request_count": len(records),
        "completed_count": len(completed),
        "error_count": len(errors),
        "error_rate": (len(errors) / len(records)) if records else None,
        "client_limited": client_limited,
        "client_ttft_p50_s": ttft_summary["p50"],
        "client_ttft_p95_s": ttft_summary["p95"],
        "client_ttft_p99_s": ttft_summary["p99"],
        "client_ttft_n": ttft_summary["n"],
        "client_e2e_p50_s": e2e_summary["p50"],
        "client_e2e_p95_s": e2e_summary["p95"],
        "client_e2e_p99_s": e2e_summary["p99"],
        "client_e2e_n": e2e_summary["n"],
        "wall_clock_completed_request_throughput_rps": (len(completed) / duration_s) if duration_s > 0 else None,
        "wall_clock_output_token_throughput_tps": (output_tokens / duration_s) if duration_s > 0 else None,
        "backend_queue_time_p50_ms": queue_summary["p50"],
        "backend_queue_time_p95_ms": queue_summary["p95"],
        "backend_queue_time_n": queue_summary["n"],
        "backend_ttft_p50_ms": backend_ttft_summary["p50"],
        "backend_ttft_p95_ms": backend_ttft_summary["p95"],
        "backend_ttft_n": backend_ttft_summary["n"],
        "response_headers_latency_p50_s": header_summary["p50"],
        "response_headers_latency_p95_s": header_summary["p95"],
        "response_headers_latency_n": header_summary["n"],
        "median_dispatch_lag_s": percentile(dispatch_lags, 50) if dispatch_lags else None,
        "status_counts": _count_by(records, "status"),
        "finish_reason_counts": _count_by(records, "finish_reason"),
    }


def mark_repeat_validity(
    summary: dict[str, Any],
    *,
    max_inflight: int | None = None,
) -> dict[str, Any]:
    """Mark whether a repeat is valid clean offered-load evidence.

    Invalid repeats stay in raw data and the repeat summary. They are excluded
    from latency/throughput that claims to describe a clean offered load.
    """
    reasons: list[str] = []
    if summary.get("aborted"):
        summary["client_limited"] = True
        reasons.append(summary.get("abort_reason") or "aborted")
    offered = summary.get("offered_rps") or 0.0
    actual = summary.get("actual_dispatch_rps")
    if actual is not None and offered > 0 and actual < 0.90 * offered:
        summary["client_limited"] = True
        reasons.append("actual_dispatch_rps_below_90pct")
    median_lag = summary.get("median_dispatch_lag_s")
    if median_lag is not None and median_lag > 0.050:
        summary["client_limited"] = True
        reasons.append("median_dispatch_lag_gt_50ms")
    status_counts = summary.get("status_counts") or {}
    if status_counts.get("client_capacity"):
        summary["client_limited"] = True
        reasons.append("httpx_pool_timeout")
    if summary.get("client_limited") and not reasons:
        reasons.append("client_limited")
    unique_reasons: list[str] = []
    for reason in reasons:
        if reason not in unique_reasons:
            unique_reasons.append(reason)
    summary["valid_offered_load"] = not bool(summary.get("client_limited")) and not bool(summary.get("aborted"))
    summary["invalid_reason"] = None if summary["valid_offered_load"] else ", ".join(unique_reasons)
    peak = summary.get("peak_inflight")
    if peak is not None and max_inflight:
        summary["peak_inflight_fraction_of_limit"] = float(peak) / float(max_inflight)
    else:
        summary.setdefault("peak_inflight_fraction_of_limit", None)
    return summary


def aggregate_repeats(
    repeat_summaries: list[dict[str, Any]],
    *,
    min_valid_repeats: int = MIN_VALID_REPEATS_FOR_CLEAN,
    slo_s: float = 1.0,
) -> dict[str, Any]:
    if not repeat_summaries:
        return {}
    offered = repeat_summaries[0]["offered_rps"]
    all_p95s = [row["client_ttft_p95_s"] for row in repeat_summaries if row.get("client_ttft_p95_s") is not None]
    valid = [row for row in repeat_summaries if row.get("valid_offered_load")]
    invalid = [row for row in repeat_summaries if not row.get("valid_offered_load")]
    valid_p95s = [row["client_ttft_p95_s"] for row in valid if row.get("client_ttft_p95_s") is not None]
    valid_p50s = [row["client_ttft_p50_s"] for row in valid if row.get("client_ttft_p50_s") is not None]
    throughputs = [
        row["wall_clock_output_token_throughput_tps"]
        for row in valid
        if row.get("wall_clock_output_token_throughput_tps") is not None
    ]
    req_tps = [
        row["wall_clock_completed_request_throughput_rps"]
        for row in valid
        if row.get("wall_clock_completed_request_throughput_rps") is not None
    ]
    valid_fail = sum(1 for value in valid_p95s if value >= slo_s)
    valid_pass = sum(1 for value in valid_p95s if value < slo_s)
    valid_waiting = [
        row.get("max_waiting_requests")
        for row in valid
        if row.get("max_waiting_requests") is not None
    ]
    return {
        "offered_rps": offered,
        "total_repeat_count": len(repeat_summaries),
        "valid_repeat_count": len(valid),
        "invalid_repeat_count": len(invalid),
        "aggregate_valid": len(valid) >= min_valid_repeats,
        "any_client_limited": any(row.get("client_limited") for row in repeat_summaries),
        "all_repeat_p95s": all_p95s,
        "valid_repeat_p95s": valid_p95s,
        "headline_p95_ttft_s_median": percentile(valid_p95s, 50),
        "headline_p95_ttft_s_min": min(valid_p95s) if valid_p95s else None,
        "headline_p95_ttft_s_max": max(valid_p95s) if valid_p95s else None,
        "headline_p50_ttft_s_median": percentile(valid_p50s, 50),
        "headline_output_token_tps_median": percentile(throughputs, 50),
        "headline_completed_rps_median": percentile(req_tps, 50),
        "slo_p95_ttft_seconds": slo_s,
        "slo_met_all_valid_repeats": bool(valid_p95s) and valid_fail == 0,
        "slo_violated_all_valid_repeats": bool(valid_p95s)
        and len(valid_p95s) >= min_valid_repeats
        and valid_pass == 0,
        "valid_repeats_mixed_slo": valid_pass > 0 and valid_fail > 0,
        "any_valid_waiting_queue": any(value and value > 0 for value in valid_waiting),
        "repeats": len(repeat_summaries),
    }


def first_clean_slo_violation(
    aggregates: list[dict[str, Any]],
    slo_s: float = 1.0,
    min_valid_repeats: int = MIN_VALID_REPEATS_FOR_CLEAN,
) -> float | None:
    """First clean offered RPS where every valid repeat misses the SLO.

    Needs at least min_valid_repeats valid repeats, all failing. Mixed
    fail/pass is instability, not a deterministic crossing.
    """
    ordered = sorted(aggregates, key=lambda row: row["offered_rps"])
    for row in ordered:
        if not row.get("aggregate_valid"):
            continue
        if int(row.get("valid_repeat_count") or 0) < min_valid_repeats:
            continue
        if row.get("slo_violated_all_valid_repeats"):
            return float(row["offered_rps"])
    return None


def first_slo_violation(
    aggregates: list[dict[str, Any]],
    slo_s: float = 1.0,
    min_valid_repeats: int = MIN_VALID_REPEATS_FOR_CLEAN,
) -> float | None:
    """Alias for first_clean_slo_violation. Invalid aggregates cannot cross."""
    return first_clean_slo_violation(aggregates, slo_s=slo_s, min_valid_repeats=min_valid_repeats)


def first_observed_valid_collapse_rps(aggregates: list[dict[str, Any]], slo_s: float = 1.0) -> float | None:
    """Lowest tested rate with at least one valid repeat whose p95 TTFT >= slo_s."""
    ordered = sorted(aggregates, key=lambda row: row["offered_rps"])
    for row in ordered:
        for value in row.get("valid_repeat_p95s") or []:
            if value is not None and value >= slo_s:
                return float(row["offered_rps"])
    return None


def instability_region_rps(aggregates: list[dict[str, Any]], slo_s: float = 1.0) -> list[float] | None:
    """Min and max tested rates that look unstable from valid repeats.

    A rate is included if valid repeats mix pass/fail, or a valid repeat
    collapsed, or a valid repeat showed a waiting queue. This is a description
    of the tested points, not a fitted threshold.
    """
    interesting: list[float] = []
    for row in sorted(aggregates, key=lambda item: item["offered_rps"]):
        p95s = [value for value in (row.get("valid_repeat_p95s") or []) if value is not None]
        if not p95s:
            continue
        passes = sum(1 for value in p95s if value < slo_s)
        fails = sum(1 for value in p95s if value >= slo_s)
        if (passes and fails) or fails or row.get("any_valid_waiting_queue"):
            interesting.append(float(row["offered_rps"]))
    if not interesting:
        return None
    return [min(interesting), max(interesting)]


def _count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        label = row.get(key)
        name = "null" if label is None else str(label)
        counts[name] = counts.get(name, 0) + 1
    return counts


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def parse_json_event(payload: str) -> dict[str, Any] | None:
    if payload == "[DONE]":
        return None
    return json.loads(payload)
