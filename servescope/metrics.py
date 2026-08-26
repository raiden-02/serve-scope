"""P1 measurement helpers: SSE parse, status, percentiles, summaries.

Percentiles use NumPy ``method='linear'`` (R type 7). That choice is fixed
for this project so aggregates stay comparable across runs.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

COMPLETED_STATUSES = frozenset({"success", "length"})

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
) -> str:
    if cancelled:
        return "cancelled"
    if timed_out:
        return "timeout"
    if stream_error:
        return "stream_error"
    if http_status is not None and http_status >= 400:
        return "http_error"
    if http_status == 200 and finish_reason == "length" and got_content and stream_done:
        return "length"
    if http_status == 200 and got_content and stream_done and finish_reason in (
        "stop",
        "eos_token",
        None,
    ):
        return "success"
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


def completed_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in records if row.get("status") in COMPLETED_STATUSES]


def summarize_repeat(records: list[dict[str, Any]], offered_rps: float, duration_s: float) -> dict[str, Any]:
    completed = completed_records(records)
    errors = [row for row in records if row.get("status") not in COMPLETED_STATUSES]
    ttfts = [row["client_ttft_s"] for row in completed if row.get("client_ttft_s") is not None]
    e2es = [row["client_e2e_s"] for row in completed if row.get("client_e2e_s") is not None]
    dispatch_lags = [row["dispatch_lag_s"] for row in records if row.get("dispatch_lag_s") is not None]
    output_tokens = sum(int(row["completion_tokens"]) for row in completed if row.get("completion_tokens") is not None)
    dispatches = [row["actual_dispatch_s"] for row in records if row.get("actual_dispatch_s") is not None]
    actual_rps = None
    if len(dispatches) >= 2:
        span = max(dispatches) - min(dispatches)
        actual_rps = (len(dispatches) - 1) / span if span > 0 else None
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
        "completed_request_throughput_rps": (len(completed) / duration_s) if duration_s > 0 else None,
        "output_token_throughput_tps": (output_tokens / duration_s) if duration_s > 0 else None,
        "backend_queue_time_p50_ms": queue_summary["p50"],
        "backend_queue_time_p95_ms": queue_summary["p95"],
        "backend_queue_time_n": queue_summary["n"],
        "backend_ttft_p50_ms": backend_ttft_summary["p50"],
        "backend_ttft_p95_ms": backend_ttft_summary["p95"],
        "backend_ttft_n": backend_ttft_summary["n"],
        "median_dispatch_lag_s": percentile(dispatch_lags, 50) if dispatch_lags else None,
        "status_counts": _count_by(records, "status"),
        "finish_reason_counts": _count_by(records, "finish_reason"),
    }


def aggregate_repeats(repeat_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not repeat_summaries:
        return {}
    offered = repeat_summaries[0]["offered_rps"]
    p95s = [row["client_ttft_p95_s"] for row in repeat_summaries if row.get("client_ttft_p95_s") is not None]
    p50s = [row["client_ttft_p50_s"] for row in repeat_summaries if row.get("client_ttft_p50_s") is not None]
    throughputs = [
        row["output_token_throughput_tps"]
        for row in repeat_summaries
        if row.get("output_token_throughput_tps") is not None
    ]
    req_tps = [
        row["completed_request_throughput_rps"]
        for row in repeat_summaries
        if row.get("completed_request_throughput_rps") is not None
    ]
    return {
        "offered_rps": offered,
        "repeats": len(repeat_summaries),
        "any_client_limited": any(row.get("client_limited") for row in repeat_summaries),
        "headline_p95_ttft_s_median": percentile(p95s, 50),
        "headline_p95_ttft_s_min": min(p95s) if p95s else None,
        "headline_p95_ttft_s_max": max(p95s) if p95s else None,
        "headline_p50_ttft_s_median": percentile(p50s, 50),
        "headline_output_token_tps_median": percentile(throughputs, 50),
        "headline_completed_rps_median": percentile(req_tps, 50),
        "repeat_p95_ttft_s": p95s,
        "slo_p95_ttft_seconds": 1.0,
        "slo_met_all_repeats": all(
            row.get("client_ttft_p95_s") is not None and row["client_ttft_p95_s"] < 1.0
            for row in repeat_summaries
        ),
    }


def first_slo_violation(aggregates: list[dict[str, Any]], slo_s: float = 1.0) -> float | None:
    """First offered RPS whose median-repeat p95 TTFT is >= slo_s. None if none cross."""
    ordered = sorted(aggregates, key=lambda row: row["offered_rps"])
    for row in ordered:
        p95 = row.get("headline_p95_ttft_s_median")
        if p95 is not None and p95 >= slo_s:
            return float(row["offered_rps"])
    return None


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
