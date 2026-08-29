"""AIMD background admission. This is not a vLLM scheduler."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from servescope.metrics import completed_records, percentile, summarize_latencies
from servescope.p3_metrics import HASHED_SOURCE_FILES

ACTION_HOLD = "hold"
ACTION_INCREASE = "increase"
ACTION_DECREASE = "decrease"

P4_HASHED_SOURCE_FILES = HASHED_SOURCE_FILES + (
    "servescope/backpressure.py",
    "scripts/p4_backpressure.py",
    "configs/p4_backpressure.json",
)


@dataclass
class PendingJob:
    index: int
    offered_arrival_s: float
    ingress_enqueue_s: float
    prompt_id: str
    prompt: str


class AimdController:
    """One congestion signal: vLLM waiting count. Additive increase, multiplicative decrease."""

    def __init__(
        self,
        *,
        initial: int = 32,
        minimum: int = 1,
        maximum: int = 256,
        increase_after_zero_samples: int = 4,
    ) -> None:
        if minimum < 1:
            raise ValueError("minimum must be at least 1")
        if maximum < minimum:
            raise ValueError("maximum must be >= minimum")
        self.minimum = int(minimum)
        self.maximum = int(maximum)
        self.increase_after_zero_samples = int(increase_after_zero_samples)
        self.limit = min(self.maximum, max(self.minimum, int(initial)))
        self.zero_wait_streak = 0

    def observe(self, waiting: float | None) -> str:
        if waiting is None:
            return ACTION_HOLD
        if waiting > 0:
            nxt = max(self.minimum, self.limit // 2)
            self.zero_wait_streak = 0
            if nxt < self.limit:
                self.limit = nxt
                return ACTION_DECREASE
            return ACTION_HOLD
        self.zero_wait_streak += 1
        if self.zero_wait_streak >= self.increase_after_zero_samples:
            nxt = min(self.maximum, self.limit + 1)
            self.zero_wait_streak = 0
            if nxt > self.limit:
                self.limit = nxt
                return ACTION_INCREASE
            return ACTION_HOLD
        return ACTION_HOLD


class PendingQueue:
    def __init__(self, capacity: int = 256) -> None:
        self.capacity = int(capacity)
        self._items: deque[PendingJob] = deque()
        self.max_depth = 0

    def __len__(self) -> int:
        return len(self._items)

    def enqueue(self, job: PendingJob) -> None:
        if len(self._items) >= self.capacity:
            raise RuntimeError(f"ServeScope pending queue is full ({self.capacity})")
        self._items.append(job)
        self.max_depth = max(self.max_depth, len(self._items))

    def dequeue(self) -> PendingJob | None:
        if not self._items:
            return None
        return self._items.popleft()


def can_admit(background_inflight: int, limit: int) -> bool:
    """Do not cancel in-flight work. Only refuse new admissions while over the limit."""
    return background_inflight < limit


def ingress_lag_s(offered_arrival_s: float, ingress_enqueue_s: float) -> float:
    return float(ingress_enqueue_s) - float(offered_arrival_s)


def admission_delay_s(ingress_enqueue_s: float, server_request_attempt_s: float) -> float:
    return float(server_request_attempt_s) - float(ingress_enqueue_s)


def background_total_e2e_s(offered_arrival_s: float, completion_s: float) -> float:
    return float(completion_s) - float(offered_arrival_s)


def annotate_native_background(record: dict[str, Any]) -> dict[str, Any]:
    """Native baseline submits immediately. Defer clocks collapse to the existing attempt."""
    offered = record.get("scheduled_arrival_s")
    attempt = record.get("request_attempt_s")
    completion = record.get("completion_s")
    record["offered_arrival_s"] = offered
    record["ingress_enqueue_s"] = attempt
    record["ingress_lag_s"] = (
        ingress_lag_s(offered, attempt) if offered is not None and attempt is not None else None
    )
    record["admission_delay_s"] = 0.0 if attempt is not None else None
    record["background_total_e2e_s"] = (
        background_total_e2e_s(offered, completion) if offered is not None and completion is not None else None
    )
    record["admission_gated"] = False
    return record


def annotate_gated_background(
    record: dict[str, Any],
    *,
    offered_arrival_s: float,
    ingress_enqueue_s: float,
) -> dict[str, Any]:
    attempt = record.get("request_attempt_s")
    completion = record.get("completion_s")
    record["offered_arrival_s"] = offered_arrival_s
    record["ingress_enqueue_s"] = ingress_enqueue_s
    record["ingress_lag_s"] = ingress_lag_s(offered_arrival_s, ingress_enqueue_s)
    record["admission_delay_s"] = (
        admission_delay_s(ingress_enqueue_s, attempt) if attempt is not None else None
    )
    record["background_total_e2e_s"] = (
        background_total_e2e_s(offered_arrival_s, completion) if completion is not None else None
    )
    record["admission_gated"] = True
    return record


def summarize_ingress(records: list[dict[str, Any]], offered_rps: float) -> dict[str, Any]:
    """Generator validity uses enqueue time, not later vLLM submission."""
    stamps = [row.get("ingress_enqueue_s") for row in records if row.get("ingress_enqueue_s") is not None]
    lags = [row.get("ingress_lag_s") for row in records if row.get("ingress_lag_s") is not None]
    actual_rps = None
    if len(stamps) >= 2:
        span = max(stamps) - min(stamps)
        actual_rps = (len(stamps) - 1) / span if span > 0 else None
    median_lag = percentile(lags, 50) if lags else None
    client_limited = False
    reasons: list[str] = []
    if actual_rps is not None and offered_rps > 0 and actual_rps < 0.90 * offered_rps:
        client_limited = True
        reasons.append("actual_ingress_rps_below_90pct")
    if median_lag is not None and median_lag > 0.050:
        client_limited = True
        reasons.append("median_ingress_lag_gt_50ms")
    return {
        "offered_rps": offered_rps,
        "actual_ingress_rps": actual_rps,
        "median_ingress_lag_s": median_lag,
        "ingress_count": len(records),
        "client_limited": client_limited,
        "valid_offered_load": not client_limited,
        "invalid_reason": None if not client_limited else ", ".join(reasons),
    }


def offered_span_goodput(records: list[dict[str, Any]], t0: float) -> dict[str, Any]:
    """Goodput from first original offered arrival to last completion."""
    completed = completed_records(records)
    output_tokens = sum(int(row["completion_tokens"]) for row in completed if row.get("completion_tokens") is not None)
    offered_stamps = [row.get("offered_arrival_s") for row in records if row.get("offered_arrival_s") is not None]
    completions = [row.get("completion_s") for row in records if row.get("completion_s") is not None]
    first_rel = (min(offered_stamps) - t0) if offered_stamps else None
    last_rel = (max(completions) - t0) if completions else None
    span = None
    if first_rel is not None and last_rel is not None and last_rel > first_rel:
        span = last_rel - first_rel
    totals = [row.get("background_total_e2e_s") for row in completed if row.get("background_total_e2e_s") is not None]
    delays = [row.get("admission_delay_s") for row in records if row.get("admission_delay_s") is not None]
    total = summarize_latencies(totals)
    delay = summarize_latencies(delays)
    return {
        "offered_count": len(records),
        "completed_count": len(completed),
        "admitted_count": sum(1 for row in records if row.get("request_attempt_s") is not None),
        "error_count": len(records) - len(completed),
        "output_tokens_sum": output_tokens,
        "wall_clock_completed_request_tps": (len(completed) / span) if span else None,
        "wall_clock_output_token_tps": (output_tokens / span) if span else None,
        "busy_span_s": span,
        "first_offered_rel_s": first_rel,
        "last_completion_rel_s": last_rel,
        "background_total_e2e_p50_s": total["p50"],
        "background_total_e2e_p95_s": total["p95"],
        "background_total_e2e_p99_s": total["p99"],
        "admission_delay_p50_s": delay["p50"],
        "admission_delay_p95_s": delay["p95"],
        "admission_delay_p99_s": delay["p99"],
    }


def controller_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    limits = [int(row["background_limit"]) for row in rows if row.get("background_limit") is not None]
    actions = [row.get("controller_action") for row in rows]
    return {
        "background_limit_min": min(limits) if limits else None,
        "background_limit_median": percentile(limits, 50) if limits else None,
        "background_limit_max": max(limits) if limits else None,
        "increase_count": sum(1 for action in actions if action == ACTION_INCREASE),
        "decrease_count": sum(1 for action in actions if action == ACTION_DECREASE),
        "hold_count": sum(1 for action in actions if action == ACTION_HOLD),
        "sample_count": len(rows),
    }
