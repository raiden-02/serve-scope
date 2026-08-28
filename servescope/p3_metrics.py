"""P3 native-priority comparison helpers. P1 clocks and P2 phases stay unchanged."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from servescope.metrics import percentile
from servescope.p2_metrics import (
    CLASS_BACKGROUND,
    CLASS_INTERACTIVE,
    PHASE_RECOVERY,
    filter_class,
    summarize_phase_latencies,
)

POLICY_FCFS = "fcfs"
POLICY_PRIORITY = "priority"
HASHED_SOURCE_FILES = (
    "scripts/p3_native_priority.py",
    "scripts/p2_mixed.py",
    "servescope/p3_metrics.py",
    "servescope/p2_metrics.py",
    "servescope/client.py",
    "servescope/metrics.py",
    "servescope/workload.py",
    "configs/p3_native_priority.json",
)


def request_priority_for_class(workload_class: str, scheduler_policy: str) -> int:
    """Native vLLM: lower integer is handled earlier. Interactive stays at 0."""
    if workload_class == CLASS_INTERACTIVE:
        return 0
    if workload_class == CLASS_BACKGROUND:
        return 1 if scheduler_policy == POLICY_PRIORITY else 0
    raise ValueError(f"unknown workload class: {workload_class}")


def should_send_priority_field(scheduler_policy: str) -> bool:
    """FCFS rejects nonzero priority. Do not send the field on the FCFS suite."""
    return scheduler_policy == POLICY_PRIORITY


def priority_protection_ratio(priority_burst_p95: float | None, fcfs_burst_p95: float | None) -> float | None:
    """priority burst p95 / FCFS burst p95. Smaller is better. Descriptive only."""
    if priority_burst_p95 is None or fcfs_burst_p95 is None:
        return None
    if fcfs_burst_p95 <= 0:
        return None
    return float(priority_burst_p95) / float(fcfs_burst_p95)


def absolute_reduction(fcfs_burst_p95: float | None, priority_burst_p95: float | None) -> float | None:
    if fcfs_burst_p95 is None or priority_burst_p95 is None:
        return None
    return float(fcfs_burst_p95) - float(priority_burst_p95)


def post_background_records(
    records: list[dict[str, Any]],
    last_background_completion_s: float | None,
) -> list[dict[str, Any]]:
    """Interactive requests attempted after the last background completion."""
    if last_background_completion_s is None:
        return []
    out = []
    for row in filter_class(records, CLASS_INTERACTIVE):
        t_rel = row.get("t_rel_s")
        if t_rel is not None and float(t_rel) > float(last_background_completion_s):
            out.append(row)
    return out


def summarize_post_background(
    records: list[dict[str, Any]],
    last_background_completion_s: float | None,
    *,
    slo_s: float = 1.0,
) -> dict[str, Any]:
    subset = post_background_records(records, last_background_completion_s)
    summary = summarize_phase_latencies(subset, slo_s=slo_s)
    summary["phase"] = "post_background"
    summary["last_background_completion_s"] = last_background_completion_s
    return summary


def source_hashes(root: Path, files: tuple[str, ...] = HASHED_SOURCE_FILES) -> dict[str, str]:
    """SHA-256 of benchmark inputs. Order is stable."""
    out: dict[str, str] = {}
    for rel in files:
        path = root / rel
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        out[rel] = digest
    return out


def pair_policies(fcfs_repeats: list[dict[str, Any]], priority_repeats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid_fcfs = [row for row in fcfs_repeats if row.get("valid_offered_load")]
    valid_pri = [row for row in priority_repeats if row.get("valid_offered_load")]
    paired = []
    for idx, pri in enumerate(valid_pri):
        fcfs = valid_fcfs[idx] if idx < len(valid_fcfs) else (valid_fcfs[-1] if valid_fcfs else None)
        if fcfs is None:
            continue
        paired.append(
            {
                "pair_index": idx + 1,
                "fcfs_repeat_id": fcfs.get("repeat_id"),
                "priority_repeat_id": pri.get("repeat_id"),
                "fcfs_burst_p95_s": fcfs.get("interactive_p95_burst_s"),
                "priority_burst_p95_s": pri.get("interactive_p95_burst_s"),
                "absolute_reduction_s": absolute_reduction(
                    fcfs.get("interactive_p95_burst_s"),
                    pri.get("interactive_p95_burst_s"),
                ),
                "priority_protection_ratio": priority_protection_ratio(
                    pri.get("interactive_p95_burst_s"),
                    fcfs.get("interactive_p95_burst_s"),
                ),
                "fcfs_max_waiting": fcfs.get("max_waiting_requests"),
                "priority_max_waiting": pri.get("max_waiting_requests"),
                "fcfs_background_output_token_tps": fcfs.get("background_output_token_goodput_tps"),
                "priority_background_output_token_tps": pri.get("background_output_token_goodput_tps"),
                "fcfs_background_request_rps": fcfs.get("background_request_goodput_rps"),
                "priority_background_request_rps": pri.get("background_request_goodput_rps"),
                "fcfs_last_background_completion_s": fcfs.get("last_background_completion_s"),
                "priority_last_background_completion_s": pri.get("last_background_completion_s"),
                "fcfs_background_e2e_p95_s": fcfs.get("background_e2e_p95_s"),
                "priority_background_e2e_p95_s": pri.get("background_e2e_p95_s"),
            }
        )
    return paired


def aggregate_numeric(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [row[key] for row in rows if row.get(key) is not None]
    return {
        "n": len(values),
        "median": percentile(values, 50) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }
