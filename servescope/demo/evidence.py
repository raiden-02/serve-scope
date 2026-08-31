"""Read benchmark summaries used by the demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

P3_COMPARISON = Path("artifacts/p3/comparison-2026-08-31T16-29-56Z/result.json")
P4_COMPARISON = Path("artifacts/p4/comparison-2026-08-31T21-15-28Z/result.json")
P4_NATIVE_SUITE = Path("artifacts/p4/native-2026-08-31T21-07-52Z/result.json")
P4_BACKPRESSURE_SUITE = Path("artifacts/p4/backpressure-2026-08-31T21-11-34Z/result.json")
SESSION_NOTE = (
    "These were separate benchmark sessions, so they should not be read as "
    "one four-stage latency progression."
)


def _median_block(payload: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    cursor: Any = payload
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    if not isinstance(cursor, dict):
        return None
    if cursor.get("median") is None:
        return None
    return cursor


def format_seconds(value: float | None) -> str | None:
    if value is None:
        return None
    seconds = float(value)
    if seconds >= 1.0:
        return f"{seconds:.2f} s"
    return f"{seconds * 1000:.0f} ms"


def format_count(value: float | None) -> str | None:
    if value is None:
        return None
    if float(value) == int(value):
        return str(int(value))
    return f"{value:.1f}"


def reduction_pct(before: float | None, after: float | None) -> int | None:
    if before is None or after is None:
        return None
    start = float(before)
    if start <= 0:
        return None
    return int(round((start - float(after)) / start * 100))


def _unavailable() -> dict[str, Any]:
    return {"available": False, "message": "Evidence unavailable"}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def completed_job_count(payload: dict[str, Any] | None) -> int | None:
    """Same completed-job count across every repeat, or None if the suite disagrees."""
    if payload is None:
        return None
    repeats = payload.get("repeats")
    if not isinstance(repeats, list) or not repeats:
        return None
    counts: list[int] = []
    for row in repeats:
        if not isinstance(row, dict):
            return None
        raw = row.get("background_completed_count")
        if raw is None:
            return None
        counts.append(int(raw))
    if min(counts) != max(counts):
        return None
    return counts[0]


def load_p3_evidence(root: Path) -> dict[str, Any]:
    path = root / P3_COMPARISON
    payload = load_json(path)
    if payload is None:
        return _unavailable()
    fcfs = _median_block(payload, "aggregates", "fcfs_burst_p95")
    priority = _median_block(payload, "aggregates", "priority_burst_p95")
    fcfs_wait = _median_block(payload, "aggregates", "fcfs_waiting")
    pri_wait = _median_block(payload, "aggregates", "priority_waiting")
    if fcfs is None or priority is None:
        return _unavailable()
    return {
        "available": True,
        "label": "Native priority baseline",
        "path": str(P3_COMPARISON).replace("\\", "/"),
        "session": "same-session FCFS vs native priority",
        "rows": [
            {
                "name": "Default FCFS",
                "burst_p95": format_seconds(fcfs["median"]),
                "burst_p95_s": fcfs["median"],
                "waiting": format_count((fcfs_wait or {}).get("median")),
            },
            {
                "name": "Native priority",
                "burst_p95": format_seconds(priority["median"]),
                "burst_p95_s": priority["median"],
                "waiting": format_count((pri_wait or {}).get("median")),
            },
        ],
    }


def load_p4_evidence(root: Path) -> dict[str, Any]:
    path = root / P4_COMPARISON
    payload = load_json(path)
    if payload is None:
        return _unavailable()
    native = _median_block(payload, "native_burst_p95")
    gated = _median_block(payload, "gated_burst_p95")
    native_wait = _median_block(payload, "native_waiting")
    gated_wait = _median_block(payload, "gated_waiting")
    native_e2e = _median_block(payload, "native_bg_e2e_p95")
    gated_e2e = _median_block(payload, "gated_bg_e2e_p95")
    native_tps = _median_block(payload, "native_bg_tok_tps")
    gated_tps = _median_block(payload, "gated_bg_tok_tps")
    pending = _median_block(payload, "gated_max_pending")
    if native is None or gated is None:
        return _unavailable()
    native_jobs = completed_job_count(load_json(root / P4_NATIVE_SUITE))
    gated_jobs = completed_job_count(load_json(root / P4_BACKPRESSURE_SUITE))
    jobs_completed = gated_jobs if native_jobs == gated_jobs else None
    return {
        "available": True,
        "ttft_reduction_pct": reduction_pct(native["median"], gated["median"]),
        "jobs_completed": jobs_completed,
        "label": "External background admission",
        "path": str(P4_COMPARISON).replace("\\", "/"),
        "session": "same-session native priority vs ServeScope admission",
        "claim_note": (
            "The measured backpressure run never halved the concurrency limit "
            "because the runtime waiting queue stayed at zero. The observed "
            "improvement came from bounded admission. The decrease path exists "
            "and is unit-tested."
        ),
        "rows": [
            {
                "name": "Native priority",
                "burst_p95": format_seconds(native["median"]),
                "burst_p95_s": native["median"],
                "waiting": format_count((native_wait or {}).get("median")),
                "background_p95": format_seconds((native_e2e or {}).get("median")),
                "output_goodput": (
                    f"{(native_tps or {}).get('median'):.0f} tok/s"
                    if (native_tps or {}).get("median") is not None
                    else None
                ),
            },
            {
                "name": "ServeScope backpressure",
                "burst_p95": format_seconds(gated["median"]),
                "burst_p95_s": gated["median"],
                "waiting": format_count((gated_wait or {}).get("median")),
                "local_pending": format_count((pending or {}).get("median")),
                "background_p95": format_seconds((gated_e2e or {}).get("median")),
                "output_goodput": (
                    f"{(gated_tps or {}).get('median'):.0f} tok/s"
                    if (gated_tps or {}).get("median") is not None
                    else None
                ),
            },
        ],
    }


def load_evidence(root: Path) -> dict[str, Any]:
    return {
        "p3": load_p3_evidence(root),
        "p4": load_p4_evidence(root),
        "note": SESSION_NOTE,
    }
