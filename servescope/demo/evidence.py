"""Load accepted P3/P4 comparison artifacts. Never invent numbers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

P3_COMPARISON = Path("artifacts/p3/comparison-2026-08-31T16-29-56Z/result.json")
P4_COMPARISON = Path("artifacts/p4/comparison-2026-08-31T21-15-28Z/result.json")
SESSION_NOTE = (
    "Comparisons are shown within the sessions in which they were measured; "
    "cross-session results are not treated as one controlled comparison."
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
        "label": "Measured benchmark: P3",
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
    return {
        "available": True,
        "label": "Measured benchmark: P4",
        "path": str(P4_COMPARISON).replace("\\", "/"),
        "session": "same-session native priority vs ServeScope backpressure",
        "claim_note": (
            "ServeScope added external background admission/backpressure with an "
            "AIMD-style concurrency policy. In the measured workload, bounded "
            "admission kept the vLLM waiting queue at zero while background jobs "
            "waited in ServeScope's local queue. The decrease path exists and is "
            "unit-tested, but was not exercised in this headline run."
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
