from __future__ import annotations

from pathlib import Path

from servescope.demo.evidence import (
    P3_COMPARISON,
    P4_COMPARISON,
    SESSION_NOTE,
    format_seconds,
    load_evidence,
    load_p3_evidence,
    load_p4_evidence,
    reduction_pct,
)

ROOT = Path(__file__).resolve().parents[1]


def test_p3_artifact_medians():
    p3 = load_p3_evidence(ROOT)
    assert p3["available"] is True
    assert p3["path"] == str(P3_COMPARISON).replace("\\", "/")
    assert p3["rows"][0]["name"] == "Default FCFS"
    assert p3["rows"][0]["burst_p95"] == "3.33 s"
    assert p3["rows"][1]["name"] == "Native priority"
    assert p3["rows"][1]["burst_p95"] == "836 ms"
    assert "client_limited" not in str(p3)


def test_p4_artifact_medians():
    p4 = load_p4_evidence(ROOT)
    assert p4["available"] is True
    assert p4["path"] == str(P4_COMPARISON).replace("\\", "/")
    assert p4["rows"][0]["burst_p95"] == "297 ms"
    assert p4["rows"][1]["burst_p95"] == "103 ms"
    assert p4["rows"][0]["waiting"] == "83"
    assert p4["rows"][1]["waiting"] == "0"
    assert p4["rows"][1]["local_pending"] == "137"
    assert p4["rows"][0]["background_p95"] == "19.11 s"
    assert p4["rows"][1]["background_p95"] == "26.08 s"
    assert "decrease" in p4["claim_note"]
    assert p4["jobs_completed"] == 240
    assert p4["ttft_reduction_pct"] == 65
    assert reduction_pct(0.297, 0.103) == 65
    assert reduction_pct(None, 0.1) is None


def test_missing_evidence():
    missing = load_p3_evidence(ROOT / "does-not-exist")
    assert missing == {"available": False, "message": "Evidence unavailable"}


def test_no_cross_session_blend():
    payload = load_evidence(ROOT)
    assert payload["note"] == SESSION_NOTE
    assert payload["p3"]["available"] and payload["p4"]["available"]
    p3_priority = payload["p3"]["rows"][1]["burst_p95_s"]
    p4_native = payload["p4"]["rows"][0]["burst_p95_s"]
    assert abs(p3_priority - 0.836) < 0.01
    assert abs(p4_native - 0.297) < 0.01
    assert p3_priority != p4_native


def test_format_seconds():
    assert format_seconds(3.32799) == "3.33 s"
    assert format_seconds(0.83631) == "836 ms"
