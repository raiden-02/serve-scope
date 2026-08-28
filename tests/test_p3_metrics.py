from __future__ import annotations

from pathlib import Path

from servescope.client import build_chat_payload
from servescope.metrics import derive_request_timings
from servescope.p2_metrics import CLASS_BACKGROUND, CLASS_INTERACTIVE, classify_phase
from servescope.p3_metrics import (
    POLICY_FCFS,
    POLICY_PRIORITY,
    HASHED_SOURCE_FILES,
    pair_policies,
    post_background_records,
    priority_protection_ratio,
    request_priority_for_class,
    should_send_priority_field,
    source_hashes,
    summarize_post_background,
)


def test_p1_clocks_unchanged():
    timings = derive_request_timings(
        scheduled_arrival_s=10.000,
        request_attempt_s=10.010,
        response_headers_s=10.500,
        first_content_s=11.000,
        completion_s=12.000,
    )
    assert abs(timings["dispatch_lag_s"] - 0.010) < 1e-12
    assert abs(timings["client_ttft_s"] - 0.990) < 1e-12


def test_p2_phase_boundaries_unchanged():
    assert classify_phase(14.999) == "pre_burst"
    assert classify_phase(15.0) == "burst_injection"
    assert classify_phase(30.0) == "recovery"


def test_priority_assigned_by_class():
    assert request_priority_for_class(CLASS_INTERACTIVE, POLICY_PRIORITY) == 0
    assert request_priority_for_class(CLASS_BACKGROUND, POLICY_PRIORITY) == 1
    assert request_priority_for_class(CLASS_INTERACTIVE, POLICY_FCFS) == 0
    assert request_priority_for_class(CLASS_BACKGROUND, POLICY_FCFS) == 0


def test_fcfs_does_not_send_nondefault_priority():
    assert should_send_priority_field(POLICY_FCFS) is False
    assert should_send_priority_field(POLICY_PRIORITY) is True
    fcfs_payload = build_chat_payload(
        model="Qwen/Qwen3-1.7B",
        prompt="hello",
        temperature=0,
        min_tokens=64,
        max_completion_tokens=64,
    )
    assert "priority" not in fcfs_payload
    pri_payload = build_chat_payload(
        model="Qwen/Qwen3-1.7B",
        prompt="hello",
        temperature=0,
        min_tokens=64,
        max_completion_tokens=64,
        priority=1,
    )
    assert pri_payload["priority"] == 1


def test_policy_tag_and_protection_ratio():
    fcfs = [
        {
            "valid_offered_load": True,
            "repeat_id": "fcfs-1",
            "interactive_p95_burst_s": 2.4,
            "max_waiting_requests": 160,
            "background_output_token_goodput_tps": 2400,
            "background_request_goodput_rps": 9.0,
            "last_background_completion_s": 42.0,
            "background_e2e_p95_s": 8.0,
        }
    ]
    priority = [
        {
            "valid_offered_load": True,
            "repeat_id": "pri-1",
            "interactive_p95_burst_s": 0.12,
            "max_waiting_requests": 20,
            "background_output_token_goodput_tps": 1800,
            "background_request_goodput_rps": 6.0,
            "last_background_completion_s": 50.0,
            "background_e2e_p95_s": 12.0,
        }
    ]
    assert abs(priority_protection_ratio(0.12, 2.4) - 0.05) < 1e-12
    paired = pair_policies(fcfs, priority)
    assert paired[0]["fcfs_repeat_id"] == "fcfs-1"
    assert paired[0]["priority_repeat_id"] == "pri-1"
    assert paired[0]["priority_protection_ratio"] == priority_protection_ratio(0.12, 2.4)
    assert paired[0]["priority_background_output_token_tps"] == 1800


def test_post_background_starts_after_final_completion():
    records = [
        {"workload_class": CLASS_INTERACTIVE, "t_rel_s": 39.0, "status": "length", "client_ttft_s": 2.0, "client_e2e_s": 3.0},
        {"workload_class": CLASS_INTERACTIVE, "t_rel_s": 44.1, "status": "length", "client_ttft_s": 0.05, "client_e2e_s": 0.9},
        {"workload_class": CLASS_INTERACTIVE, "t_rel_s": 50.0, "status": "length", "client_ttft_s": 0.06, "client_e2e_s": 0.9},
        {"workload_class": CLASS_BACKGROUND, "t_rel_s": 20.0, "status": "length", "client_ttft_s": 1.0, "client_e2e_s": 20.0},
    ]
    subset = post_background_records(records, 44.0)
    assert [row["t_rel_s"] for row in subset] == [44.1, 50.0]
    summary = summarize_post_background(records, 44.0)
    assert summary["count"] == 2
    assert summary["phase"] == "post_background"
    assert abs(summary["client_ttft_p50_s"] - 0.055) < 1e-12


def test_source_hashes_are_deterministic():
    root = Path(__file__).resolve().parents[1]
    first = source_hashes(root, HASHED_SOURCE_FILES)
    second = source_hashes(root, HASHED_SOURCE_FILES)
    assert first == second
    assert set(first) == set(HASHED_SOURCE_FILES)
    assert all(len(value) == 64 for value in first.values())
