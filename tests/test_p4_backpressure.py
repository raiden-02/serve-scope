from __future__ import annotations

from pathlib import Path

from servescope.backpressure import (
    P4_HASHED_SOURCE_FILES,
    AimdController,
    annotate_gated_background,
    background_total_e2e_s,
    can_admit,
    ingress_lag_s,
    offered_span_goodput,
    summarize_ingress,
)
from servescope.client import build_chat_payload
from servescope.metrics import derive_request_timings
from servescope.p2_metrics import CLASS_BACKGROUND, CLASS_INTERACTIVE, classify_phase
from servescope.p3_metrics import (
    POLICY_FCFS,
    POLICY_PRIORITY,
    request_priority_for_class,
    should_send_priority_field,
    source_hashes,
)


def test_multiplicative_decrease_floors_at_one():
    ctl = AimdController(initial=32, minimum=1, maximum=256)
    assert ctl.observe(5) == "decrease" and ctl.limit == 16
    assert ctl.observe(2) == "decrease" and ctl.limit == 8
    assert ctl.observe(1) == "decrease" and ctl.limit == 4
    assert ctl.observe(1) == "decrease" and ctl.limit == 2
    assert ctl.observe(1) == "decrease" and ctl.limit == 1
    assert ctl.observe(9) == "hold" and ctl.limit == 1


def test_additive_increase_after_four_zero_wait_samples():
    ctl = AimdController(initial=32, minimum=1, maximum=256)
    assert ctl.observe(0) == "hold"
    assert ctl.observe(0) == "hold"
    assert ctl.observe(0) == "hold"
    assert ctl.observe(0) == "increase"
    assert ctl.limit == 33
    assert ctl.zero_wait_streak == 0
    assert ctl.observe(0) == "hold"


def test_limit_never_exceeds_maximum():
    ctl = AimdController(initial=256, minimum=1, maximum=256)
    assert ctl.observe(0) == "hold"
    assert ctl.observe(0) == "hold"
    assert ctl.observe(0) == "hold"
    assert ctl.observe(0) == "hold"
    assert ctl.limit == 256


def test_decrease_does_not_cancel_existing_inflight():
    assert can_admit(10, 8) is False
    assert can_admit(7, 8) is True
    assert can_admit(8, 8) is False


def test_admission_delay_is_not_ingress_lag():
    offered = 15.000
    enqueue = 15.002
    attempt = 22.500
    assert abs(ingress_lag_s(offered, enqueue) - 0.002) < 1e-12
    assert ingress_lag_s(offered, enqueue) <= 0.050
    record = {
        "request_attempt_s": attempt,
        "completion_s": 30.0,
        "dispatch_lag_s": attempt - offered,
    }
    annotate_gated_background(record, offered_arrival_s=offered, ingress_enqueue_s=enqueue)
    assert abs(record["admission_delay_s"] - 7.498) < 1e-12
    assert record["ingress_lag_s"] < 0.050
    assert record["dispatch_lag_s"] > 7.0
    summary = summarize_ingress([record], 16.0)
    assert summary["valid_offered_load"] is True
    assert summary["median_ingress_lag_s"] < 0.050


def test_total_background_latency_includes_local_defer():
    offered = 15.0
    completion = 40.0
    assert abs(background_total_e2e_s(offered, completion) - 25.0) < 1e-12
    record = {
        "workload_class": CLASS_BACKGROUND,
        "offered_arrival_s": offered,
        "ingress_enqueue_s": 15.001,
        "request_attempt_s": 20.0,
        "completion_s": completion,
        "background_total_e2e_s": 25.0,
        "status": "length",
        "completion_tokens": 256,
    }
    goodput = offered_span_goodput([record], t0=0.0)
    assert abs(goodput["background_total_e2e_p50_s"] - 25.0) < 1e-12
    assert abs(goodput["first_offered_rel_s"] - 15.0) < 1e-12
    assert abs(goodput["busy_span_s"] - 25.0) < 1e-12


def test_p1_clocks_and_p2_phases_unchanged():
    timings = derive_request_timings(
        scheduled_arrival_s=10.000,
        request_attempt_s=10.010,
        response_headers_s=10.500,
        first_content_s=11.000,
        completion_s=12.000,
    )
    assert abs(timings["dispatch_lag_s"] - 0.010) < 1e-12
    assert abs(timings["client_ttft_s"] - 0.990) < 1e-12
    assert classify_phase(14.999) == "pre_burst"
    assert classify_phase(15.0) == "burst_injection"
    assert classify_phase(30.0) == "recovery"


def test_p3_priority_assignment_unchanged():
    assert request_priority_for_class(CLASS_INTERACTIVE, POLICY_PRIORITY) == 0
    assert request_priority_for_class(CLASS_BACKGROUND, POLICY_PRIORITY) == 1
    assert request_priority_for_class(CLASS_BACKGROUND, POLICY_FCFS) == 0
    assert should_send_priority_field(POLICY_PRIORITY) is True
    assert should_send_priority_field(POLICY_FCFS) is False
    payload = build_chat_payload(
        model="Qwen/Qwen3-1.7B",
        prompt="hello",
        temperature=0,
        min_tokens=64,
        max_completion_tokens=64,
    )
    assert "priority" not in payload


def test_p4_source_hashes_are_deterministic():
    root = Path(__file__).resolve().parents[1]
    first = source_hashes(root, P4_HASHED_SOURCE_FILES)
    second = source_hashes(root, P4_HASHED_SOURCE_FILES)
    assert first == second
    assert set(first) == set(P4_HASHED_SOURCE_FILES)
    assert all(len(value) == 64 for value in first.values())
