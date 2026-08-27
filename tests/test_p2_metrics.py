from __future__ import annotations

from servescope.metrics import derive_request_timings, mark_repeat_validity
from servescope.p2_metrics import (
    CLASS_BACKGROUND,
    CLASS_INTERACTIVE,
    PHASE_BURST,
    PHASE_PRE,
    PHASE_RECOVERY,
    absolute_ttft_increase,
    attach_phase,
    background_goodput,
    classify_phase,
    filter_class,
    interference_ratio,
    mark_class_validity,
    summarize_phase_latencies,
    sustained_waiting_queue,
)
from servescope.workload import (
    select_background_prompt,
    select_prompt,
    windowed_arrivals,
)


def test_p1_timing_semantics_unchanged():
    timings = derive_request_timings(
        scheduled_arrival_s=10.000,
        request_attempt_s=10.010,
        response_headers_s=10.500,
        first_content_s=11.000,
        completion_s=12.000,
    )
    assert abs(timings["dispatch_lag_s"] - 0.010) < 1e-12
    assert abs(timings["response_headers_latency_s"] - 0.490) < 1e-12
    assert abs(timings["client_ttft_s"] - 0.990) < 1e-12
    assert abs(timings["client_e2e_s"] - 1.990) < 1e-12
    assert timings["actual_dispatch_s"] == 10.010


def test_phase_boundaries():
    assert classify_phase(0.0) == PHASE_PRE
    assert classify_phase(14.999) == PHASE_PRE
    assert classify_phase(15.0) == PHASE_BURST
    assert classify_phase(29.999) == PHASE_BURST
    assert classify_phase(30.0) == PHASE_RECOVERY
    assert classify_phase(60.0) == PHASE_RECOVERY


def test_phase_uses_attempt_not_completion():
    row = {
        "request_attempt_s": 100.0 + 15.0,
        "scheduled_arrival_s": 100.0 + 14.9,
        "completion_s": 100.0 + 40.0,
    }
    attach_phase(row, t0=100.0)
    assert row["phase"] == PHASE_BURST
    assert abs(row["t_rel_s"] - 15.0) < 1e-12


def test_interactive_and_background_tags_are_distinct():
    interactive = {"workload_class": CLASS_INTERACTIVE, "prompt_id": "p1_short_00"}
    background = {"workload_class": CLASS_BACKGROUND, "prompt_id": "p2_bg_meeting_notes_00"}
    records = [interactive, background]
    assert [row["prompt_id"] for row in filter_class(records, CLASS_INTERACTIVE)] == ["p1_short_00"]
    assert [row["prompt_id"] for row in filter_class(records, CLASS_BACKGROUND)] == ["p2_bg_meeting_notes_00"]
    assert select_prompt(20260830, 0)[0].startswith("p1_short_")
    assert select_background_prompt(20260830, 0)[0].startswith("p2_bg_")
    assert select_prompt(20260830, 0) != select_background_prompt(20260830, 0)


def test_independent_class_schedules():
    t0 = 1000.0
    interactive = windowed_arrivals(64.0, t0, 0.0, 60.0)
    background = windowed_arrivals(8.0, t0, 15.0, 30.0)
    assert abs(interactive[0] - t0) < 1e-12
    assert abs(interactive[-1] - (t0 + 3839.0 / 64.0)) < 1e-9
    assert len(interactive) == 3840
    assert abs(background[0] - (t0 + 15.0)) < 1e-12
    assert background[0] >= t0 + 15.0
    assert background[-1] < t0 + 30.0
    assert len(background) == 120
    assert all(t0 <= t < t0 + 60.0 for t in interactive)
    assert all(t0 + 15.0 <= t < t0 + 30.0 for t in background)


def test_control_schedule_has_no_background_arrivals():
    t0 = 0.0
    interactive = windowed_arrivals(64.0, t0, 0.0, 60.0)
    background = []
    assert len(interactive) == 3840
    assert background == []


def test_phase_percentile_aggregation():
    records = [
        {"status": "length", "client_ttft_s": 0.04, "client_e2e_s": 0.9, "phase": PHASE_PRE},
        {"status": "length", "client_ttft_s": 0.05, "client_e2e_s": 0.9, "phase": PHASE_PRE},
        {"status": "length", "client_ttft_s": 0.40, "client_e2e_s": 1.2, "phase": PHASE_BURST},
        {"status": "length", "client_ttft_s": 0.50, "client_e2e_s": 1.3, "phase": PHASE_BURST},
        {"status": "timeout", "client_ttft_s": None, "client_e2e_s": None, "phase": PHASE_BURST},
        {"status": "length", "client_ttft_s": 0.06, "client_e2e_s": 0.9, "phase": PHASE_RECOVERY},
    ]
    pre = summarize_phase_latencies([row for row in records if row["phase"] == PHASE_PRE], slo_s=1.0)
    burst = summarize_phase_latencies([row for row in records if row["phase"] == PHASE_BURST], slo_s=1.0)
    assert pre["count"] == 2
    assert pre["slo_met"] is True
    assert burst["count"] == 3
    assert burst["error_count"] == 1
    assert burst["client_ttft_n"] == 2
    assert abs(burst["client_ttft_p50_s"] - 0.45) < 1e-12
    assert burst["slo_met"] is True


def test_interference_ratio_is_descriptive():
    assert abs(interference_ratio(0.20, 0.05) - 4.0) < 1e-12
    assert abs(absolute_ttft_increase(0.20, 0.05) - 0.15) < 1e-12
    assert interference_ratio(0.20, 0.0) is None
    assert interference_ratio(None, 0.05) is None


def test_class_specific_validity():
    interactive = {
        "offered_rps": 64.0,
        "actual_dispatch_rps": 64.0,
        "median_dispatch_lag_s": 0.002,
        "client_limited": False,
        "aborted": False,
        "status_counts": {},
    }
    background = {
        "offered_rps": 8.0,
        "actual_dispatch_rps": 4.0,
        "median_dispatch_lag_s": 0.002,
        "client_limited": False,
        "aborted": False,
        "status_counts": {},
    }
    mark_class_validity(interactive, max_inflight=2048)
    mark_class_validity(background, max_inflight=512)
    assert interactive["valid_offered_load"] is True
    assert background["valid_offered_load"] is False
    assert "actual_dispatch_rps_below_90pct" in (background["invalid_reason"] or "")


def test_pooltimeout_invalidates_class():
    row = {
        "offered_rps": 64.0,
        "actual_dispatch_rps": 64.0,
        "median_dispatch_lag_s": 0.001,
        "client_limited": False,
        "aborted": False,
        "status_counts": {"client_capacity": 2},
    }
    mark_repeat_validity(row, max_inflight=2048)
    assert row["valid_offered_load"] is False
    assert "httpx_pool_timeout" in (row["invalid_reason"] or "")


def test_background_completion_after_injection_is_preserved():
    t0 = 50.0
    records = [
        {
            "status": "length",
            "request_attempt_s": t0 + 15.0,
            "completion_s": t0 + 44.0,
            "completion_tokens": 256,
            "prompt_tokens": 800,
            "client_e2e_s": 29.0,
        }
    ]
    summary = background_goodput(records, t0)
    assert abs(summary["last_completion_rel_s"] - 44.0) < 1e-12
    assert summary["last_completion_rel_s"] > 30.0
    assert summary["completed_count"] == 1
    assert summary["output_tokens_sum"] == 256


def test_sustained_queue_requires_more_than_a_blip():
    assert sustained_waiting_queue({"max_waiting_requests": 4, "waiting_positive_samples": 6}) is True
    assert sustained_waiting_queue({"max_waiting_requests": 2, "waiting_positive_samples": 1}) is False
    assert sustained_waiting_queue({"max_waiting_requests": 0, "waiting_positive_samples": 0}) is False
