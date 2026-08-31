from __future__ import annotations

from servescope.metrics import (
    aggregate_repeats,
    classify_status,
    derive_request_timings,
    extract_backend_metrics,
    extract_finish_reason,
    extract_usage,
    first_clean_slo_violation,
    first_observed_valid_collapse_rps,
    first_slo_violation,
    is_nonempty_generated_content,
    mark_repeat_validity,
    percentile,
    summarize_repeat,
    sse_data_payloads,
)
from servescope.workload import scheduled_arrivals, select_prompt


def test_sse_data_payloads_ignores_comments():
    raw = "event: message\ndata: {\"ok\": true}\n\ndata: [DONE]\n"
    assert sse_data_payloads(raw) == ['{"ok": true}', "[DONE]"]


def test_first_content_ignores_role_and_empty():
    assert is_nonempty_generated_content({"role": "assistant"}) is False
    assert is_nonempty_generated_content({"content": ""}) is False
    assert is_nonempty_generated_content({"content": None}) is False
    assert is_nonempty_generated_content({"reasoning_content": "think"}) is False
    assert is_nonempty_generated_content({"content": "Hello"}) is True


def test_finish_reason_and_usage_extraction():
    event = {
        "choices": [{"finish_reason": "length", "delta": {}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 64, "total_tokens": 75},
        "metrics": {"queue_time_ms": 2.5, "time_to_first_token_ms": 40.0},
    }
    assert extract_finish_reason(event) == "length"
    assert extract_usage(event)["completion_tokens"] == 64
    backend = extract_backend_metrics(event)
    assert backend["backend_queue_time_ms"] == 2.5
    assert backend["backend_time_to_first_token_ms"] == 40.0


def test_status_length_is_not_generic_success():
    assert (
        classify_status(
            http_status=200,
            finish_reason="length",
            got_content=True,
            stream_done=True,
            timed_out=False,
            cancelled=False,
            stream_error=False,
            error_message=None,
        )
        == "length"
    )
    assert (
        classify_status(
            http_status=200,
            finish_reason="stop",
            got_content=True,
            stream_done=True,
            timed_out=False,
            cancelled=False,
            stream_error=False,
            error_message=None,
        )
        == "success"
    )
    assert (
        classify_status(
            http_status=200,
            finish_reason=None,
            got_content=True,
            stream_done=False,
            timed_out=True,
            cancelled=False,
            stream_error=False,
            error_message="timeout",
        )
        == "timeout"
    )
    assert (
        classify_status(
            http_status=500,
            finish_reason=None,
            got_content=False,
            stream_done=False,
            timed_out=False,
            cancelled=False,
            stream_error=False,
            error_message="boom",
        )
        == "http_error"
    )
    assert (
        classify_status(
            http_status=200,
            finish_reason=None,
            got_content=False,
            stream_done=False,
            timed_out=False,
            cancelled=False,
            stream_error=True,
            error_message="bad json",
        )
        == "stream_error"
    )


def test_percentile_linear_known_values():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile([], 95) is None
    assert percentile(values, 50) == 2.5
    assert percentile([10.0], 99) == 10.0


def test_arrival_schedule_is_open_loop_fixed_interval():
    times = scheduled_arrivals(4, offered_rps=4.0, t0=100.0)
    assert times == [100.0, 100.25, 100.50, 100.75]


def test_prompt_selection_is_deterministic():
    assert select_prompt(7, 3) == select_prompt(7, 3)
    ids = [select_prompt(1, i)[0] for i in range(20)]
    assert len(set(ids)) > 1


def test_repeat_summary_marks_client_limited_and_preserves_counts():
    records = [
        {
            "status": "length",
            "client_ttft_s": 0.2,
            "client_e2e_s": 0.8,
            "request_attempt_s": 0.0,
            "actual_dispatch_s": 0.0,
            "dispatch_lag_s": 0.001,
            "completion_tokens": 64,
            "finish_reason": "length",
        },
        {
            "status": "length",
            "client_ttft_s": 0.3,
            "client_e2e_s": 0.9,
            "request_attempt_s": 0.25,
            "actual_dispatch_s": 0.25,
            "dispatch_lag_s": 0.001,
            "completion_tokens": 64,
            "finish_reason": "length",
        },
        {
            "status": "timeout",
            "client_ttft_s": None,
            "client_e2e_s": 2.0,
            "request_attempt_s": 0.50,
            "actual_dispatch_s": 0.50,
            "dispatch_lag_s": 0.001,
            "completion_tokens": None,
            "finish_reason": None,
        },
    ]
    summary = summarize_repeat(records, offered_rps=4.0, duration_s=1.0)
    assert summary["completed_count"] == 2
    assert summary["error_count"] == 1
    assert summary["client_limited"] is False
    assert summary["wall_clock_output_token_throughput_tps"] == 128.0
    assert summary["client_ttft_n"] == 2


def test_premature_eof_without_finish_reason_is_not_success():
    assert (
        classify_status(
            http_status=200,
            finish_reason=None,
            got_content=True,
            stream_done=False,
            timed_out=False,
            cancelled=False,
            stream_error=False,
            error_message=None,
        )
        == "stream_error"
    )
    assert (
        classify_status(
            http_status=200,
            finish_reason=None,
            got_content=True,
            stream_done=True,
            timed_out=False,
            cancelled=False,
            stream_error=False,
            error_message=None,
        )
        == "stream_error"
    )


def test_pool_timeout_is_client_capacity():
    assert (
        classify_status(
            http_status=None,
            finish_reason=None,
            got_content=False,
            stream_done=False,
            timed_out=False,
            cancelled=False,
            stream_error=False,
            error_message="PoolTimeout",
            client_capacity=True,
        )
        == "client_capacity"
    )


def _repeat(*, rps, p95, valid=True, aborted=False, waiting=0.0, tok_tps=100.0, req_tps=10.0, p50=0.05):
    row = {
        "offered_rps": rps,
        "client_ttft_p50_s": p50,
        "client_ttft_p95_s": p95,
        "wall_clock_output_token_throughput_tps": tok_tps,
        "wall_clock_completed_request_throughput_rps": req_tps,
        "client_limited": not valid,
        "aborted": aborted,
        "max_waiting_requests": waiting,
        "actual_dispatch_rps": rps if valid else rps * 0.5,
        "median_dispatch_lag_s": 0.001,
        "status_counts": {},
    }
    return mark_repeat_validity(row, max_inflight=4096)


def test_client_limited_repeats_excluded_from_headline():
    summaries = [
        _repeat(rps=132, p95=0.08, valid=True),
        _repeat(rps=132, p95=0.09, valid=True),
        _repeat(rps=132, p95=14.7, valid=False, aborted=True),
    ]
    agg = aggregate_repeats(summaries, min_valid_repeats=3, slo_s=1.0)
    assert agg["total_repeat_count"] == 3
    assert agg["valid_repeat_count"] == 2
    assert agg["invalid_repeat_count"] == 1
    assert agg["aggregate_valid"] is False
    assert agg["all_repeat_p95s"] == [0.08, 0.09, 14.7]
    assert agg["valid_repeat_p95s"] == [0.08, 0.09]
    assert abs(agg["headline_p95_ttft_s_median"] - 0.085) < 1e-12
    assert first_clean_slo_violation([agg], slo_s=1.0) is None
    assert first_slo_violation([agg], slo_s=1.0) is None


def test_header_delay_is_not_dispatch_lag():
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


def test_actual_dispatch_rps_uses_request_attempt_not_headers():
    """Late response headers must not look like missed dispatch."""
    records = []
    for i in range(5):
        scheduled = 10.0 + i * 0.25
        attempt = scheduled + 0.002
        headers = attempt + 0.490
        timings = derive_request_timings(
            scheduled_arrival_s=scheduled,
            request_attempt_s=attempt,
            response_headers_s=headers,
            first_content_s=headers + 0.05,
            completion_s=headers + 0.40,
        )
        records.append(
            {
                "status": "length",
                "completion_tokens": 64,
                "finish_reason": "length",
                **timings,
            }
        )
    summary = summarize_repeat(records, offered_rps=4.0, duration_s=1.0)
    assert abs(summary["actual_dispatch_rps"] - 4.0) < 1e-9
    assert summary["median_dispatch_lag_s"] < 0.010
    assert summary["response_headers_latency_p95_s"] > 0.4
    assert summary["client_limited"] is False
    marked = mark_repeat_validity(summary, max_inflight=4096)
    assert marked["valid_offered_load"] is True


def test_clean_crossing_requires_three_valid_failures():
    failing = [
        _repeat(rps=132, p95=2.0, valid=True),
        _repeat(rps=132, p95=2.1, valid=True),
        _repeat(rps=132, p95=2.2, valid=True),
    ]
    agg = aggregate_repeats(failing, min_valid_repeats=3, slo_s=1.0)
    assert agg["aggregate_valid"] is True
    assert agg["slo_violated_all_valid_repeats"] is True
    assert first_clean_slo_violation([agg], slo_s=1.0) == 132.0


def test_mixed_valid_fail_pass_is_instability_not_clean_crossing():
    mixed = [
        _repeat(rps=128, p95=2.0, valid=True),
        _repeat(rps=128, p95=2.1, valid=True),
        _repeat(rps=128, p95=0.08, valid=True),
    ]
    agg = aggregate_repeats(mixed, min_valid_repeats=3, slo_s=1.0)
    assert agg["aggregate_valid"] is True
    assert agg["slo_violated_all_valid_repeats"] is False
    assert agg["valid_repeats_mixed_slo"] is True
    assert first_clean_slo_violation([agg], slo_s=1.0) is None


def test_invalid_aggregate_cannot_be_first_clean_crossing():
    mixed_invalid = [
        _repeat(rps=132, p95=19.0, valid=True),
        _repeat(rps=132, p95=14.7, valid=False, aborted=True),
        _repeat(rps=132, p95=21.0, valid=True),
    ]
    agg = aggregate_repeats(mixed_invalid, min_valid_repeats=3, slo_s=1.0)
    assert agg["aggregate_valid"] is False
    assert first_clean_slo_violation([agg], slo_s=1.0) is None


def test_mixed_valid_repeats_are_not_universally_healthy():
    mixed = [
        _repeat(rps=128, p95=0.08, valid=True, waiting=0.0),
        _repeat(rps=128, p95=0.09, valid=True, waiting=0.0),
        _repeat(rps=128, p95=12.0, valid=True, waiting=300.0),
    ]
    agg = aggregate_repeats(mixed, min_valid_repeats=3, slo_s=1.0)
    assert agg["aggregate_valid"] is True
    assert agg["slo_met_all_valid_repeats"] is False
    assert agg["valid_repeats_mixed_slo"] is True
    assert agg["any_valid_waiting_queue"] is True
    assert first_observed_valid_collapse_rps([agg], slo_s=1.0) == 128.0


def test_first_slo_violation_does_not_invent_crossing():
    ok = {
        "offered_rps": 2.0,
        "aggregate_valid": True,
        "valid_repeat_count": 3,
        "headline_p95_ttft_s_median": 0.4,
        "valid_repeat_p95s": [0.3, 0.4, 0.4],
    }
    bad = {
        "offered_rps": 8.0,
        "aggregate_valid": True,
        "valid_repeat_count": 3,
        "headline_p95_ttft_s_median": 1.4,
        "valid_repeat_p95s": [1.2, 1.4, 1.5],
        "slo_violated_all_valid_repeats": True,
    }
    assert first_clean_slo_violation([ok, bad], slo_s=1.0) == 8.0
    assert first_slo_violation([ok], slo_s=1.0) is None
