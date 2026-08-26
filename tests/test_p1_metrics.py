from __future__ import annotations

from servescope.metrics import (
    classify_status,
    extract_backend_metrics,
    extract_finish_reason,
    extract_usage,
    first_slo_violation,
    is_nonempty_generated_content,
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
            "actual_dispatch_s": 0.0,
            "dispatch_lag_s": 0.001,
            "completion_tokens": 64,
            "finish_reason": "length",
        },
        {
            "status": "length",
            "client_ttft_s": 0.3,
            "client_e2e_s": 0.9,
            "actual_dispatch_s": 0.25,
            "dispatch_lag_s": 0.001,
            "completion_tokens": 64,
            "finish_reason": "length",
        },
        {
            "status": "timeout",
            "client_ttft_s": None,
            "client_e2e_s": 2.0,
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
    assert summary["output_token_throughput_tps"] == 128.0
    assert summary["client_ttft_n"] == 2


def test_first_slo_violation_does_not_invent_crossing():
    ok = {"offered_rps": 2.0, "headline_p95_ttft_s_median": 0.4}
    bad = {"offered_rps": 8.0, "headline_p95_ttft_s_median": 1.4}
    assert first_slo_violation([ok, bad], slo_s=1.0) == 8.0
    assert first_slo_violation([ok], slo_s=1.0) is None
