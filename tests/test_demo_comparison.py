from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from servescope.client import build_chat_payload
from servescope.demo.comparison import (
    OVERALL_COMPLETE,
    OVERALL_INVALID,
    P4_CONFIG_REL,
    ComparisonBlocked,
    ComparisonBusy,
    LiveComparison,
    load_p4_config,
    phase_for_elapsed,
    ttft_delta_pct,
    workload_identity,
)
from servescope.metrics import derive_request_timings, percentile
from servescope.p3_metrics import POLICY_PRIORITY

ROOT = Path(__file__).resolve().parents[1]
P4_FILES = (
    ROOT / "artifacts/p4/comparison-2026-08-31T21-15-28Z/result.json",
    ROOT / "artifacts/p4/native-2026-08-31T21-07-52Z/result.json",
    ROOT / "artifacts/p4/backpressure-2026-08-31T21-11-34Z/result.json",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _final(
    *,
    p95,
    wait,
    pending=0,
    e2e=1.0,
    offered=240,
    done=240,
    failed=0,
    interactive_failed=0,
    valid=True,
    reason=None,
):
    return {
        "valid": valid,
        "invalid_reason": reason,
        "final": {
            "burst_p95_ttft_s": p95,
            "max_vllm_waiting": wait,
            "peak_local_pending": pending,
            "background_e2e_p95_s": e2e,
            "background_output_goodput_tps": 1500.0,
            "interactive_offered": 3840,
            "interactive_completed": 3840,
            "interactive_failed": interactive_failed,
            "background_offered": offered,
            "background_admitted": offered if pending == 0 else done,
            "background_completed": done,
            "background_failed": failed,
            "valid_offered_load": valid,
            "aborted": False,
        },
        "history": [{"t_s": 16.0, "vllm_waiting": wait, "servescope_pending": pending}],
    }


def test_workload_comes_from_accepted_p4_config():
    config = load_p4_config(ROOT)
    ident = workload_identity(config)
    assert ident["source"] == P4_CONFIG_REL
    assert ident["interactive"]["offered_rps"] == 64
    assert ident["interactive"]["priority"] == 0
    assert ident["background"]["offered_rps"] == 16
    assert ident["background"]["priority"] == 1
    assert ident["background"]["max_completion_tokens"] == 256
    assert ident["controller"]["initial_background_limit"] == 32
    assert ident["scheduler_policy"] == POLICY_PRIORITY
    assert ident["pre_burst_end_s"] == 15
    assert ident["burst_end_s"] == 30
    assert ident["scenario_duration_s"] == 60


def test_priorities_match_native_priority_policy():
    payload_ix = build_chat_payload(
        model="Qwen/Qwen3-1.7B",
        prompt="x",
        temperature=0,
        min_tokens=64,
        max_completion_tokens=64,
        priority=0,
    )
    payload_bg = build_chat_payload(
        model="Qwen/Qwen3-1.7B",
        prompt="x",
        temperature=0,
        min_tokens=256,
        max_completion_tokens=256,
        priority=1,
    )
    assert payload_ix["priority"] == 0
    assert payload_bg["priority"] == 1


def test_ttft_clock_semantics_unchanged():
    clocks = derive_request_timings(
        scheduled_arrival_s=10.0,
        request_attempt_s=10.04,
        response_headers_s=10.20,
        first_content_s=10.30,
        completion_s=10.80,
    )
    assert abs(clocks["client_ttft_s"] - 0.26) < 1e-9
    assert abs(clocks["response_headers_latency_s"] - 0.16) < 1e-9
    assert abs(clocks["dispatch_lag_s"] - 0.04) < 1e-9


def test_p95_uses_numpy_linear():
    assert abs(percentile([0.10, 0.20, 0.30, 0.40], 95) - 0.385) < 1e-9


def test_phase_timeline():
    assert phase_for_elapsed(3, pre_end_s=15, burst_end_s=30, duration_s=60, inflight=1, arrivals_done=False) == "baseline"
    assert phase_for_elapsed(20, pre_end_s=15, burst_end_s=30, duration_s=60, inflight=1, arrivals_done=False) == "burst"
    assert phase_for_elapsed(40, pre_end_s=15, burst_end_s=30, duration_s=60, inflight=1, arrivals_done=False) == "recovery"
    assert phase_for_elapsed(61, pre_end_s=15, burst_end_s=30, duration_s=60, inflight=2, arrivals_done=True) == "draining"
    assert phase_for_elapsed(61, pre_end_s=15, burst_end_s=30, duration_s=60, inflight=0, arrivals_done=True) == "complete"


def test_native_runs_before_servescope_and_idle_between():
    async def body():
        order = []

        async def probe():
            return True

        async def native(side):
            order.append("native")
            assert cmp.servescope.state == "waiting"
            return _final(p95=0.30, wait=80)

        async def gated(side):
            order.append("servescope")
            assert "native" in cmp.order
            return _final(p95=0.10, wait=0, pending=120, e2e=2.0)

        async def idle(label):
            order.append(f"idle:{label}")
            return True

        async def warmup():
            order.append("warmup")

        cmp = LiveComparison(
            root=ROOT,
            probe=probe,
            run_native=native,
            run_gated=gated,
            wait_idle=idle,
            warmup=warmup,
            sleep=lambda _s: asyncio.sleep(0),
        )
        cmp.start(connected=True)
        await cmp._task
        assert order[:5] == ["warmup", "idle:before_native", "native", "idle:between_sides", "servescope"]
        assert cmp.order == ["native", "servescope"]
        assert cmp.controller_resets == 1
        assert cmp.overall_state == OVERALL_COMPLETE
        assert cmp.comparison["available"] is True
        assert cmp.comparison["ttft_delta_pct"] is not None
        assert cmp.comparison["ttft_delta_pct"] > 0
        assert "lower" in cmp.comparison["headline"]
        assert cmp.native.peak_local_pending == 0
        assert cmp.servescope.peak_local_pending == 120
        assert cmp.native.max_vllm_waiting == 80
        assert cmp.servescope.background_completed == 240

    asyncio.run(body())


def test_no_simultaneous_start():
    async def body():
        async def hang(_side):
            await asyncio.Event().wait()
            return _final(p95=0.3, wait=1)

        async def idle(_label):
            return True

        async def yes():
            return True

        async def nothing():
            return None

        cmp = LiveComparison(
            root=ROOT,
            probe=yes,
            run_native=hang,
            run_gated=hang,
            wait_idle=idle,
            warmup=nothing,
            sleep=lambda _s: asyncio.sleep(0),
        )
        cmp.start(connected=True)
        try:
            cmp.start(connected=True)
            raise AssertionError("second start should fail")
        except ComparisonBusy:
            pass
        await cmp.cancel()

    asyncio.run(body())


def test_disconnected_start_is_blocked():
    cmp = LiveComparison(root=ROOT)
    try:
        cmp.start(connected=False)
        raise AssertionError("disconnected start should fail")
    except ComparisonBlocked as exc:
        assert "disconnected" in str(exc)
    assert cmp.overall_state == "idle"


def test_invalid_run_has_no_positive_headline():
    async def body():
        async def native(_side):
            return _final(p95=0.30, wait=80, valid=False, reason="client-side saturation")

        async def gated(_side):
            return _final(p95=0.10, wait=0, pending=10)

        async def yes():
            return True

        async def nothing():
            return None

        cmp = LiveComparison(
            root=ROOT,
            probe=yes,
            run_native=native,
            run_gated=gated,
            wait_idle=lambda _l: yes(),
            warmup=nothing,
            sleep=lambda _s: asyncio.sleep(0),
        )
        cmp.start(connected=True)
        await cmp._task
        assert cmp.overall_state == OVERALL_INVALID
        assert cmp.comparison["available"] is False
        assert cmp.comparison["headline"] == "Live run was not valid for comparison."
        assert "client-side saturation" in cmp.comparison["reason"]
        assert cmp.comparison["ttft_delta_pct"] is None

    asyncio.run(body())


def test_honest_delta_when_servescope_is_worse():
    assert ttft_delta_pct(0.10, 0.20) < 0
    async def body():
        async def yes():
            return True

        async def nothing():
            return None

        async def native(_s):
            return _final(p95=0.10, wait=1)

        async def gated(_s):
            return _final(p95=0.20, wait=2, pending=9)

        cmp = LiveComparison(
            root=ROOT,
            probe=yes,
            run_native=native,
            run_gated=gated,
            wait_idle=lambda _l: yes(),
            warmup=nothing,
            sleep=lambda _s: asyncio.sleep(0),
        )
        cmp.start(connected=True)
        await cmp._task
        assert cmp.comparison["available"] is True
        assert "higher" in cmp.comparison["headline"]

    asyncio.run(body())


def test_live_run_does_not_touch_p4_artifacts():
    before = {path: _sha(path) for path in P4_FILES}

    async def body():
        async def yes():
            return True

        async def nothing():
            return None

        async def native(_s):
            return _final(p95=0.3, wait=4)

        async def gated(_s):
            return _final(p95=0.1, wait=0, pending=3)

        cmp = LiveComparison(
            root=ROOT,
            probe=yes,
            run_native=native,
            run_gated=gated,
            wait_idle=lambda _l: yes(),
            warmup=nothing,
            sleep=lambda _s: asyncio.sleep(0),
        )
        cmp.start(connected=True)
        await cmp._task
        assert cmp.wrote_paths == []

    asyncio.run(body())
    after = {path: _sha(path) for path in P4_FILES}
    assert before == after


def test_chat_stays_locked_only_while_running():
    async def body():
        started = asyncio.Event()
        release = asyncio.Event()

        async def yes():
            return True

        async def nothing():
            return None

        async def native(_side):
            started.set()
            await release.wait()
            return _final(p95=0.3, wait=1)

        async def gated(_side):
            return _final(p95=0.1, wait=0, pending=1)

        cmp = LiveComparison(
            root=ROOT,
            probe=yes,
            run_native=native,
            run_gated=gated,
            wait_idle=lambda _l: yes(),
            warmup=nothing,
            sleep=lambda _s: asyncio.sleep(0),
        )
        cmp.start(connected=True)
        await started.wait()
        assert cmp.chat_locked is True
        release.set()
        await cmp._task
        assert cmp.chat_locked is False

    asyncio.run(body())


def test_interactive_failures_invalidate_comparison():
    async def body():
        async def yes():
            return True

        async def nothing():
            return None

        async def native(_side):
            return _final(p95=0.30, wait=80, interactive_failed=3)

        async def gated(_side):
            return _final(p95=0.10, wait=0, pending=10)

        cmp = LiveComparison(
            root=ROOT,
            probe=yes,
            run_native=native,
            run_gated=gated,
            wait_idle=lambda _l: yes(),
            warmup=nothing,
            sleep=lambda _s: asyncio.sleep(0),
        )
        cmp.start(connected=True)
        await cmp._task
        assert cmp.overall_state == OVERALL_INVALID
        assert cmp.comparison["available"] is False
        assert cmp.comparison["headline"] == "Live run was not valid for comparison."
        assert cmp.comparison["ttft_delta_pct"] is None
        assert "interactive_request_failures=3" in cmp.comparison["reason"]
        assert "%" not in (cmp.comparison["headline"] or "")

    asyncio.run(body())


def test_background_failures_invalidate_comparison():
    async def body():
        async def yes():
            return True

        async def nothing():
            return None

        async def native(_side):
            return _final(p95=0.30, wait=80)

        async def gated(_side):
            return _final(p95=0.10, wait=0, pending=10, failed=2)

        cmp = LiveComparison(
            root=ROOT,
            probe=yes,
            run_native=native,
            run_gated=gated,
            wait_idle=lambda _l: yes(),
            warmup=nothing,
            sleep=lambda _s: asyncio.sleep(0),
        )
        cmp.start(connected=True)
        await cmp._task
        assert cmp.overall_state == OVERALL_INVALID
        assert cmp.comparison["available"] is False
        assert cmp.comparison["headline"] == "Live run was not valid for comparison."
        assert cmp.comparison["ttft_delta_pct"] is None
        assert "background_request_failures=2" in cmp.comparison["reason"]
        assert "%" not in (cmp.comparison["headline"] or "")

    asyncio.run(body())


def test_missing_burst_p95_invalidates_comparison():
    async def body():
        async def yes():
            return True

        async def nothing():
            return None

        async def native(_side):
            return _final(p95=None, wait=80)

        async def gated(_side):
            return _final(p95=0.10, wait=0, pending=10)

        cmp = LiveComparison(
            root=ROOT,
            probe=yes,
            run_native=native,
            run_gated=gated,
            wait_idle=lambda _l: yes(),
            warmup=nothing,
            sleep=lambda _s: asyncio.sleep(0),
        )
        cmp.start(connected=True)
        await cmp._task
        assert cmp.overall_state == OVERALL_INVALID
        assert cmp.comparison["available"] is False
        assert cmp.comparison["headline"] == "Live run was not valid for comparison."
        assert cmp.comparison["ttft_delta_pct"] is None
        assert "missing_burst_p95" in cmp.comparison["reason"]
        assert "%" not in (cmp.comparison["headline"] or "")

    asyncio.run(body())


def test_native_telemetry_is_observable_before_runner_returns():
    """Native live lists must update while the side is still SIDE_RUNNING."""
    from unittest.mock import patch

    from servescope.demo import comparison as cmp_mod
    from servescope.p2_metrics import CLASS_BACKGROUND, CLASS_INTERACTIVE, PHASE_BURST
    from servescope.workload import select_prompt

    seen = {}

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    async def fake_offer(_client, _config, _class_cfg, *, records, prompt_fn, **_kwargs):
        cls = CLASS_INTERACTIVE if prompt_fn is select_prompt else CLASS_BACKGROUND
        records.append(
            {
                "workload_class": cls,
                "status": "success",
                "phase": PHASE_BURST,
                "client_ttft_s": 0.22,
                "request_index": 1,
            }
        )
        await asyncio.sleep(1.2)
        return {"aborted": False, "abort_reason": None, "peak_inflight": 1}

    async def fake_runtime(_client, _url, _interval, stop, out, _t0):
        out.append(
            {
                "t_s": 0.2,
                "num_requests_running": 1,
                "num_requests_waiting": 17,
                "kv_cache_usage_perc": 8,
            }
        )
        await stop.wait()

    async def fake_gpu(_interval, stop, _out, _t0):
        await stop.wait()

    def fake_summarize(records, runtime_rows, gpu_rows, meta, config, **_kwargs):
        return {
            "repeat": {
                "valid_offered_load": True,
                "interactive_p95_burst_s": 0.22,
                "max_waiting_requests": 17,
                "background_total_e2e_p95_s": 1.0,
                "background_completed_count": 1,
            }
        }

    async def body():
        async def yes():
            return True

        async def nothing():
            return None

        async def gated(_side):
            return _final(p95=0.10, wait=0, pending=4)

        cmp = LiveComparison(
            root=ROOT,
            probe=yes,
            run_gated=gated,
            wait_idle=lambda _l: yes(),
            warmup=nothing,
            sleep=lambda _s: asyncio.sleep(0),
        )
        assert cmp._run_native is None

        with (
            patch.object(cmp_mod.p2, "make_client", lambda *_a, **_k: DummyClient()),
            patch.object(cmp_mod.p2, "offer_class", fake_offer),
            patch.object(cmp_mod.p2, "sample_runtime", fake_runtime),
            patch.object(cmp_mod.p2, "sample_gpu", fake_gpu),
            patch.object(cmp_mod.p2, "summarize_scenario", fake_summarize),
            patch.object(cmp_mod.p4, "annotate_native_records", lambda *_a, **_k: None),
            patch.object(cmp_mod.p4, "enrich_p4", lambda *_a, **_k: None),
        ):
            cmp.start(connected=True)
            deadline = asyncio.get_event_loop().time() + 3.0
            while asyncio.get_event_loop().time() < deadline:
                if cmp.native.state == "running":
                    snap = cmp.native.public()["metrics"]
                    live = (
                        (snap.get("burst_p95_sample_count") or 0) > 0
                        or snap.get("vllm_waiting") is not None
                        or (cmp.native.history and len(cmp.native.history) > 0)
                        or (snap.get("interactive_completed") or 0) > 0
                        or (snap.get("background_completed") or 0) > 0
                    )
                    if live:
                        seen.update(snap)
                        seen["history_points"] = len(cmp.native.history)
                        seen["state"] = cmp.native.state
                        break
                await asyncio.sleep(0.1)
            else:
                raise AssertionError("Native telemetry was not visible before the runner returned")
            await cmp._task

        assert seen["state"] == "running"
        assert (seen.get("burst_p95_sample_count") or 0) > 0
        assert seen.get("vllm_waiting") is not None
        assert (seen.get("history_points") or 0) > 0
        assert (seen.get("interactive_completed") or 0) > 0

    asyncio.run(body())


def test_cancel_stops_tracked_child_tasks():
    async def body():
        started = asyncio.Event()

        async def yes():
            return True

        async def nothing():
            return None

        async def native(_side):
            child = asyncio.create_task(asyncio.Event().wait())
            cmp._track(child)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                seen["child_done"] = child.done()
                seen["child_cancelled"] = child.cancelled()
            return _final(p95=0.3, wait=1)

        async def gated(_side):
            return _final(p95=0.1, wait=0, pending=1)

        seen = {}
        cmp = LiveComparison(
            root=ROOT,
            probe=yes,
            run_native=native,
            run_gated=gated,
            wait_idle=lambda _l: yes(),
            warmup=nothing,
            sleep=lambda _s: asyncio.sleep(0),
        )
        cmp.start(connected=True)
        await started.wait()
        await cmp.cancel()
        assert cmp.overall_state == "cancelled"
        assert seen.get("child_cancelled") is True or seen.get("child_done") is True
        assert all(task.done() for task in cmp._child_tasks)

    asyncio.run(body())
