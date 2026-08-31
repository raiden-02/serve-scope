from __future__ import annotations

from servescope.demo.state import (
    BURST_COMPLETE,
    BURST_DRAINING,
    BURST_IDLE,
    BURST_INJECTING,
    MODE_BACKPRESSURE,
    MODE_NATIVE,
    BurstBusy,
    DemoState,
    ModeLocked,
    connected_runtime,
    disconnected_runtime,
)


def test_initial_values():
    state = DemoState()
    snap = state.snapshot()
    assert snap["mode"] == MODE_NATIVE
    assert snap["burst_state"] == BURST_IDLE
    assert snap["background_offered"] == 0
    assert snap["background_pending"] == 0
    assert snap["background_running"] == 0
    assert snap["burst_allowed"] is True
    assert snap["mode_switch_allowed"] is True


def test_one_burst_at_a_time():
    state = DemoState()
    state.start_burst(2)
    assert state.burst_state == BURST_INJECTING
    try:
        state.start_burst(2)
        raise AssertionError("second burst should be blocked")
    except BurstBusy:
        pass


def test_mode_switch_blocked_while_draining():
    state = DemoState()
    state.set_mode(MODE_BACKPRESSURE)
    state.start_burst(1)
    state.offer_one()
    state.admit_one()
    state.mark_inject_finished()
    assert state.burst_state == BURST_DRAINING
    try:
        state.set_mode(MODE_NATIVE)
        raise AssertionError("mode switch should be locked")
    except ModeLocked:
        pass
    state.finish_one(ok=True)
    assert state.burst_state == BURST_COMPLETE
    state.set_mode(MODE_NATIVE)
    assert state.mode == MODE_NATIVE


def test_native_has_no_local_queue():
    state = DemoState()
    state.start_burst(3)
    assert state.offer_one()
    assert state.offer_one()
    assert state.background_pending == 0
    assert state.background_admitted == 2
    assert state.background_running == 2
    assert state.can_admit() is False


def test_backpressure_uses_local_admission():
    state = DemoState()
    state.set_mode(MODE_BACKPRESSURE)
    state.controller_limit = 1
    state.start_burst(2)
    state.offer_one()
    state.offer_one()
    assert state.background_pending == 2
    assert state.background_admitted == 0
    assert state.admit_one() is True
    assert state.background_pending == 1
    assert state.background_running == 1
    assert state.admit_one() is False
    state.finish_one(ok=True)
    assert state.admit_one() is True
    assert state.background_pending == 0


def test_counters_never_exceed_offered():
    state = DemoState()
    state.set_mode(MODE_BACKPRESSURE)
    state.start_burst(2)
    assert state.offer_one()
    assert state.offer_one()
    assert state.offer_one() is False
    assert state.background_offered == 2
    state.admit_one()
    state.admit_one()
    state.finish_one(ok=True)
    state.finish_one(ok=False)
    state.finish_one(ok=True)
    assert state.accounted() == state.background_offered
    assert state.background_completed + state.background_failed == 2


def test_disconnected_does_not_fabricate_zeros():
    runtime = disconnected_runtime()
    assert runtime["server"] == "disconnected"
    assert runtime["vllm_running"] is None
    assert runtime["vllm_waiting"] is None
    assert runtime["gpu_util_pct"] is None
    assert runtime["vram_used_mib"] is None
    connected = connected_runtime(model="Qwen/Qwen3-1.7B", gpu=None, running=3.0, waiting=1.0)
    assert connected["gpu_status"] == "unavailable"
    assert connected["gpu_util_pct"] is None
    assert connected["vllm_running"] == 3.0
