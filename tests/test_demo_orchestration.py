from __future__ import annotations

import asyncio

from servescope.demo.app import DemoApp
from servescope.demo.state import (
    BURST_COMPLETE,
    BURST_IDLE,
    MODE_BACKPRESSURE,
    MODE_NATIVE,
)


def _config(*, burst_rps: float = 10.0, burst_duration_s: float = 0.3, **extra) -> dict:
    cfg = {
        "model": "Qwen/Qwen3-1.7B",
        "base_url": "http://127.0.0.1:9",
        "chat_path": "/v1/chat/completions",
        "metrics_path": "/metrics",
        "demo_host": "127.0.0.1",
        "demo_port": 8080,
        "seed": 20260830,
        "chat_max_tokens": 8,
        "background_min_tokens": 8,
        "background_max_tokens": 8,
        "burst_rps": burst_rps,
        "burst_duration_s": burst_duration_s,
        "request_timeout_s": 5,
        "telemetry_interval_s": 0.5,
        "controller": {
            "initial_background_limit": 32,
            "minimum_background_limit": 1,
            "maximum_background_limit": 256,
            "increase_after_zero_samples": 4,
        },
    }
    cfg.update(extra)
    return cfg


def _install_fake_jobs(demo: DemoApp, handler):
    async def fake_job(index: int) -> None:
        await handler(index)

    demo.run_background_job = fake_job  # type: ignore[method-assign]


def test_admission_overlaps_injection():
    async def body():
        demo = DemoApp(_config(burst_rps=5.0, burst_duration_s=0.6))
        overlap = []

        async def handler(index: int) -> None:
            async with demo.lock:
                if (
                    demo.state.burst_state == "injecting"
                    and demo.state.background_offered < demo.burst_jobs
                ):
                    overlap.append(
                        {
                            "index": index,
                            "offered": demo.state.background_offered,
                            "admitted": demo.state.background_admitted,
                        }
                    )
            await asyncio.sleep(0.05)
            async with demo.lock:
                demo.state.finish_one(ok=True)

        _install_fake_jobs(demo, handler)
        demo.state.set_mode(MODE_BACKPRESSURE)
        demo.state.start_burst(demo.burst_jobs)
        await demo.run_burst()
        try:
            assert overlap, "a job should be admitted before the last arrival"
            assert overlap[0]["offered"] < demo.burst_jobs
            assert overlap[0]["admitted"] >= 1
        finally:
            await demo.close()

    asyncio.run(body())


def test_no_idle_cap_drift():
    demo = DemoApp(_config())
    demo.state.set_mode(MODE_BACKPRESSURE)
    assert demo.state.burst_state == BURST_IDLE
    for _ in range(12):
        demo.apply_controller_sample(0.0)
    assert demo.controller.limit == 32
    assert demo.state.controller_limit == 32
    demo.state.burst_state = BURST_COMPLETE
    for _ in range(8):
        demo.apply_controller_sample(0.0)
    assert demo.controller.limit == 32
    asyncio.run(demo.close())


def test_controller_resets_at_burst_start():
    async def body():
        demo = DemoApp(_config(burst_rps=20.0, burst_duration_s=0.1))
        limits_at_offer: list[int] = []
        real_offer = demo.state.offer_one

        def offer():
            limits_at_offer.append(demo.controller.limit)
            return real_offer()

        demo.state.offer_one = offer  # type: ignore[method-assign]

        async def handler(_index: int) -> None:
            async with demo.lock:
                demo.state.finish_one(ok=True)

        _install_fake_jobs(demo, handler)
        demo.state.set_mode(MODE_BACKPRESSURE)
        demo.controller.limit = 47
        demo.state.controller_limit = 47
        demo.controller.zero_wait_streak = 3
        demo.state.start_burst(demo.burst_jobs)
        await demo.run_burst()
        try:
            assert limits_at_offer
            assert limits_at_offer[0] == 32
            assert demo.controller.limit == 32 or demo.state.burst_state == BURST_COMPLETE
        finally:
            await demo.close()

    asyncio.run(body())


def test_native_mode_does_not_mutate_controller():
    demo = DemoApp(_config())
    demo.state.set_mode(MODE_NATIVE)
    demo.state.start_burst(4)
    for _ in range(8):
        demo.apply_controller_sample(0.0)
        demo.apply_controller_sample(9.0)
    assert demo.controller.limit == 32
    assert demo.state.controller_limit == 32
    assert demo.state.controller_action == "hold"
    asyncio.run(demo.close())


def test_active_controller_still_decreases():
    demo = DemoApp(_config())
    demo.state.set_mode(MODE_BACKPRESSURE)
    demo.state.start_burst(4)
    demo.apply_controller_sample(5.0)
    assert demo.controller.limit == 16
    assert demo.state.controller_action == "decrease"
    asyncio.run(demo.close())


def test_burst_completion_accounting():
    async def body():
        demo = DemoApp(_config(burst_rps=10.0, burst_duration_s=0.3))

        async def handler(index: int) -> None:
            await asyncio.sleep(0.01)
            async with demo.lock:
                demo.state.finish_one(ok=index != 1)

        _install_fake_jobs(demo, handler)
        demo.state.set_mode(MODE_BACKPRESSURE)
        demo.state.start_burst(demo.burst_jobs)
        await demo.run_burst()
        try:
            assert demo.state.burst_state == BURST_COMPLETE
            assert demo.state.background_completed + demo.state.background_failed == demo.state.background_offered
            assert demo.state.background_pending == 0
            assert demo.state.background_running == 0
            assert demo.state.background_offered == demo.burst_jobs
        finally:
            await demo.close()

    asyncio.run(body())
