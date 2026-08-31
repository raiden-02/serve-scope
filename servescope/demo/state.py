"""Bounded live-demo state. One burst at a time."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

MODE_NATIVE = "native"
MODE_BACKPRESSURE = "backpressure"
BURST_IDLE = "idle"
BURST_INJECTING = "injecting"
BURST_DRAINING = "draining"
BURST_COMPLETE = "complete"
ACTIVE_BURST = (BURST_INJECTING, BURST_DRAINING)


class ModeLocked(RuntimeError):
    pass


class BurstBusy(RuntimeError):
    pass


@dataclass
class DemoState:
    mode: str = MODE_NATIVE
    burst_state: str = BURST_IDLE
    background_offered: int = 0
    background_admitted: int = 0
    background_running: int = 0
    background_pending: int = 0
    background_completed: int = 0
    background_failed: int = 0
    burst_target: int = 0
    controller_limit: int = 32
    controller_action: str = "hold"
    history: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=120))

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "burst_state": self.burst_state,
            "background_offered": self.background_offered,
            "background_admitted": self.background_admitted,
            "background_running": self.background_running,
            "background_pending": self.background_pending,
            "background_completed": self.background_completed,
            "background_failed": self.background_failed,
            "burst_target": self.burst_target,
            "controller_limit": self.controller_limit,
            "controller_action": self.controller_action,
            "mode_switch_allowed": self.can_switch_mode(),
            "burst_allowed": self.can_start_burst(),
        }

    def can_switch_mode(self) -> bool:
        return self.burst_state not in ACTIVE_BURST

    def can_start_burst(self) -> bool:
        return self.burst_state not in ACTIVE_BURST

    def set_mode(self, mode: str) -> None:
        if mode not in (MODE_NATIVE, MODE_BACKPRESSURE):
            raise ValueError(f"unknown mode: {mode}")
        if not self.can_switch_mode():
            raise ModeLocked("mode switch blocked until the current burst drains")
        self.mode = mode
        if mode == MODE_NATIVE:
            self.background_pending = 0

    def start_burst(self, job_count: int) -> None:
        if not self.can_start_burst():
            raise BurstBusy("only one demo burst may run at a time")
        if job_count < 1:
            raise ValueError("burst needs at least one job")
        self.background_offered = 0
        self.background_admitted = 0
        self.background_running = 0
        self.background_pending = 0
        self.background_completed = 0
        self.background_failed = 0
        self.burst_target = int(job_count)
        self.burst_state = BURST_INJECTING

    def offer_one(self) -> bool:
        if self.burst_state != BURST_INJECTING:
            return False
        if self.background_offered >= self.burst_target:
            return False
        self.background_offered += 1
        if self.mode == MODE_NATIVE:
            self.background_admitted += 1
            self.background_running += 1
        else:
            self.background_pending += 1
        return True

    def can_admit(self) -> bool:
        if self.mode != MODE_BACKPRESSURE:
            return False
        if self.background_pending <= 0:
            return False
        return self.background_running < self.controller_limit

    def admit_one(self) -> bool:
        if not self.can_admit():
            return False
        self.background_pending -= 1
        self.background_admitted += 1
        self.background_running += 1
        return True

    def finish_one(self, *, ok: bool) -> None:
        if self.background_running <= 0:
            return
        self.background_running -= 1
        if ok:
            self.background_completed += 1
        else:
            self.background_failed += 1
        self.refresh_burst_state()

    def mark_inject_finished(self) -> None:
        if self.burst_state == BURST_INJECTING and self.background_offered >= self.burst_target:
            self.refresh_burst_state()

    def refresh_burst_state(self) -> None:
        if self.burst_state == BURST_IDLE:
            return
        inflight = self.background_running + self.background_pending
        if self.background_offered < self.burst_target:
            self.burst_state = BURST_INJECTING
            return
        if inflight > 0:
            self.burst_state = BURST_DRAINING
            return
        self.burst_state = BURST_COMPLETE

    def accounted(self) -> int:
        return (
            self.background_running
            + self.background_pending
            + self.background_completed
            + self.background_failed
        )

    def record_sample(self, t_s: float, vllm_waiting: float | None) -> None:
        self.history.append(
            {
                "t_s": t_s,
                "vllm_waiting": vllm_waiting,
                "servescope_pending": self.background_pending if self.mode == MODE_BACKPRESSURE else 0,
            }
        )


def disconnected_runtime() -> dict[str, Any]:
    """Real missing telemetry. Do not fill zeros."""
    return {
        "server": "disconnected",
        "model": None,
        "gpu_name": None,
        "gpu_util_pct": None,
        "vram_used_mib": None,
        "vram_total_mib": None,
        "vllm_running": None,
        "vllm_waiting": None,
    }


def connected_runtime(*, model: str, gpu: dict[str, Any] | None, running: float | None, waiting: float | None) -> dict[str, Any]:
    return {
        "server": "connected",
        "model": model,
        "gpu_name": None if gpu is None else gpu.get("name"),
        "gpu_util_pct": None if gpu is None else gpu.get("utilization_gpu_pct"),
        "vram_used_mib": None if gpu is None else gpu.get("memory_used_mib"),
        "vram_total_mib": None if gpu is None else gpu.get("memory_total_mib"),
        "vllm_running": running,
        "vllm_waiting": waiting,
        "gpu_status": "unavailable" if gpu is None else "ok",
    }
