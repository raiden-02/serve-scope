"""Headful paired P4-style comparison. Native always runs before ServeScope."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from servescope.backpressure import AimdController
from servescope.metrics import COMPLETED_STATUSES, percentile
from servescope.p2_metrics import (
    CLASS_BACKGROUND,
    CLASS_INTERACTIVE,
    PHASE_BURST as RECORD_PHASE_BURST,
    filter_class,
    last_completion_rel_s,
)
from servescope.p3_metrics import POLICY_PRIORITY, request_priority_for_class
from servescope.workload import select_background_prompt, select_prompt, windowed_arrivals

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import p2_mixed as p2  # noqa: E402
import p4_backpressure as p4  # noqa: E402

P4_CONFIG_REL = "configs/p4_backpressure.json"

OVERALL_IDLE = "idle"
OVERALL_RUNNING = "running"
OVERALL_COMPLETE = "complete"
OVERALL_INVALID = "invalid"
OVERALL_CANCELLED = "cancelled"
OVERALL_FAILED = "failed"

SIDE_READY = "ready"
SIDE_WAITING = "waiting"
SIDE_RUNNING = "running"
SIDE_COMPLETE = "complete"
SIDE_FAILED = "failed"

PHASE_IDLE = "idle"
PHASE_WARMUP = "warmup"
PHASE_BASELINE = "baseline"
PHASE_BURST = "burst"
PHASE_RECOVERY = "recovery"
PHASE_DRAINING = "draining"
PHASE_COMPLETE = "complete"

ACTIVE_OVERALL = (OVERALL_RUNNING,)


class ComparisonBusy(RuntimeError):
    pass


class ComparisonBlocked(RuntimeError):
    pass


def load_p4_config(root: Path | None = None) -> dict[str, Any]:
    path = (root or ROOT) / P4_CONFIG_REL
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("P4 config must be an object")
    return payload


def workload_identity(config: dict[str, Any]) -> dict[str, Any]:
    interactive = config["interactive"]
    background = config["background"]
    controller = config["controller"]
    return {
        "source": P4_CONFIG_REL,
        "warmup_requests": int(config["warmup_requests"]),
        "scenario_duration_s": float(config["scenario_duration_s"]),
        "pre_burst_end_s": float(config["pre_burst_end_s"]),
        "burst_end_s": float(config["burst_end_s"]),
        "interactive": {
            "offered_rps": float(interactive["offered_rps"]),
            "min_tokens": int(interactive["min_tokens"]),
            "max_completion_tokens": int(interactive["max_completion_tokens"]),
            "temperature": interactive["temperature"],
            "priority": int(interactive["priority"]),
        },
        "background": {
            "offered_rps": float(background["offered_rps"]),
            "min_tokens": int(background["min_tokens"]),
            "max_completion_tokens": int(background["max_completion_tokens"]),
            "temperature": background["temperature"],
            "priority": int(background["priority"]),
        },
        "controller": {
            "initial_background_limit": int(controller["initial_background_limit"]),
            "minimum_background_limit": int(controller["minimum_background_limit"]),
            "maximum_background_limit": int(controller["maximum_background_limit"]),
            "increase_after_zero_samples": int(controller["increase_after_zero_samples"]),
            "sample_interval_s": float(controller["sample_interval_s"]),
            "pending_queue_capacity": int(controller["pending_queue_capacity"]),
        },
        "scheduler_policy": POLICY_PRIORITY,
        "interactive_priority": request_priority_for_class(CLASS_INTERACTIVE, POLICY_PRIORITY),
        "background_priority": request_priority_for_class(CLASS_BACKGROUND, POLICY_PRIORITY),
    }


def phase_for_elapsed(
    elapsed_s: float | None,
    *,
    pre_end_s: float,
    burst_end_s: float,
    duration_s: float,
    inflight: int,
    arrivals_done: bool,
) -> str:
    if elapsed_s is None:
        return PHASE_IDLE
    if elapsed_s < pre_end_s:
        return PHASE_BASELINE
    if elapsed_s < burst_end_s:
        return PHASE_BURST
    if elapsed_s < duration_s:
        return PHASE_RECOVERY
    if inflight > 0 or not arrivals_done:
        return PHASE_DRAINING
    return PHASE_COMPLETE


def evaluate_side_validity(
    *,
    valid_offered_load: bool,
    aborted: bool,
    interactive_failed: int,
    background_failed: int,
    burst_p95_ttft_s: float | None,
    extra_reasons: list[str] | None = None,
) -> tuple[bool, str | None]:
    reasons: list[str] = []
    if not valid_offered_load:
        reasons.append("invalid_offered_load")
    if aborted:
        reasons.append("aborted")
    if interactive_failed:
        reasons.append(f"interactive_request_failures={interactive_failed}")
    if background_failed:
        reasons.append(f"background_request_failures={background_failed}")
    if burst_p95_ttft_s is None:
        reasons.append("missing_burst_p95")
    if extra_reasons:
        reasons.extend(reason for reason in extra_reasons if reason)
    if not reasons:
        return True, None
    return False, ", ".join(reasons)


def ttft_delta_pct(native_s: float | None, gated_s: float | None) -> float | None:
    if native_s is None or gated_s is None:
        return None
    start = float(native_s)
    if start <= 0:
        return None
    return (start - float(gated_s)) / start * 100.0


def count_status(records: list[dict[str, Any]], workload_class: str) -> tuple[int, int]:
    rows = filter_class(records, workload_class)
    completed = sum(1 for row in rows if row.get("status") in COMPLETED_STATUSES)
    failed = len(rows) - completed
    return completed, failed


def live_burst_p95(records: list[dict[str, Any]]) -> tuple[float | None, int]:
    burst = [
        row["client_ttft_s"]
        for row in filter_class(records, CLASS_INTERACTIVE)
        if row.get("phase") == RECORD_PHASE_BURST
        and row.get("status") in COMPLETED_STATUSES
        and row.get("client_ttft_s") is not None
    ]
    return percentile(burst, 95), len(burst)


def waiting_stats(runtime_rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    vals = [row.get("num_requests_waiting") for row in runtime_rows if row.get("num_requests_waiting") is not None]
    if not vals:
        return None, None
    return float(vals[-1]), float(max(vals))


def make_controller(config: dict[str, Any]) -> AimdController:
    return p4.make_controller(config)


@dataclass
class SideView:
    name: str
    gated: bool
    state: str = SIDE_WAITING
    phase: str = PHASE_IDLE
    elapsed_s: float | None = None
    status_label: str = "Waiting"
    interactive_target: int = 0
    interactive_offered: int = 0
    interactive_completed: int = 0
    interactive_failed: int = 0
    background_target: int = 0
    background_offered: int = 0
    background_admitted: int = 0
    background_completed: int = 0
    background_failed: int = 0
    vllm_waiting: float | None = None
    max_vllm_waiting: float | None = None
    local_pending: int = 0
    peak_local_pending: int = 0
    burst_p95_ttft_s: float | None = None
    burst_p95_sample_count: int = 0
    background_e2e_p95_s: float | None = None
    background_output_goodput_tps: float | None = None
    valid: bool | None = None
    invalid_reason: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    final: dict[str, Any] | None = None

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "gated": self.gated,
            "state": self.state,
            "phase": self.phase,
            "elapsed_s": self.elapsed_s,
            "status_label": self.status_label,
            "metrics": {
                "burst_p95_ttft_s": self.burst_p95_ttft_s,
                "burst_p95_sample_count": self.burst_p95_sample_count,
                "vllm_waiting": self.vllm_waiting,
                "max_vllm_waiting": self.max_vllm_waiting,
                "local_pending": self.local_pending if self.gated else 0,
                "peak_local_pending": self.peak_local_pending if self.gated else 0,
                "interactive_target": self.interactive_target,
                "interactive_offered": self.interactive_offered,
                "interactive_completed": self.interactive_completed,
                "interactive_failed": self.interactive_failed,
                "background_target": self.background_target,
                "background_offered": self.background_offered,
                "background_admitted": self.background_admitted,
                "background_completed": self.background_completed,
                "background_failed": self.background_failed,
                "background_e2e_p95_s": self.background_e2e_p95_s,
                "background_output_goodput_tps": self.background_output_goodput_tps,
            },
            "history": list(self.history),
            "final": self.final,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
        }


class LiveComparison:
    def __init__(
        self,
        *,
        root: Path | None = None,
        config: dict[str, Any] | None = None,
        probe: Callable[[], Awaitable[bool]] | None = None,
        run_native: Callable[..., Awaitable[dict[str, Any]]] | None = None,
        run_gated: Callable[..., Awaitable[dict[str, Any]]] | None = None,
        wait_idle: Callable[..., Awaitable[bool]] | None = None,
        warmup: Callable[..., Awaitable[None]] | None = None,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        clock=time.perf_counter,
    ) -> None:
        self.root = root or ROOT
        self.config = config or load_p4_config(self.root)
        self.workload = workload_identity(self.config)
        self._probe = probe
        self._run_native = run_native
        self._run_gated = run_gated
        self._wait_idle = wait_idle
        self._warmup = warmup
        self._sleep = sleep
        self.clock = clock
        self.order: list[str] = []
        self.idle_checks: list[str] = []
        self.controller_resets: int = 0
        self.wrote_paths: list[str] = []
        self.overall_state = OVERALL_IDLE
        self.native = SideView("native", gated=False, state=SIDE_READY, status_label="Ready")
        self.servescope = SideView("servescope", gated=True, state=SIDE_WAITING, status_label="Waiting")
        self.comparison: dict[str, Any] = {"available": False, "ttft_delta_pct": None, "headline": None, "reason": None}
        self.chat_locked = False
        self.error: str | None = None
        self._task: asyncio.Task | None = None
        self._cancel = asyncio.Event()
        self._child_tasks: list[asyncio.Task] = []

    def is_active(self) -> bool:
        return self.overall_state in ACTIVE_OVERALL

    def snapshot(self) -> dict[str, Any]:
        return {
            "overall_state": self.overall_state,
            "chat_locked": self.chat_locked,
            "workload": self.workload,
            "native": self.native.public(),
            "servescope": self.servescope.public(),
            "comparison": dict(self.comparison),
            "error": self.error,
            "order": list(self.order),
        }

    def start(self, *, connected: bool) -> dict[str, Any]:
        if self.is_active():
            raise ComparisonBusy("only one live comparison may run at a time")
        if not connected:
            raise ComparisonBlocked("vLLM is disconnected")
        self._reset_views()
        self.overall_state = OVERALL_RUNNING
        self.chat_locked = True
        self.native.state = SIDE_READY
        self.native.status_label = "Starting"
        self.servescope.state = SIDE_WAITING
        self.servescope.status_label = "Waiting for Native run"
        self._cancel = asyncio.Event()
        self._task = asyncio.create_task(self._run())
        return self.snapshot()

    async def cancel(self) -> dict[str, Any]:
        if not self.is_active():
            return self.snapshot()
        self._cancel.set()
        await self._cancel_children()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.overall_state = OVERALL_CANCELLED
        self.chat_locked = False
        self.comparison = {
            "available": False,
            "ttft_delta_pct": None,
            "headline": "Live run was not valid for comparison.",
            "reason": "cancelled",
        }
        return self.snapshot()

    def _reset_views(self) -> None:
        self.native = SideView("native", gated=False, state=SIDE_READY, status_label="Ready")
        self.servescope = SideView("servescope", gated=True, state=SIDE_WAITING, status_label="Waiting")
        self.comparison = {"available": False, "ttft_delta_pct": None, "headline": None, "reason": None}
        self.error = None
        self.order.clear()
        self.idle_checks.clear()
        self.controller_resets = 0
        self.wrote_paths.clear()
        self._child_tasks.clear()

    async def _cancel_children(self) -> None:
        for task in list(self._child_tasks):
            if not task.done():
                task.cancel()
        if self._child_tasks:
            await asyncio.gather(*self._child_tasks, return_exceptions=True)
        self._child_tasks.clear()

    def _track(self, task: asyncio.Task) -> asyncio.Task:
        self._child_tasks.append(task)
        return task

    async def _run(self) -> None:
        try:
            if self._probe is not None and not await self._probe():
                raise ComparisonBlocked("vLLM is disconnected")
            await self._warmup_if_needed()
            await self._ensure_idle("before_native")
            native_final = await self._run_side(self.native, gated=False)
            await self._ensure_idle("between_sides")
            await self._sleep(float(self.config.get("inter_run_idle_s") or 0.0))
            gated_final = await self._run_side(self.servescope, gated=True)
            self._finish_pair(native_final, gated_final)
        except asyncio.CancelledError:
            await self._cancel_children()
            self.overall_state = OVERALL_CANCELLED
            self.chat_locked = False
            self.comparison = {
                "available": False,
                "ttft_delta_pct": None,
                "headline": "Live run was not valid for comparison.",
                "reason": "cancelled",
            }
            raise
        except ComparisonBlocked as exc:
            self.overall_state = OVERALL_FAILED
            self.error = str(exc)
            self.chat_locked = False
            self.comparison = {
                "available": False,
                "ttft_delta_pct": None,
                "headline": "Live run was not valid for comparison.",
                "reason": str(exc),
            }
        except Exception as exc:
            self.overall_state = OVERALL_FAILED
            self.error = str(exc)
            self.chat_locked = False
            self.comparison = {
                "available": False,
                "ttft_delta_pct": None,
                "headline": "Live run was not valid for comparison.",
                "reason": str(exc),
            }

    async def _warmup_if_needed(self) -> None:
        self.native.phase = PHASE_WARMUP
        self.native.status_label = "Warmup"
        if self._warmup is not None:
            await self._warmup()
            return
        interactive = self.config["interactive"]
        async with p2.make_client(self.config, interactive) as client:
            await p2.warmup(client, self.config)

    async def _ensure_idle(self, label: str) -> None:
        self.idle_checks.append(label)
        if self._wait_idle is not None:
            ok = await self._wait_idle(label)
        else:
            async with p2.make_client(self.config, self.config["telemetry"]) as client:
                ok = await p2.wait_until_idle(client, self.config)
        if not ok:
            raise ComparisonBlocked("runtime did not become idle before the next side")

    async def _run_side(self, side: SideView, *, gated: bool) -> dict[str, Any]:
        if gated:
            self.controller_resets += 1
        self.order.append("servescope" if gated else "native")
        side.state = SIDE_RUNNING
        side.status_label = "Running"
        side.phase = PHASE_BASELINE
        if gated:
            runner = self._run_gated if self._run_gated is not None else self._default_gated
        else:
            runner = self._run_native if self._run_native is not None else self._default_native
        result = await runner(side)
        final = result.get("final") or {}
        raw_reason = result.get("invalid_reason")
        generated = raw_reason and any(
            token in str(raw_reason)
            for token in (
                "interactive_request_failures=",
                "background_request_failures=",
                "missing_burst_p95",
                "invalid_offered_load",
            )
        )
        extras = [str(raw_reason)] if raw_reason and not generated else None
        offered_valid = final.get("valid_offered_load")
        if offered_valid is None:
            offered_valid = bool(result.get("valid"))
        ok, reason = evaluate_side_validity(
            valid_offered_load=bool(offered_valid),
            aborted=bool(final.get("aborted")),
            interactive_failed=int(final.get("interactive_failed") or 0),
            background_failed=int(final.get("background_failed") or 0),
            burst_p95_ttft_s=final.get("burst_p95_ttft_s"),
            extra_reasons=extras,
        )
        result["valid"] = ok
        result["invalid_reason"] = reason
        side.final = result.get("final")
        side.valid = ok
        side.invalid_reason = reason
        self._apply_final_metrics(side, result)
        side.state = SIDE_COMPLETE if side.valid else SIDE_FAILED
        side.phase = PHASE_COMPLETE
        side.status_label = "Complete" if side.valid else "Invalid"
        return result

    async def _default_native(self, side: SideView) -> dict[str, Any]:
        return await self._execute_p4_side(side, gated=False)

    async def _run_native_live(
        self,
        interactive_client,
        background_client,
        telemetry_client,
        records: list[dict[str, Any]],
        runtime_rows: list[dict[str, Any]],
        gpu_rows: list[dict[str, Any]],
        t0_box: dict[str, float | None],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """Same Native mixed path as p2.run_scenario, using shared live lists."""
        config = self.config
        clock = self.clock
        interactive_cfg = dict(config["interactive"])
        background_cfg = dict(config["background"])
        duration_s = float(config["scenario_duration_s"])
        pre_end = float(config["pre_burst_end_s"])
        burst_end = float(config["burst_end_s"])
        t0 = clock()
        t0_box["t0"] = t0
        interactive_cfg["offered_rps"] = float(interactive_cfg["offered_rps"])
        background_rps = float(background_cfg["offered_rps"])
        background_cfg["offered_rps"] = background_rps
        interactive_targets = windowed_arrivals(interactive_cfg["offered_rps"], t0, 0.0, duration_s)
        background_targets = windowed_arrivals(background_rps, t0, pre_end, burst_end)
        stop = asyncio.Event()
        metrics_url = config["base_url"].rstrip("/") + config["metrics_path"]
        priorities = p4.class_priorities()
        runtime_task = self._track(
            asyncio.create_task(
                p2.sample_runtime(
                    telemetry_client,
                    metrics_url,
                    float(config["runtime_sample_interval_s"]),
                    stop,
                    runtime_rows,
                    t0,
                )
            )
        )
        gpu_task = self._track(
            asyncio.create_task(p2.sample_gpu(float(config["gpu_sample_interval_s"]), stop, gpu_rows, t0))
        )
        interactive_inflight = p2.Inflight(int(interactive_cfg["max_inflight"]))
        background_inflight = p2.Inflight(int(background_cfg["max_inflight"]))
        interactive_offer = self._track(
            asyncio.create_task(
                p2.offer_class(
                    interactive_client,
                    config,
                    interactive_cfg,
                    suite_id="live-comparison",
                    repeat_id="native",
                    scenario="mixed",
                    t0=t0,
                    targets=interactive_targets,
                    clock=clock,
                    records=records,
                    inflight=interactive_inflight,
                    prompt_fn=select_prompt,
                    seed=int(config["seed"]),
                    request_priority=priorities[CLASS_INTERACTIVE],
                    scheduler_policy=POLICY_PRIORITY,
                    send_priority_field=True,
                )
            )
        )
        background_offer = self._track(
            asyncio.create_task(
                p2.offer_class(
                    background_client,
                    config,
                    background_cfg,
                    suite_id="live-comparison",
                    repeat_id="native",
                    scenario="mixed",
                    t0=t0,
                    targets=background_targets,
                    clock=clock,
                    records=records,
                    inflight=background_inflight,
                    prompt_fn=select_background_prompt,
                    seed=int(config["seed"]) + 1_000_000,
                    request_priority=priorities[CLASS_BACKGROUND],
                    scheduler_policy=POLICY_PRIORITY,
                    send_priority_field=True,
                )
            )
        )
        try:
            interactive_meta, background_meta = await asyncio.gather(interactive_offer, background_offer)
        except asyncio.CancelledError:
            await self._cancel_children()
            raise
        finally:
            stop.set()
            await asyncio.gather(runtime_task, gpu_task, return_exceptions=True)
        duration_wall = clock() - t0
        records.sort(key=lambda row: (row.get("workload_class") or "", row.get("request_index") or 0))
        bg_rows = filter_class(records, CLASS_BACKGROUND)
        meta = {
            "t0": t0,
            "aborted": bool(interactive_meta["aborted"] or background_meta["aborted"]),
            "abort_reason": interactive_meta["abort_reason"] or background_meta["abort_reason"],
            "duration_s": duration_wall,
            "peak_inflight_interactive": interactive_meta["peak_inflight"],
            "peak_inflight_background": background_meta["peak_inflight"],
            "last_background_completion_s": last_completion_rel_s(bg_rows, t0),
            "background_offered_rps": background_cfg.get("offered_rps") or 0.0,
            "interactive_offered_count": len(interactive_targets),
            "background_offered_count": len(background_targets),
        }
        return records, runtime_rows, gpu_rows, meta

    async def _default_gated(self, side: SideView) -> dict[str, Any]:
        return await self._execute_p4_side(side, gated=True)

    async def _run_gated_live(
        self,
        interactive_client,
        background_client,
        telemetry_client,
        records: list[dict[str, Any]],
        runtime_rows: list[dict[str, Any]],
        gpu_rows: list[dict[str, Any]],
        controller_rows: list[dict[str, Any]],
        pending_holder: dict[str, Any],
        t0_box: dict[str, float | None],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """Same admission path as p4.run_gated_scenario, with live queue handles."""
        from servescope.backpressure import PendingQueue, offered_span_goodput

        config = self.config
        clock = self.clock
        interactive_cfg = dict(config["interactive"])
        background_cfg = dict(config["background"])
        duration_s = float(config["scenario_duration_s"])
        pre_end = float(config["pre_burst_end_s"])
        burst_end = float(config["burst_end_s"])
        t0 = clock()
        t0_box["t0"] = t0
        interactive_cfg["offered_rps"] = float(interactive_cfg["offered_rps"])
        interactive_targets = windowed_arrivals(interactive_cfg["offered_rps"], t0, 0.0, duration_s)
        background_rps = float(background_cfg["offered_rps"])
        background_cfg["offered_rps"] = background_rps
        background_targets = windowed_arrivals(background_rps, t0, pre_end, burst_end)
        stop = asyncio.Event()
        ingress_done = asyncio.Event()
        pending = PendingQueue(int(config["controller"]["pending_queue_capacity"]))
        controller = make_controller(config)
        counts = {"inflight": 0, "admitted": 0, "completed": 0}
        pending_holder["pending"] = pending
        pending_holder["counts"] = counts
        pending_holder["controller"] = controller
        safety = p2.Inflight(int(background_cfg["max_inflight"]))
        metrics_url = config["base_url"].rstrip("/") + config["metrics_path"]
        priorities = p4.class_priorities()
        runtime_task = self._track(asyncio.create_task(
            p4.sample_runtime_and_control(
                telemetry_client,
                metrics_url,
                float(config["controller"]["sample_interval_s"]),
                stop,
                runtime_rows,
                controller_rows,
                t0,
                controller,
                counts,
                pending,
            )
        ))
        gpu_task = self._track(asyncio.create_task(p2.sample_gpu(float(config["gpu_sample_interval_s"]), stop, gpu_rows, t0)))
        interactive_inflight = p2.Inflight(int(interactive_cfg["max_inflight"]))
        interactive_offer = self._track(asyncio.create_task(
            p2.offer_class(
                interactive_client,
                config,
                interactive_cfg,
                suite_id="live-comparison",
                repeat_id="servescope",
                scenario="mixed",
                t0=t0,
                targets=interactive_targets,
                clock=clock,
                records=records,
                inflight=interactive_inflight,
                prompt_fn=select_prompt,
                seed=int(config["seed"]),
                request_priority=priorities[CLASS_INTERACTIVE],
                scheduler_policy=POLICY_PRIORITY,
                send_priority_field=True,
            )
        ))
        ingress_offer = self._track(asyncio.create_task(
            p4.offer_background_ingress(
                t0=t0,
                targets=background_targets,
                clock=clock,
                pending=pending,
                seed=int(config["seed"]) + 1_000_000,
            )
        ))
        admit_task = self._track(asyncio.create_task(
            p4.admit_background(
                background_client,
                config,
                suite_id="live-comparison",
                repeat_id="servescope",
                scenario="mixed",
                t0=t0,
                clock=clock,
                records=records,
                pending=pending,
                controller=controller,
                counts=counts,
                safety=safety,
                ingress_done=ingress_done,
                request_priority=priorities[CLASS_BACKGROUND],
            )
        ))
        try:
            interactive_meta, ingress_meta = await asyncio.gather(interactive_offer, ingress_offer)
            ingress_done.set()
            background_meta = await admit_task
        except asyncio.CancelledError:
            await self._cancel_children()
            raise
        finally:
            stop.set()
            await asyncio.gather(runtime_task, gpu_task, return_exceptions=True)
        duration_wall = clock() - t0
        records.sort(key=lambda row: (row.get("workload_class") or "", row.get("request_index") or 0))
        bg_rows = filter_class(records, CLASS_BACKGROUND)
        meta = {
            "t0": t0,
            "aborted": bool(interactive_meta["aborted"] or ingress_meta["aborted"] or background_meta["aborted"]),
            "abort_reason": interactive_meta["abort_reason"]
            or ingress_meta["abort_reason"]
            or background_meta["abort_reason"],
            "duration_s": duration_wall,
            "peak_inflight_interactive": interactive_meta["peak_inflight"],
            "peak_inflight_background": background_meta["peak_inflight"],
            "last_background_completion_s": offered_span_goodput(bg_rows, t0)["last_completion_rel_s"],
            "background_offered_rps": background_cfg.get("offered_rps") or 0.0,
            "interactive_offered_count": len(interactive_targets),
            "background_offered_count": len(background_targets),
            "max_local_pending_depth": pending.max_depth,
        }
        return records, runtime_rows, gpu_rows, controller_rows, meta

    async def _execute_p4_side(self, side: SideView, *, gated: bool) -> dict[str, Any]:
        config = self.config
        clock = self.clock
        records: list[dict[str, Any]] = []
        runtime_rows: list[dict[str, Any]] = []
        gpu_rows: list[dict[str, Any]] = []
        controller_rows: list[dict[str, Any]] = []
        pending_holder: dict[str, Any] = {"pending": None, "counts": None, "controller": None}
        t0_box: dict[str, float | None] = {"t0": clock()}
        stop_poll = asyncio.Event()

        async def poll() -> None:
            while not stop_poll.is_set():
                self._refresh_live(side, records, runtime_rows, pending_holder, t0_box)
                try:
                    await asyncio.wait_for(stop_poll.wait(), timeout=0.5)
                except TimeoutError:
                    continue

        poll_task = self._track(asyncio.create_task(poll()))
        interactive = p2.make_client(config, config["interactive"])
        background = p2.make_client(config, config["background"])
        telemetry = p2.make_client(config, config["telemetry"])
        try:
            async with interactive, background, telemetry:
                if gated:
                    records, runtime_rows, gpu_rows, controller_rows, meta = await self._run_gated_live(
                        interactive,
                        background,
                        telemetry,
                        records,
                        runtime_rows,
                        gpu_rows,
                        controller_rows,
                        pending_holder,
                        t0_box,
                    )
                else:
                    records, runtime_rows, gpu_rows, meta = await self._run_native_live(
                        interactive,
                        background,
                        telemetry,
                        records,
                        runtime_rows,
                        gpu_rows,
                        t0_box,
                    )
                    p4.annotate_native_records(records)
        finally:
            stop_poll.set()
            await poll_task

        packed = p2.summarize_scenario(
            records,
            runtime_rows,
            gpu_rows,
            meta,
            config,
            suite_id="live-comparison",
            repeat_id="servescope" if gated else "native",
            scenario="mixed",
            background_rps=float(config["background"]["offered_rps"]),
        )
        p4.enrich_p4(
            packed,
            records,
            meta,
            config,
            runtime_rows,
            gated=gated,
            controller_rows=controller_rows,
        )
        summary = packed["repeat"]
        extras = []
        if summary.get("invalid_reason"):
            extras.append(str(summary["invalid_reason"]))
        if meta.get("abort_reason"):
            extras.append(str(meta["abort_reason"]))
        ix_done, ix_fail = count_status(records, CLASS_INTERACTIVE)
        bg_done, bg_fail = count_status(records, CLASS_BACKGROUND)
        valid, invalid_reason = evaluate_side_validity(
            valid_offered_load=bool(summary.get("valid_offered_load")),
            aborted=bool(meta.get("aborted")),
            interactive_failed=ix_fail,
            background_failed=bg_fail,
            burst_p95_ttft_s=summary.get("interactive_p95_burst_s"),
            extra_reasons=extras or None,
        )
        history = [
            {
                "t_s": row.get("t_s"),
                "vllm_waiting": row.get("num_requests_waiting"),
                "servescope_pending": (row.get("local_pending_depth") if gated else 0),
            }
            for row in (controller_rows if gated and controller_rows else runtime_rows)
        ]
        if gated and not history:
            history = [
                {"t_s": row.get("t_s"), "vllm_waiting": row.get("num_requests_waiting"), "servescope_pending": 0}
                for row in runtime_rows
            ]
        return {
            "valid": valid,
            "invalid_reason": invalid_reason,
            "final": {
                "burst_p95_ttft_s": summary.get("interactive_p95_burst_s"),
                "max_vllm_waiting": summary.get("max_waiting_requests"),
                "peak_local_pending": 0 if not gated else (meta.get("max_local_pending_depth") or 0),
                "background_e2e_p95_s": summary.get("background_total_e2e_p95_s") or summary.get("background_e2e_p95_s"),
                "background_output_goodput_tps": summary.get("background_output_token_goodput_tps"),
                "interactive_offered": meta.get("interactive_offered_count"),
                "interactive_completed": ix_done,
                "interactive_failed": ix_fail,
                "background_offered": meta.get("background_offered_count"),
                "background_admitted": summary.get("background_admitted_count")
                if gated
                else meta.get("background_offered_count"),
                "background_completed": summary.get("background_completed_count"),
                "background_failed": bg_fail,
                "valid_offered_load": summary.get("valid_offered_load"),
                "client_limited": bool(summary.get("interactive_client_limited") or summary.get("background_client_limited")),
                "aborted": bool(meta.get("aborted")),
            },
            "history": history,
        }

    def _refresh_live(
        self,
        side: SideView,
        records: list[dict[str, Any]],
        runtime_rows: list[dict[str, Any]],
        pending_holder: dict[str, Any],
        t0_box: dict[str, float | None],
    ) -> None:
        pre_end = float(self.config["pre_burst_end_s"])
        burst_end = float(self.config["burst_end_s"])
        duration = float(self.config["scenario_duration_s"])
        t0 = t0_box.get("t0")
        elapsed = None if t0 is None else self.clock() - t0
        ix_done, ix_fail = count_status(records, CLASS_INTERACTIVE)
        bg_done, bg_fail = count_status(records, CLASS_BACKGROUND)
        target_ix = int(round(float(self.config["interactive"]["offered_rps"]) * duration))
        target_bg = int(
            round(
                float(self.config["background"]["offered_rps"])
                * (burst_end - pre_end)
            )
        )
        current, maximum = waiting_stats(runtime_rows)
        pending = pending_holder.get("pending")
        pending_depth = len(pending) if pending is not None else 0
        p95, n = live_burst_p95(records)
        inflight = (ix_done + ix_fail < target_ix) or (bg_done + bg_fail < target_bg)
        arrivals_done = elapsed is not None and elapsed >= duration
        side.elapsed_s = elapsed
        side.phase = phase_for_elapsed(
            elapsed,
            pre_end_s=pre_end,
            burst_end_s=burst_end,
            duration_s=duration,
            inflight=int(inflight),
            arrivals_done=arrivals_done,
        )
        if side.phase == PHASE_WARMUP:
            side.status_label = "Warmup"
        elif side.state == SIDE_RUNNING:
            labels = {
                PHASE_BASELINE: "Baseline",
                PHASE_BURST: "Background burst",
                PHASE_RECOVERY: "Recovery",
                PHASE_DRAINING: "Draining",
                PHASE_COMPLETE: "Complete",
            }
            side.status_label = labels.get(side.phase, "Running")
        side.interactive_target = target_ix
        side.interactive_offered = max(target_ix, ix_done + ix_fail)
        side.interactive_completed = ix_done
        side.interactive_failed = ix_fail
        side.background_target = target_bg
        side.background_offered = max(target_bg, bg_done + bg_fail)
        side.background_completed = bg_done
        side.background_failed = bg_fail
        if side.gated:
            counts = pending_holder.get("counts") or {}
            side.background_admitted = int(counts.get("admitted") or bg_done + bg_fail)
            side.local_pending = pending_depth
            side.peak_local_pending = max(side.peak_local_pending, pending_depth)
        else:
            side.background_admitted = bg_done + bg_fail
            side.local_pending = 0
        side.vllm_waiting = current
        if maximum is not None:
            side.max_vllm_waiting = maximum if side.max_vllm_waiting is None else max(side.max_vllm_waiting, maximum)
        side.burst_p95_ttft_s = p95
        side.burst_p95_sample_count = n
        if runtime_rows:
            last = runtime_rows[-1]
            side.history.append(
                {
                    "t_s": last.get("t_s"),
                    "vllm_waiting": last.get("num_requests_waiting"),
                    "servescope_pending": pending_depth if side.gated else 0,
                }
            )

    def _apply_final_metrics(self, side: SideView, result: dict[str, Any]) -> None:
        final = result.get("final") or {}
        side.burst_p95_ttft_s = final.get("burst_p95_ttft_s")
        side.max_vllm_waiting = final.get("max_vllm_waiting")
        side.peak_local_pending = final.get("peak_local_pending") or 0
        side.background_e2e_p95_s = final.get("background_e2e_p95_s")
        side.background_output_goodput_tps = final.get("background_output_goodput_tps")
        side.interactive_target = int(final.get("interactive_offered") or side.interactive_target)
        side.interactive_offered = int(final.get("interactive_offered") or side.interactive_offered)
        side.interactive_completed = int(final.get("interactive_completed") or 0)
        side.interactive_failed = int(final.get("interactive_failed") or 0)
        side.background_target = int(final.get("background_offered") or side.background_target)
        side.background_offered = int(final.get("background_offered") or side.background_offered)
        side.background_admitted = int(final.get("background_admitted") or side.background_admitted)
        side.background_completed = int(final.get("background_completed") or 0)
        side.background_failed = int(final.get("background_failed") or 0)
        if result.get("history"):
            side.history = list(result["history"])

    def _finish_pair(self, native_final: dict[str, Any], gated_final: dict[str, Any]) -> None:
        self.chat_locked = False
        native_ok = bool(native_final.get("valid"))
        gated_ok = bool(gated_final.get("valid"))
        if not native_ok or not gated_ok:
            parts = [part for part in (native_final.get("invalid_reason"), gated_final.get("invalid_reason")) if part]
            reason = ", ".join(parts) or "invalid"
            self.overall_state = OVERALL_INVALID
            self.comparison = {
                "available": False,
                "ttft_delta_pct": None,
                "headline": "Live run was not valid for comparison.",
                "reason": reason,
            }
            return
        native_p95 = (native_final.get("final") or {}).get("burst_p95_ttft_s")
        gated_p95 = (gated_final.get("final") or {}).get("burst_p95_ttft_s")
        delta = ttft_delta_pct(native_p95, gated_p95)
        headline = None
        if delta is not None:
            pct = abs(round(delta))
            word = "lower" if delta >= 0 else "higher"
            headline = f"{pct}% {word} p95 first-token delay in this live run"
        self.overall_state = OVERALL_COMPLETE
        self.comparison = {
            "available": True,
            "ttft_delta_pct": delta,
            "headline": headline,
            "reason": None,
        }
