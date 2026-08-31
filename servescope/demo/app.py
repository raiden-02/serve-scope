"""ServeScope local demo: browser → FastAPI → vLLM."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from servescope.backpressure import AimdController
from servescope.client import build_chat_payload
from servescope.demo.evidence import load_evidence
from servescope.demo.state import (
    MODE_BACKPRESSURE,
    BurstBusy,
    DemoState,
    ModeLocked,
    connected_runtime,
    disconnected_runtime,
)
from servescope.metrics import (
    METRIC_RUNNING,
    METRIC_WAITING,
    is_nonempty_generated_content,
    parse_json_event,
    parse_prometheus_gauges,
    sse_data_payloads,
)
from servescope.workload import select_background_prompt

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
CONFIG_PATH = ROOT / "configs" / "demo.json"


class ChatIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)


class ModeIn(BaseModel):
    mode: str


def load_demo_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def nvidia_smi_snapshot() -> dict[str, Any] | None:
    try:
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        name, util, used, total = [part.strip() for part in text.split(",")]
        return {
            "name": name,
            "utilization_gpu_pct": float(util),
            "memory_used_mib": float(used),
            "memory_total_mib": float(total),
        }
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


class DemoApp:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.state = DemoState()
        ctl = config["controller"]
        self.controller = AimdController(
            initial=int(ctl["initial_background_limit"]),
            minimum=int(ctl["minimum_background_limit"]),
            maximum=int(ctl["maximum_background_limit"]),
            increase_after_zero_samples=int(ctl["increase_after_zero_samples"]),
        )
        self.state.controller_limit = self.controller.limit
        self.lock = asyncio.Lock()
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(float(config["request_timeout_s"])))
        self.runtime = disconnected_runtime()
        self.started_at = time.monotonic()
        self.burst_task: asyncio.Task | None = None

    @property
    def chat_url(self) -> str:
        return self.config["base_url"].rstrip("/") + self.config["chat_path"]

    @property
    def metrics_url(self) -> str:
        return self.config["base_url"].rstrip("/") + self.config["metrics_path"]

    @property
    def burst_jobs(self) -> int:
        return int(round(float(self.config["burst_rps"]) * float(self.config["burst_duration_s"])))

    async def probe_vllm(self) -> bool:
        try:
            response = await self.client.get(self.config["base_url"].rstrip("/") + "/health", timeout=2.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def refresh_runtime(self) -> None:
        connected = await self.probe_vllm()
        if not connected:
            self.runtime = disconnected_runtime()
            return
        gauges: dict[str, float] = {}
        try:
            response = await self.client.get(self.metrics_url, timeout=2.0)
            response.raise_for_status()
            gauges = parse_prometheus_gauges(response.text, {METRIC_RUNNING, METRIC_WAITING})
        except httpx.HTTPError:
            gauges = {}
        gpu = await asyncio.to_thread(nvidia_smi_snapshot)
        running = gauges.get(METRIC_RUNNING)
        waiting = gauges.get(METRIC_WAITING)
        self.runtime = connected_runtime(
            model=self.config["model"],
            gpu=gpu,
            running=running,
            waiting=waiting,
        )
        async with self.lock:
            if self.state.mode == MODE_BACKPRESSURE:
                action = self.controller.observe(waiting)
                self.state.controller_limit = self.controller.limit
                self.state.controller_action = action
            self.state.record_sample(time.monotonic() - self.started_at, waiting)

    async def telemetry_loop(self) -> None:
        while True:
            await self.refresh_runtime()
            await asyncio.sleep(float(self.config["telemetry_interval_s"]))

    def live_payload(self) -> dict[str, Any]:
        burst = {
            "rps": self.config["burst_rps"],
            "duration_s": self.config["burst_duration_s"],
            "jobs": self.burst_jobs,
        }
        return {
            **self.state.snapshot(),
            **self.runtime,
            "burst_preset": burst,
            "history": list(self.state.history),
        }

    async def stream_completion(self, prompt: str, *, priority: int, min_tokens: int, max_tokens: int):
        payload = build_chat_payload(
            model=self.config["model"],
            prompt=prompt,
            temperature=0,
            min_tokens=min_tokens,
            max_completion_tokens=max_tokens,
            priority=priority,
        )
        attempt = time.perf_counter()
        first_content = None
        async with self.client.stream("POST", self.chat_url, json=payload) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise HTTPException(status_code=502, detail=body[:300] or "vLLM rejected the request")
            async for line in response.aiter_lines():
                for text in sse_data_payloads(line):
                    if text == "[DONE]":
                        return
                    event = parse_json_event(text)
                    if event is None:
                        continue
                    choices = event.get("choices") or [{}]
                    delta = choices[0].get("delta") if choices else None
                    if not is_nonempty_generated_content(delta):
                        continue
                    if first_content is None:
                        first_content = time.perf_counter()
                        yield {
                            "type": "first",
                            "backend_ttft_ms": (first_content - attempt) * 1000.0,
                        }
                    yield {"type": "token", "text": delta.get("content")}

    async def run_background_job(self, index: int) -> None:
        _prompt_id, prompt = select_background_prompt(int(self.config["seed"]), index)
        ok = False
        try:
            async for _event in self.stream_completion(
                prompt,
                priority=1,
                min_tokens=int(self.config["background_min_tokens"]),
                max_tokens=int(self.config["background_max_tokens"]),
            ):
                pass
            ok = True
        except Exception:
            ok = False
        async with self.lock:
            self.state.finish_one(ok=ok)

    async def run_burst(self) -> None:
        rps = float(self.config["burst_rps"])
        n = self.burst_jobs
        t0 = time.perf_counter()
        for i in range(n):
            target = t0 + i / rps
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            async with self.lock:
                started = self.state.offer_one()
                native = self.state.mode != MODE_BACKPRESSURE
            if started and native:
                asyncio.create_task(self.run_background_job(i))
        async with self.lock:
            self.state.mark_inject_finished()
        while True:
            async with self.lock:
                if self.state.mode == MODE_BACKPRESSURE:
                    while self.state.can_admit():
                        if not self.state.admit_one():
                            break
                        index = self.state.background_admitted - 1
                        asyncio.create_task(self.run_background_job(index))
                done = self.state.burst_state not in ("injecting", "draining")
            if done:
                break
            await asyncio.sleep(0.02)

    async def close(self) -> None:
        await self.client.aclose()


def create_app(config: dict[str, Any] | None = None) -> FastAPI:
    config = config or load_demo_config()
    demo = DemoApp(config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await demo.refresh_runtime()
        task = asyncio.create_task(demo.telemetry_loop())
        try:
            yield
        finally:
            task.cancel()
            await demo.close()

    app = FastAPI(title="ServeScope demo", lifespan=lifespan)

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return demo.live_payload()

    @app.get("/api/live")
    async def live() -> dict[str, Any]:
        return demo.live_payload()

    @app.get("/api/evidence")
    async def evidence() -> dict[str, Any]:
        return load_evidence(ROOT)

    @app.post("/api/mode")
    async def set_mode(body: ModeIn) -> dict[str, Any]:
        async with demo.lock:
            try:
                demo.state.set_mode(body.mode)
            except ModeLocked as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return demo.live_payload()

    @app.post("/api/burst")
    async def burst() -> dict[str, Any]:
        async with demo.lock:
            try:
                demo.state.start_burst(demo.burst_jobs)
            except BurstBusy as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        demo.burst_task = asyncio.create_task(demo.run_burst())
        return demo.live_payload()

    @app.post("/api/chat")
    async def chat(body: ChatIn) -> StreamingResponse:
        if demo.runtime.get("server") != "connected":
            raise HTTPException(status_code=503, detail="vLLM is disconnected")

        async def events():
            try:
                async for item in demo.stream_completion(
                    body.prompt,
                    priority=0,
                    min_tokens=1,
                    max_tokens=int(demo.config["chat_max_tokens"]),
                ):
                    yield f"data: {json.dumps(item)}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            except HTTPException as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': exc.detail})}\n\n"
            except httpx.HTTPError as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")
    app.state.demo = demo
    return app


app = create_app()
