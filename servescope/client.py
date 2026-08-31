"""Async OpenAI-compatible streaming client."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from servescope.metrics import (
    classify_status,
    derive_request_timings,
    extract_backend_metrics,
    extract_finish_reason,
    extract_usage,
    is_nonempty_generated_content,
    parse_json_event,
    sse_data_payloads,
)


def build_chat_payload(
    *,
    model: str,
    prompt: str,
    temperature: float,
    min_tokens: int,
    max_completion_tokens: int,
    priority: int | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "min_tokens": min_tokens,
        "max_completion_tokens": max_completion_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
        "include_reasoning": False,
    }
    if priority is not None:
        payload["priority"] = int(priority)
    return payload


async def stream_chat_request(
    client: httpx.AsyncClient,
    *,
    url: str,
    payload: dict[str, Any],
    request_id: str,
    prompt_id: str,
    workload_class: str,
    scheduled_s: float,
    timeout_s: float,
    clock,
) -> dict[str, Any]:
    request_attempt_s = clock()
    response_headers_s = None
    first_content_s = None
    complete_s = None
    http_status = None
    finish_reason = None
    usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    backend = extract_backend_metrics({})
    got_content = False
    stream_done = False
    timed_out = False
    cancelled = False
    stream_error = False
    client_capacity = False
    error_message = None
    response_id = None

    try:
        async with client.stream("POST", url, json=payload, timeout=timeout_s) as response:
            response_headers_s = clock()
            http_status = response.status_code
            if http_status >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                error_message = body[:500]
            else:
                async for line in response.aiter_lines():
                    now = clock()
                    for payload_text in sse_data_payloads(line):
                        if payload_text == "[DONE]":
                            stream_done = True
                            complete_s = now
                            continue
                        try:
                            event = parse_json_event(payload_text)
                        except ValueError as exc:
                            stream_error = True
                            error_message = f"JSONDecodeError: {exc}"
                            complete_s = now
                            break
                        if event is None:
                            continue
                        if response_id is None:
                            response_id = event.get("id")
                        reason = extract_finish_reason(event)
                        if reason:
                            finish_reason = reason
                        event_usage = extract_usage(event)
                        if any(value is not None for value in event_usage.values()):
                            usage = event_usage
                        event_metrics = extract_backend_metrics(event)
                        if any(value is not None for value in event_metrics.values()):
                            backend = event_metrics
                        choices = event.get("choices") or [{}]
                        delta = choices[0].get("delta") if choices else None
                        if is_nonempty_generated_content(delta) and first_content_s is None:
                            first_content_s = now
                            got_content = True
                if complete_s is None:
                    complete_s = clock()
                    if http_status == 200 and not stream_done and not stream_error:
                        stream_error = True
                        error_message = error_message or "premature EOF: stream ended without a terminal finish reason"
    except httpx.PoolTimeout as exc:
        client_capacity = True
        error_message = f"PoolTimeout: {exc}"
        complete_s = clock()
    except httpx.TimeoutException as exc:
        timed_out = True
        error_message = f"TimeoutException: {exc}"
        complete_s = clock()
    except httpx.HTTPError as exc:
        stream_error = True
        error_message = f"{type(exc).__name__}: {exc}"
        complete_s = clock()
    except asyncio.CancelledError:
        cancelled = True
        error_message = "CancelledError"
        complete_s = clock()
        raise
    except Exception as exc:  # noqa: BLE001 - record unexpected client errors
        error_message = f"{type(exc).__name__}: {exc}"
        complete_s = clock()

    status = classify_status(
        http_status=http_status,
        finish_reason=finish_reason,
        got_content=got_content,
        stream_done=stream_done,
        timed_out=timed_out,
        cancelled=cancelled,
        stream_error=stream_error,
        error_message=error_message,
        client_capacity=client_capacity,
    )
    timings = derive_request_timings(
        scheduled_arrival_s=scheduled_s,
        request_attempt_s=request_attempt_s,
        response_headers_s=response_headers_s,
        first_content_s=first_content_s,
        completion_s=complete_s,
    )
    return {
        "request_id": request_id,
        "backend_request_id": response_id,
        "prompt_id": prompt_id,
        "workload_class": workload_class,
        **timings,
        "http_status": http_status,
        "finish_reason": finish_reason,
        "status": status,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        **backend,
        "error": error_message,
    }
