# P3 native vLLM 0.28.0 verification

Installed package: vLLM 0.28.0 at
`/home/mayur/serve-scope/.venv/lib/python3.12/site-packages/vllm`.
No patch. No other vLLM version. No custom scheduler class.

## CLI

`vllm serve --help` lists:

```text
--scheduling-policy {fcfs,priority}
```

Default is `fcfs`. The priority server added only this flag.

FCFS process log `non-default args`:

```text
enable_per_request_metrics, enforce_eager, gpu_memory_utilization=0.85
```

Priority process log `non-default args`:

```text
enable_per_request_metrics, enforce_eager, gpu_memory_utilization=0.85,
scheduling_policy=priority
```

## Request mechanism

Installed OpenAI-compatible API accepts a chat-body field `priority`.
Header `X-Vllm-Priority` also exists and would override the body.
P3 uses the request-body field only. No proxy. No client-side queue.

## Ordering (from installed source)

`vllm/v1/request.py` `Request.__lt__`:

1. lower `priority` integer is earlier
2. then earlier `arrival_time`
3. then smaller `request_id`

P3 uses interactive `0` and background `1`.

## Preemption (from installed source)

`vllm/v1/core/sched/scheduler.py` around the KV-allocation loop:

- FCFS: if a request cannot get KV slots, the last running request is
  preempted (`self.running.pop()`).
- Priority: the victim is `max(running, key=(priority, arrival_time))`,
  so the lowest-priority (and then latest-arriving) running request is
  preempted to make room.

This is waiting-queue reordering plus possible preemption under KV
pressure. It is not an SLO, QoS, or isolation guarantee.

## Smoke

Invalid stale-server smoke (timeouts, discarded for headlines):
`artifacts/p3/smoke-2026-08-31T14-40-03Z/`.

Valid smoke on a clean FCFS server:
`artifacts/p3/smoke-2026-08-31T16-07-27Z/`.

- no priority field: HTTP 200, `status=length`
- `priority: 1` on FCFS: also HTTP 200, `status=length`

The protocol text says a nonzero priority should be rejected on FCFS.
The installed 0.28.0 server accepted it. FCFS benchmark requests still
omit the field and record `request_priority=0`.

Tiny accepted check on the priority server (interactive 0, background 1):
both HTTP 200, `finish=length`.

## Suites

- FCFS mixed: `artifacts/p3/fcfs-2026-08-31T16-07-48Z/`
- Priority mixed + one control: `artifacts/p3/priority-2026-08-31T16-24-57Z/`
- Comparison: this directory

Conclusion from measured medians: native priority substantially solves
the P2 seconds-scale interactive collapse, at a real background cost.
Burst p95 still sits near 836 ms versus a 47 ms healthy control.
