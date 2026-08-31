# ServeScope

ServeScope is a local LLM inference lab for seeing what happens when interactive users and background AI jobs share one GPU. It measures first-token latency and runtime queues, compares vLLM's native priority scheduling, and adds an external backpressure gate for background work.

It runs a real OpenAI-compatible vLLM server on this machine's RTX 4080 SUPER. It is a measurement lab, not a production serving product and not a new inference runtime.

## Problem

One GPU can serve a person waiting on a reply and longer background jobs at the same time. If the background jobs flood the server, the person stares at a blank reply for seconds.

## What I built

- A streaming measurement harness with explicit request clocks
- Mixed interactive + background workloads on real vLLM 0.28.0
- A fair test of vLLM's own `--scheduling-policy priority`
- An external AIMD-style background admission gate (not a vLLM scheduler)
- A local live lab at `http://127.0.0.1:8080`

## Headline results

Same-session comparisons only. Do not read these as one three-step experiment.

**P3, one session:** default FCFS mixed burst p95 **3.33 s** → native priority **836 ms**.

**P4, a later session:** native priority mixed burst p95 **297 ms** → ServeScope backpressure **103 ms**. Runtime waiting **83 → 0**. Background jobs sat in ServeScope's local queue (peak **137**). Background p95 total E2E **19.1 s → 26.1 s**. Output goodput **1869 → 1510 tok/s**. All 240 background jobs still finished.

The P4 run used bounded admission. The AIMD decrease path exists and is unit-tested. It was not exercised in that headline workload (`decrease_count = 0`).

P3 native (836 ms) and P4 native (297 ms) are different sessions. They are not one controlled pair.

## How to run the demo

Terminal 1, WSL2, from `~/serve-scope`:

```bash
scripts/_p3_start_priority.sh
```

Terminal 2:

```bash
source .venv/bin/activate
python scripts/run_demo.py
```

Open `http://127.0.0.1:8080`.

The page still renders if vLLM is down. It shows Disconnected / Unavailable. It does not invent telemetry.

Live modes (same already-running priority server):

- **Native priority:** background jobs go straight to vLLM at priority 1
- **ServeScope backpressure:** the same server, but background jobs wait in a local admission queue

Default FCFS is shown only as measured P3 evidence. Changing a UI toggle cannot change `--scheduling-policy` on a running server.

The burst button is a short demo (`8 jobs/s × 5 s = 40 jobs`). It is not a P4 benchmark replay.

## Screenshot

See `docs/demo.png` if present. That file is a capture of the running app, not a mockup.

## Experiment history

Clock definitions, validity rules, P0-P4 methods, and artifact paths live in [`docs/experiments.md`](docs/experiments.md).

Accepted comparison files:

- P3: `artifacts/p3/comparison-2026-08-31T16-29-56Z/result.json`
- P4: `artifacts/p4/comparison-2026-08-31T21-15-28Z/result.json`
