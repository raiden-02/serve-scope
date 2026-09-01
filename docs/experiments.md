# ServeScope experiment notes

Workload definitions, clocks, validity rules, and reproduction steps. The short project page is [`README.md`](../README.md).

Artifact directories keep names like `p1` and `p4`. Those are file-tree identifiers, not a required reading order.

## Environment and runtime setup

- GPU: NVIDIA GeForce RTX 4080 SUPER, 16 GB
- Host: Windows 11 + WSL2 Ubuntu 24.04
- vLLM 0.28.0, PyTorch 2.13.0+cu130, Python 3.12.14
- Model: `Qwen/Qwen3-1.7B`, BF16
- `nvidia-smi` CUDA UMD 13.3 is driver compatibility. `torch.version.cuda` is 13.0. Those are different numbers.

WSLg `/Xwayland` shares the GPU with vLLM.

`--enforce-eager` and `VLLM_USE_FLASHINFER_SAMPLER=0` are WSL limitations, not a tuned serving mode. `--gpu-memory-utilization 0.85` is required because WSLg already holds about 1.2 GiB VRAM.

This machine needed a host C compiler for Triton. gcc 13.3 was extracted into a user-local prefix because `sudo apt` was not available. Do not install a Linux NVIDIA display driver inside WSL. Do not install the CUDA Toolkit just to drop `--enforce-eager`.

Environment capture: `artifacts/p0/environment.txt`.

First check that one streaming chat request works:

```bash
cd ~/serve-scope
export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH"
export HF_HOME="$HOME/.cache/huggingface"
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
uv pip install vllm --torch-backend=cu130
python scripts/p0_stream_smoke.py
```

Smoke artifact: `artifacts/p0/smoke.json`. That smoke treats "received any content" as success. Later measurements use a stricter completion rule.

## Terms

- **TTFT (time to first token):** `first_content_s - request_attempt_s`. Time from actually starting the HTTP request until the first nonempty generated text. Waiting for response headers is inside this number.
- **Offered load:** requests scheduled per second, whether earlier requests have finished. Not the same as concurrency.
- **Throughput:** completed work per second. Interactive saturation reports wall-clock completed requests/sec and output tokens/sec over the full repeat, including drain after the last scheduled arrival. That is not a kernel-only decode rate.
- **Queue:** requests waiting because the server is already busy. vLLM exposes this as `vllm:num_requests_waiting`.
- **p95:** 95% of measured values are at or below this number.
- **SLO used here:** p95 client TTFT under 1.0 second. A chosen target, not an industry standard.

## Interactive saturation

Artifact group: `p1`. Config: `configs/p1_short_chat.json`.

Question: at what offered request load does this server stop meeting p95 client TTFT under 1 s for a controlled short interactive workload?

One class, `interactive`. Qwen3 thinking is off (`chat_template_kwargs.enable_thinking=false`, `reasoning_effort=none`). Each request asks for exactly 64 output tokens (`min_tokens=64`, `max_completion_tokens=64`, `temperature=0`, streaming). Synthetic serving load, not a stand-in for every chat app.

Arrivals are open-loop: request *i* is scheduled at `t0 + i / offered_rps` on a monotonic clock. The client does not wait for completion before sending the next request.

There is no first clean tested SLO violation. `first_clean_slo_violation_rps` is null. A clean crossing needs three valid repeats at one offered rate, and every one of those must miss p95 client TTFT < 1 s. A mix of fail and pass is instability, not a cliff.

Corrected evidence: `artifacts/p1/validation-2026-08-31T11-26-17Z/`.

| Offered RPS | Valid / attempts | Valid p95 client TTFT | What happened |
|---|---|---|---|
| 120 | 3/3 | 56 ms, 60 ms, 70 ms. All pass | clean. No waiting queue. Attempt dispatch held at ~120 RPS |
| 128 | 3/3 | 26 s fail, then 70 ms pass, 70 ms pass | valid offered load, mixed SLO. Instability, not a clean crossing |
| 132 | 2/5 | 21 s and 84 s on the two valid repeats | both valid repeats collapsed. Three other attempts had real request-attempt dispatch lag > 50 ms, so they are invalid |
| 136 | 1/5 | 54 ms on the one valid repeat | four attempts were client-limited. The one valid repeat stayed healthy |

An early sweep accidentally treated response-header delay as load-generator dispatch lag. The timing point was moved to immediately before request submission. The affected suite is `artifacts/p1/validation-2026-08-31T02-15-05Z/` and is superseded for timing conclusions.

What the corrected clocks show:

- At 128 RPS the load generator can dispatch on schedule while vLLM grows a waiting queue. That collapse is a server/path result, not fake client lag.
- The same 128 RPS can also stay near 70 ms.
- At 132 and 136 RPS some repeats still have a real client-side ceiling: the event loop misses the schedule, or in-flight hits 4096 and the run aborts. Late response headers alone do not make a run client-limited.
- 136 RPS also produced one valid healthy repeat. That rate is not a proven server cliff.

This is one synthetic 64-token workload on one model and one machine. It is not maximum GPU capacity.

The original final suite (`artifacts/p1/final-2026-08-31T01-43-39Z/`) and the superseded validation stay on disk. Smoke and pilot directories are range-finding only.

## Mixed-workload interference

Artifact group: `p2`. Config: `configs/p2_mixed.json`.

Question: can background AI work damage interactive responsiveness even when the interactive workload is healthy by itself?

Interactive traffic stays at 64 RPS for the whole 60 s scenario. That rate was comfortably healthy in the interactive-only sweep. This experiment does not put interactive traffic on the 128 RPS instability region and then blame background work.

Background traffic is a second synthetic class: longer documents (about 924-933 server-reported prompt tokens) and exactly 256 output tokens. Thinking stays off. Arrivals are injected only from 15-30 s. Existing background requests are not cancelled when injection stops.

Control is the same 64 RPS interactive stream with zero background requests. Final evidence is 3 valid control repeats and 3 valid mixed repeats, measured together.

Interactive and background use separate HTTP clients and separate in-flight caps so one shared connection pool cannot fake interference.

**Background work damaged interactive responsiveness.** After a 16 RPS background burst started at t=15 s, a vLLM waiting queue appeared and interactive p95 TTFT moved from tens of milliseconds to a few seconds.

Evidence: `artifacts/p2/final-2026-08-31T13-09-14Z/`. Pilot that selected 16 RPS: `artifacts/p2/pilot-2026-08-31T13-03-48Z/`.

| Window | Control interactive p95 TTFT | Mixed interactive p95 TTFT |
|---|---|---|
| pre (0-15 s) | 46-48 ms | 49 ms. SLO met |
| burst (15-30 s) | 45-47 ms | 1.88 s, 2.41 s, 3.97 s. SLO missed |
| recovery (30-60 s) | 47-49 ms | 5.18-5.72 s. SLO still missed |

Median interference ratio (mixed burst p95 / control aligned-window p95) is about 54. Descriptive for this workload, not a universal slowdown factor.

Waiting queue went from 0 in control to 152-290 in mixed. All 240 background requests completed. Last background completion was about 40-44 s after scenario start, so work continued into the nominal recovery window. Background output-token goodput was about 2100-2500 tok/s.

4 RPS and 8 RPS background bursts in the pilot did not grow a sustained waiting queue. 16 RPS was the lowest tested rate that did, with both client schedules still valid.

The server stayed on default scheduling. No ServeScope policy ran here.

## Native priority baseline

Artifact group: `p3`. Config: `configs/p3_native_priority.json`.

Question: how much of the mixed-workload interference can vLLM's own native priority scheduler solve before ServeScope adds a policy?

Same mixed scenario as above. Interactive and background stay independent open-loop streams. The only added server flag is `--scheduling-policy priority`. Priority values travel in the OpenAI chat request body. The client does not reorder requests. Interactive uses `priority=0`. Background uses `priority=1`. Lower integer is handled earlier. FCFS benchmark requests did not send a priority field.

**Native priority substantially mitigated the interference.** It met the 1 s SLO. It did not restore the healthy tens-of-milliseconds burst TTFT, and background work paid for it.

This session's FCFS mixed burst median was 3.33 s, a bit slower than the earlier mixed-workload median of 2.41 s. The two policies are compared as measured together, not against the older mixed-workload numbers.

Evidence: FCFS `artifacts/p3/fcfs-2026-08-31T16-07-48Z/`, native priority `artifacts/p3/priority-2026-08-31T16-24-57Z/`, comparison `artifacts/p3/comparison-2026-08-31T16-29-56Z/`.

| Window | FCFS mixed interactive p95 TTFT | Native-priority mixed interactive p95 TTFT |
|---|---|---|
| pre (0-15 s) | 49-55 ms. SLO met | 47-51 ms. SLO met |
| burst (15-30 s) | 3.32-3.36 s. SLO missed | 827-838 ms. SLO met |
| 30-60 s backlog window | 5.36-5.62 s. SLO missed | 806-824 ms. SLO met |
| after last background completion | 5.31-5.54 s. SLO missed | 58-61 ms. SLO met |

Median burst p95 went from 3.33 s to 836 ms. Protection ratio (priority / FCFS) is 0.25. Absolute reduction is about 2.49 s.

One interactive-only control on the priority server stayed at 47 ms burst p95 with a waiting queue of 0.

Waiting queue: FCFS 235-242, priority 144-146. All 240 background requests completed in every repeat. Interactive stream errors: 3-6 under FCFS, 0 under priority.

Background cost: output-token goodput fell from about 2240 tok/s to 1583 tok/s. Request goodput fell from about 8.75 to 6.18 rps. Last background completion moved from about 42.4 s to 53.8 s. Background p95 E2E moved from about 12.8 s to 26.3 s.

This is native vLLM priority, not isolation. Burst p95 is still about 18 times the 47 ms healthy control, and a waiting queue remains. ServeScope is not a claim about this result.

## External background admission

Artifact group: `p4`. Config: `configs/p4_backpressure.json`.

Question: can ServeScope keep excess background work out of vLLM by admitting it only when the server is not already queued, while keeping native priority as the runtime baseline?

All 240 background jobs still arrive at ServeScope on the original 16/s, t=15-30 s schedule. The controller may defer them in a local pending queue. It does not drop, cancel, shorten, or rewrite arrivals. Interactive traffic goes straight to vLLM.

Policy: watch `vllm:num_requests_waiting` every 500 ms. If waiting > 0, halve the background concurrency limit (floor 1). After 4 consecutive zero-wait samples, add 1 (ceiling 256). Already admitted jobs are never cancelled. If the limit falls below current in-flight work, new admissions stop until in-flight drops.

**External admission moved the backlog off vLLM and cut this session's mixed burst p95 from about 297 ms to 103 ms.** Background jobs waited in a ServeScope queue instead, then still finished.

This session's native-priority baseline was healthier than the earlier native-priority suite (297 ms / wait 83 vs 836 ms / wait 144). The two modes here are compared as measured together. The FCFS vs native-priority evidence stays in the `p3` artifacts.

Evidence: native `artifacts/p4/native-2026-08-31T21-07-52Z/`, backpressure `artifacts/p4/backpressure-2026-08-31T21-11-34Z/`, comparison `artifacts/p4/comparison-2026-08-31T21-15-28Z/`. Pilot: `artifacts/p4/pilot-2026-08-31T21-05-55Z/`.

| Window | Native priority mixed p95 TTFT | Priority + ServeScope admission p95 TTFT |
|---|---|---|
| pre (0-15 s) | 47 ms. SLO met | 50-54 ms. SLO met |
| burst (15-30 s) | 297-312 ms. SLO met | 102-104 ms. SLO met |
| 30-60 s backlog window | 159-255 ms. SLO met | 104-113 ms. SLO met |
| after last background completion | 47-50 ms. SLO met | 50-52 ms. SLO met |

Median burst p95: 297 ms to 103 ms. Max vLLM waiting: 83 to 0. Max ServeScope pending: 137. The work did not disappear.

Background cost, from original offered arrival to completion: p95 E2E 19.1 s to 26.1 s. Output-token goodput 1869 tok/s to 1510 tok/s. Last completion 47.9 s to 55.7 s. Admission-delay p95 was about 18 s. All 240 jobs entered ServeScope on schedule and all 240 completed.

The limit started at 32, never dropped (`decrease_count = 0` because waiting stayed 0), and rose to 62. That is additive increase while the server stayed empty. The measured improvement came from bounded admission, not from multiplicative decrease reacting to congestion. The decrease path is unit-tested.

vLLM still owns priority scheduling, batching, KV cache, and kernels.

## Measurement definitions

Each request keeps five clocks:

- `scheduled_arrival_s`: open-loop target `t0 + i / offered_rps`
- `request_attempt_s`: immediately before `client.stream(...)`. Real dispatch / attempt time
- `response_headers_s`: immediately after entering the streaming-response context. Headers are already available. Diagnostic only. Not dispatch
- `first_content_s`: first nonempty generated `delta.content`
- `completion_s`: actual terminal stream completion

Derived:

- `dispatch_lag_s = request_attempt_s - scheduled_arrival_s`
- `response_headers_latency_s = response_headers_s - request_attempt_s` when headers exist
- `client_ttft_s = first_content_s - request_attempt_s`
- `client_e2e_s = completion_s - request_attempt_s`

`actual_dispatch_rps` in `repeat_summary.csv` is computed from `request_attempt_s`, not from header timestamps.

- **Client TTFT:** first nonempty `delta.content` minus `request_attempt_s`. Role-only, empty, and reasoning-only chunks do not count. Header wait is part of this number.
- **Response-headers latency:** diagnostic. It is part of client-observed TTFT. Late headers are server/network/setup time, not load-generator dispatch lag.
- **Client E2E:** stream completion minus `request_attempt_s`.
- **Backend queue time / backend scheduled-to-first-token:** copied from vLLM `metrics` on the usage chunk when `--enable-per-request-metrics` is on. Keep these separate from client TTFT and from header latency.
- **Completed request:** status `success` or `length` with an explicit terminal finish reason (`length`, `stop`, or `eos_token`). HTTP 200 plus some content plus a dropped connection is a stream error, not success.
- **Percentiles:** NumPy `percentile(..., method="linear")` (R type 7). Every aggregate keeps its sample count.
- **Valid offered-load repeat:** not aborted, and request-attempt dispatch held (actual attempt rate at least 90% of offered, median attempt dispatch lag at most 50 ms), and no `PoolTimeout` / `client_capacity` event. Late response headers do not invalidate a repeat.
- **Client-limited run:** attempt dispatch rate < 90% of offered, or median attempt dispatch lag > 50 ms, or the in-flight safety cap aborted the run, or httpx hit `PoolTimeout`. Those runs stay in diagnostics. They do not establish a clean threshold.
- **Clean SLO violation:** at least three valid repeats, and every valid repeat has p95 client TTFT >= 1 s (`slo_violated_all_valid_repeats`). fail + fail + pass is `valid_repeats_mixed_slo`, not a crossing.
- **Wall-clock throughput:** completed requests or output tokens divided by the full repeat wall-clock, including drain after the last scheduled arrival.
- **Mixed-workload phase:** `pre_burst` is [0, 15) s, `burst_injection` is [15, 30) s, `recovery` is [30, 60) s, from the request's attempt time relative to scenario start. Completion time does not move a request into a later phase.
- **Interference ratio:** mixed burst-phase interactive p95 TTFT divided by the control repeat's aligned 15-30 s p95. Descriptive only.
- **Admission ingress lag:** `ingress_enqueue_s - offered_arrival_s`. Generator schedule into the ServeScope pending queue. Not admission delay.
- **Admission delay:** `request_attempt_s - ingress_enqueue_s`. Intentional ServeScope defer. It does not fail a repeat.
- **Background total E2E:** `completion_s - offered_arrival_s`. Includes local defer. Background latency numbers in the README use this, not server-attempt E2E.

## Reproduction

Work inside WSL2 Ubuntu. Keep the Hugging Face cache on the Linux filesystem, not `/mnt/c`.

### Server

```bash
cd ~/serve-scope
export PATH="$HOME/.local/gcc/usr/bin:$HOME/.local/bin:/usr/lib/wsl/lib:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/gcc/usr/lib/x86_64-linux-gnu:/usr/lib/wsl/lib"
export LIBRARY_PATH="$HOME/.local/gcc/usr/lib/x86_64-linux-gnu"
export CPATH="$HOME/.local/gcc/usr/include/x86_64-linux-gnu:$HOME/.local/gcc/usr/include"
export C_INCLUDE_PATH="$CPATH"
export CC="$HOME/.local/gcc/usr/bin/gcc"
export HF_HOME="$HOME/.cache/huggingface"
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve Qwen/Qwen3-1.7B --gpu-memory-utilization 0.85 --enforce-eager --enable-per-request-metrics
```

`--enable-per-request-metrics` lets streaming usage chunks carry vLLM queue/TTFT timings.

Priority server, same flags plus `--scheduling-policy priority`. The live demo starts that server with `scripts/start_server.sh`. Historical benchmark launchers remain at `scripts/_p3_start_fcfs.sh` and `scripts/_p3_start_priority.sh`.

### Harness

```bash
cd ~/serve-scope
source .venv/bin/activate
python -m pytest tests/test_p1_metrics.py
python scripts/p1_sweep.py --mode smoke
python scripts/p1_sweep.py --mode pilot
python scripts/p1_sweep.py --mode final --rates 8,64,128,132,136,144
python scripts/p1_sweep.py --mode validation --rates 120,128,132,136
python scripts/p2_mixed.py --mode pilot
python scripts/p2_mixed.py --mode final --pilot-dir artifacts/p2/pilot-<UTC>/
python scripts/p3_native_priority.py --mode smoke
python scripts/p3_native_priority.py --mode fcfs
# stop FCFS, start the priority server, then:
python scripts/p3_native_priority.py --mode priority
python scripts/p3_native_priority.py --mode compare --fcfs-dir artifacts/p3/fcfs-<UTC>/ --priority-dir artifacts/p3/priority-<UTC>/
python scripts/p4_backpressure.py --mode pilot
python scripts/p4_backpressure.py --mode native
python scripts/p4_backpressure.py --mode backpressure
python scripts/p4_backpressure.py --mode compare --native-dir artifacts/p4/native-<UTC>/ --backpressure-dir artifacts/p4/backpressure-<UTC>/
```

Configs: `configs/p1_short_chat.json`, `configs/p2_mixed.json`, `configs/p3_native_priority.json`, `configs/p4_backpressure.json`.

Each suite writes `artifacts/<group>/<mode>-<UTC>/` with `run_manifest.json`, `config.json`, `requests.jsonl`, summaries, and plots. `requests.jsonl` is the source of truth for request metrics.

Mixed-workload suites add `phase_summary.csv` and timeline plots. Native-priority comparisons add `fcfs_vs_priority_ttft.png`, `fcfs_vs_priority_queue.png`, and `tradeoff.png`. Every request row there includes `scheduler_policy` and `request_priority`. Manifests store SHA-256 `source_hashes`.

Admission suites add `controller_metrics.csv` and clocks `offered_arrival_s`, `ingress_enqueue_s`, `admission_delay_s`, `background_total_e2e_s`. Comparison plots are `interactive_protection.png`, `queue_movement.png`, and `controller_window.png`.

## Scope

ServeScope measures the runtime and, for background work only, gates admission. It does not patch vLLM, replace the runtime scheduler, drop or cancel jobs, or deploy to the cloud. The live page is local FastAPI and plain HTML.
