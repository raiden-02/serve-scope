# ServeScope

ServeScope is a local LLM inference lab for seeing when interactive responsiveness breaks down as a GPU-backed server is given more work.

It runs a real OpenAI-compatible vLLM server on this machine's RTX 4080 SUPER. P1 measures offered request load against a chosen first-token target. It is a measurement lab, not a production serving product and not a new inference runtime.

## What the words mean

- **TTFT (Time To First Token / first content):** time from sending the request until the first nonempty generated text arrives. In plain English: how long the user stares at a blank reply.
- **Offered load:** how many requests per second we *schedule* to the server, whether earlier requests have finished or not. This is not the same as concurrency.
- **Throughput:** completed work per second. P1 reports **wall-clock** completed requests/sec and output tokens/sec: completed work divided by the full repeat duration, including the tail after the last scheduled arrival. That is not a kernel-only decode-rate measurement.
- **Queue:** requests waiting because the server is already busy. vLLM exposes this as `vllm:num_requests_waiting`.
- **p95:** 95% of measured values are at or below this number. The slowest 5% are worse.
- **SLO:** a chosen target, not a universal industry standard. P1 uses **p95 client TTFT < 1.0 second**.

## P1 result

On this machine, with this model and this synthetic 64-token interactive workload:

**No single deterministic offered-load cliff was established.** The server is stable at 120 requests/sec. Around 128 to 136 RPS it enters an unstable saturation region: the same offered load can stay near 70 to 80 ms p95 TTFT, or grow a vLLM waiting queue and take many seconds to start a reply. When a run collapses, the load generator often cannot keep the intended arrival rate either.

A clean crossing needs three valid repeats at that rate. Invalid means aborted, client-limited, or dispatch timing that missed the offered schedule. Those repeats stay in the raw files. They do not get to vote on a headline threshold.

| Offered RPS | Valid repeats | Median valid p95 TTFT | What happened |
|---|---|---|---|
| 8 and 64 | 3/3 in the original final suite | 47 to 51 ms | healthy baseline from `artifacts/p1/final-2026-08-31T01-43-39Z/` |
| 120 | 3/3 | 70 ms (68 to 71 ms) | clean, no waiting queue |
| 128 | 2/5 | 76 ms on the two valid repeats | two healthy repeats, three collapsed and invalid. Not a clean crossing |
| 132 | 0/5 | n/a | every attempt was client-limited. Waiting queue and multi-second TTFT appeared, but the generator was no longer a clean offered-load source |
| 136 | 0/5 | n/a | same as 132, including one abort at the 4096 in-flight cap |

`first_clean_slo_violation_rps` is **null**. 132 RPS is not the P1 answer.

Corrected evidence: `artifacts/p1/validation-2026-08-31T02-15-05Z/`.

The earlier final suite is still on disk. Do not use it for the headline threshold. Smoke and pilot directories are range-finding only.

## What P1 is

P1 answers one question:

> At what offered request load does this server stop meeting p95 client TTFT < 1 s for a controlled short interactive workload?

It uses one workload class, `interactive`. Qwen3 thinking is turned off in the request (`chat_template_kwargs.enable_thinking=false`, `reasoning_effort=none`). Each request asks for exactly 64 output tokens (`min_tokens=64`, `max_completion_tokens=64`, `temperature=0`, streaming). That is a synthetic serving workload. It is not a stand-in for every chat app.

Arrivals are open-loop: request *i* is scheduled at `t0 + i / offered_rps` on a monotonic clock. The client does not wait for completion before sending the next request.

P1 does not include mixed interactive/background traffic, priority scheduling, admission control, a dashboard, or a custom scheduler.

## Reproduce P1

Work inside WSL2 Ubuntu. Keep the Hugging Face cache on the Linux filesystem, not `/mnt/c`.

### Server

Same performance-affecting flags as P0, plus `--enable-per-request-metrics` so streaming usage chunks can carry vLLM queue/TTFT timings.

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

`--enforce-eager` and the FlashInfer-off env var are WSL limitations, not a tuned serving mode. `--gpu-memory-utilization 0.85` is required because WSLg already holds about 1.2 GiB VRAM.

### Harness

```bash
cd ~/serve-scope
source .venv/bin/activate
python -m pytest tests/test_p1_metrics.py
python scripts/p1_sweep.py --mode smoke
python scripts/p1_sweep.py --mode pilot
python scripts/p1_sweep.py --mode final --rates 8,64,128,132,136,144
python scripts/p1_sweep.py --mode validation --rates 120,128,132,136
```

Config: `configs/p1_short_chat.json`.

Each suite writes an immutable directory:

```text
artifacts/p1/<mode>-<UTC>/
  run_manifest.json
  config.json
  requests.jsonl
  repeat_summary.csv
  aggregate_summary.csv
  runtime_metrics.csv
  gpu_metrics.csv
  ttft_vs_load.png
  throughput_vs_load.png
```

`requests.jsonl` is the source of truth for request metrics.

## Measurement definitions

- **Client TTFT:** first nonempty `delta.content` timestamp minus client dispatch. Role-only, empty, and reasoning-only chunks do not count.
- **Client E2E:** stream completion minus client dispatch.
- **Backend queue time / backend scheduled-to-first-token:** copied from vLLM `metrics` on the usage chunk when `--enable-per-request-metrics` is on. These are not client TTFT.
- **Completed request:** status `success` or `length` with an explicit terminal finish reason (`length`, `stop`, or `eos_token`). HTTP 200 plus some content plus a dropped connection is a stream error, not success.
- **Percentiles:** NumPy `percentile(..., method="linear")` (R type 7). Every aggregate keeps its sample count.
- **Valid offered-load repeat:** not aborted, not client-limited, and dispatch timing held (actual dispatch rate at least 90% of offered, median dispatch lag at most 50 ms). A clean crossing needs three such repeats.
- **Client-limited run:** actual dispatch rate < 90% of offered, or median dispatch lag > 50 ms, or the in-flight safety cap aborted the run, or httpx hit `PoolTimeout`. Those runs stay in diagnostics. They do not establish a clean threshold.
- **Wall-clock throughput:** completed requests or output tokens divided by the full repeat wall-clock, including drain after the last scheduled arrival.

## Environment

- GPU: NVIDIA GeForce RTX 4080 SUPER, 16 GB
- Host: Windows 11 + WSL2 Ubuntu 24.04
- vLLM 0.28.0, PyTorch 2.13.0+cu130, Python 3.12.14
- Model: `Qwen/Qwen3-1.7B`, BF16
- `nvidia-smi` CUDA UMD 13.3 is driver compatibility. `torch.version.cuda` is 13.0. Those are not the same number.

WSLg `/Xwayland` shares the GPU with vLLM. That is recorded, not hidden.

P0 environment capture: `artifacts/p0/environment.txt`.

## Reproduce P0

P0 only proves that one streaming chat request works. Its `success = received any content` rule is not used by P1.

```bash
cd ~/serve-scope
export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH"
export HF_HOME="$HOME/.cache/huggingface"
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
uv pip install vllm --torch-backend=cu130
```

Start the P0 server (no per-request metrics flag) with the same env block as above, then:

```bash
python scripts/p0_stream_smoke.py
```

P0 smoke artifact: `artifacts/p0/smoke.json`.

This machine needed a host C compiler for Triton. gcc 13.3 was extracted into `/home/mayur/.local/gcc` because `sudo apt` was not available. Do not install a Linux NVIDIA display driver inside WSL. Do not install the CUDA Toolkit just to drop `--enforce-eager`.

## What is not here

P1 does not implement mixed interactive/background traffic, request priority, admission control, ServeScope scheduling, a frontend, Prometheus/Grafana, or a second inference backend. Those are later checkpoints if the measurements justify them.
