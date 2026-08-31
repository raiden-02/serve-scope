# ServeScope

ServeScope is a local LLM inference lab for seeing when interactive responsiveness breaks down as a GPU-backed server is given more work.

It runs a real OpenAI-compatible vLLM server on this machine's RTX 4080 SUPER. P1 measures offered request load against a chosen first-token target. It is a measurement lab, not a production serving product and not a new inference runtime.

## What the words mean

- **TTFT (Time To First Token / first content):** time from sending the request until the first nonempty generated text arrives. In plain English: how long the user stares at a blank reply.
- **Offered load:** how many requests per second we *schedule* to the server, whether earlier requests have finished or not. This is not the same as concurrency.
- **Throughput:** completed work per second. P1 reports completed requests/sec and output tokens/sec.
- **Queue:** requests waiting because the server is already busy. vLLM exposes this as `vllm:num_requests_waiting`.
- **p95:** 95% of measured values are at or below this number. The slowest 5% are worse.
- **SLO:** a chosen target, not a universal industry standard. P1 uses **p95 client TTFT < 1.0 second**.

## P1 result

On this machine, with this model and this synthetic 64-token interactive workload:

**The first tested offered load that fails the chosen p95 TTFT < 1 s target is 132 requests/sec.**

Headline p95 is the **median of three independent repeat p95s**. That is what "first tested rate" refers to.

| Offered RPS | Median repeat p95 TTFT | Repeat p95 range | Waiting queue appeared | Notes |
|---|---|---|---|---|
| 8 | 47 ms | 47 to 48 ms | no | low-load baseline |
| 64 | 51 ms | 51 to 56 ms | no | still comfortable |
| 128 | 93 ms | 80 ms to 12.3 s | yes, on the collapsed repeat | approaching the knee. One of three repeats collapsed |
| 132 | 19.7 s | 14.7 to 21.8 s | yes | first rate where every repeat misses the SLO |
| 136 | 80 ms | 79 ms to 19.3 s | yes, on the collapsed repeat | median still passes. One repeat collapsed |
| 144 | 26.8 s | 22.9 to 75.2 s | yes | overloaded. All three repeats were client-limited or aborted |

The useful story is not a single clean cliff at one integer. From 8 through 64 RPS the server stays near 50 ms p95 TTFT and throughput scales up. Near 128 to 136 RPS the same offered load can either stay responsive or fall into a multi-second waiting queue. 132 RPS is the first tested point where that collapse happened on every repeat. 144 RPS is worse, and the client could not always sustain the intended arrival rate.

Peak healthy output-token throughput in the final suite was about 8000 to 8500 tokens/sec (128 to 136 RPS on the healthy repeats). When a run queued, completed-request throughput dropped into the 80 to 100 req/s range.

Raw evidence: `artifacts/p1/final-2026-08-31T01-43-39Z/`.

Pilot and smoke directories under `artifacts/p1/` are disposable range-finding. Do not mix them into the headline table.

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
- **Completed request:** status `success` or `length`. A 64-token run ending with `finish_reason=length` is expected. It is not recorded as a generic `success`.
- **Percentiles:** NumPy `percentile(..., method="linear")` (R type 7). Every aggregate keeps its sample count.
- **Client-limited run:** actual dispatch rate < 90% of offered, or median dispatch lag > 50 ms, or the in-flight safety cap aborted the run. Those runs are marked invalid as GPU-saturation evidence.

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
