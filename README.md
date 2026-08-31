# ServeScope

ServeScope will investigate how a real local GPU-backed LLM server behaves when interactive requests share the GPU with other work.

This repository is at **P0 only**: it proves that a real model can stream a real response from a real local OpenAI-compatible backend on this machine's RTX 4080 SUPER. It is an environment spike, not a lab product.

## What P0 uses

- Backend: vLLM 0.28.0, OpenAI-compatible HTTP server
- Model: `Qwen/Qwen3-1.7B` at default precision (`torch.bfloat16`)
- Environment: Windows 11, WSL2 Ubuntu 24.04, isolated Python 3.12.14, native Linux vLLM
- GPU: NVIDIA GeForce RTX 4080 SUPER, Windows NVIDIA driver 610.88, CUDA exposed into WSL

`nvidia-smi` reports CUDA UMD 13.3. That is driver compatibility. The installed PyTorch build is `2.13.0+cu130`, so `torch.version.cuda` is `13.0`. Do not treat those as the same number.

P0 timing numbers are smoke-test evidence that the stack works. They are not benchmark results and they are not performance claims.

## Reproduce P0

Work inside WSL2 Ubuntu. Keep the Hugging Face cache on the Linux filesystem, not `/mnt/c`.

This machine needed a host C compiler for Triton's small runtime compile. `sudo` was not available, so gcc 13.3 was extracted into `/home/mayur/.local/gcc`. If you can install packages, `sudo apt install -y build-essential` is the simpler path. Do not install a Linux NVIDIA display driver inside WSL. Do not install the full CUDA Toolkit just to get `nvcc`.

```bash
cd ~/serve-scope
export PATH="$HOME/.local/bin:/usr/lib/wsl/lib:$PATH"
export HF_HOME="$HOME/.cache/huggingface"

# If nvidia-smi is not on PATH, --torch-backend=auto may install CPU torch.
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
uv pip install vllm --torch-backend=cu130
```

If `uv pip install vllm --torch-backend=auto` already pulled CPU torch, fix it with:

```bash
uv pip install --upgrade torch torchvision torchaudio --torch-backend=cu130
```

Start the server. The extra flags are WSL/shared-display workarounds, not performance tuning:

```bash
export PATH="$HOME/.local/gcc/usr/bin:$HOME/.local/bin:/usr/lib/wsl/lib:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/gcc/usr/lib/x86_64-linux-gnu:/usr/lib/wsl/lib"
export LIBRARY_PATH="$HOME/.local/gcc/usr/lib/x86_64-linux-gnu"
export CPATH="$HOME/.local/gcc/usr/include/x86_64-linux-gnu:$HOME/.local/gcc/usr/include"
export C_INCLUDE_PATH="$CPATH"
export CC="$HOME/.local/gcc/usr/bin/gcc"
export HF_HOME="$HOME/.cache/huggingface"
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve Qwen/Qwen3-1.7B --gpu-memory-utilization 0.85 --enforce-eager
```

`--gpu-memory-utilization 0.85` is required because Windows/WSLg already holds about 1.2 GiB VRAM, so the default 0.92 reservation does not fit. `--enforce-eager` and `VLLM_USE_FLASHINFER_SAMPLER=0` avoid FlashInfer/Inductor paths that ask for `nvcc`. `VLLM_WSL2_ENABLE_PIN_MEMORY=1` is required for the vLLM 0.28 V2 runner on WSL2.

In a second WSL shell, after the server prints `Application startup complete`:

```bash
cd ~/serve-scope
source .venv/bin/activate
python scripts/p0_stream_smoke.py
```

The smoke script sends one streaming chat request, prints chunks as they arrive, and writes `artifacts/p0/smoke.json`.

Captured machine metadata lives in `artifacts/p0/environment.txt`.

## What is not here yet

P0 does not include workload generators, saturation sweeps, mixed interactive/background traffic, dashboards, frontend code, admission control, ServeScope scheduling, or a benchmark harness.

Those belong to later checkpoints. Do not treat this repo as if they already exist.
