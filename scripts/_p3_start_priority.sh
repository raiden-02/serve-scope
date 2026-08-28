#!/bin/bash
set -euo pipefail
cd /home/mayur/serve-scope
export PATH="$HOME/.local/gcc/usr/bin:$HOME/.local/bin:/usr/lib/wsl/lib:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/gcc/usr/lib/x86_64-linux-gnu:/usr/lib/wsl/lib"
export LIBRARY_PATH="$HOME/.local/gcc/usr/lib/x86_64-linux-gnu"
export CPATH="$HOME/.local/gcc/usr/include/x86_64-linux-gnu:$HOME/.local/gcc/usr/include"
export C_INCLUDE_PATH="$CPATH"
export CC="$HOME/.local/gcc/usr/bin/gcc"
export HF_HOME="$HOME/.cache/huggingface"
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
ulimit -n 10240
exec .venv/bin/vllm serve Qwen/Qwen3-1.7B \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --enable-per-request-metrics \
  --scheduling-policy priority
