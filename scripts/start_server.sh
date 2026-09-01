#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

LOCAL_GCC="$HOME/.local/gcc"
LOCAL_CC="$LOCAL_GCC/usr/bin/gcc"

prepend() {
  local var="$1"
  local dir="$2"
  local current="${!var:-}"
  if [ ! -d "$dir" ]; then
    return
  fi
  case ":${current}:" in
    *":${dir}:"*) ;;
    *)
      if [ -n "$current" ]; then
        export "${var}=${dir}:${current}"
      else
        export "${var}=${dir}"
      fi
      ;;
  esac
}

usable_cc() {
  local c="${1:-}"
  [ -n "$c" ] || return 1
  [ -x "$c" ] && return 0
  command -v "$c" >/dev/null 2>&1
}

use_local_gcc() {
  export CC="$LOCAL_CC"
  prepend PATH "$LOCAL_GCC/usr/bin"
  prepend LD_LIBRARY_PATH "$LOCAL_GCC/usr/lib/x86_64-linux-gnu"
  prepend LIBRARY_PATH "$LOCAL_GCC/usr/lib/x86_64-linux-gnu"
  prepend CPATH "$LOCAL_GCC/usr/include/x86_64-linux-gnu"
  prepend CPATH "$LOCAL_GCC/usr/include"
  export C_INCLUDE_PATH="${CPATH:-}"
}

if usable_cc "${CC:-}"; then
  if [ -x "$LOCAL_CC" ] && [ "$CC" = "$LOCAL_CC" ]; then
    use_local_gcc
  fi
elif [ -x "$LOCAL_CC" ]; then
  use_local_gcc
elif command -v gcc >/dev/null 2>&1; then
  export CC="$(command -v gcc)"
else
  echo "No usable C compiler found. vLLM/Triton needs gcc." >&2
  echo "Install a system compiler or set CC." >&2
  exit 1
fi

prepend PATH /usr/lib/wsl/lib
prepend LD_LIBRARY_PATH /usr/lib/wsl/lib

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0

if [ ! -x .venv/bin/vllm ]; then
  echo "vLLM is not installed in .venv." >&2
  echo "See README.md -> First-time setup." >&2
  exit 1
fi

ulimit -n 10240
exec .venv/bin/vllm serve Qwen/Qwen3-1.7B \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --enable-per-request-metrics \
  --scheduling-policy priority
