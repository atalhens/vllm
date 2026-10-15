#!/usr/bin/env bash
# Start vLLM with a CPU/GPU prefix-pruning profile.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: ./server.sh <profile> [gpu] [port] [model] [extra vllm serve args...]

Profiles:
  lru              Native GPU prefix-cache behavior.
  cpu-prefix       CPU offload with prefix-tier eviction.
  cpu-sessions     CPU offload with session-aware prefix-tier eviction.
  gpu-prefix       GPU suffix-preserving prefix pruning, no CPU offload.
  gpu-sessions     GPU pruning with per-session EWA liveness.
  gpu-global       GPU pruning with global-quantile liveness.
  cpu-gpu-prefix   Prefix pruning in both CPU and GPU cache tiers.
  cpu-gpu-sessions Session-aware pruning in both CPU and GPU tiers.

Environment:
  MODEL, HOST, MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION, CPU_CACHE_GB,
  BLOCK_SIZE, PREFIX_TIER_TOKENS, PREFIX_TIER_STEP_TOKENS, PREFIX_MAX_TIERS,
  SESSION_DEAD_TTL_S, SESSION_DETECTOR, SESSION_EWA_ALPHA, SESSION_DEAD_K,
  SESSION_MIN_GAPS, SESSION_QUANTILE, SESSION_GAP_WINDOW, GPU_IDLE_TTL_S,
  VLLM_BIN, LOG_DIR, CONFIG_ONLY=1, DRY_RUN=1.

Set SESSION_ID on each request belonging to the same conversation for
session-aware profiles. Consult PREFIX_PRUNING_README.md for request examples
and configuration details.
EOF
}

PROFILE="${1:-help}"
if [[ "$PROFILE" == "-h" || "$PROFILE" == "--help" || "$PROFILE" == "help" ]]; then
  usage
  exit 0
fi
shift

GPU="${1:-0}"; (($#)) && shift
PORT="${1:-8000}"; (($#)) && shift
MODEL="${1:-${MODEL:-meta-llama/Llama-3.1-8B-Instruct}}"; (($#)) && shift
EXTRA_ARGS=("$@")

HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
CPU_CACHE_GB="${CPU_CACHE_GB:-8}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
PREFIX_TIER_TOKENS="${PREFIX_TIER_TOKENS:-16384}"
PREFIX_TIER_STEP_TOKENS="${PREFIX_TIER_STEP_TOKENS:-16384}"
PREFIX_MAX_TIERS="${PREFIX_MAX_TIERS:-8}"
SESSION_DEAD_TTL_S="${SESSION_DEAD_TTL_S:-200}"
SESSION_DETECTOR="${SESSION_DETECTOR:-ewa}"
SESSION_EWA_ALPHA="${SESSION_EWA_ALPHA:-0.3}"
SESSION_DEAD_K="${SESSION_DEAD_K:-3}"
SESSION_MIN_GAPS="${SESSION_MIN_GAPS:-2}"
SESSION_QUANTILE="${SESSION_QUANTILE:-0.95}"
SESSION_GAP_WINDOW="${SESSION_GAP_WINDOW:-256}"
GPU_IDLE_TTL_S="${GPU_IDLE_TTL_S:-60}"

CPU_POLICY=""
GPU_POLICY=0
NEEDS_SESSION=0
case "$PROFILE" in
  lru) ;;
  cpu-prefix) CPU_POLICY=prefix_pruning; SESSION_DEAD_TTL_S=0 ;;
  cpu-sessions) CPU_POLICY=prefix_pruning_on_sessions; NEEDS_SESSION=1 ;;
  gpu-prefix) GPU_POLICY=1; SESSION_DEAD_TTL_S=0 ;;
  gpu-sessions) GPU_POLICY=1; NEEDS_SESSION=1 ;;
  gpu-global) GPU_POLICY=1; SESSION_DETECTOR=global; NEEDS_SESSION=1 ;;
  cpu-gpu-prefix)
    CPU_POLICY=prefix_pruning; GPU_POLICY=1; SESSION_DEAD_TTL_S=0 ;;
  cpu-gpu-sessions)
    CPU_POLICY=prefix_pruning_on_sessions; GPU_POLICY=1; NEEDS_SESSION=1 ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    usage >&2
    exit 2
    ;;
esac

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_SUFFIX_PRESERVING_EVICTION=0
if ((GPU_POLICY)); then
  export VLLM_SUFFIX_PRESERVING_EVICTION=1
  export VLLM_SUFFIX_EVICTION_TIER_TOKENS="$PREFIX_TIER_TOKENS"
  export VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS="$PREFIX_TIER_STEP_TOKENS"
  export VLLM_SUFFIX_EVICTION_MAX_TIERS="$PREFIX_MAX_TIERS"
  export VLLM_SUFFIX_EVICTION_IDLE_TTL_S="$GPU_IDLE_TTL_S"
  export VLLM_SUFFIX_SESSION_DEAD_TTL_S="$SESSION_DEAD_TTL_S"
  export VLLM_SUFFIX_SESSION_DETECTOR="$SESSION_DETECTOR"
  export VLLM_SUFFIX_SESSION_EWA_ALPHA="$SESSION_EWA_ALPHA"
  export VLLM_SUFFIX_SESSION_DEAD_K="$SESSION_DEAD_K"
  export VLLM_SUFFIX_SESSION_MIN_GAPS="$SESSION_MIN_GAPS"
  export VLLM_SUFFIX_SESSION_QUANTILE="$SESSION_QUANTILE"
  export VLLM_SUFFIX_SESSION_GAP_WINDOW="$SESSION_GAP_WINDOW"
fi

KV_TRANSFER_CONFIG=""
if [[ -n "$CPU_POLICY" ]]; then
  export CPU_POLICY CPU_CACHE_GB BLOCK_SIZE PREFIX_TIER_TOKENS
  export PREFIX_TIER_STEP_TOKENS PREFIX_MAX_TIERS SESSION_DEAD_TTL_S
  export SESSION_DETECTOR SESSION_EWA_ALPHA SESSION_DEAD_K SESSION_MIN_GAPS
  export SESSION_QUANTILE SESSION_GAP_WINDOW
  KV_TRANSFER_CONFIG="$(python3 - <<'PY'
import json
import os

def number(name, default, cast=int):
    return cast(os.environ.get(name, default))

extra = {
    "block_size": number("BLOCK_SIZE", "16"),
    "cpu_bytes_to_use": int(float(os.environ["CPU_CACHE_GB"]) * 1024**3),
    "eviction_policy": os.environ["CPU_POLICY"],
    "first_tier_tokens": number("PREFIX_TIER_TOKENS", "16384"),
    "tier_step_tokens": number("PREFIX_TIER_STEP_TOKENS", "16384"),
    "max_tiers": number("PREFIX_MAX_TIERS", "8"),
    "session_dead_ttl_s": number("SESSION_DEAD_TTL_S", "200", float),
    "session_ewa_alpha": number("SESSION_EWA_ALPHA", "0.3", float),
    "session_dead_k": number("SESSION_DEAD_K", "3", float),
    "session_min_gaps": number("SESSION_MIN_GAPS", "2"),
    "session_detector": os.environ["SESSION_DETECTOR"],
    "session_quantile": number("SESSION_QUANTILE", "0.95", float),
    "session_gap_window": number("SESSION_GAP_WINDOW", "256"),
}
print(json.dumps({
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": extra,
}, separators=(",", ":")))
PY
)"
fi

if [[ "${CONFIG_ONLY:-0}" == 1 ]]; then
  if [[ -n "$KV_TRANSFER_CONFIG" ]]; then
    printf '%s\n' "$KV_TRANSFER_CONFIG"
  elif ((GPU_POLICY)); then
    env | grep '^VLLM_SUFFIX_' | sort
  else
    echo "Profile '$PROFILE' uses native cache behavior."
  fi
  exit 0
fi

VLLM_BIN="${VLLM_BIN:-$ROOT/.venv/bin/vllm}"
if [[ ! -x "$VLLM_BIN" ]]; then
  if command -v vllm >/dev/null 2>&1; then
    VLLM_BIN="$(command -v vllm)"
  else
    echo "Cannot find vLLM CLI at '$VLLM_BIN' or on PATH." >&2
    echo "Install this checkout or set VLLM_BIN=/path/to/vllm." >&2
    exit 1
  fi
fi

COMMAND=("$VLLM_BIN" serve "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --enable-prefix-caching
)
if [[ -n "$KV_TRANSFER_CONFIG" ]]; then
  COMMAND+=(--kv-transfer-config "$KV_TRANSFER_CONFIG")
fi
COMMAND+=("${EXTRA_ARGS[@]}")

echo "profile=$PROFILE gpu=$GPU host=$HOST port=$PORT model=$MODEL"
if ((NEEDS_SESSION)); then
  echo "Session-aware mode: send the same session_id for every request in a conversation."
fi
printf 'command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  exit 0
fi
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${PROFILE}-gpu${GPU}-port${PORT}-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "log=$LOG"
exec "${COMMAND[@]}" 2>&1 | tee "$LOG"
