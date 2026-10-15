#!/usr/bin/env bash
# Start vLLM with one of this branch's KV-cache eviction profiles.
#
# Usage:
#   ./server.sh <profile> [gpu] [port] [model] [extra vllm serve args...]
#
# Profiles:
#   lru                 Native GPU prefix-cache LRU; no offloading connector.
#   cpu-prefix          CPU offload tier with plain prefix pruning.
#   cpu-sessions        CPU offload tier with session-aware prefix pruning.
#   cpu-fs-prefix       CPU + bounded filesystem tier, prefix pruning on both.
#   cpu-fs-sessions     CPU + bounded filesystem tier, session-aware on both.
#   gpu-prefix          GPU suffix-preserving prefix pruning, GPU-only profile.
#   gpu-sessions        GPU prefix pruning with per-session EWA liveness.
#   gpu-sessions-global GPU prefix pruning with global-quantile liveness.
#   cpu-gpu-prefix      Plain prefix pruning in both GPU and CPU tiers.
#   cpu-gpu-sessions    Session-aware pruning in both GPU and CPU tiers.
#   cpu-gpu-fs-prefix   Plain prefix pruning in GPU, CPU, and bounded FS tiers.
#   cpu-gpu-fs-sessions Session-aware pruning in GPU, CPU, and bounded FS tiers.
#
# Configuration is controlled with environment variables; see --help below.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Start a local vLLM server with one of this branch's KV eviction profiles.

Usage:
  ./server.sh <profile> [gpu] [port] [model] [extra vllm serve args...]

Profiles:
  lru                  Native GPU prefix-cache LRU (baseline).
  cpu-prefix           GPU + CPU offload, plain CPU prefix-pruning policy.
  cpu-sessions         GPU + CPU offload, session-aware CPU prefix pruning.
  cpu-fs-prefix        CPU prefix pruning + bounded FS prefix pruning.
  cpu-fs-sessions      Session-aware pruning in CPU and bounded FS tiers.
  gpu-prefix           GPU prefix pruning without CPU/FS offloading.
  gpu-sessions         GPU prefix pruning with per-session EWA detection.
  gpu-sessions-global  GPU prefix pruning with global-quantile detection.
  cpu-gpu-prefix       Plain prefix pruning in both GPU and CPU tiers.
  cpu-gpu-sessions     Session-aware pruning in both GPU and CPU tiers.
  cpu-gpu-fs-prefix    Plain prefix pruning in GPU, CPU, and bounded FS tiers.
  cpu-gpu-fs-sessions  Session-aware pruning in GPU, CPU, and bounded FS tiers.

Examples:
  ./server.sh cpu-prefix 0 8000 meta-llama/Llama-3.1-8B-Instruct
  ./server.sh cpu-sessions 0 8000 meta-llama/Llama-3.1-8B-Instruct
  ./server.sh cpu-fs-sessions 0 8000 meta-llama/Llama-3.1-8B-Instruct
  ./server.sh cpu-gpu-prefix 0 8000 meta-llama/Llama-3.1-8B-Instruct
  ./server.sh cpu-gpu-sessions 0 8000 meta-llama/Llama-3.1-8B-Instruct
  ./server.sh cpu-gpu-fs-sessions 0 8000 meta-llama/Llama-3.1-8B-Instruct
  ./server.sh gpu-prefix 0 8000 meta-llama/Llama-3.1-8B-Instruct
  ./server.sh gpu-sessions 0 8000 meta-llama/Llama-3.1-8B-Instruct
  ./server.sh gpu-sessions-global 0 8000 meta-llama/Llama-3.1-8B-Instruct

Configuration:
  MODEL                    Default model: meta-llama/Llama-3.1-8B-Instruct
  HOST                     Bind address (default 0.0.0.0)
  MAX_MODEL_LEN            Model context limit (default 32768)
  GPU_MEMORY_UTILIZATION   GPU memory fraction (default 0.90)
  CPU_CACHE_GB             CPU tier budget in GiB across workers (default 8)
  BLOCK_SIZE               Offload chunk block size (default 16 tokens)
  PREFIX_TIER_TOKENS       Initial prefix tier width (default 16384)
  PREFIX_TIER_STEP_TOKENS  Tier growth (default 16384)
  PREFIX_MAX_TIERS         Maximum prefix tiers (default 8)
  SESSION_DEAD_TTL_S       Session idle floor (default 200 for session profiles)
  SESSION_DETECTOR         CPU detector: ewa or global (default ewa)
  SESSION_EWA_ALPHA        EWA smoothing factor (default 0.3)
  SESSION_DEAD_K           EWA standard-deviation multiplier (default 3)
  SESSION_MIN_GAPS         Gaps required before adaptive detection (default 2)
  SESSION_QUANTILE         Global idle-gap quantile (default 0.95)
  SESSION_GAP_WINDOW       Global detector sample window (default 256)
  FS_ROOT                  FS tier directory (default /tmp/vllm-kv-cache)
  FS_CACHE_GB              FS tier capacity in GiB (default 64)
  FS_POLICY                FS policy: prefix_pruning, prefix_pruning_on_sessions,
                           lru, reject (default follows the selected profile)
  VLLM_BIN                 vLLM CLI path (default .venv/bin/vllm, then PATH)
  LOG_DIR                  Server log directory (default ./logs)
  CONFIG_ONLY=1            Print the generated connector JSON and exit.
  DRY_RUN=1                Print the resolved profile/command without launching.

Pass additional vLLM CLI options after the model, e.g.:
  ./server.sh cpu-prefix 0 8000 MODEL --trust-remote-code --dtype half

Session-aware profiles need a stable session_id on every request belonging to
the same conversation. Example:
  curl http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"MODEL","session_id":"conversation-42","messages":[{"role":"user","content":"Hello"}]}'

The cpu-gpu-* profiles enable GPU eviction and the CPU offload policy in one
server; cpu-gpu-fs-* profiles also add an independently configured FS tier.
The GPU tier prunes local GPU blocks, while OffloadingConnector manages
offloaded CPU/FS copies and suffix-hit/gap-recompute. GPU-only profiles use
the device cache's suffix-hit path.
EOF
}

PROFILE="${1:-help}"
if [[ "$PROFILE" == "-h" || "$PROFILE" == "--help" || "$PROFILE" == "help" ]]; then
  usage
  exit 0
fi
shift

GPU="${1:-0}"
if (($#)); then shift; fi
PORT="${1:-8000}"
if (($#)); then shift; fi
MODEL="${1:-${MODEL:-meta-llama/Llama-3.1-8B-Instruct}}"
if (($#)); then shift; fi
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
FS_ROOT="${FS_ROOT:-/tmp/vllm-kv-cache}"
FS_CACHE_GB="${FS_CACHE_GB:-64}"
FS_READ_THREADS="${FS_READ_THREADS:-4}"
FS_WRITE_THREADS="${FS_WRITE_THREADS:-4}"
FS_POLICY_OVERRIDE="${FS_POLICY:-}"

CPU_POLICY=""
FS_POLICY=""
GPU_POLICY=0
NEEDS_SESSION=0
case "$PROFILE" in
  lru)
    ;;
  cpu-prefix)
    CPU_POLICY=prefix_pruning
    SESSION_DEAD_TTL_S=0
    ;;
  cpu-sessions)
    CPU_POLICY=prefix_pruning_on_sessions
    NEEDS_SESSION=1
    ;;
  cpu-fs-prefix)
    CPU_POLICY=prefix_pruning
    FS_POLICY="${FS_POLICY_OVERRIDE:-prefix_pruning}"
    SESSION_DEAD_TTL_S=0
    ;;
  cpu-fs-sessions)
    CPU_POLICY=prefix_pruning_on_sessions
    FS_POLICY="${FS_POLICY_OVERRIDE:-prefix_pruning_on_sessions}"
    NEEDS_SESSION=1
    ;;
  cpu-gpu-prefix)
    CPU_POLICY=prefix_pruning
    GPU_POLICY=1
    SESSION_DEAD_TTL_S=0
    ;;
  cpu-gpu-sessions)
    CPU_POLICY=prefix_pruning_on_sessions
    GPU_POLICY=1
    NEEDS_SESSION=1
    ;;
  cpu-gpu-fs-prefix)
    CPU_POLICY=prefix_pruning
    FS_POLICY="${FS_POLICY_OVERRIDE:-prefix_pruning}"
    GPU_POLICY=1
    SESSION_DEAD_TTL_S=0
    ;;
  cpu-gpu-fs-sessions)
    CPU_POLICY=prefix_pruning_on_sessions
    FS_POLICY="${FS_POLICY_OVERRIDE:-prefix_pruning_on_sessions}"
    GPU_POLICY=1
    NEEDS_SESSION=1
    ;;
  gpu-prefix)
    GPU_POLICY=1
    SESSION_DEAD_TTL_S=0
    ;;
  gpu-sessions)
    GPU_POLICY=1
    NEEDS_SESSION=1
    ;;
  gpu-sessions-global)
    GPU_POLICY=1
    SESSION_DETECTOR=global
    NEEDS_SESSION=1
    ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    usage >&2
    exit 2
    ;;
esac

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_SUFFIX_PRESERVING_EVICTION=0
unset VLLM_SUFFIX_EVICTION_TIER_TOKENS
unset VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS
unset VLLM_SUFFIX_EVICTION_MAX_TIERS
unset VLLM_SUFFIX_EVICTION_IDLE_TTL_S
unset VLLM_SUFFIX_SESSION_DEAD_TTL_S
unset VLLM_SUFFIX_SESSION_DETECTOR
unset VLLM_SUFFIX_SESSION_EWA_ALPHA
unset VLLM_SUFFIX_SESSION_DEAD_K
unset VLLM_SUFFIX_SESSION_MIN_GAPS
unset VLLM_SUFFIX_SESSION_QUANTILE
unset VLLM_SUFFIX_SESSION_GAP_WINDOW

if ((GPU_POLICY)); then
  export VLLM_SUFFIX_PRESERVING_EVICTION=1
  export VLLM_SUFFIX_EVICTION_TIER_TOKENS="$PREFIX_TIER_TOKENS"
  export VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS="$PREFIX_TIER_STEP_TOKENS"
  export VLLM_SUFFIX_EVICTION_MAX_TIERS="$PREFIX_MAX_TIERS"
  export VLLM_SUFFIX_EVICTION_IDLE_TTL_S="${GPU_IDLE_TTL_S:-60}"
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
  export CPU_POLICY FS_POLICY CPU_CACHE_GB BLOCK_SIZE
  export PREFIX_TIER_TOKENS PREFIX_TIER_STEP_TOKENS PREFIX_MAX_TIERS
  export SESSION_DEAD_TTL_S SESSION_DETECTOR SESSION_EWA_ALPHA SESSION_DEAD_K
  export SESSION_MIN_GAPS SESSION_QUANTILE SESSION_GAP_WINDOW
  export FS_ROOT FS_CACHE_GB FS_READ_THREADS FS_WRITE_THREADS
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
if os.environ.get("FS_POLICY"):
    extra["spec_name"] = "TieringOffloadingSpec"
    extra["secondary_tiers"] = [{
        "type": "fs",
        "root_dir": os.environ["FS_ROOT"],
        "capacity_bytes": int(float(os.environ["FS_CACHE_GB"]) * 1024**3),
        "eviction_policy": os.environ["FS_POLICY"],
        "n_read_threads": number("FS_READ_THREADS", "4"),
        "n_write_threads": number("FS_WRITE_THREADS", "4"),
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
    }]

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
    if ((GPU_POLICY)); then
      echo "GPU eviction settings for '$PROFILE':" >&2
      echo "  VLLM_SUFFIX_PRESERVING_EVICTION=$VLLM_SUFFIX_PRESERVING_EVICTION" >&2
      echo "  VLLM_SUFFIX_EVICTION_TIER_TOKENS=$VLLM_SUFFIX_EVICTION_TIER_TOKENS" >&2
      echo "  VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS=$VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS" >&2
      echo "  VLLM_SUFFIX_EVICTION_MAX_TIERS=$VLLM_SUFFIX_EVICTION_MAX_TIERS" >&2
      echo "  VLLM_SUFFIX_EVICTION_IDLE_TTL_S=$VLLM_SUFFIX_EVICTION_IDLE_TTL_S" >&2
      echo "  VLLM_SUFFIX_SESSION_DEAD_TTL_S=$VLLM_SUFFIX_SESSION_DEAD_TTL_S" >&2
      echo "  VLLM_SUFFIX_SESSION_DETECTOR=$VLLM_SUFFIX_SESSION_DETECTOR" >&2
    fi
    printf '%s\n' "$KV_TRANSFER_CONFIG"
  elif ((GPU_POLICY)); then
    printf '%s\n' \
      "VLLM_SUFFIX_PRESERVING_EVICTION=$VLLM_SUFFIX_PRESERVING_EVICTION" \
      "VLLM_SUFFIX_EVICTION_TIER_TOKENS=$VLLM_SUFFIX_EVICTION_TIER_TOKENS" \
      "VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS=$VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS" \
      "VLLM_SUFFIX_EVICTION_MAX_TIERS=$VLLM_SUFFIX_EVICTION_MAX_TIERS" \
      "VLLM_SUFFIX_EVICTION_IDLE_TTL_S=$VLLM_SUFFIX_EVICTION_IDLE_TTL_S" \
      "VLLM_SUFFIX_SESSION_DEAD_TTL_S=$VLLM_SUFFIX_SESSION_DEAD_TTL_S" \
      "VLLM_SUFFIX_SESSION_DETECTOR=$VLLM_SUFFIX_SESSION_DETECTOR" \
      "VLLM_SUFFIX_SESSION_EWA_ALPHA=$VLLM_SUFFIX_SESSION_EWA_ALPHA" \
      "VLLM_SUFFIX_SESSION_DEAD_K=$VLLM_SUFFIX_SESSION_DEAD_K" \
      "VLLM_SUFFIX_SESSION_MIN_GAPS=$VLLM_SUFFIX_SESSION_MIN_GAPS" \
      "VLLM_SUFFIX_SESSION_QUANTILE=$VLLM_SUFFIX_SESSION_QUANTILE" \
      "VLLM_SUFFIX_SESSION_GAP_WINDOW=$VLLM_SUFFIX_SESSION_GAP_WINDOW"
  else
    echo "Profile '$PROFILE' does not use --kv-transfer-config."
  fi
  exit 0
fi

VLLM_BIN="${VLLM_BIN:-$ROOT/.venv/bin/vllm}"
if [[ ! -x "$VLLM_BIN" ]]; then
  if command -v vllm >/dev/null 2>&1; then
    VLLM_BIN="$(command -v vllm)"
  else
    echo "Cannot find vLLM CLI at '$VLLM_BIN' or on PATH." >&2
    echo "Install this checkout into its environment or set VLLM_BIN=/path/to/vllm." >&2
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
  echo "Session-aware mode is enabled. Send the same session_id on each conversation request."
fi
if ((GPU_POLICY)); then
  echo "GPU eviction: tier_tokens=$VLLM_SUFFIX_EVICTION_TIER_TOKENS tier_step_tokens=$VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS max_tiers=$VLLM_SUFFIX_EVICTION_MAX_TIERS idle_ttl_s=$VLLM_SUFFIX_EVICTION_IDLE_TTL_S session_dead_ttl_s=$VLLM_SUFFIX_SESSION_DEAD_TTL_S detector=$VLLM_SUFFIX_SESSION_DETECTOR"
fi
if [[ -n "$CPU_POLICY" ]]; then
  echo "CPU eviction: policy=$CPU_POLICY cache_gb=$CPU_CACHE_GB"
fi
if [[ -n "$FS_POLICY" ]]; then
  echo "FS eviction: policy=$FS_POLICY capacity_gb=$FS_CACHE_GB root=$FS_ROOT"
fi
printf 'command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  exit 0
fi

LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"
if [[ -n "$FS_POLICY" ]]; then
  mkdir -p "$FS_ROOT"
fi
LOG="$LOG_DIR/${PROFILE}-gpu${GPU}-port${PORT}-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "log=$LOG"
exec "${COMMAND[@]}" 2>&1 | tee "$LOG"
