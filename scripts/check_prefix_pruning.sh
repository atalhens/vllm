#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"
"$PYTHON" -m pytest -q \
  tests/v1/core/test_kv_cache_utils.py::test_tiered_free_block_queue_orders_and_demotes_idle_blocks \
  tests/v1/core/test_kv_cache_utils.py::test_gpu_pool_reuses_dead_session_prefix_before_live_session_prefix \
  tests/v1/kv_offload/cpu/policies/test_prefix_pruning.py::test_prefix_policy_evicts_shallow_tiers_before_lru_tail \
  tests/v1/kv_offload/cpu/policies/test_prefix_pruning.py::test_dead_session_is_evicted_before_live_prefix \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py::test_prefix_pruning_offload_loads_disjoint_suffix_spans \
  tests/v1/core/test_scheduler.py::test_scheduler_jumps_over_cached_suffix_spans

echo "Prefix-pruning checks passed (CPU/GPU policy ordering and CPU suffix-load planning)."
echo "This is a CPU-side logic check; it does not run a model or measure GPU prefill latency."
