# CPU/GPU Prefix Pruning on vLLM 0.30

This file is the implementation handoff for the stable vLLM 0.30 port. It
records the branch provenance, design, v0.30 API adaptations, configuration,
known boundaries, and the validation performed against this checkout. The
0.30 code is an API-aware port, not a cherry-pick of the older feature patch.

## Branches and source references

| Work | Repository / branch | Base or reference |
| --- | --- | --- |
| CPU prefix pruning and suffix-reuse source | `/home/ubuntu/sneh/vllm`, `feat/prefix-based-eviction-policy-cpu` | v0.31.1rc0-era source |
| Combined CPU/GPU and session-aware source | `/home/ubuntu/sneh/vllm`, `feat/cpu-gpu-prefix-pruning` | Derived from the CPU branch; original feature commit `344354e2f4d985bb186f8945d8031565768ea96e` |
| Older GPU design reference | `/home/ubuntu/sneh/opensource/eviction/vllm` | `feat/idle-demote-protect-suffix`, v0.26.0-era |
| This stable port | `/home/ubuntu/sneh/vllm-v0.30`, `feat/cpu-gpu-prefix-pruning-v0.30` | Official vLLM tag `v0.30.0`, commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |

The source CPU/GPU feature branch was v0.31.1rc0-derived. The separate GPU
reference was older and was used for design comparison only. The stable port
branch was created from the official 0.30.0 tag; the earlier failed
cherry-pick was aborted and contributed no changes. The changes now described
below were reimplemented against the 0.30.0 APIs.

## Behavior and design

The pruning policy favors cache regions useful to returning requests:

1. Cached blocks are tagged with the session that used them and their end-token
   position when available.
2. CPU eviction can use either prefix tiers alone (`prefix_pruning`) or
   session-aware prefix tiers (`prefix_pruning_on_sessions`). Session mode
   promotes cached blocks from sessions predicted idle/dead.
3. GPU eviction maintains prefix tiers in its free-block queues and can demote
   long-idle blocks or promote blocks from dead sessions.
4. Suffix lookup can keep a cached run after an evicted/missing gap. The
   scheduler computes the gap, then advances past the loaded suffix rather
   than running the model over it a second time.

CPU and GPU use different mechanisms. CPU policy ordering is implemented by
the offload cache policy; GPU ordering is implemented by the KV block pool.
Enabling one does not implicitly enable the other. Session-aware behavior
requires a stable `session_id` on each request in the conversation.

Session liveness uses either a per-session exponentially weighted gap model
(`ewa`) or a global gap quantile (`global`), with a configurable static
minimum idle threshold. An in-progress request is not considered dead.

## v0.30-specific implementation changes

| Surface | Port behavior |
| --- | --- |
| GPU block pool | Adds per-block freed/session metadata, tier-aware free queues, idle demotion, and dead-session promotion. |
| GPU KV managers | Adds prefix-tier setup and suffix lookup/gap block handling for the supported full-attention cache configuration. |
| Scheduler | Observes session arrivals/completions, separates final completion from preemption, limits work to the gap before a GPU or CPU suffix, and advances over cached suffix spans. Deferred request frees preserve session/tier metadata. |
| CPU cache policy | Registers `prefix_pruning` and `prefix_pruning_on_sessions`; configuration is passed through CPU offload `extra_config`. |
| Request/offload context | Extends `ReqContext` with `session_id` and offload-key token positions, populated from the v0.30 `Request` and connector key generation. |
| CPU suffix lookup | For a single full-attention group and the prefix-pruning CPU policy, scans cached runs and loads them as disjoint GPU block segments. |
| CPU/GPU transfers | Extends `GPULoadStoreSpec` with segment metadata and teaches the GPU transfer worker to copy each disjoint segment. |
| Operations | Adds `server.sh` profiles for baseline, CPU-only, GPU-only, and combined CPU/GPU modes. |

v0.30 already has `Request.session_id`, a pluggable CPU `CachePolicy`, and
generic offload load/store support. Those existing APIs did **not** provide the
session gap detector, GPU queue tiers, offload-key position metadata, or the
prefix-pruning-specific cached-suffix/gap scheduling needed here.

## Configuration

`server.sh` accepts a profile, GPU index, port, model, and additional `vllm
serve` arguments:

```bash
./server.sh cpu-prefix 0 8000 MODEL
./server.sh cpu-sessions 0 8000 MODEL
./server.sh gpu-prefix 0 8000 MODEL
./server.sh gpu-sessions 0 8000 MODEL
./server.sh gpu-global 0 8000 MODEL
./server.sh cpu-gpu-prefix 0 8000 MODEL
./server.sh cpu-gpu-sessions 0 8000 MODEL
./server.sh lru 0 8000 MODEL
```

Set `CONFIG_ONLY=1` to print the connector JSON and GPU environment variables
without starting a server. Set `DRY_RUN=1` to print the resolved command.
`VLLM_BIN` can point to a CLI installed for this checkout; avoid mixing the
executable from one worktree with Python imports from another.

### Fast logic check (no model or GPU required)

Run the focused deterministic checks from this checkout:

```bash
PYTHON=/path/to/checkout/python ./scripts/check_prefix_pruning.sh
```

If the checkout's Python is already active, `./scripts/check_prefix_pruning.sh`
is enough. The checks assert that CPU prefix tiers evict shallower entries
first, a finished session's CPU entry is evicted while another entry remains,
the GPU block pool promotes/reuses the finished session's block before the
active session's cached block, CPU suffix spans are planned into GPU block
segments, and the scheduler advances over those cached spans. These tests
exercise eviction selection and transfer planning; they do not copy KV data
on actual GPU hardware or demonstrate serving latency savings.

The script exposes the following tuning variables:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `CPU_CACHE_GB` | `8` | CPU offload capacity |
| `BLOCK_SIZE` | `16` | Offload chunk block size |
| `PREFIX_TIER_TOKENS` | `16384` | First prefix tier width |
| `PREFIX_TIER_STEP_TOKENS` | `16384` | Subsequent tier width |
| `PREFIX_MAX_TIERS` | `8` | Maximum prefix tiers; zero leaves tiers uncapped |
| `SESSION_DEAD_TTL_S` | `200` | Session idle floor; zero disables session promotion |
| `SESSION_DETECTOR` | `ewa` | Session detector: `ewa` or `global` |
| `SESSION_EWA_ALPHA` | `0.3` | EWA smoothing factor |
| `SESSION_DEAD_K` | `3` | EWA standard-deviation multiplier |
| `SESSION_MIN_GAPS` | `2` | Samples before adaptive thresholds apply |
| `SESSION_QUANTILE` | `0.95` | Global idle-gap quantile |
| `SESSION_GAP_WINDOW` | `256` | Maximum global gap samples |
| `GPU_IDLE_TTL_S` | `60` | GPU idle-block demotion delay |

Session-aware profiles require the same session ID on each request in the same
conversation. Use the request's supported session field or pass `session_id`
through the API in a way that populates vLLM's `Request.session_id`; an
arbitrary client-only JSON field is not sufficient unless the serving API
forwards it.

## Compatibility boundaries

- GPU suffix-preserving eviction currently validates for prefix caching and a
  single full-attention cache group. Speculative decoding, pipeline/context
  parallelism, and encoder-decoder requests are rejected by the feature
  validation.
- CPU non-contiguous suffix lookup is enabled only for the prefix-pruning CPU
  policies and a single full-attention offload group. Other cache policies
  retain their normal prefix lookup behavior.
- Mixed CPU+GPU operation uses the GPU suffix hit first when one is already
  available; otherwise the offload connector can find CPU suffix spans.
- Filesystem/storage-tier prefix scoring from the newer source branch has not
  been ported here. The profiles in this branch intentionally cover CPU and
  GPU, not FS.
- A live model-serving run on actual supported GPU hardware is still needed
  to validate end-to-end load timing and measured prefill savings.

## Validation record

Commands run from `/home/ubuntu/sneh/vllm-v0.30`:

```text
python -m compileall -q vllm \
  tests/v1/core/test_kv_cache_utils.py \
  tests/v1/core/test_scheduler.py \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py \
  tests/v1/kv_offload/cpu/policies/test_prefix_pruning.py
bash -n server.sh
CONFIG_ONLY=1 ./server.sh cpu-gpu-sessions
```

The compile and launcher checks passed. `CONFIG_ONLY` printed the expected
session-aware CPU offload policy and GPU-tier settings.

Focused pytest validation:

```text
tests/v1/core/test_kv_cache_utils.py                         116 passed
tests/v1/core/test_scheduler.py -k cached_suffix_spans         1 passed
tests/v1/kv_offload/cpu/policies/test_prefix_pruning.py         3 passed
tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py 221 passed
```

An earlier focused CPU policy/manager and KV manager batch passed 196 tests.
The checkout's test environment emits PyTorch/JIT deprecation warnings on
Python 3.14.

A previous broad mixed scheduler/connector run had scheduler test setup
failures while inspecting Llava model configurations in this environment. The
offloading connector suite has since passed in full; scheduler validation
currently covers the targeted cached-suffix jump and previously passing
focused scheduling/deferred-free tests, not the entire scheduler suite. No
model server has been run on supported GPU hardware, so live end-to-end
serving and measured prefill savings remain unverified.
