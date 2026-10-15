# CPU and GPU Prefix-Pruning Implementation Notes

This document records the branch layout, design, implementation boundaries,
configuration, and validation status for the CPU/GPU prefix-pruning work. It is
intended as a practical handoff and reference when comparing this checkout
with the separate GPU reference checkout.

## 1. Branch and source-version record

### Combined implementation branch

- Repository: `/home/ubuntu/sneh/vllm`
- Branch: `feat/cpu-gpu-prefix-pruning`
- Intended starting point: `feat/prefix-based-eviction-policy-cpu`
- Base commit: `5d0aeb5b9cfe507680f28f2eddae578e1a3cfc43`
- At the time this document was written, that commit was also the tip of
  `feat/prefix-based-eviction-policy-cpu`. The implementation and documentation
  edits described below were worktree changes, not commits on this branch yet.
- The selected base already contains the CPU/offload suffix lookup and
  recomputation work that the user asked to retain.

### GPU reference checkout

- Repository: `/home/ubuntu/sneh/opensource/eviction/vllm`
- Branch at inspection: `feat/idle-demote-protect-suffix`
- Reference commit: `75e061ae2f`
- `git describe --tags` at inspection: `v0.26.0-7-g75e061ae2f`

### Version caveat

The original concern was that the GPU reference might be based on vLLM 0.26 and
the CPU feature branch on vLLM 0.28. The local Git metadata inspected for this
handoff does **not** confirm that distinction: both the selected CPU-feature
base and the GPU reference worktree describe from the `v0.26.0` tag. The
combined checkout described as `v0.26.0-7-g5d0aeb5b9c` at its base.

`git describe` identifies the nearest reachable tag, not necessarily the exact
version used when an implementation was first authored or tested. If 0.28 refers
to a different historical branch/checkout, record that checkout's commit or
tag before treating it as the CPU implementation's source version. In either
case, code should not be copied mechanically across vLLM versions: scheduler,
block-pool, connector, and request lifecycle APIs can differ. This combined
branch adapts behavior to the APIs in the selected checkout.

## 2. What the feature does

The implementation tries to evict less useful KV cache before more useful KV
cache:

1. Prefer shallow/leading prefix blocks for eviction.
2. Preserve deeper suffix blocks when possible, so a returning prompt may hit a
   cached suffix.
3. Recompute only the missing gap, then resume from the surviving suffix.
4. Optionally identify finished conversation sessions and prioritize their
   cached blocks over blocks belonging to sessions likely to return.

The broad policy is shared across tiers, but the **eviction mechanism is
tier-specific**:

| Tier | Where policy is applied | How it is enabled |
| --- | --- | --- |
| GPU | GPU block pool's free-block queue | `VLLM_SUFFIX_PRESERVING_EVICTION=1` |
| CPU host memory | Offloading connector's CPU cache policy | `eviction_policy` in `--kv-transfer-config` |
| Filesystem secondary tier | Bounded FS tier's eviction index | FS `eviction_policy` in `secondary_tiers` |

GPU prefix pruning and the `OffloadingConnector` can be enabled together.
GPU eviction orders local free GPU blocks; the connector independently manages
CPU/secondary-tier copies. Combined operation is exposed as `cpu-gpu-*`
profiles in `server.sh`. The connector suffix lookup and gap-recomputation
path remains responsible for suffixes loaded from CPU or secondary storage.

## 3. Design and lifecycle

### Prefix tiers

Freed hashed blocks are ordered by their token position in the completed
request. Leading blocks enter the earliest eviction tier; following ranges
enter progressively later tiers; blocks beyond a configured tier cap fall back
to the LRU tail. Within a tier, the earlier prefix blocks are selected first.
This makes the evicted region tend to grow from the front of a sequence while
leaving a later suffix available for reuse.

The CPU policy and filesystem index use shared scoring/session-liveness
primitives in `vllm/v1/kv_offload/eviction/prefix_pruning_core.py`. The GPU
queue applies the same high-level intent to free GPU blocks, but it is not the
same queue or data path as CPU offloading.

### Session liveness

`vllm/v1/core/session_liveness.py` provides a shared session think-time
tracker:

- It observes intervals between completed turns and later arrivals carrying
  the same session ID.
- The per-session `ewa` detector estimates an expected gap and variation, then
  compares idle time against the configured floor and learned estimate.
- The `global` detector uses a quantile of recent observed gaps across
  sessions.
- A positive `session_dead_ttl_s` is the minimum idle threshold. A value of
  zero disables session-dead promotion.
- `session_min_gaps` prevents adapting from too little history.

For GPU eviction, free blocks remember their session owner. On allocation
pressure, idle TTL demotion is applied before dead-session promotion. Dead
session blocks are then moved ahead of other cached free blocks so they are
reclaimed first. Cache reset and block reuse clear the associated metadata.

Scheduler preemption is not treated as the end of a conversation: it releases
KV but does not complete the session. A completed request turn updates
session-finish state. Session-aware behavior requires callers to send a stable
session ID across requests in the same conversation.

### Suffix lookup and gap recomputation

Eviction intentionally may leave a non-contiguous cache: an early range is
missing while later blocks remain. Lookup therefore cannot stop permanently at
the first prefix miss.

- The GPU prefix-cache path can report a surviving suffix and arrange for the
  scheduler to compute the missing gap before jumping over the cached range.
- The CPU/offload connector scans for reusable hit runs, loads the surviving
  suffix, and keeps the missing prefix/gap available for recomputation and
  subsequent storage.
- The connector's suffix lookup and gap-recompute path remains available when
  GPU suffix-preserving eviction is disabled.

The important correctness condition is that cached tokens after a hole are not
treated as a contiguous prefix hit. The gap must be computed before execution
resumes from the later hit.

## 4. Change inventory

### Existing work retained from the CPU feature branch

The chosen base already carried CPU/offload prefix-pruning and suffix lookup /
gap-recompute behavior. This branch was intentionally created from that base
instead of dropping those changes and rebuilding from `main`.

### Combined CPU/GPU session-aware implementation

The changes on `feat/cpu-gpu-prefix-pruning` add or adjust:

- **Shared session tracker** — added `vllm/v1/core/session_liveness.py`; the
  CPU/storage scorer now reuses it rather than owning a separate implementation.
- **GPU block ownership and ordering** — GPU free blocks retain session
  identity and freed time; idle demotion and dead-session promotion are wired
  into the GPU tiered queue.
- **GPU lifecycle wiring** — the scheduler and KV-cache manager record request
  arrivals and completed turns; preemptions do not falsely finish sessions.
- **Policy distinction** — CPU `prefix_pruning` now means plain
  prefix-first eviction by default; `prefix_pruning_on_sessions` opts in to
  session-dead promotion. Both keep the same suffix lookup/recompute behavior.
- **Environment configuration** — GPU pruning and session detector settings
  are read through `vllm/envs.py`.
- **Tests** — coverage for plain CPU policy defaults, GPU dead-session
  precedence, scheduler session-arrival wiring, and suffix/gap behavior.
- **Documentation and launch helper** — `server.sh` provides GPU-only,
  CPU-offload, and CPU-plus-filesystem profiles. The feature docs point to it.

Specific edited paths, grouped by responsibility:

| Responsibility | Files |
| --- | --- |
| GPU queue and block metadata | `vllm/v1/core/block_pool.py`, `vllm/v1/core/kv_cache_utils.py` |
| GPU manager/coordinator integration | `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/kv_cache_coordinator.py`, `vllm/v1/core/single_type_kv_cache_manager.py` |
| Scheduler and environment wiring | `vllm/v1/core/sched/scheduler.py`, `vllm/envs.py` |
| Shared session detection and CPU policy | `vllm/v1/core/session_liveness.py`, `vllm/v1/kv_offload/eviction/prefix_pruning_core.py`, `vllm/v1/kv_offload/cpu/policies/factory.py`, `vllm/v1/kv_offload/cpu/policies/prefix_pruning.py` |
| Tests | `tests/v1/core/test_prefix_caching.py`, `tests/v1/core/test_scheduler.py`, `tests/v1/kv_offload/cpu/policies/test_prefix_pruning.py` |
| Operation and profile docs | `server.sh`, `docs/features/kv_offloading_usage.md`, `docs/features/kv_offload_prefix_pruning_lookup.md` |

At the time of writing, `scripts/tier_ttft_bench.py` was also modified in the
worktree, but that edit predated this implementation work and was deliberately
left untouched. Check `git status` before staging so it is not accidentally
bundled with the feature unless intended.

## 5. How to run the profiles

From the repository root:

```bash
./server.sh --help
```

Examples:

```bash
# Native GPU prefix-cache behavior as a baseline.
./server.sh lru 0 8000 <model>

# CPU-tier prefix-first eviction, without session-dead promotion.
./server.sh cpu-prefix 0 8000 <model>

# CPU-tier prefix pruning with session-dead promotion.
./server.sh cpu-sessions 0 8000 <model>

# CPU and bounded filesystem tier, both session-aware.
./server.sh cpu-fs-sessions 0 8000 <model>

# GPU-only prefix pruning.
./server.sh gpu-prefix 0 8000 <model>

# GPU-only per-session EWA or global-quantile session detection.
./server.sh gpu-sessions 0 8000 <model>
./server.sh gpu-sessions-global 0 8000 <model>

# Enable GPU and CPU prefix pruning together, with or without session tracking.
./server.sh cpu-gpu-prefix 0 8000 <model>
./server.sh cpu-gpu-sessions 0 8000 <model>

# Enable all three tiers (GPU, CPU, filesystem).
./server.sh cpu-gpu-fs-sessions 0 8000 <model>

# Run independent implementations concurrently on separate GPUs/ports.
./server.sh gpu-prefix 0 8000 <model> &
./server.sh cpu-sessions 1 8001 <model> &
```

The second positional argument is the GPU index, and the third is the HTTP
port. `server.sh` defaults to `meta-llama/Llama-3.1-8B-Instruct`; replace
`<model>` with a model accessible to your environment. It uses
`.venv/bin/vllm` if present, then falls back to `PATH`; set `VLLM_BIN` to select
a particular executable.

### CPU and filesystem profiles

`cpu-prefix` and `cpu-sessions` use the CPU primary tier of
`OffloadingConnector`. The `cpu-fs-*` profiles also configure a capacity-bounded
filesystem tier. Useful overrides include:

```bash
CPU_CACHE_GB=16 BLOCK_SIZE=16 \
PREFIX_TIER_TOKENS=16384 PREFIX_TIER_STEP_TOKENS=16384 PREFIX_MAX_TIERS=8 \
./server.sh cpu-sessions 0 8000 <model>
```

For filesystem profiles, set `FS_ROOT`, `FS_CACHE_GB`, and optionally
`FS_POLICY` (for example, `FS_POLICY=lru`) to adjust the FS tier independently
of the CPU tier.

### GPU profiles

The GPU and cpu-gpu profiles set the `VLLM_SUFFIX_*` variables for the server process.
They use `PREFIX_TIER_TOKENS`, `PREFIX_TIER_STEP_TOKENS`,
`PREFIX_MAX_TIERS`, and `GPU_IDLE_TTL_S`. Session-aware profiles also use
`SESSION_DEAD_TTL_S`, `SESSION_DETECTOR`, `SESSION_EWA_ALPHA`,
`SESSION_DEAD_K`, `SESSION_MIN_GAPS`, `SESSION_QUANTILE`, and
`SESSION_GAP_WINDOW`.

Inspect a generated profile without starting a server:

```bash
CONFIG_ONLY=1 ./server.sh cpu-sessions 0 8000 <model>
CONFIG_ONLY=1 ./server.sh cpu-fs-sessions 0 8000 <model>
CONFIG_ONLY=1 ./server.sh gpu-sessions-global 0 8000 <model>
```

The CPU commands print the connector JSON; GPU commands print the resolved
eviction environment variables. `DRY_RUN=1 VLLM_BIN=/bin/echo ...` prints the
actual launch command without starting the model.

### Sending stable session IDs

Use the same `session_id` on each request belonging to the same conversation.
For example, the Chat Completions API accepts the field:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<model>",
    "session_id": "conversation-42",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Alternatively, the serving code can obtain session identity from the
`x-session-id` header. Use a stable, non-empty identifier; a new ID for every
turn prevents the liveness tracker from correlating requests.

## 6. Configuration reference

### CPU/FS connector profiles

The CPU settings are in `--kv-transfer-config` / `kv_connector_extra_config`:

| Setting | Purpose |
| --- | --- |
| `eviction_policy` | CPU policy: `prefix_pruning` or `prefix_pruning_on_sessions`. |
| `cpu_bytes_to_use` | CPU tier budget, in bytes across workers. `server.sh` exposes `CPU_CACHE_GB`. |
| `first_tier_tokens` | Width of the first prefix tier. |
| `tier_step_tokens` | Width added for each later tier. |
| `max_tiers` | Number of prefix tiers before falling back to the LRU tail; zero means uncapped. |
| `session_dead_ttl_s` | Minimum idle threshold; zero disables session promotion. |
| `session_detector` | `ewa` per-session detector or `global` gap quantile detector. |
| `session_ewa_alpha`, `session_dead_k` | EWA smoothing and deviation multiplier. |
| `session_min_gaps` | Minimum observed gaps before adapting the threshold. |
| `session_quantile`, `session_gap_window` | Global detector quantile and sample-window size. |

These fields may be supplied at the top level or in
`eviction_policy_config`. `cpu-prefix` selects
`eviction_policy: "prefix_pruning"` and forces a zero session TTL.
`cpu-sessions` selects `"prefix_pruning_on_sessions"` and uses the configured
session TTL.

For filesystem profiles, the settings inside the `fs` entry in
`secondary_tiers` are independent of the CPU policy. In particular, the FS
`eviction_policy` and `capacity_bytes` determine its eviction behavior and
capacity bound.

### GPU environment variables

| Variable | Purpose |
| --- | --- |
| `VLLM_SUFFIX_PRESERVING_EVICTION` | Enables GPU prefix-first eviction. |
| `VLLM_SUFFIX_EVICTION_PREFIX_FRACTION` | Fraction of GPU cache blocks managed as prefix-pruning tiers. |
| `VLLM_SUFFIX_EVICTION_TIER_TOKENS` | Initial tier width. |
| `VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS` | Growth per subsequent tier. |
| `VLLM_SUFFIX_EVICTION_MAX_TIERS` | Maximum prefix tiers; zero leaves them uncapped. |
| `VLLM_SUFFIX_EVICTION_IDLE_TTL_S` | Time before old free blocks are demoted by idle TTL. |
| `VLLM_SUFFIX_SESSION_DEAD_TTL_S` | Minimum idle time before a finished session can be promoted for eviction; zero disables session promotion. |
| `VLLM_SUFFIX_SESSION_DETECTOR` | `ewa` or `global`. |
| `VLLM_SUFFIX_SESSION_EWA_ALPHA`, `VLLM_SUFFIX_SESSION_DEAD_K` | Per-session EWA parameters. |
| `VLLM_SUFFIX_SESSION_MIN_GAPS` | Minimum gaps before adaptive detection. |
| `VLLM_SUFFIX_SESSION_QUANTILE`, `VLLM_SUFFIX_SESSION_GAP_WINDOW` | Global gap-quantile parameters. |

The provided GPU profiles default to a 60-second idle-demotion TTL. The
session-aware GPU profiles set a positive session-dead TTL. The script allows
overrides; see `./server.sh --help`.

## 7. Limitations and interpretation

- GPU suffix-preserving eviction can run with `OffloadingConnector`; the
  `cpu-gpu-*` profiles enable both tiers. It remains incompatible with
  speculative decoding, context/pipeline parallelism, and encoder-decoder
  models in this implementation. The scheduler checks these constraints.
- GPU and CPU/FS settings are not interchangeable: GPU controls are
  environment variables; CPU and FS controls are connector JSON.
- Session-aware behavior depends on stable session IDs and completed turns.
  Unidentified requests can use plain prefix pruning but cannot be grouped into
  sessions.
- The policy is an eviction preference, not a guarantee that a suffix remains
  resident; capacity pressure and other cache lifecycle rules still apply.
- GPU session promotion runs after idle demotion so a later demotion does not
  undo the dead-session priority ordering.
- A GPU cache reset clears liveness state and per-block session/freed metadata.

## 8. Validation performed and remaining environment issues

Focused checks on this branch completed:

- CPU prefix-pruning policy tests: **13 passed**.
- GPU suffix-prefix eviction and dead-session precedence tests: **7 passed**.
- Combined focused CPU/GPU pruning selection: **20 passed**.
- Scheduler session-arrival and suffix-gap recomputation tests: **2 passed**.
- Connector suffix-run lookup tests: **5 passed**.
- Combined connector/GPU suffix-reload variants: **4 passed** with optional
  Humming detection disabled in the test process. The installed Python 3.14
  `humming` package is incompatible with this checkout (missing `float4e0m3`)
  and otherwise breaks the CPU-backend test fixture before test execution.
  These variants exercise the connector's mocked offload worker with GPU
  prefix pruning enabled; they are not a live-model serving test.
- Python compile checks and `git diff --check`: passed.

The larger attempted test selection did not pass as a whole. It included
unrelated hybrid-cache/configuration failures and Python/environment-dependent
failures. A live-model server combining the GPU and CPU policies has not been
launched in this environment. The focused suffix lookup unit tests passed;
validate the combined profiles with the target runtime and hardware before
deployment.

Before deployment, run the tests under the repository's supported Python and
dependency versions, and validate with the target model, connector backend,
GPU hardware, block size, and memory budgets.

## 9. Useful related files

- `server.sh` — profile launcher and inline configuration help.
- `docs/features/kv_offloading_usage.md` — connector and CPU-tier settings.
- `docs/features/kv_offload_prefix_pruning_lookup.md` — suffix lookup,
  gap-recompute, and GPU pruning details.
- `vllm/v1/core/session_liveness.py` — shared session gap/liveness tracking.
- `vllm/v1/core/block_pool.py` — GPU tier queue, session tagging, and
  dead-session promotion.
- `vllm/v1/core/sched/scheduler.py` — GPU eviction setup and scheduler
  lifecycle integration.
- `tests/v1/kv_connector/unit/offloading_connector/test_suffix_repro.py` —
  connector suffix-reload coverage with GPU pruning both disabled and enabled.
- `vllm/v1/kv_offload/eviction/prefix_pruning_core.py` — shared CPU/FS scoring
  and session-liveness primitives.
- `vllm/v1/kv_offload/cpu/policies/prefix_pruning.py` — CPU policy variants.
- `tests/v1/core/test_prefix_caching.py`,
  `tests/v1/core/test_scheduler.py`, and
  `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py` —
  focused GPU lifecycle, recomputation, and suffix-lookup tests.
