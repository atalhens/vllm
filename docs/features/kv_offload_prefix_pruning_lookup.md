# Prefix-Pruning Lookup: Gap Recompute and Suffix Reload

This document describes how the [`OffloadingConnector`](kv_offloading_usage.md)
**looks up, reloads, and recomputes** KV when storage-tier (or CPU-tier)
**prefix pruning** has deleted a shallow prefix while leaving a deeper suffix
cached. It is the read-path companion to
[Storage-Tier Eviction](kv_offload_storage_eviction.md), which covers *what* gets
deleted; this guide covers *what happens on the next request* that needs those
blocks.

For the GPU-side analogue (suffix-preserving eviction with
`VLLM_SUFFIX_PRESERVING_EVICTION`), see the same gap/jump contract in
`FullAttentionManager.find_suffix_hit` — this offload path mirrors that
semantics without enabling GPU suffix eviction.

---

## Why maximal prefix matching is not enough

Standard offload lookup is **maximal contiguous prefix**: walk chunk keys from
the start of the request and stop at the first `MISS`. That is correct when
eviction (or capacity pressure) removes a **suffix** of the cached sequence, or
when the entire prefix is still present:

```text
Need:     A B C D E
Cached:   A B C · ·     → load ABC, recompute DE   (contiguous prefix)
Cached:   · · · · ·     → load nothing, recompute all
```

**Prefix pruning** deliberately peels **shallow** blocks first (shared prompts,
early context). After pressure, the resident set often looks like:

```text
Need:     A B C D E
Cached:   · · C D E     → AB gone, CDE still on disk/CPU
```

A pure maximal-prefix scan sees `A = MISS` and returns **0 hits**, even though
`CDE` are still valuable. Without a second pass, the engine would recompute the
entire prompt and ignore the cached suffix.

The logic below closes that hole: **find the first contiguous hit run after the
prefix miss, load only that suffix, recompute the gap, then jump past the
loaded tokens.**

---

## End-to-end picture (ABCDE example)

Assume one full-attention group, one token-chunk per letter (`A`…`E`), local GPU
prefix cache empty, and storage holding only `C`,`D`,`E`.

```mermaid
sequenceDiagram
    participant Sched as Scheduler
    participant Off as OffloadingConnector
    participant Stor as Storage/CPU tier
    participant GPU as GPU KV blocks

    Sched->>Off: get_num_new_matched_tokens(local=0)
    Off->>Stor: maximal prefix lookup (A)
    Stor-->>Off: MISS
    Off->>Stor: suffix scan (B…E)
    Stor-->>Off: MISS, HIT, HIT, HIT  (CDE)
    Off-->>Sched: external=E (span through E), async load
    Note over Off: suffix_gap_end=C, suffix_end=E+1

    Sched->>GPU: allocate slots for A…E
    Sched->>Off: update_state_after_alloc(external=E)
    Off->>Stor: prepare_load(CDE only)
    Off->>GPU: copy CDE into blocks C…E
    Note over Sched: WAITING_FOR_REMOTE_KVS

    Stor-->>Off: load complete
    Off-->>Sched: finished_recving
    Sched->>GPU: register suffix hashes; num_computed=0
    Note over Sched: _suffix_hits=(C,E+1), _offload_suffix_ready

    Sched->>GPU: schedule gap AB (recompute)
    Note over Sched: after gap scheduled, jump num_computed → E+1
    Sched->>GPU: continue prefill/decode past E
```

Token layout after a successful path:

| Region | Tokens | Source | Action |
| --- | --- | --- | --- |
| Gap | `[0, gap_end)` e.g. AB | Not in storage | **Recompute** on GPU |
| Suffix | `[gap_end, suffix_end)` e.g. CDE | Storage/CPU hit | **Load** into GPU, then **jump** |
| Remainder | `[suffix_end, prompt)` | — | Prefill as usual |

---

## Phase 1 — Lookup

**Where:** `OffloadingConnectorScheduler._lookup_complete_chunks` →
`_maximal_prefix_lookup` → (on prefix miss) `_find_suffix_hit`

**Entry:** Main scheduler calls `connector.get_num_new_matched_tokens(request,
local_hit)` while admitting a waiting request with
`num_computed_tokens == 0`.

### 1.1 Maximal prefix (unchanged)

For full-attention groups, scan offload keys from the local computed boundary:

| Lookup result | Effect |
| --- | --- |
| `HIT` / `HIT_PENDING` | Extend contiguous prefix; pending defers the whole lookup |
| `RETRY` | Keep scanning (kick async lookups) but defer the answer |
| `MISS` | **Stop** the prefix scan |

If the contiguous prefix length is `N > 0`, the **old path** runs: return
`N * tokens_per_chunk` (subject to caps / eagle pop / multi-group tighten). No
suffix search.

### 1.2 Suffix search (prefix-pruning path)

Only when the contiguous prefix length is **0**, and only when all of:

- the group is full-attention (`sliding_window_size_in_chunks is None`);
- the connector has **no** sliding-window lookup groups;
- the group is not in an unverified eagle/MTP pop.

…then `_find_suffix_hit` scans **after** the miss that ended the prefix
(starting at local index `1` when prefix hits are `0`):

1. Skip leading `MISS`es (the gap, e.g. `B` if `A` already missed).
2. Take the **first** contiguous run of `HIT` / `HIT_PENDING`.
3. Stop that run at the next `MISS` (or defer on `RETRY` / pending).

Returns:

| Result | Meaning |
| --- | --- |
| `None` | Backend deferred → scheduler retries later (`external is None`) |
| `(0, 0)` | No later run → same as today: **0** external tokens |
| `(start, count)` | Suffix found at local chunk `start` for `count` chunks |

Absolute token bounds stored on `RequestOffloadState`:

```text
suffix_gap_end = (start_chunk_idx + start) * tokens_per_chunk
suffix_end     = round_down(min(cap, gap_end + count * tokens_per_chunk),
                            tokens_per_chunk)
```

**Returned external token count** is `suffix_end - local`, **not** just the
suffix length. The span intentionally includes the gap so the scheduler
**allocates GPU blocks for A…E** in one shot (gap slots empty, suffix slots to
be filled by the load).

Partial-tail extension is **skipped** when a suffix hit is active (the suffix
already ends on a chunk boundary; extending past a peeled gap is undefined).

### 1.3 What other eviction policies see

| Storage state | Path |
| --- | --- |
| Contiguous prefix `ABC··` | Maximal prefix only (suffix search never runs) |
| Full miss `·····` | Prefix 0, suffix `(0,0)` → 0 external |
| Hole `··CDE` (prefix pruning) | Prefix 0, suffix hit → gap+suffix path |
| LRU/reject leaving a contiguous prefix | Unchanged maximal-prefix behavior |

Eviction policy code is not branched here; the hole path activates only when
the **resident key set** has a gap after a leading miss.

### 1.4 Current scope limits

- **Unitary full-attention** offload groups only. Hybrid SWA / Mamba /
  multi-group convergence still require a contiguous prefix (suffix search
  disabled).
- Suffix search runs only after a **complete** prefix miss (`num_hit_chunks ==
  0`), not after a short non-zero prefix followed by a later hole.
- Mirrors GPU `find_suffix_hit` intent for the common peel shape (shallow
  gone, deep kept).

---

## Phase 2 — Allocate and plan the load

**Where:** Main scheduler `allocate_slots` →
`connector.update_state_after_alloc` →
`OffloadingConnectorScheduler.update_state_after_alloc`

### 2.1 Allocation

With `load_kv_async=True` and `num_external_computed_tokens = suffix_end -
local`:

- GPU blocks are reserved through `suffix_end` (gap + suffix).
- `num_new_tokens = 0` for this step (no forward while loading).
- Request enters `WAITING_FOR_REMOTE_KVS` with
  `num_computed_tokens = suffix_end` (optimistic bound until the transfer
  finishes).

### 2.2 Load only the suffix

`update_state_after_alloc` normally starts loading at the first fresh block
after the local hit. For a suffix hit it **overrides** the start:

```text
load_start_gpu_block_idx = suffix_gap_end // tokens_per_block
```

So only keys/blocks in `[suffix_gap_end, suffix_end)` are passed to
`prepare_load`. Gap blocks stay allocated but empty.

`load_key_ranges` record the loaded ranges for hit-source accounting (gap tokens
are not counted as external hits in `VLLM_KV_CONNECTOR_HIT_LOG` stats; only
`suffix_end - gap_end` is).

### 2.3 Store-index bookkeeping

For `OffloadPolicy.CHUNK_LEVEL`, a normal contiguous hit advances
`next_stored_chunk_idx` past the loaded prefix so those chunks are not
re-stored. For a **suffix** hit that advance is **skipped**: the gap must remain
eligible to store after recompute. Already-resident suffix keys are skipped by
`prepare_store` when they are still on the tier.

---

## Phase 3 — After the async load

**Where:** `Scheduler._update_waiting_for_remote_kv`

On successful finish, if the connector still has a pending offload suffix
(`take_offload_suffix_hit`):

1. **`register_offload_suffix_hit(gap_end, suffix_end)`**
   (`KVCacheManager`):
   - Hash the loaded suffix blocks into the GPU prefix-cache pool.
   - Do **not** mark the empty gap as cached.
   - Set `FullAttentionManager._suffix_hit_reqs[req] = (gap_end_block,
     suffix_end_block)` so a later `cache_blocks` of the recomputed gap jumps
     past the suffix (same contract as a GPU suffix hit).
2. Set `request.num_computed_tokens = 0` (start of gap).
3. Record `scheduler._suffix_hits[req] = (gap_end, suffix_end)`.
4. Add `req` to `scheduler._offload_suffix_ready`.

On load failure, suffix state is cleared and the usual invalid-block / free
path runs.

Request status returns to `WAITING` (or `PREEMPTED`).

---

## Phase 4 — Gap recompute and jump

**Where:** next `Scheduler.schedule` for that request, then
`_update_after_schedule`

### 4.1 Resume without a second lookup

If `req_id in _offload_suffix_ready` and `num_computed_tokens == 0`:

- Skip GPU prefix lookup and connector `get_num_new_matched_tokens` (avoids
  re-loading CDE).
- Keep existing allocated blocks.
- Treat `suffix_hit = (gap_end, suffix_end)` like a GPU suffix hit:
  `num_new_tokens = gap_end - num_computed` (schedule only the gap).
- Stay in `_offload_suffix_ready` until `allocate_slots` succeeds so a
  watermark reject can retry this resume path.

### 4.2 Forward pass on the gap

The worker recomputes tokens in `[0, gap_end)` into the already-allocated gap
blocks. Attention can read the loaded suffix once the gap is filled and the
computed cursor jumps (standard causal prefill with KV already present for
later positions after the jump).

### 4.3 Jump past the suffix

In `_update_after_schedule`, after advancing by the scheduled gap tokens:

```text
if num_computed_tokens >= suffix_hit[0]:   # gap finished
    num_computed_tokens = suffix_hit[1]    # jump to suffix_end
    drop _suffix_hits[req]
```

Subsequent steps prefill/decode from `suffix_end` onward. When the gap is
cached via `cache_blocks`, `_suffix_hit_reqs` causes the manager to cache the
gap then mark the suffix range already cached (no double-hash of CDE).

---

## State summary

| Object | Field | Role |
| --- | --- | --- |
| `RequestOffloadState` | `suffix_gap_end`, `suffix_end` | Set at lookup; peeked for stats; taken after load |
| `OffloadingConnector` | `get_offload_suffix_hit` / `take_offload_suffix_hit` | Scheduler-facing API (also forwarded by `MultiConnector`) |
| `Scheduler` | `_suffix_hits` | `(gap_end, suffix_end)` until gap scheduled |
| `Scheduler` | `_offload_suffix_ready` | Resume gate: blocks loaded, skip second lookup |
| `FullAttentionManager` | `_suffix_hit_reqs` | Prefix-cache bookkeeping for gap then suffix |

---

## Comparison to contiguous offload hits

| Step | Contiguous prefix hit (`ABC··`) | Suffix hit after peel (`··CDE`) |
| --- | --- | --- |
| Lookup | Maximal prefix length | Prefix 0 + first later run |
| External tokens reported | Prefix length | Span through `suffix_end` (gap+suffix) |
| Blocks loaded | From local boundary through hit | Only `[gap_end, suffix_end)` |
| After load | `num_computed = hit_end` | `num_computed = 0`, `_suffix_hits` set |
| Next compute | Continue after hit | Recompute gap, then jump to `suffix_end` |
| Stats (`external=`) | Loaded prefix tokens | `suffix_end - gap_end` only |

---

## Observability

With `VLLM_KV_CONNECTOR_HIT_LOG=1`, the main scheduler logs (same gate as
ordinary connector hit lines):

```text
Prefix cache hits: req_id=… device=0 external=<suffix_tokens> by_source=…
Offload suffix hit: req_id=… suffix_tokens=… gap_tokens=…
  gap=[local, gap_end) suffix=[gap_end, suffix_end)
```

After the transfer completes:

```text
Offload suffix loaded: req_id=… suffix_tokens=… gap_tokens=…
  gap=[0, gap_end) suffix=[gap_end, suffix_end); will recompute gap then jump
```

Debug-level detail also exists inside the offloading scheduler at discovery
time (`offload suffix hit: gap … load …`).

---

## Code map

| Concern | Primary location |
| --- | --- |
| Suffix discovery | `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` (`_find_suffix_hit`, `_lookup_complete_chunks`) |
| Load plan | same file (`update_state_after_alloc`, `take_suffix_hit` / `peek_suffix_hit`) |
| Connector API | `…/offloading_connector.py`, `…/multi_connector.py` |
| Post-load resume + jump | `vllm/v1/core/sched/scheduler.py` (`_update_waiting_for_remote_kv`, waiting admit, `_update_after_schedule`) |
| Prefix-cache registration | `vllm/v1/core/kv_cache_manager.py` (`register_offload_suffix_hit`) |
| Unit tests | `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py` (`TestOffloadSuffixHit`) |

## GPU prefix-pruning profiles

Set `VLLM_SUFFIX_PRESERVING_EVICTION=1` to enable GPU prefix-first eviction.
`VLLM_SUFFIX_EVICTION_TIER_TOKENS` and
`VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS` set the initial tier and its growth;
`VLLM_SUFFIX_EVICTION_MAX_TIERS` caps prefix tiers; and
`VLLM_SUFFIX_EVICTION_IDLE_TTL_S` enables idle demotion. For session-aware
eviction, set `VLLM_SUFFIX_SESSION_DEAD_TTL_S` to a positive value. Finished
sessions past that floor are promoted ahead of live prefixes. The detector
defaults to per-session EWA (`VLLM_SUFFIX_SESSION_DETECTOR=ewa`), with alpha
0.3, `k=3`, and two gaps required before adapting the threshold. Requests need
a session ID for this policy to distinguish sessions. `session_dead_ttl_s=0`
leaves plain prefix pruning enabled without session promotion.

From the repository root, [`server.sh`](../../server.sh) provides runnable
`gpu-prefix`, `gpu-sessions`, and `gpu-sessions-global` profiles, alongside
CPU/offload profiles. The GPU profiles use the device-cache suffix lookup and
gap-recompute path and must not be launched with an `OffloadingConnector`.

This GPU eviction path is incompatible with KV connectors, speculative
decoding, context/pipeline parallelism, and encoder-decoder models. The
connector's own suffix lookup and gap-recompute path remains available when
GPU suffix-preserving eviction is disabled.

---

## Related docs

- [KV Offloading Usage](kv_offloading_usage.md) — connector config, tiers, basic load/store
- [Storage-Tier Eviction](kv_offload_storage_eviction.md) — prefix-pruning eviction order and knobs
- GPU suffix-preserving eviction (`VLLM_SUFFIX_PRESERVING_EVICTION`) — same
  gap/jump idea on the device cache; incompatible with KV connectors when that
  GPU feature is enabled, which is why the offload path implements the hole
  handling inside the connector instead
