# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU offload policy for prefix tiers plus session affinity.

The CPU cache is bounded by ``cpu_bytes_to_use`` (a chunk capacity).
When that capacity is full, eviction follows the GPU
``vllm-prefix-pruning-on-sessions`` order, without the GPU idle-TTL
demote. Idle demote is a small-GPU pressure valve; on CPU it would
peel a live prefix before a dead session's suffix.

Order, lowest first:

* every chunk of a session predicted dead, still in prefix-tier order
* live prefix tiers, low end-token tier first
* the LRU tail: chunks past ``max_tiers``, older free first

A session is dead when the time since its last finish exceeds
``max(session_dead_ttl_s, ewa + k·std)``. The gap is measured from
that finish to the next arrival. The first gap seeds the average.
Fewer than ``session_min_gaps`` samples keep the static floor. A turn
still in flight is not dead.

``session_dead_ttl_s=0`` disables the wipe. ``max_tiers=0`` leaves
the prefix tiers uncapped.

The eviction order (session liveness and prefix-tier scoring) lives in
``vllm.v1.kv_offload.eviction.prefix_pruning_core`` and is shared with the
storage tier's ``StorageEvictionIndex``.
"""

from __future__ import annotations

import heapq
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import (
    CachePolicy,
    ChunkStatus,
    order_request_keys,
)
from vllm.v1.kv_offload.eviction.prefix_pruning_core import (
    UNKNOWN_POSITION,
    PrefixTierScorer,
    SessionLivenessTracker,
    resolve_session_id,
)

__all__ = [
    "PrefixPruningOnSessionsCachePolicy",
    "PrefixPruningCachePolicy",
    "resolve_session_id",
]


class PrefixPruningOnSessionsCachePolicy(CachePolicy):
    """Prefix-first peel plus session-dead wipe.

    Defaults follow the GPU sessions arm's tiers and think-time
    detector: 16384-token steps, 8 tiers, 200s floor, EWA alpha 0.3,
    k 3, and 2 gaps before the adaptive threshold.
    """

    def __init__(
        self,
        cache_capacity: int,
        first_tier_tokens: int = 16384,
        tier_step_tokens: int = 16384,
        max_tiers: int = 8,
        session_dead_ttl_s: float = 200.0,
        session_ewa_alpha: float = 0.3,
        session_dead_k: float = 3.0,
        session_min_gaps: int = 2,
        session_detector: str = "ewa",
        session_quantile: float = 0.95,
        session_gap_window: int = 256,
    ) -> None:
        super().__init__(cache_capacity)
        self._scorer = PrefixTierScorer(
            first_tier_tokens=first_tier_tokens,
            tier_step_tokens=tier_step_tokens,
            max_tiers=max_tiers,
        )
        self._liveness = SessionLivenessTracker(
            session_dead_ttl_s=session_dead_ttl_s,
            session_ewa_alpha=session_ewa_alpha,
            session_dead_k=session_dead_k,
            session_min_gaps=session_min_gaps,
            session_detector=session_detector,
            session_quantile=session_quantile,
            session_gap_window=session_gap_window,
        )
        self._time_fn = time.monotonic

        self.chunks: dict[OffloadKey, ChunkStatus] = {}
        self._ranks: dict[OffloadKey, int] = {}
        self._position: dict[OffloadKey, int] = {}
        self._key_session: dict[OffloadKey, str] = {}
        self._session_keys: dict[str, set[OffloadKey]] = defaultdict(set)
        self._evictable: set[OffloadKey] = set()
        self._next_rank = 0

    # Introspection accessors kept for config reporting and tests; the
    # authoritative values live in the scorer and liveness tracker.
    @property
    def first_tier_tokens(self) -> int:
        return self._scorer.first_tier_tokens

    @property
    def tier_step_tokens(self) -> int:
        return self._scorer.tier_step_tokens

    @property
    def max_tiers(self) -> int:
        return self._scorer.max_tiers

    @property
    def session_dead_ttl_s(self) -> float:
        return self._liveness.session_dead_ttl_s

    @property
    def session_ewa_alpha(self) -> float:
        return self._liveness.session_ewa_alpha

    @property
    def session_dead_k(self) -> float:
        return self._liveness.session_dead_k

    @property
    def session_min_gaps(self) -> int:
        return self._liveness.session_min_gaps

    @property
    def session_detector(self) -> str:
        return self._liveness.session_detector

    @property
    def session_quantile(self) -> float:
        return self._liveness.session_quantile

    @property
    def session_gap_window(self) -> int:
        return self._liveness.session_gap_window

    def _score(
        self, key: OffloadKey, dead: set[str]
    ) -> tuple[int, int, int, OffloadKey]:
        position = self._position.get(key, UNKNOWN_POSITION)
        rank = self._ranks.get(key, 0)
        sid = self._key_session.get(key)
        is_dead = sid is not None and sid in dead
        return self._scorer.score(position, rank, key, is_dead)

    def _assign_new_rank(self, key: OffloadKey) -> None:
        if key not in self.chunks:
            return
        self._next_rank += 1
        self._ranks[key] = self._next_rank

    def _note_position(self, key: OffloadKey, req_context: ReqContext) -> None:
        position = req_context.get_offload_key_position(key)
        if position is not None:
            self._position[key] = position

    def _tag_session(self, key: OffloadKey, sid: str | None) -> None:
        if not sid or key not in self.chunks:
            return
        prev = self._key_session.get(key)
        if prev is not None and prev != sid:
            self._session_keys[prev].discard(key)
        self._key_session[key] = sid
        self._session_keys[sid].add(key)

    def _forget_key(self, key: OffloadKey) -> None:
        sid = self._key_session.pop(key, None)
        if sid is not None:
            keys = self._session_keys.get(sid)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    del self._session_keys[sid]
        self._position.pop(key, None)
        self._ranks.pop(key, None)
        self._evictable.discard(key)

    @override
    def get(self, key: OffloadKey) -> ChunkStatus | None:
        return self.chunks.get(key)

    @override
    def insert(self, key: OffloadKey, chunk: ChunkStatus) -> None:
        self.chunks[key] = chunk
        self._assign_new_rank(key)
        if chunk.ref_cnt == 0:
            self._evictable.add(key)

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.chunks[key]
        self._forget_key(key)

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        """Record a session arrival. Does not refresh eviction rank.

        On the GPU, a hit removes the block from the free queue via its
        refcount. Rank changes when the chunk is freed, not when it is
        touched.
        """
        now = self._time_fn()
        sid = resolve_session_id(req_context)
        self._liveness.on_arrival(sid, now)
        for key in keys:
            self._note_position(key, req_context)
            self._tag_session(key, sid)

    @override
    def on_request_finished(
        self,
        key_groups: Sequence[Sequence[OffloadKey]],
        insertion_only_keys: set[OffloadKey],
        reused_keys: set[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        del insertion_only_keys, reused_keys
        now = self._time_fn()
        sid = resolve_session_id(req_context)
        self._liveness.on_finish(sid, now)
        for key in order_request_keys(key_groups, req_context):
            self._note_position(key, req_context)
            self._tag_session(key, sid)
            self._assign_new_rank(key)

    @override
    def clear(self) -> None:
        self.chunks.clear()
        self._ranks.clear()
        self._position.clear()
        self._key_session.clear()
        self._session_keys.clear()
        self._liveness.clear()
        self._evictable.clear()
        self._next_rank = 0

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        if n == 0:
            return []
        now = self._time_fn()
        dead = self._liveness.dead_sessions(now)
        scored: list[tuple[int, int, int, OffloadKey]] = []
        for key in self._evictable:
            if key in protected:
                continue
            scored.append(self._score(key, dead))
        if len(scored) < n:
            return None
        picked = heapq.nsmallest(n, scored)
        candidates: list[tuple[OffloadKey, ChunkStatus]] = []
        for *_rest, key in picked:
            chunk = self.chunks[key]
            assert chunk.ref_cnt == 0
            del self.chunks[key]
            self._forget_key(key)
            candidates.append((key, chunk))
        return candidates

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        self._evictable.add(key)

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        self._evictable.discard(key)


# Live-benchmark / extra_config alias: same class.
PrefixPruningCachePolicy = PrefixPruningOnSessionsCachePolicy
