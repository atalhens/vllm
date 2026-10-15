# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU cache eviction that peels deep prefix tiers before hot prefixes."""

from __future__ import annotations

import heapq
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from typing_extensions import override

from vllm.v1.core.session_liveness import SessionLivenessTracker
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy, ChunkStatus
from vllm.v1.kv_offload.eviction.prefix_pruning_core import (
    UNKNOWN_POSITION,
    PrefixTierScorer,
    resolve_session_id,
)


class PrefixPruningOnSessionsCachePolicy(CachePolicy):
    supports_suffix_pruning = True

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
            first_tier_tokens, tier_step_tokens, max_tiers
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
        self._positions: dict[OffloadKey, int] = {}
        self._sessions: dict[OffloadKey, str] = {}
        self._session_keys: dict[str, set[OffloadKey]] = defaultdict(set)
        self._evictable: set[OffloadKey] = set()
        self._next_rank = 0

    @property
    def first_tier_tokens(self) -> int:
        return self._scorer.first_tier_tokens

    @property
    def tier_step_tokens(self) -> int:
        return self._scorer.tier_step_tokens

    @property
    def max_tiers(self) -> int:
        return self._scorer.max_tiers

    @override
    def get(self, key: OffloadKey) -> ChunkStatus | None:
        return self.chunks.get(key)

    @override
    def insert(self, key: OffloadKey, chunk: ChunkStatus) -> None:
        self.chunks[key] = chunk
        self._next_rank += 1
        self._ranks[key] = self._next_rank
        if chunk.ref_cnt == 0:
            self._evictable.add(key)

    def _forget(self, key: OffloadKey) -> None:
        session_id = self._sessions.pop(key, None)
        if session_id:
            keys = self._session_keys[session_id]
            keys.discard(key)
            if not keys:
                self._session_keys.pop(session_id, None)
        self._positions.pop(key, None)
        self._ranks.pop(key, None)
        self._evictable.discard(key)

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.chunks[key]
        self._forget(key)

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        session_id = resolve_session_id(req_context)
        for key in keys:
            if key not in self.chunks:
                continue
            position = req_context.get_offload_key_position(key)
            if position is not None:
                self._positions[key] = position
            if session_id:
                old_session = self._sessions.get(key)
                if old_session and old_session != session_id:
                    self._session_keys[old_session].discard(key)
                self._sessions[key] = session_id
                self._session_keys[session_id].add(key)

    @override
    def on_new_request(self, req_context: ReqContext) -> None:
        self._liveness.on_arrival(resolve_session_id(req_context), self._time_fn())

    @override
    def on_chunks_stored(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> None:
        self.touch(keys, req_context)

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        self._liveness.on_finish(resolve_session_id(req_context), self._time_fn())

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        self._evictable.add(key)
        self._next_rank += 1
        self._ranks[key] = self._next_rank

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        self._evictable.discard(key)

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        if n == 0:
            return []
        now = self._time_fn()
        dead = self._liveness.dead_sessions(now)
        candidates = []
        for key in self._evictable - protected:
            session_id = self._sessions.get(key)
            candidates.append(
                self._scorer.score(
                    self._positions.get(key, UNKNOWN_POSITION),
                    self._ranks.get(key, 0),
                    key,
                    session_id is not None and session_id in dead,
                )
            )
        if len(candidates) < n:
            return None
        evicted = []
        for *_, key in heapq.nsmallest(n, candidates):
            chunk = self.chunks.pop(key)
            self._forget(key)
            evicted.append((key, chunk))
        return evicted

    @override
    def clear(self) -> None:
        self.chunks.clear()
        self._ranks.clear()
        self._positions.clear()
        self._sessions.clear()
        self._session_keys.clear()
        self._evictable.clear()
        self._liveness.clear()
        self._next_rank = 0


class PrefixPruningCachePolicy(PrefixPruningOnSessionsCachePolicy):
    def __init__(
        self,
        cache_capacity: int,
        first_tier_tokens: int = 16384,
        tier_step_tokens: int = 16384,
        max_tiers: int = 8,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("session_dead_ttl_s", None)
        super().__init__(
            cache_capacity=cache_capacity,
            first_tier_tokens=first_tier_tokens,
            tier_step_tokens=tier_step_tokens,
            max_tiers=max_tiers,
            session_dead_ttl_s=0.0,
            **kwargs,
        )
