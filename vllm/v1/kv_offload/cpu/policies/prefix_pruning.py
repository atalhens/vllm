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
"""

from __future__ import annotations

import heapq
import math
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import (
    CachePolicy,
    ChunkStatus,
    order_request_keys,
)

_UNKNOWN_POSITION = 1 << 30


def resolve_session_id(req_context: ReqContext) -> str | None:
    """Session id from ReqContext, then kv_transfer_params."""
    if req_context.session_id:
        return req_context.session_id
    params = req_context.kv_transfer_params
    if params is not None:
        value = params.get("session_id")
        if isinstance(value, str) and value:
            return value
    return None


@dataclass
class _SessionState:
    last_finish: float | None = None
    last_arrival: float | None = None
    n_gaps: int = 0
    ewa_gap: float = 0.0
    ewa_gap_sq: float = 0.0

    @property
    def std(self) -> float:
        var = max(0.0, self.ewa_gap_sq - self.ewa_gap * self.ewa_gap)
        return math.sqrt(var)


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
        if first_tier_tokens < 1:
            raise ValueError(f"first_tier_tokens must be >= 1, got {first_tier_tokens}")
        if tier_step_tokens < 1:
            raise ValueError(f"tier_step_tokens must be >= 1, got {tier_step_tokens}")
        if max_tiers < 0:
            raise ValueError(f"max_tiers must be >= 0, got {max_tiers}")
        if session_dead_ttl_s < 0:
            raise ValueError(
                f"session_dead_ttl_s must be >= 0, got {session_dead_ttl_s}"
            )
        if not 0.0 < session_ewa_alpha <= 1.0:
            raise ValueError(
                f"session_ewa_alpha must be in (0, 1], got {session_ewa_alpha}"
            )
        if session_dead_k < 0:
            raise ValueError(f"session_dead_k must be >= 0, got {session_dead_k}")
        if session_min_gaps < 0:
            raise ValueError(f"session_min_gaps must be >= 0, got {session_min_gaps}")
        if session_detector not in ("ewa", "global"):
            raise ValueError(
                f"session_detector must be 'ewa' or 'global', got {session_detector!r}"
            )
        if not 0.0 < session_quantile < 1.0:
            raise ValueError(
                f"session_quantile must be in (0, 1), got {session_quantile}"
            )
        if session_gap_window < 1:
            raise ValueError(
                f"session_gap_window must be >= 1, got {session_gap_window}"
            )
        self.first_tier_tokens = first_tier_tokens
        self.tier_step_tokens = tier_step_tokens
        self.max_tiers = max_tiers
        self.session_dead_ttl_s = session_dead_ttl_s
        self.session_ewa_alpha = session_ewa_alpha
        self.session_dead_k = session_dead_k
        self.session_min_gaps = session_min_gaps
        self.session_detector = session_detector
        self.session_quantile = session_quantile
        self.session_gap_window = session_gap_window
        self._time_fn = time.monotonic

        self.chunks: dict[OffloadKey, ChunkStatus] = {}
        self._ranks: dict[OffloadKey, int] = {}
        self._position: dict[OffloadKey, int] = {}
        self._key_session: dict[OffloadKey, str] = {}
        self._session_keys: dict[str, set[OffloadKey]] = defaultdict(set)
        self._sessions: dict[str, _SessionState] = {}
        self._global_gaps: deque[float] = deque(maxlen=session_gap_window)
        self._evictable: set[OffloadKey] = set()
        self._next_rank = 0

    def _tier(self, position: int) -> int:
        """Prefix tier of a chunk end-token, matching GPU block tiers.

        A chunk whose end token is ``first_tier_tokens`` is still tier 0.
        ``max_tiers`` is applied by the scorer, which sends that tier and
        everything past it to the LRU tail.
        """
        if position >= _UNKNOWN_POSITION:
            return _UNKNOWN_POSITION
        if position <= self.first_tier_tokens:
            return 0
        return 1 + (position - self.first_tier_tokens - 1) // self.tier_step_tokens

    def _is_tail(self, tier: int) -> bool:
        return self.max_tiers > 0 and tier >= self.max_tiers

    def _score(
        self, key: OffloadKey, dead: set[str]
    ) -> tuple[int, int, int, OffloadKey]:
        """Dead session, then live prefix tier, then the LRU tail."""
        position = self._position.get(key, _UNKNOWN_POSITION)
        tier = self._tier(position)
        rank = self._ranks.get(key, 0)
        sid = self._key_session.get(key)
        is_dead = sid is not None and sid in dead
        if is_dead:
            return (0, tier, rank, key)
        if self._is_tail(tier):
            return (2, 0, rank, key)
        return (1, tier, rank, key)

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

    def _observe_gap(self, state: _SessionState, gap: float) -> None:
        """Match ``SessionLivenessTracker._observe_gap`` on the GPU."""
        self._global_gaps.append(gap)
        if state.n_gaps == 0:
            state.ewa_gap = gap
            state.ewa_gap_sq = gap * gap
        else:
            a = self.session_ewa_alpha
            state.ewa_gap = a * gap + (1.0 - a) * state.ewa_gap
            state.ewa_gap_sq = a * gap * gap + (1.0 - a) * state.ewa_gap_sq
        state.n_gaps += 1

    def _on_arrival(self, sid: str | None, now: float) -> None:
        if not sid or self.session_dead_ttl_s <= 0:
            return
        state = self._sessions.setdefault(sid, _SessionState())
        # A second touch before finish is the same turn. A finish catches
        # up to the arrival, so the next touch is a new turn and records
        # one gap. ``>=`` would swallow that gap when both share a timestamp.
        same_turn = (
            state.last_arrival is not None
            and state.last_finish is not None
            and state.last_arrival > state.last_finish
        )
        if state.last_finish is not None and not same_turn:
            self._observe_gap(state, max(0.0, now - state.last_finish))
        state.last_arrival = now

    def _on_finish(self, sid: str | None, now: float) -> None:
        if not sid or self.session_dead_ttl_s <= 0:
            return
        state = self._sessions.setdefault(sid, _SessionState())
        state.last_finish = now

    def _threshold(self, state: _SessionState) -> float:
        if self.session_detector == "global":
            if len(self._global_gaps) < self.session_min_gaps:
                return self.session_dead_ttl_s
            return max(self.session_dead_ttl_s, self._global_quantile())
        if state.n_gaps < self.session_min_gaps:
            return self.session_dead_ttl_s
        return max(
            self.session_dead_ttl_s,
            state.ewa_gap + self.session_dead_k * state.std,
        )

    def _global_quantile(self) -> float:
        if not self._global_gaps:
            return self.session_dead_ttl_s
        xs = sorted(self._global_gaps)
        idx = int((len(xs) - 1) * self.session_quantile)
        return xs[idx]

    def _dead_sessions(self, now: float) -> set[str]:
        if self.session_dead_ttl_s <= 0:
            return set()
        dead: set[str] = set()
        for sid, state in self._sessions.items():
            if state.last_finish is None:
                continue
            if (
                state.last_arrival is not None
                and state.last_arrival >= state.last_finish
            ):
                continue
            if now - state.last_finish >= self._threshold(state):
                dead.add(sid)
        return dead

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
        self._on_arrival(sid, now)
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
        self._on_finish(sid, now)
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
        self._sessions.clear()
        self._global_gaps.clear()
        self._evictable.clear()
        self._next_rank = 0

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        if n == 0:
            return []
        now = self._time_fn()
        dead = self._dead_sessions(now)
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
