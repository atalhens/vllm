# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-bounded session prefix-pruning eviction index for the storage tier.

Unlike the CPU primary tier, the filesystem tier has no ref-counted physical
slots: each key is one block-sized file on disk. This index tracks the keys
written this session, enforces a byte budget (``capacity_bytes``), and selects
eviction victims using the same order as the CPU
``PrefixPruningOnSessionsCachePolicy`` via the shared
``vllm.v1.kv_offload.eviction.prefix_pruning_core`` primitives:

* every chunk of a session predicted dead, still in prefix-tier order
* live prefix tiers, low end-token tier first
* the LRU tail: chunks past ``max_tiers``, older first

All methods run on the scheduler thread (no internal locking). The owning
``FileSystemTierManager`` is responsible for the actual file deletion and for
excluding keys with in-flight I/O from the ``protected`` set.
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.eviction.prefix_pruning_core import (
    UNKNOWN_POSITION,
    PrefixTierScorer,
    SessionLivenessTracker,
    resolve_session_id,
)


@dataclass
class _Entry:
    position: int = UNKNOWN_POSITION
    session_id: str | None = None
    rank: int = 0
    # False while the store is still in flight, True once committed to disk.
    resident: bool = False


@dataclass
class EvictionStats:
    """Cumulative eviction counters, broken down by scoring bucket."""

    total: int = 0
    dead_session: int = 0  # bucket 0: blocks of a session predicted dead
    live_prefix: int = 0  # bucket 1: live prefix tiers
    lru_tail: int = 0  # bucket 2: recency tail past max_tiers


class StorageEvictionIndex:
    """Tracks storage-tier keys and picks prefix-pruning eviction victims.

    Capacity is a byte budget. Every key occupies one uniform block, so the
    budget reduces to a resident-key cap of ``capacity_bytes // block_size``.
    A key counts against the budget from the moment it is noted (store
    submitted) so concurrent stores cannot overshoot the cap.
    """

    def __init__(
        self,
        block_size_bytes: int,
        capacity_bytes: int,
        scorer: PrefixTierScorer,
        liveness: SessionLivenessTracker,
    ) -> None:
        if block_size_bytes < 1:
            raise ValueError(
                f"block_size_bytes must be >= 1, got {block_size_bytes}"
            )
        if capacity_bytes < block_size_bytes:
            raise ValueError(
                "capacity_bytes must be at least one block "
                f"({block_size_bytes}), got {capacity_bytes}"
            )
        self._block_size_bytes = block_size_bytes
        self._capacity_bytes = capacity_bytes
        self._max_keys = capacity_bytes // block_size_bytes
        self._scorer = scorer
        self._liveness = liveness
        self._time_fn = time.monotonic

        self._chunks: dict[OffloadKey, _Entry] = {}
        self._next_rank = 0
        self._stats = EvictionStats()

    @property
    def capacity_bytes(self) -> int:
        return self._capacity_bytes

    @property
    def max_keys(self) -> int:
        return self._max_keys

    @property
    def tracked_chunks(self) -> int:
        return len(self._chunks)

    @property
    def tracked_bytes(self) -> int:
        return len(self._chunks) * self._block_size_bytes

    @property
    def stats(self) -> EvictionStats:
        return self._stats

    def _note_metadata(self, entry: _Entry, key: OffloadKey, req_context: ReqContext):
        position = req_context.get_offload_key_position(key)
        if position is not None:
            entry.position = position
        sid = resolve_session_id(req_context)
        if sid:
            entry.session_id = sid

    def note_store(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        """Register keys for an in-flight store and record their metadata.

        New keys are assigned a fresh recency rank and counted against the
        budget. Known keys only refresh their position/session metadata.
        """
        for key in keys:
            entry = self._chunks.get(key)
            if entry is None:
                self._next_rank += 1
                entry = _Entry(rank=self._next_rank)
                self._chunks[key] = entry
            self._note_metadata(entry, key, req_context)

    def on_store_committed(self, keys: Iterable[OffloadKey]) -> None:
        for key in keys:
            entry = self._chunks.get(key)
            if entry is not None:
                entry.resident = True

    def on_store_failed(self, keys: Iterable[OffloadKey]) -> None:
        """Drop keys whose store failed before committing to disk.

        A key that was already resident from an earlier successful store is
        kept: the failed job did not remove its file.
        """
        for key in keys:
            entry = self._chunks.get(key)
            if entry is not None and not entry.resident:
                del self._chunks[key]

    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        """Record a session arrival and refresh metadata; rank is unchanged."""
        now = self._time_fn()
        self._liveness.on_arrival(resolve_session_id(req_context), now)
        for key in keys:
            entry = self._chunks.get(key)
            if entry is not None:
                self._note_metadata(entry, key, req_context)

    def on_request_finished(self, req_context: ReqContext) -> None:
        now = self._time_fn()
        self._liveness.on_finish(resolve_session_id(req_context), now)

    def forget(self, keys: Iterable[OffloadKey]) -> None:
        for key in keys:
            self._chunks.pop(key, None)

    def overflow(self) -> int:
        """How many keys exceed the budget right now (>= 0)."""
        return max(0, len(self._chunks) - self._max_keys)

    def select_victims(
        self, n: int, protected: set[OffloadKey]
    ) -> list[OffloadKey]:
        """Return up to ``n`` resident keys to evict, lowest score first.

        Only resident (on-disk) keys are candidates; keys with in-flight I/O
        must be passed in ``protected``. Best-effort: returns fewer than ``n``
        if too few candidates are available.
        """
        if n <= 0:
            return []
        now = self._time_fn()
        dead = self._liveness.dead_sessions(now)
        scored: list[tuple[int, int, int, OffloadKey]] = []
        for key, entry in self._chunks.items():
            if not entry.resident or key in protected:
                continue
            is_dead = entry.session_id is not None and entry.session_id in dead
            scored.append(
                self._scorer.score(entry.position, entry.rank, key, is_dead)
            )
        picked = heapq.nsmallest(min(n, len(scored)), scored)
        for bucket, _tier, _rank, _key in picked:
            self._stats.total += 1
            if bucket == 0:
                self._stats.dead_session += 1
            elif bucket == 1:
                self._stats.live_prefix += 1
            else:
                self._stats.lru_tail += 1
        return [key for *_rest, key in picked]

    def clear(self) -> None:
        self._chunks.clear()
        self._liveness.clear()
        self._next_rank = 0
