# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-bounded LRU eviction index for the storage tier.

Tracks keys written this session against a ``capacity_bytes`` budget and
selects the least-recently-used resident keys for deletion. Shares the same
scheduler-thread contract as ``StorageEvictionIndex`` (prefix pruning) so
``FileSystemTierManager`` can drive either policy through one code path:

* a key counts against the budget from ``note_store`` (store submitted)
* only resident keys without in-flight I/O are eviction candidates
* ``touch`` / re-store bump recency so recently used blocks survive

All methods run on the scheduler thread (no internal locking). The owning
manager performs file deletion and supplies the ``protected`` set.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.v1.kv_offload.base import OffloadKey, ReqContext


@dataclass
class _Entry:
    rank: int = 0
    # False while the store is still in flight, True once committed to disk.
    resident: bool = False


@dataclass
class LRUEvictionStats:
    """Cumulative LRU eviction counter."""

    total: int = 0


class StorageLRUIndex:
    """Tracks storage-tier keys and picks LRU eviction victims.

    Capacity is a byte budget. Every key occupies one uniform block, so the
    budget reduces to a resident-key cap of ``capacity_bytes // block_size``.
    A key counts against the budget from the moment it is noted (store
    submitted) so concurrent stores cannot overshoot the cap.
    """

    def __init__(self, block_size_bytes: int, capacity_bytes: int) -> None:
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

        self._chunks: dict[OffloadKey, _Entry] = {}
        self._next_rank = 0
        self._stats = LRUEvictionStats()

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
    def stats(self) -> LRUEvictionStats:
        return self._stats

    def _bump_rank(self, entry: _Entry) -> None:
        self._next_rank += 1
        entry.rank = self._next_rank

    def note_store(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        """Register keys for an in-flight store; new/re-stored keys are MRU."""
        del req_context  # LRU has no session/position metadata
        for key in keys:
            entry = self._chunks.get(key)
            if entry is None:
                entry = _Entry()
                self._chunks[key] = entry
            self._bump_rank(entry)

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
        """Mark known keys as most-recently used."""
        del req_context
        for key in keys:
            entry = self._chunks.get(key)
            if entry is not None:
                self._bump_rank(entry)

    def on_request_finished(self, req_context: ReqContext) -> None:
        del req_context

    def forget(self, keys: Iterable[OffloadKey]) -> None:
        for key in keys:
            self._chunks.pop(key, None)

    def overflow(self) -> int:
        """How many keys exceed the budget right now (>= 0)."""
        return max(0, len(self._chunks) - self._max_keys)

    def select_victims(
        self, n: int, protected: set[OffloadKey]
    ) -> list[OffloadKey]:
        """Return up to ``n`` resident keys to evict, oldest (lowest rank) first.

        Only resident (on-disk) keys are candidates; keys with in-flight I/O
        must be passed in ``protected``. Best-effort: returns fewer than ``n``
        if too few candidates are available.
        """
        if n <= 0:
            return []
        scored: list[tuple[int, OffloadKey]] = []
        for key, entry in self._chunks.items():
            if not entry.resident or key in protected:
                continue
            scored.append((entry.rank, key))
        picked = heapq.nsmallest(min(n, len(scored)), scored)
        self._stats.total += len(picked)
        return [key for _rank, key in picked]

    def clear(self) -> None:
        self._chunks.clear()
        self._next_rank = 0
