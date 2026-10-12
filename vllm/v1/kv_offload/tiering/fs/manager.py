# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FileSystemTierManager: pure-Python filesystem tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then os.replace'd to the final path (without .tmp).

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import functools
import json
import os
import time
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

try:
    from vllm.fs_io_C import batch_lookup as batch_lookup_C

    _HAS_BATCH_LOOKUP_C = True
except ImportError:
    _HAS_BATCH_LOOKUP_C = False

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.eviction.prefix_pruning_core import (
    PrefixTierScorer,
    SessionLivenessTracker,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.backpressure import BackpressureDetector
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
    TransferJob,
)
from vllm.v1.kv_offload.tiering.fs.eviction import StorageEvictionIndex
from vllm.v1.kv_offload.tiering.fs.io import (
    batch_delete_block,
    batch_load_block,
    batch_store_block,
    probe_o_direct,
)
from vllm.v1.kv_offload.tiering.fs.lru_eviction import StorageLRUIndex
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

# Eviction policy names that enable the storage prefix-pruning index.
_PREFIX_PRUNING_POLICIES = ("prefix_pruning", "prefix_pruning_on_sessions")
# Policy that enables LRU eviction when capacity_bytes is set.
_LRU_POLICIES = ("lru",)
# Policy that enables reject-on-full (no eviction) when capacity_bytes is set.
_REJECT_POLICIES = ("reject", "none")

StorageEvictionBackend = StorageEvictionIndex | StorageLRUIndex

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


@dataclass
class RejectStats:
    """Cumulative reject-on-full counters for the hard capacity guard."""

    blocks_rejected: int = 0
    blocks_accepted: int = 0
    stores_full_reject: int = 0  # jobs where every new write was dropped
    stores_partial_reject: int = 0  # jobs that stored some keys, dropped others


class _HardCapacityGuard:
    """Session-local hard byte budget that rejects stores over capacity.

    Unlike ``StorageEvictionIndex``, this never deletes resident blocks.
    New keys that would exceed ``capacity_bytes`` are dropped; keys already
    counted against the budget may be re-stored. Capacity is reserved when a
    store is accepted and released if that store fails before commit.
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
        # key -> resident (True once a store has committed)
        self._keys: dict[OffloadKey, bool] = {}
        self._stats = RejectStats()

    @property
    def capacity_bytes(self) -> int:
        return self._capacity_bytes

    @property
    def max_keys(self) -> int:
        return self._max_keys

    @property
    def tracked_chunks(self) -> int:
        return len(self._keys)

    @property
    def tracked_bytes(self) -> int:
        return len(self._keys) * self._block_size_bytes

    @property
    def stats(self) -> RejectStats:
        return self._stats

    def reserve(self, keys: list[OffloadKey]) -> list[OffloadKey]:
        """Accept keys that fit; reserve capacity for newly accepted keys."""
        accepted: list[OffloadKey] = []
        for key in keys:
            if key in self._keys:
                accepted.append(key)
                continue
            if len(self._keys) >= self._max_keys:
                continue
            self._keys[key] = False
            accepted.append(key)
        n_rejected = len(keys) - len(accepted)
        if n_rejected:
            self._stats.blocks_rejected += n_rejected
            if not accepted:
                self._stats.stores_full_reject += 1
            else:
                self._stats.stores_partial_reject += 1
        self._stats.blocks_accepted += len(accepted)
        return accepted

    def on_store_committed(self, keys: Iterable[OffloadKey]) -> None:
        for key in keys:
            if key in self._keys:
                self._keys[key] = True

    def on_store_failed(self, keys: Iterable[OffloadKey]) -> None:
        for key in keys:
            resident = self._keys.get(key)
            if resident is False:
                del self._keys[key]


class FsAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for FileSystemTierManager."""

    def __init__(
        self,
        tier: "FileSystemTierManager",
        tier_type: str,
    ) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        paths = [self._tier.file_mapper.get_file_name(k) for k in keys]
        if _HAS_BATCH_LOOKUP_C:
            # C extension: GIL released for the entire faccessat() batch.
            return batch_lookup_C(paths)
        return (os.path.exists(p) for p in paths)


class FileSystemTierManager(SecondaryTierManager):
    """Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        KV cache sharing between multiple vLLM instances using the same
        ``root_dir`` (e.g., via a shared PVC) works by default: ``NONE_HASH``
        (the chain-hash seed for block content hashes) is derived from a fixed
        default seed, so identical token content produces identical block
        filenames across instances. Setting the ``PYTHONHASHSEED`` environment
        variable to the same value on all instances overrides the default seed,
        and is required to share a cache when using a non-cryptographic
        prefix-caching hash algorithm, which seeds ``NONE_HASH`` randomly.
    """

    medium: ClassVar[Medium] = Medium.STORAGE

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        enable_kv_events: bool = False,
        locality: str | None = None,
        backpressure_detector: BackpressureDetector | None = None,
        capacity_bytes: int | None = None,
        eviction_policy: str = "lru",
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
    ):
        """Args:
        offloading_spec: Contains normalized offloading configuration and
            blocks_per_chunk.
        primary_kv_view: Memoryview of the primary tier's CPU KV cache.
        tier_type: Tier type identifier, set by SecondaryTierFactory.
        root_dir: Root directory for block files.
        n_read_threads: Number of read-priority I/O threads.
        n_write_threads: Number of write-priority I/O threads.
        enable_kv_events: Emit BlockStored KV events for blocks
            successfully stored to this tier. Effective only when KV
            cache events are enabled globally (kv_events_config).
        locality: Whether this tier's storage is LOCAL or REMOTE relative
            to the publishing vLLM instance.
        backpressure_detector: Optional backpressure detector.
        capacity_bytes: Byte budget for this tier. When set with an eviction
            policy (``lru`` or prefix-pruning), stores beyond the budget evict
            resident blocks. When set with ``reject`` / ``none``, stores that
            would exceed the budget are rejected. When None (default), the tier
            is unbounded.
        eviction_policy: ``"lru"`` (default) enables LRU eviction;
            ``"prefix_pruning"`` / ``"prefix_pruning_on_sessions"`` enables
            session prefix-pruning eviction; ``"reject"`` / ``"none"`` enables
            reject-on-full. All require ``capacity_bytes``.
        first_tier_tokens, tier_step_tokens, max_tiers, session_dead_ttl_s,
        session_ewa_alpha, session_dead_k, session_min_gaps, session_detector,
        session_quantile, session_gap_window: prefix-pruning knobs, matching
            the CPU-tier PrefixPruningOnSessionsCachePolicy defaults.

        """
        super().__init__(
            offloading_spec, primary_kv_view, tier_type, backpressure_detector
        )
        self.locality = Locality(locality) if locality is not None else None

        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}
        # Keys of in-flight store jobs, tracked only when eviction is enabled.
        # Used to feed the eviction index on completion and to protect blocks
        # with an in-flight write from being chosen as eviction victims.
        self._inflight_store_keys: dict[JobId, list[OffloadKey]] = {}
        # Keys of in-flight load (promotion) jobs, so a failed load can mark
        # its own cached lookup verdicts False (see get_finished_jobs).
        self._load_job_keys: dict[JobId, list[OffloadKey]] = {}
        # Block count per in-flight job, used to report transfer_bytes.
        self._job_block_counts: dict[JobId, int] = {}
        # Per load job: how many blocks loaded before a failure (partial keep).
        # Written by the pool worker inside the load task before it raises (so
        # before task_done publishes the job); read on the scheduler thread in
        # get_finished_jobs only for job ids the finished queue returned. Under
        # the GIL that read cannot observe the finished job without the prior
        # write, so no extra lock is needed (get_finished is itself lock-free).
        self._load_progress: dict[JobId, int] = {}

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        # Byte budget: eviction index (_index: LRU or prefix-pruning) or
        # reject-on-full (_capacity_guard). Both None => unbounded.
        self._index: StorageEvictionBackend | None = None
        self._capacity_guard: _HardCapacityGuard | None = None
        self._build_capacity(
            capacity_bytes=capacity_bytes,
            eviction_policy=eviction_policy,
            first_tier_tokens=first_tier_tokens,
            tier_step_tokens=tier_step_tokens,
            max_tiers=max_tiers,
            session_dead_ttl_s=session_dead_ttl_s,
            session_ewa_alpha=session_ewa_alpha,
            session_dead_k=session_dead_k,
            session_min_gaps=session_min_gaps,
            session_detector=session_detector,
            session_quantile=session_quantile,
            session_gap_window=session_gap_window,
        )
        # Stores rejected by the hard capacity guard complete synchronously so
        # the tiering manager can unpin primary chunks without touching disk.
        self._rejected_store_jobs: list[tuple[JobId, bool, float]] = []
        # Accepted keys for capacity-guarded store jobs (may be a subset of the
        # TransferJob keys when some writes were rejected for capacity).
        self._guard_store_keys: dict[JobId, list[OffloadKey]] = {}
        # Fire-and-forget delete jobs reuse the thread pool but must not be
        # surfaced to the tiering manager as transfer JobResults; they are
        # tagged with negative ids and filtered out in get_finished_jobs.
        self._delete_job_ids: set[int] = set()
        self._next_delete_job_id: int = -1

        # Eviction / reject logging is throttled: the first event always logs at
        # INFO, then at most once per interval, so a full tier does not flood.
        self._eviction_log_interval_s: float = 30.0
        self._eviction_log_count: int = 0
        self._last_eviction_log_t: float = 0.0
        self._reject_log_count: int = 0
        self._last_reject_log_t: float = 0.0

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=True,
        )

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        # Prefer O_DIRECT to bypass the page cache, but fall back to buffered
        # I/O on filesystems that reject it (e.g. overlayfs, some NFS mounts)
        # rather than failing every block.
        self._use_o_direct = probe_o_direct(os.path.dirname(config_path))
        if not self._use_o_direct:
            logger.warning(
                "O_DIRECT is not supported at '%s'; falling back to buffered "
                "I/O for the '%s' KV offload tier.",
                root_dir,
                tier_type,
            )

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)

    def _build_capacity(
        self,
        capacity_bytes: int | None,
        eviction_policy: str,
        first_tier_tokens: int,
        tier_step_tokens: int,
        max_tiers: int,
        session_dead_ttl_s: float,
        session_ewa_alpha: float,
        session_dead_k: float,
        session_min_gaps: int,
        session_detector: str,
        session_quantile: float,
        session_gap_window: int,
    ) -> None:
        if capacity_bytes is None:
            if (
                eviction_policy in _PREFIX_PRUNING_POLICIES
                or eviction_policy in _LRU_POLICIES
                or eviction_policy in _REJECT_POLICIES
            ):
                # "lru" is the constructor default; only warn for an explicit
                # non-default policy that cannot take effect without a budget.
                if eviction_policy != "lru":
                    logger.warning(
                        "eviction_policy '%s' is set on secondary tier '%s' but "
                        "capacity_bytes is not; the tier stays unbounded (no "
                        "eviction).",
                        eviction_policy,
                        self.tier_type,
                    )
            return

        if eviction_policy in _LRU_POLICIES:
            self._index = StorageLRUIndex(
                block_size_bytes=self._block_size,
                capacity_bytes=capacity_bytes,
            )
            logger.info(
                "Storage tier '%s' LRU eviction enabled: "
                "capacity=%d bytes (%d blocks).",
                self.tier_type,
                capacity_bytes,
                self._index.max_keys,
            )
            return

        if eviction_policy in _REJECT_POLICIES:
            self._capacity_guard = _HardCapacityGuard(
                block_size_bytes=self._block_size,
                capacity_bytes=capacity_bytes,
            )
            logger.info(
                "Storage tier '%s' hard capacity enabled (reject-on-full): "
                "capacity=%d bytes (%d blocks).",
                self.tier_type,
                capacity_bytes,
                self._capacity_guard.max_keys,
            )
            return

        if eviction_policy in _PREFIX_PRUNING_POLICIES:
            scorer = PrefixTierScorer(
                first_tier_tokens=first_tier_tokens,
                tier_step_tokens=tier_step_tokens,
                max_tiers=max_tiers,
            )
            liveness = SessionLivenessTracker(
                session_dead_ttl_s=session_dead_ttl_s,
                session_ewa_alpha=session_ewa_alpha,
                session_dead_k=session_dead_k,
                session_min_gaps=session_min_gaps,
                session_detector=session_detector,
                session_quantile=session_quantile,
                session_gap_window=session_gap_window,
            )
            self._index = StorageEvictionIndex(
                block_size_bytes=self._block_size,
                capacity_bytes=capacity_bytes,
                scorer=scorer,
                liveness=liveness,
            )
            logger.info(
                "Storage tier '%s' prefix-pruning eviction enabled: "
                "capacity=%d bytes (%d blocks).",
                self.tier_type,
                capacity_bytes,
                self._index.max_keys,
            )
            return

        raise ValueError(
            f"Unknown storage eviction_policy {eviction_policy!r} on tier "
            f"'{self.tier_type}'. Supported: 'lru', 'reject'/'none', "
            f"'prefix_pruning'/'prefix_pruning_on_sessions'."
        )

    def _protected_keys(self) -> set[OffloadKey]:
        """Keys with in-flight store/load I/O; never safe to delete."""
        protected: set[OffloadKey] = set()
        for in_flight in self._inflight_store_keys.values():
            protected.update(in_flight)
        for in_flight in self._load_job_keys.values():
            protected.update(in_flight)
        return protected

    def _maybe_evict(self) -> None:
        """Delete resident blocks over the byte budget (async, best-effort)."""
        assert self._index is not None
        n = self._index.overflow()
        if n <= 0:
            return
        victims = self._index.select_victims(n, self._protected_keys())
        if not victims:
            # Over budget but every candidate is pinned by in-flight I/O; the
            # tier is transiently above capacity until those stores complete.
            logger.debug(
                "Storage tier '%s' is over capacity by %d block(s) but all "
                "candidates have in-flight I/O; deferring eviction.",
                self.tier_type,
                n,
            )
            return
        self._index.forget(victims)
        self._lookup_manager.invalidate(victims)
        paths = [self.file_mapper.get_file_name(key) for key in victims]
        self._enqueue_delete(paths)
        self._log_eviction(len(victims))

    def _log_eviction(self, n_evicted: int) -> None:
        """Log an eviction event: always at DEBUG, throttled at INFO."""
        assert self._index is not None
        stats = self._index.stats
        logger.debug(
            "Storage tier '%s' evicted %d block(s) to disk budget.",
            self.tier_type,
            n_evicted,
        )
        now = time.monotonic()
        first = self._eviction_log_count == 0
        self._eviction_log_count += 1
        since_last = now - self._last_eviction_log_t
        if not first and since_last < self._eviction_log_interval_s:
            return
        self._last_eviction_log_t = now
        if isinstance(self._index, StorageEvictionIndex):
            logger.info(
                "Storage tier '%s' eviction active: evicted %d block(s) this "
                "pass; cumulative total=%d (dead_session=%d, live_prefix=%d, "
                "lru_tail=%d); occupancy=%d/%d blocks (%d/%d bytes).",
                self.tier_type,
                n_evicted,
                stats.total,
                stats.dead_session,
                stats.live_prefix,
                stats.lru_tail,
                self._index.tracked_chunks,
                self._index.max_keys,
                self._index.tracked_bytes,
                self._index.capacity_bytes,
            )
        else:
            logger.info(
                "Storage tier '%s' LRU eviction active: evicted %d block(s) "
                "this pass; cumulative total=%d; occupancy=%d/%d blocks "
                "(%d/%d bytes).",
                self.tier_type,
                n_evicted,
                stats.total,
                self._index.tracked_chunks,
                self._index.max_keys,
                self._index.tracked_bytes,
                self._index.capacity_bytes,
            )

    def _enqueue_delete(self, paths: list[str]) -> None:
        job_id = self._next_delete_job_id
        self._next_delete_job_id -= 1
        self._delete_job_ids.add(job_id)
        self._pool.enqueue_store(
            job_id, 1, [functools.partial(batch_delete_block, paths)]
        )

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        if self._index is not None:
            self._index.touch(keys, req_context)

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    def _log_capacity_reject(self, n_rejected: int) -> None:
        """Log a capacity reject: always at DEBUG, throttled at INFO.

        Mirrors ``_log_eviction``: first reject always logs at INFO, then at
        most once per ``_eviction_log_interval_s`` (default 30s).
        """
        assert self._capacity_guard is not None
        stats = self._capacity_guard.stats
        logger.debug(
            "Storage tier '%s' rejected %d block store(s) (at capacity).",
            self.tier_type,
            n_rejected,
        )
        now = time.monotonic()
        first = self._reject_log_count == 0
        self._reject_log_count += 1
        since_last = now - self._last_reject_log_t
        if not first and since_last < self._eviction_log_interval_s:
            return
        self._last_reject_log_t = now
        logger.info(
            "Storage tier '%s' capacity reject active: rejected %d block(s) "
            "this pass; cumulative rejected=%d accepted=%d "
            "(full_reject_stores=%d, partial_reject_stores=%d); "
            "occupancy=%d/%d blocks (%d/%d bytes).",
            self.tier_type,
            n_rejected,
            stats.blocks_rejected,
            stats.blocks_accepted,
            stats.stores_full_reject,
            stats.stores_partial_reject,
            self._capacity_guard.tracked_chunks,
            self._capacity_guard.max_keys,
            self._capacity_guard.tracked_bytes,
            self._capacity_guard.capacity_bytes,
        )

    @override
    def submit_store(self, job_metadata: TransferJob) -> None:
        keys = list(job_metadata.keys)
        chunk_ids = list(job_metadata.chunk_ids)
        if self._capacity_guard is not None:
            accepted = self._capacity_guard.reserve(keys)
            n_rejected = len(keys) - len(accepted)
            if n_rejected:
                self._log_capacity_reject(n_rejected)
            if not accepted:
                # No room: finish the job immediately so primary pins release.
                self._rejected_store_jobs.append((job_metadata.job_id, False, 0.0))
                self._job_block_counts[job_metadata.job_id] = 0
                return
            if n_rejected:
                accepted_set = set(accepted)
                chunk_ids = [
                    cid for key, cid in zip(keys, chunk_ids) if key in accepted_set
                ]
                keys = accepted
            self._guard_store_keys[job_metadata.job_id] = keys
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = keys
        if self._index is not None:
            self._inflight_store_keys[job_metadata.job_id] = keys
            self._index.note_store(keys, job_metadata.req_context)
        task = functools.partial(
            batch_store_block,
            [self.file_mapper.get_file_name(key) for key in keys],
            self._primary_kv_view,
            [int(cid) * self._block_size for cid in chunk_ids],
            self._block_size,
            self._use_o_direct,
        )
        self._job_block_counts[job_metadata.job_id] = len(keys)
        self._pool.enqueue_store(job_metadata.job_id, 1, [task])
        if self._index is not None:
            self._maybe_evict()

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        job_id = job_metadata.job_id
        # Track this load's keys so a failed promotion can mark only its failed
        # keys as a miss (see get_finished_jobs).
        keys = list(job_metadata.keys)
        self._load_job_keys[job_id] = keys
        self._job_block_counts[job_id] = len(keys)
        paths = [self.file_mapper.get_file_name(key) for key in keys]
        offsets = [int(cid) * self._block_size for cid in job_metadata.chunk_ids]

        def load_task() -> None:
            try:
                batch_load_block(
                    paths,
                    self._primary_kv_view,
                    offsets,
                    self._block_size,
                    self._use_o_direct,
                )
            except OSError as exc:
                # Runs on the pool worker thread. Record how many blocks loaded
                # before the failure so get_finished_jobs can keep them; this
                # write precedes task_done, so the scheduler reads it safely
                # under the GIL once the finished queue hands back this job.
                num_succeeded = getattr(exc, "num_succeeded", 0)
                self._load_progress[job_id] = num_succeeded
                # Surfaces errno (e.g. EMFILE "Too many open files") for both
                # the C and Python load paths.
                logger.debug(
                    "Load of %d blocks for job %s failed at block %d: %s",
                    len(paths),
                    job_id,
                    num_succeeded,
                    exc,
                )
                raise

        self._pool.enqueue_load(job_id, 1, [load_task])

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Collect finished jobs; a failed promotion marks only its failed keys
        as a miss here (scheduler thread)."""
        results = []
        # Capacity rejects complete without touching the thread pool.
        while self._rejected_store_jobs:
            job_id, success, transfer_time = self._rejected_store_jobs.pop(0)
            block_count = self._job_block_counts.pop(job_id, 0)
            results.append(
                JobResult(
                    job_id=job_id,
                    success=success,
                    transfer_time=transfer_time,
                    transfer_bytes=block_count * self._block_size
                    if block_count
                    else None,
                )
            )
        for job_id, success, transfer_time in self._pool.get_finished():
            if job_id in self._delete_job_ids:
                # Fire-and-forget eviction delete; not a tracked transfer job.
                self._delete_job_ids.discard(job_id)
                continue
            if self._index is not None:
                store_keys = self._inflight_store_keys.pop(job_id, None)
                if store_keys is not None:
                    if success:
                        self._index.on_store_committed(store_keys)
                    else:
                        self._index.on_store_failed(store_keys)
            if self._capacity_guard is not None:
                guard_keys = self._guard_store_keys.pop(job_id, None)
                if guard_keys is not None:
                    if success:
                        self._capacity_guard.on_store_committed(guard_keys)
                    else:
                        self._capacity_guard.on_store_failed(guard_keys)
            block_count = self._job_block_counts.pop(job_id, 0)
            transfer_bytes = block_count * self._block_size if block_count else None
            if self.events is not None:
                keys = self._store_job_keys.pop(job_id, None)
                if success and keys:
                    self.events.append(
                        OffloadingEvent(
                            keys=keys,
                            medium=self.medium,
                            removed=False,
                            locality=self.locality,
                        )
                    )
            load_keys = self._load_job_keys.pop(job_id, None)
            num_succeeded = self._load_progress.pop(job_id, 0)
            if load_keys is not None and not success:
                # A batched load stops at the first bad block and reports how
                # many loaded before it. Those earlier blocks are kept in the
                # primary tier (reported via successful_keys); only this block
                # and the ones after it are marked a miss and recomputed.
                successful = load_keys[:num_succeeded]
                failed = load_keys[num_succeeded:]
                self._lookup_manager.mark_miss(failed)
                results.append(
                    JobResult(
                        job_id=job_id,
                        success=False,
                        successful_keys=tuple(successful) if successful else None,
                        transfer_time=transfer_time,
                        transfer_bytes=transfer_bytes,
                    )
                )
                continue
            results.append(
                JobResult(
                    job_id=job_id,
                    success=success,
                    transfer_time=transfer_time,
                    transfer_bytes=transfer_bytes,
                )
            )
        return results

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)
        if self._index is not None:
            self._index.on_request_finished(req_context)

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()

    @override
    def shutdown(self) -> None:
        """Release resources held by this tier.

        Shuts down the lookup manager and the thread pool,
        clearing pending tasks and waiting for active threads to complete.
        """
        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)
