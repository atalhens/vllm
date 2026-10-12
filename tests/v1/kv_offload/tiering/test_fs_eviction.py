# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Storage-tier session prefix-pruning eviction.

Unit tests for StorageEvictionIndex ordering/budget/in-flight protection, plus
integration tests that a byte-bounded FileSystemTierManager deletes victim
files, re-probes them as MISS, and never deletes blocks with in-flight I/O.
"""

import mmap
import os
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingKVEventsConfig,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.eviction.prefix_pruning_core import (
    PrefixTierScorer,
    SessionLivenessTracker,
)
from vllm.v1.kv_offload.tiering.base import TransferJob
from vllm.v1.kv_offload.tiering.fs.eviction import StorageEvictionIndex
from vllm.v1.kv_offload.tiering.fs.lru_eviction import StorageLRUIndex
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

_BLOCK = 100


def key(n: int) -> OffloadKey:
    return make_offload_key(n.to_bytes(8, "big"), 0)


def _ctx(session_id=None, positions=None) -> ReqContext:
    ctx = ReqContext(req_id="r", session_id=session_id)
    for k, pos in (positions or {}).items():
        ctx.set_offload_key_position(k, pos)
    return ctx


def _index(capacity_blocks: int, *, ttl: float = 0.0, max_tiers: int = 8):
    return StorageEvictionIndex(
        block_size_bytes=_BLOCK,
        capacity_bytes=capacity_blocks * _BLOCK,
        scorer=PrefixTierScorer(
            first_tier_tokens=16, tier_step_tokens=16, max_tiers=max_tiers
        ),
        liveness=SessionLivenessTracker(session_dead_ttl_s=ttl),
    )


def _store_committed(index: StorageEvictionIndex, keys, ctx: ReqContext) -> None:
    index.note_store(keys, ctx)
    index.on_store_committed(keys)


# ---------------------------------------------------------------------------
# StorageEvictionIndex unit tests
# ---------------------------------------------------------------------------


def test_capacity_derives_block_cap():
    index = _index(4)
    assert index.max_keys == 4
    assert index.capacity_bytes == 4 * _BLOCK


def test_capacity_below_one_block_rejected():
    with pytest.raises(ValueError, match="at least one block"):
        StorageEvictionIndex(
            block_size_bytes=_BLOCK,
            capacity_bytes=_BLOCK - 1,
            scorer=PrefixTierScorer(),
            liveness=SessionLivenessTracker(),
        )


def test_evicts_leading_prefix_before_suffix():
    index = _index(8)
    k0, k1, k2, k3 = key(0), key(1), key(2), key(3)
    _store_committed(
        index, [k0, k1, k2, k3], _ctx("s", {k0: 16, k1: 32, k2: 48, k3: 64})
    )
    assert index.select_victims(2, protected=set()) == [k0, k1]


def test_overflow_counts_keys_over_budget():
    index = _index(2)
    _store_committed(index, [key(0), key(1)], _ctx("s", {key(0): 16, key(1): 32}))
    assert index.overflow() == 0
    index.note_store([key(2)], _ctx("s", {key(2): 48}))
    assert index.overflow() == 1


def test_in_flight_keys_are_protected():
    index = _index(1)
    hot, cold = key(0), key(1)
    _store_committed(index, [cold], _ctx("s", {cold: 16}))
    index.note_store([hot], _ctx("s", {hot: 32}))  # in flight
    # Over budget by one; hot is in flight (protected), so cold is the victim.
    assert index.overflow() == 1
    assert index.select_victims(1, protected={hot}) == [cold]
    # A key that is not resident is never selected, even if unprotected.
    assert hot not in index.select_victims(5, protected=set())


def test_pending_key_never_selected():
    index = _index(4)
    pending = key(0)
    index.note_store([pending], _ctx("s", {pending: 16}))  # not committed
    assert index.select_victims(1, protected=set()) == []


def test_store_failed_drops_only_pending():
    index = _index(4)
    resident, pending = key(0), key(1)
    _store_committed(index, [resident], _ctx(None))
    index.note_store([pending], _ctx(None))
    index.on_store_failed([resident, pending])
    # resident survives (its file was not removed); pending is dropped.
    assert index.select_victims(5, protected=set()) == [resident]


def test_dead_session_evicted_before_live_prefix():
    clock = {"t": 0.0}
    index = _index(2, ttl=10.0)
    index._time_fn = lambda: clock["t"]
    suffix, prefix = key(0), key(1)

    _store_committed(index, [suffix], _ctx("dead", {suffix: 64}))
    index.on_request_finished(_ctx("dead"))
    clock["t"] = 30.0
    _store_committed(index, [prefix], _ctx("live", {prefix: 16}))
    index.on_request_finished(_ctx("live"))

    clock["t"] = 31.0  # dead idle 31s > ttl 10s
    assert index.select_victims(1, protected=set()) == [suffix]


def test_touch_marks_session_alive():
    clock = {"t": 0.0}
    index = _index(2, ttl=10.0)
    index._time_fn = lambda: clock["t"]
    a, b = key(0), key(1)
    _store_committed(index, [a], _ctx("sa", {a: 16}))
    index.on_request_finished(_ctx("sa"))
    _store_committed(index, [b], _ctx("sb", {b: 16}))
    index.on_request_finished(_ctx("sb"))

    clock["t"] = 100.0  # both idle well past ttl -> both dead by time alone
    index.touch([a], _ctx("sa", {a: 16}))  # a's session arrives again (alive)
    victims = index.select_victims(1, protected=set())
    assert victims == [b]  # the still-dead session loses first


def test_stats_breakdown_by_bucket():
    clock = {"t": 0.0}
    index = _index(2, ttl=10.0, max_tiers=1)
    index._time_fn = lambda: clock["t"]
    suffix, prefix = key(0), key(1)
    _store_committed(index, [suffix], _ctx("dead", {suffix: 64}))
    index.on_request_finished(_ctx("dead"))
    clock["t"] = 30.0
    _store_committed(index, [prefix], _ctx("live", {prefix: 16}))
    index.on_request_finished(_ctx("live"))

    clock["t"] = 31.0  # dead session idle past ttl
    index.select_victims(1, protected=set())
    assert index.stats.total == 1
    assert index.stats.dead_session == 1
    assert index.stats.live_prefix == 0
    assert index.stats.lru_tail == 0


def test_max_tiers_tail_on_recency():
    index = _index(8, max_tiers=1)
    deep, mid, prefix = key(0), key(1), key(2)
    # Insert deep first so it is the older tail entry by rank.
    _store_committed(index, [deep], _ctx(None, {deep: 1000}))
    _store_committed(index, [mid], _ctx(None, {mid: 32}))
    _store_committed(index, [prefix], _ctx(None, {prefix: 16}))
    # prefix (tier 0, live) peeled first, then the older tail entry (deep).
    assert index.select_victims(2, protected=set()) == [prefix, deep]


# ---------------------------------------------------------------------------
# FileSystemTierManager integration tests
# ---------------------------------------------------------------------------

_NUM_BLOCKS = 8
_BLOCK_ELEMENTS = 128 * mmap.PAGESIZE


def _make_offloading_spec() -> MagicMock:
    spec = MagicMock()
    spec.config = OffloadingConfig(
        groups=(),
        worker_kv_bytes_per_block=0,
        enable_kv_cache_events=False,
        extra_config={},
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float32"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=1,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=False,
        ),
        replicated_layout=False,
    )
    spec.blocks_per_chunk = 1
    spec.kv_events_config = OffloadingKVEventsConfig(
        enable_kv_cache_events=False,
        self_describing_kv_events=False,
    )
    return spec


def _page_aligned_zero_tensor(num_blocks: int, block_elements: int) -> torch.Tensor:
    page_size = mmap.PAGESIZE
    dtype_num_bytes = torch.finfo(torch.float32).bits // 8
    num_bytes = num_blocks * block_elements * dtype_num_bytes
    t = torch.zeros(num_bytes + page_size, dtype=torch.uint8)
    shift = page_size - (t.data_ptr() % page_size)
    t = t[shift : shift + num_bytes]
    return t.view(torch.float32).view(num_blocks, block_elements)


def _job(job_id: int, keys, chunk_ids, ctx: ReqContext, is_promotion=False):
    return TransferJob(
        job_id=job_id,
        keys=keys,
        chunk_ids=np.array(chunk_ids, dtype=np.int64),
        is_promotion=is_promotion,
        req_context=ctx,
    )


def _drain(tier: FileSystemTierManager) -> list:
    tier.drain_jobs()
    return list(tier.get_finished_jobs())


def _lookup_and_wait(
    tier: FileSystemTierManager,
    keys,
    ctx: ReqContext,
    timeout: float = 1.0,
) -> list:
    """Run a full async lookup cycle and return the resolved verdicts."""
    for k in keys:
        tier.lookup(k, ctx)
    tier.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not tier._lookup_manager._pending_results.empty():
            break
        time.sleep(0.01)
    return [tier.lookup(k, ctx) for k in keys]


@pytest.fixture
def bounded_tier(tmp_path):
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    block_bytes = memoryview(tensor.numpy()).strides[0]
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=2,
        n_write_threads=2,
        capacity_bytes=2 * block_bytes,
        eviction_policy="prefix_pruning",
        session_dead_ttl_s=0.0,
        first_tier_tokens=16,
        tier_step_tokens=16,
    )
    yield tier
    tier.shutdown()


def test_no_eviction_without_capacity(tmp_path):
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
    )
    try:
        assert tier._index is None
        assert tier._capacity_guard is None
    finally:
        tier.shutdown()


def test_prefix_pruning_without_capacity_warns_and_disables(tmp_path):
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        eviction_policy="prefix_pruning",
    )
    try:
        assert tier._index is None
        assert tier._capacity_guard is None
    finally:
        tier.shutdown()


def test_capacity_with_lru_enables_lru_eviction(tmp_path):
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    block_bytes = memoryview(tensor.numpy()).strides[0]
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        capacity_bytes=2 * block_bytes,
        eviction_policy="lru",
    )
    try:
        assert isinstance(tier._index, StorageLRUIndex)
        assert tier._capacity_guard is None
        assert tier._index.max_keys == 2
    finally:
        tier.shutdown()


def test_capacity_with_reject_enables_reject_on_full(tmp_path):
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    block_bytes = memoryview(tensor.numpy()).strides[0]
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        capacity_bytes=2 * block_bytes,
        eviction_policy="reject",
    )
    try:
        assert tier._index is None
        assert tier._capacity_guard is not None
        assert tier._capacity_guard.max_keys == 2
    finally:
        tier.shutdown()


def test_reject_on_full_drops_stores_past_capacity(tmp_path):
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    block_bytes = memoryview(tensor.numpy()).strides[0]
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=2,
        n_write_threads=2,
        capacity_bytes=2 * block_bytes,
        eviction_policy="reject",
    )
    try:
        k0, k1, k2, k3 = key(0), key(1), key(2), key(3)
        ctx = _ctx("s", {k0: 16, k1: 32, k2: 48, k3: 64})

        results = _drain(tier)  # nothing pending
        assert results == []

        tier.submit_store(_job(1, [k0, k1], [0, 1], ctx))
        assert all(r.success for r in _drain(tier))
        assert os.path.exists(tier.file_mapper.get_file_name(k0))
        assert os.path.exists(tier.file_mapper.get_file_name(k1))

        # Over capacity: entire job rejected, no new files, prior files kept.
        tier.submit_store(_job(2, [k2, k3], [2, 3], ctx))
        finished = _drain(tier)
        assert len(finished) == 1
        assert finished[0].job_id == 2
        assert finished[0].success is False
        assert not os.path.exists(tier.file_mapper.get_file_name(k2))
        assert not os.path.exists(tier.file_mapper.get_file_name(k3))
        assert os.path.exists(tier.file_mapper.get_file_name(k0))
        assert os.path.exists(tier.file_mapper.get_file_name(k1))
        assert tier._capacity_guard.tracked_chunks == 2
    finally:
        tier.shutdown()


def test_reject_on_full_partial_job_stores_what_fits(tmp_path):
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    block_bytes = memoryview(tensor.numpy()).strides[0]
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=2,
        n_write_threads=2,
        capacity_bytes=2 * block_bytes,
        eviction_policy="reject",
    )
    try:
        k0, k1, k2 = key(0), key(1), key(2)
        ctx = _ctx("s", {k0: 16, k1: 32, k2: 48})

        tier.submit_store(_job(1, [k0], [0], ctx))
        assert all(r.success for r in _drain(tier))

        # One slot left: first new key stored, second rejected.
        tier.submit_store(_job(2, [k1, k2], [1, 2], ctx))
        finished = _drain(tier)
        assert len(finished) == 1
        assert finished[0].success is True
        assert os.path.exists(tier.file_mapper.get_file_name(k0))
        assert os.path.exists(tier.file_mapper.get_file_name(k1))
        assert not os.path.exists(tier.file_mapper.get_file_name(k2))
        assert tier._capacity_guard.tracked_chunks == 2
    finally:
        tier.shutdown()


def test_reject_on_full_emits_info_log_with_cumulative(tmp_path, caplog):
    import logging

    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    block_bytes = memoryview(tensor.numpy()).strides[0]
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=2,
        n_write_threads=2,
        capacity_bytes=2 * block_bytes,
        eviction_policy="reject",
    )
    try:
        k0, k1 = key(0), key(1)
        tier.submit_store(
            _job(1, [k0, k1], [0, 1], _ctx("s", {k0: 16, k1: 32}))
        )
        _drain(tier)

        with caplog.at_level(logging.INFO):
            k2, k3 = key(2), key(3)
            tier.submit_store(
                _job(2, [k2, k3], [2, 3], _ctx("s", {k2: 48, k3: 64}))
            )
        _drain(tier)

        msgs = [r.getMessage() for r in caplog.records]
        assert any(
            "capacity reject active" in m and "cumulative rejected=" in m
            for m in msgs
        ), f"expected throttled INFO reject log, got: {msgs}"
        stats = tier._capacity_guard.stats
        assert stats.blocks_rejected == 2
        assert stats.stores_full_reject == 1
    finally:
        tier.shutdown()


def test_store_over_budget_evicts_leading_prefix(bounded_tier):
    tier = bounded_tier
    assert tier._index is not None and tier._index.max_keys == 2

    k0, k1 = key(0), key(1)
    ctx1 = _ctx("s", {k0: 16, k1: 32})
    tier.submit_store(_job(1, [k0, k1], [0, 1], ctx1))
    assert all(r.success for r in _drain(tier))
    assert os.path.exists(tier.file_mapper.get_file_name(k0))
    assert os.path.exists(tier.file_mapper.get_file_name(k1))

    k2, k3 = key(2), key(3)
    ctx2 = _ctx("s", {k2: 48, k3: 64})
    tier.submit_store(_job(2, [k2, k3], [2, 3], ctx2))
    _drain(tier)  # waits for both the stores and the async deletes

    # The two leading-prefix blocks are evicted; the fresh suffix survives.
    assert not os.path.exists(tier.file_mapper.get_file_name(k0))
    assert not os.path.exists(tier.file_mapper.get_file_name(k1))
    assert os.path.exists(tier.file_mapper.get_file_name(k2))
    assert os.path.exists(tier.file_mapper.get_file_name(k3))


def test_eviction_emits_info_log(bounded_tier, caplog):
    import logging

    tier = bounded_tier
    k0, k1 = key(0), key(1)
    tier.submit_store(_job(1, [k0, k1], [0, 1], _ctx("s", {k0: 16, k1: 32})))
    _drain(tier)

    with caplog.at_level(logging.INFO):
        k2, k3 = key(2), key(3)
        tier.submit_store(_job(2, [k2, k3], [2, 3], _ctx("s", {k2: 48, k3: 64})))
    _drain(tier)

    assert any(
        "eviction active" in r.getMessage() for r in caplog.records
    ), "expected an INFO eviction log line"
    assert tier._index is not None and tier._index.stats.total >= 2


def test_evicted_key_relooks_up_as_miss(bounded_tier):
    tier = bounded_tier
    k0, k1 = key(0), key(1)
    tier.submit_store(_job(1, [k0, k1], [0, 1], _ctx("s", {k0: 16, k1: 32})))
    _drain(tier)

    k2, k3 = key(2), key(3)
    tier.submit_store(_job(2, [k2, k3], [2, 3], _ctx("s", {k2: 48, k3: 64})))
    _drain(tier)

    assert _lookup_and_wait(tier, [k0, k1], _ctx("s")) == [
        LookupResult.MISS,
        LookupResult.MISS,
    ]


# ---------------------------------------------------------------------------
# StorageLRUIndex unit + integration tests
# ---------------------------------------------------------------------------


def _lru_index(capacity_blocks: int) -> StorageLRUIndex:
    return StorageLRUIndex(
        block_size_bytes=_BLOCK,
        capacity_bytes=capacity_blocks * _BLOCK,
    )


def _lru_store_committed(index: StorageLRUIndex, keys, ctx: ReqContext) -> None:
    index.note_store(keys, ctx)
    index.on_store_committed(keys)


def test_lru_evicts_oldest_first():
    index = _lru_index(2)
    k0, k1, k2 = key(0), key(1), key(2)
    ctx = _ctx()
    _lru_store_committed(index, [k0, k1], ctx)
    index.note_store([k2], ctx)
    assert index.overflow() == 1
    # k2 is still in-flight; only resident victims. Oldest resident is k0.
    assert index.select_victims(1, protected=set()) == [k0]


def test_lru_touch_protects_from_eviction():
    index = _lru_index(2)
    k0, k1, k2 = key(0), key(1), key(2)
    ctx = _ctx()
    _lru_store_committed(index, [k0, k1], ctx)
    index.touch([k0], ctx)  # k0 becomes MRU; k1 is LRU
    index.note_store([k2], ctx)
    index.on_store_committed([k2])
    assert index.select_victims(1, protected=set()) == [k1]


def test_lru_protects_inflight_keys():
    index = _lru_index(1)
    k0, k1 = key(0), key(1)
    ctx = _ctx()
    _lru_store_committed(index, [k0], ctx)
    index.note_store([k1], ctx)
    assert index.select_victims(1, protected={k0}) == []
    assert index.select_victims(1, protected=set()) == [k0]


@pytest.fixture
def lru_bounded_tier(tmp_path):
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    block_bytes = memoryview(tensor.numpy()).strides[0]
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=2,
        n_write_threads=2,
        capacity_bytes=2 * block_bytes,
        eviction_policy="lru",
    )
    yield tier
    tier.shutdown()


def test_lru_store_over_budget_evicts_oldest(lru_bounded_tier):
    tier = lru_bounded_tier
    assert isinstance(tier._index, StorageLRUIndex)
    assert tier._index.max_keys == 2

    k0, k1 = key(0), key(1)
    tier.submit_store(_job(1, [k0, k1], [0, 1], _ctx()))
    assert all(r.success for r in _drain(tier))
    assert os.path.exists(tier.file_mapper.get_file_name(k0))
    assert os.path.exists(tier.file_mapper.get_file_name(k1))

    k2, k3 = key(2), key(3)
    tier.submit_store(_job(2, [k2, k3], [2, 3], _ctx()))
    _drain(tier)

    # Oldest two blocks evicted; newest survive.
    assert not os.path.exists(tier.file_mapper.get_file_name(k0))
    assert not os.path.exists(tier.file_mapper.get_file_name(k1))
    assert os.path.exists(tier.file_mapper.get_file_name(k2))
    assert os.path.exists(tier.file_mapper.get_file_name(k3))


def test_lru_touch_keeps_recently_used_block(lru_bounded_tier):
    tier = lru_bounded_tier
    k0, k1, k2 = key(0), key(1), key(2)
    tier.submit_store(_job(1, [k0, k1], [0, 1], _ctx()))
    _drain(tier)

    # Touch k0 so k1 is the LRU victim when k2 arrives.
    tier.touch([k0], _ctx())
    tier.submit_store(_job(2, [k2], [2], _ctx()))
    _drain(tier)

    assert os.path.exists(tier.file_mapper.get_file_name(k0))
    assert not os.path.exists(tier.file_mapper.get_file_name(k1))
    assert os.path.exists(tier.file_mapper.get_file_name(k2))


def test_lru_eviction_emits_info_log(lru_bounded_tier, caplog):
    import logging

    tier = lru_bounded_tier
    k0, k1 = key(0), key(1)
    tier.submit_store(_job(1, [k0, k1], [0, 1], _ctx()))
    _drain(tier)

    with caplog.at_level(logging.INFO):
        k2, k3 = key(2), key(3)
        tier.submit_store(_job(2, [k2, k3], [2, 3], _ctx()))
    _drain(tier)

    assert any(
        "LRU eviction active" in r.getMessage() for r in caplog.records
    ), "expected an INFO LRU eviction log line"
    assert tier._index is not None and tier._index.stats.total >= 2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
