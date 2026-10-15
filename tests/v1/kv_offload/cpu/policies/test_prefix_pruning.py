# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.policies.factory import instantiate_cache_policy
from vllm.v1.kv_offload.cpu.policies.prefix_pruning import (
    PrefixPruningCachePolicy,
    PrefixPruningOnSessionsCachePolicy,
)


def _insert(policy, key: OffloadKey, position: int, session_id: str | None = None):
    chunk = ChunkStatus(len(policy.chunks))
    chunk.ref_cnt = 0
    policy.insert(key, chunk)
    policy.mark_evictable(key)
    context = ReqContext(
        req_id="test",
        session_id=session_id,
    )
    context.set_offload_key_position(key, position)
    policy.on_chunks_stored([key], context)


def test_prefix_policy_evicts_shallow_tiers_before_lru_tail():
    policy = PrefixPruningCachePolicy(
        cache_capacity=4,
        first_tier_tokens=4,
        tier_step_tokens=4,
        max_tiers=2,
    )
    hot = OffloadKey(b"hot")
    deep = OffloadKey(b"deep")
    tail = OffloadKey(b"tail")
    _insert(policy, hot, position=1)
    _insert(policy, deep, position=6)
    _insert(policy, tail, position=12)

    evicted = policy.evict(2, protected=set())

    assert evicted is not None
    assert [key for key, _ in evicted] == [hot, deep]


def test_dead_session_is_evicted_before_live_prefix():
    policy = PrefixPruningOnSessionsCachePolicy(
        cache_capacity=2,
        first_tier_tokens=4,
        tier_step_tokens=4,
        session_dead_ttl_s=10,
        session_min_gaps=0,
    )
    policy._time_fn = lambda: 0.0
    dead = OffloadKey(b"dead")
    live = OffloadKey(b"live")
    _insert(policy, dead, position=100, session_id="finished")
    _insert(policy, live, position=1)
    context = ReqContext(req_id="finished", session_id="finished")
    policy.on_new_request(context)
    policy._time_fn = lambda: 1.0
    policy.on_request_finished(context)
    policy._time_fn = lambda: 20.0

    evicted = policy.evict(1, protected=set())

    assert evicted is not None
    assert evicted[0][0] == dead
    assert policy.get(dead) is None
    assert policy.get(live) is not None


def test_policy_config_is_passed_to_builtin_policy():
    policy = instantiate_cache_policy(
        PrefixPruningCachePolicy,
        cache_capacity=8,
        policy_config={"first_tier_tokens": 32, "max_tiers": 3},
    )

    assert policy.first_tier_tokens == 32
    assert policy.max_tiers == 3
