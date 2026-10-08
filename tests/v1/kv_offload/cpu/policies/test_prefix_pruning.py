# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrefixPruningCachePolicy: peel leading CPU chunks; wipe a dead session."""

from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.policies.factory import (
    CachePolicyFactory,
    cache_policy_config_from_extra,
)
from vllm.v1.kv_offload.cpu.policies.prefix_pruning import (
    PrefixPruningCachePolicy,
    PrefixPruningOnSessionsCachePolicy,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


def _key(tag: str) -> bytes:
    raw = tag.encode("ascii")
    return make_offload_key(raw + b"\x00" * (32 - len(raw)), 0)


def _ctx(req_id: str, session_id: str | None, positions: dict) -> ReqContext:
    ctx = ReqContext(req_id=req_id, session_id=session_id)
    for key, end_token in positions.items():
        ctx.set_offload_key_position(key, end_token)
    return ctx


def _insert_ready(policy: PrefixPruningCachePolicy, key) -> None:
    chunk = ChunkStatus(chunk_id=len(policy.chunks))
    chunk.ref_cnt = 0
    policy.insert(key, chunk)


def test_defaults_match_sessions_arm() -> None:
    policy = PrefixPruningOnSessionsCachePolicy(cache_capacity=4)
    assert policy.first_tier_tokens == 16384
    assert policy.tier_step_tokens == 16384
    assert policy.max_tiers == 8
    assert policy.session_dead_ttl_s == 200.0
    assert policy.session_ewa_alpha == 0.3
    assert policy.session_dead_k == 3.0
    assert policy.session_min_gaps == 2
    assert policy.session_detector == "ewa"


def test_factory_resolves_prefix_pruning() -> None:
    assert (
        CachePolicyFactory.get_cache_policy_cls("prefix_pruning")
        is PrefixPruningOnSessionsCachePolicy
    )
    assert (
        CachePolicyFactory.get_cache_policy_cls("prefix_pruning_on_sessions")
        is PrefixPruningOnSessionsCachePolicy
    )
    manager = CPUOffloadingManager(
        num_chunks=4,
        cache_policy="prefix_pruning_on_sessions",
        cache_policy_config={"session_dead_ttl_s": 120.0, "first_tier_tokens": 8192},
    )
    assert isinstance(manager._policy, PrefixPruningOnSessionsCachePolicy)
    assert manager._policy.session_dead_ttl_s == 120.0
    assert manager._policy.first_tier_tokens == 8192


def test_evicts_leading_prefix_before_suffix() -> None:
    policy = PrefixPruningCachePolicy(
        cache_capacity=8, first_tier_tokens=16, tier_step_tokens=16
    )
    k0, k1, k2, k3 = _key("p0"), _key("p1"), _key("p2"), _key("p3")
    for key in (k0, k1, k2, k3):
        _insert_ready(policy, key)
    ctx = _ctx("r1", "s1", {k0: 16, k1: 32, k2: 48, k3: 64})
    policy.on_request_finished(((k0, k1, k2, k3),), set(), {k0, k1, k2, k3}, ctx)

    evicted = policy.evict(2, protected=set())
    assert evicted is not None
    assert [key for key, _ in evicted] == [k0, k1]


def test_protected_keys_are_skipped() -> None:
    policy = PrefixPruningCachePolicy(
        cache_capacity=4, first_tier_tokens=16, tier_step_tokens=16
    )
    k0, k1 = _key("a0"), _key("a1")
    _insert_ready(policy, k0)
    _insert_ready(policy, k1)
    ctx = _ctx("r", "s", {k0: 16, k1: 32})
    policy.on_request_finished(((k0, k1),), set(), {k0, k1}, ctx)
    evicted = policy.evict(1, protected={k0})
    assert evicted is not None
    assert evicted[0][0] == k1


def test_session_dead_wipes_before_live_prefix() -> None:
    clock = {"now": 0.0}
    policy = PrefixPruningCachePolicy(
        cache_capacity=8,
        first_tier_tokens=16,
        tier_step_tokens=16,
        session_dead_ttl_s=50.0,
    )
    policy._time_fn = lambda: clock["now"]

    dead_suffix = _key("d1")
    live_prefix = _key("l0")
    _insert_ready(policy, dead_suffix)
    _insert_ready(policy, live_prefix)

    clock["now"] = 1.0
    policy.on_request_finished(
        ((dead_suffix,),),
        set(),
        {dead_suffix},
        _ctx("rd", "dead", {dead_suffix: 64}),
    )
    clock["now"] = 2.0
    policy.on_request_finished(
        ((live_prefix,),),
        set(),
        {live_prefix},
        _ctx("rl", "live", {live_prefix: 16}),
    )

    clock["now"] = 55.0
    policy.touch([live_prefix], _ctx("rl", "live", {live_prefix: 16}))
    clock["now"] = 60.0
    evicted = policy.evict(1, protected=set())
    assert evicted is not None
    assert evicted[0][0] == dead_suffix


def test_extra_config_moves_session_ttl_and_tiers() -> None:
    cfg = cache_policy_config_from_extra(
        {
            "eviction_policy": "prefix_pruning_on_sessions",
            "session_dead_ttl_s": 90,
            "eviction_policy_config": {"first_tier_tokens": 4096},
            "tier_step_tokens": 8192,
        }
    )
    assert cfg == {
        "first_tier_tokens": 4096,
        "tier_step_tokens": 8192,
        "session_dead_ttl_s": 90,
    }


def test_ttl_zero_is_prefix_pruning_without_session_wipe() -> None:
    clock = {"now": 0.0}
    policy = PrefixPruningCachePolicy(
        cache_capacity=8,
        first_tier_tokens=16,
        tier_step_tokens=16,
        session_dead_ttl_s=0.0,
    )
    policy._time_fn = lambda: clock["now"]
    dead_suffix = _key("d1")
    live_prefix = _key("l0")
    _insert_ready(policy, dead_suffix)
    _insert_ready(policy, live_prefix)
    clock["now"] = 1.0
    policy.on_request_finished(
        ((dead_suffix,),),
        set(),
        {dead_suffix},
        _ctx("rd", "dead", {dead_suffix: 64}),
    )
    clock["now"] = 2.0
    policy.on_request_finished(
        ((live_prefix,),),
        set(),
        {live_prefix},
        _ctx("rl", "live", {live_prefix: 16}),
    )
    clock["now"] = 10_000.0
    evicted = policy.evict(1, protected=set())
    assert evicted is not None
    assert evicted[0][0] == live_prefix


def test_chunk_ending_on_tier_boundary_stays_in_that_tier() -> None:
    policy = PrefixPruningCachePolicy(
        cache_capacity=4,
        first_tier_tokens=16,
        tier_step_tokens=16,
        session_dead_ttl_s=0,
    )
    boundary, nxt = _key("b0"), _key("b1")
    _insert_ready(policy, boundary)
    _insert_ready(policy, nxt)
    policy.on_request_finished(
        ((boundary, nxt),),
        set(),
        {boundary, nxt},
        _ctx("r", "s", {boundary: 16, nxt: 32}),
    )
    evicted = policy.evict(1, protected=set())
    assert evicted is not None
    assert evicted[0][0] == boundary


def test_max_tiers_puts_the_tail_on_recency() -> None:
    policy = PrefixPruningCachePolicy(
        cache_capacity=8,
        first_tier_tokens=16,
        tier_step_tokens=16,
        max_tiers=1,
        session_dead_ttl_s=0,
    )
    deep, mid, prefix = _key("deep"), _key("mid"), _key("pre")
    for key in (deep, mid, prefix):
        _insert_ready(policy, key)
    # Deep is freed first, so it is the older tail entry.
    policy.on_request_finished(
        ((deep,),), set(), {deep}, _ctx("d", None, {deep: 1000})
    )
    policy.on_request_finished(
        ((mid,),), set(), {mid}, _ctx("m", None, {mid: 32})
    )
    policy.on_request_finished(
        ((prefix,),), set(), {prefix}, _ctx("p", None, {prefix: 16})
    )
    evicted = policy.evict(2, protected=set())
    assert evicted is not None
    assert [key for key, _ in evicted] == [prefix, deep]


def test_dead_session_beats_a_fresh_live_prefix() -> None:
    clock = {"now": 0.0}
    policy = PrefixPruningCachePolicy(
        cache_capacity=4,
        first_tier_tokens=16,
        tier_step_tokens=16,
        session_dead_ttl_s=10.0,
    )
    policy._time_fn = lambda: clock["now"]
    suffix, prefix = _key("suf"), _key("pre")
    _insert_ready(policy, suffix)
    _insert_ready(policy, prefix)
    clock["now"] = 0.0
    policy.on_request_finished(
        ((suffix,),), set(), {suffix}, _ctx("a", "dead", {suffix: 64})
    )
    clock["now"] = 30.0
    policy.on_request_finished(
        ((prefix,),), set(), {prefix}, _ctx("b", "live", {prefix: 16})
    )
    evicted = policy.evict(1, protected=set())
    assert evicted is not None
    assert evicted[0][0] == suffix


def test_ewa_seeds_the_first_gap() -> None:
    clock = {"now": 0.0}
    policy = PrefixPruningCachePolicy(
        cache_capacity=4,
        first_tier_tokens=16,
        tier_step_tokens=16,
        session_dead_ttl_s=10.0,
        session_dead_k=0.0,
        session_min_gaps=2,
        session_ewa_alpha=0.3,
    )
    policy._time_fn = lambda: clock["now"]
    suffix, prefix = _key("suf"), _key("pre")
    _insert_ready(policy, suffix)
    _insert_ready(policy, prefix)

    clock["now"] = 0.0
    policy.on_request_finished(
        ((suffix,),), set(), {suffix}, _ctx("a", "sess", {suffix: 64})
    )
    clock["now"] = 40.0
    policy.touch([suffix], _ctx("a2", "sess", {suffix: 64}))
    clock["now"] = 41.0
    policy.on_request_finished(
        ((suffix,),), set(), {suffix}, _ctx("a2", "sess", {suffix: 64})
    )
    clock["now"] = 50.0
    policy.touch([suffix], _ctx("a3", "sess", {suffix: 64}))
    clock["now"] = 51.0
    policy.on_request_finished(
        ((suffix,),), set(), {suffix}, _ctx("a3", "sess", {suffix: 64})
    )
    clock["now"] = 70.0
    policy.on_request_finished(
        ((prefix,),), set(), {prefix}, _ctx("b", "other", {prefix: 16})
    )
    # Idle 20s. Seeded EWA is 30.7s, so the session is still alive and the
    # fresh prefix is peeled. Blending the first gap into 0 would already
    # have declared it dead and taken the suffix.
    clock["now"] = 71.0
    evicted = policy.evict(1, protected=set())
    assert evicted is not None
    assert evicted[0][0] == prefix

    later = _key("lat")
    _insert_ready(policy, later)
    clock["now"] = 82.0
    policy.on_request_finished(
        ((later,),), set(), {later}, _ctx("c", "other", {later: 16})
    )
    clock["now"] = 83.0
    evicted = policy.evict(1, protected=set())
    assert evicted is not None
    assert evicted[0][0] == suffix


def test_atomic_evict_when_not_enough() -> None:
    policy = PrefixPruningCachePolicy(cache_capacity=2)
    k0 = _key("only")
    _insert_ready(policy, k0)
    assert policy.evict(2, protected=set()) is None
    assert policy.get(k0) is not None


if __name__ == "__main__":
    tests = [
        test_defaults_match_sessions_arm,
        test_factory_resolves_prefix_pruning,
        test_evicts_leading_prefix_before_suffix,
        test_protected_keys_are_skipped,
        test_session_dead_wipes_before_live_prefix,
        test_extra_config_moves_session_ttl_and_tiers,
        test_ttl_zero_is_prefix_pruning_without_session_wipe,
        test_chunk_ending_on_tier_boundary_stays_in_that_tier,
        test_max_tiers_puts_the_tail_on_recency,
        test_dead_session_beats_a_fresh_live_prefix,
        test_ewa_seeds_the_first_gap,
        test_atomic_evict_when_not_enough,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")
