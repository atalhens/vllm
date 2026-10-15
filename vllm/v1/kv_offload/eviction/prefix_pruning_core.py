# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tier-agnostic core for session-based prefix-pruning eviction.

Both the CPU primary tier (``PrefixPruningOnSessionsCachePolicy``) and the
storage tier (``StorageEvictionIndex``) share the same eviction order:

* every chunk of a session predicted dead, still in prefix-tier order
* live prefix tiers, low end-token tier first
* the LRU tail: chunks past ``max_tiers``, older free first

This module owns the ``PrefixTierScorer`` that maps a chunk end-token position
to its (bucket, tier) score and uses the shared ``SessionLivenessTracker``.
Callers own chunk storage, recency ranks, and the eviction trigger.
"""

from __future__ import annotations

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.core.session_liveness import SessionLivenessTracker

# End-token position used when a chunk's position is unknown. Larger than any
# realistic context so unknown chunks sort to the deepest tier.
UNKNOWN_POSITION = 1 << 30


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


class PrefixTierScorer:
    """Maps a chunk end-token position to its eviction score.

    A chunk whose end token is ``first_tier_tokens`` is still tier 0.
    ``max_tiers`` sends that tier and everything past it to the LRU tail.
    ``max_tiers=0`` leaves the prefix tiers uncapped.
    """

    def __init__(
        self,
        first_tier_tokens: int = 16384,
        tier_step_tokens: int = 16384,
        max_tiers: int = 8,
    ) -> None:
        if first_tier_tokens < 1:
            raise ValueError(f"first_tier_tokens must be >= 1, got {first_tier_tokens}")
        if tier_step_tokens < 1:
            raise ValueError(f"tier_step_tokens must be >= 1, got {tier_step_tokens}")
        if max_tiers < 0:
            raise ValueError(f"max_tiers must be >= 0, got {max_tiers}")
        self.first_tier_tokens = first_tier_tokens
        self.tier_step_tokens = tier_step_tokens
        self.max_tiers = max_tiers

    def tier(self, position: int) -> int:
        if position >= UNKNOWN_POSITION:
            return UNKNOWN_POSITION
        if position <= self.first_tier_tokens:
            return 0
        return 1 + (position - self.first_tier_tokens - 1) // self.tier_step_tokens

    def is_tail(self, tier: int) -> bool:
        return self.max_tiers > 0 and tier >= self.max_tiers

    def score(
        self, position: int, rank: int, key: OffloadKey, is_dead: bool
    ) -> tuple[int, int, int, OffloadKey]:
        """Dead session, then live prefix tier, then the LRU tail."""
        tier = self.tier(position)
        if is_dead:
            return (0, tier, rank, key)
        if self.is_tail(tier):
            return (2, 0, rank, key)
        return (1, tier, rank, key)
