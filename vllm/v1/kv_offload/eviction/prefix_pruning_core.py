# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared prefix-tier scoring for CPU KV-cache eviction."""

from __future__ import annotations

from vllm.v1.core.session_liveness import SessionLivenessTracker
from vllm.v1.kv_offload.base import OffloadKey, ReqContext

UNKNOWN_POSITION = 1 << 30


def resolve_session_id(req_context: ReqContext) -> str | None:
    if req_context.session_id:
        return req_context.session_id
    params = req_context.kv_transfer_params
    if params:
        session_id = params.get("session_id")
        if isinstance(session_id, str) and session_id:
            return session_id
    return None


class PrefixTierScorer:
    def __init__(
        self,
        first_tier_tokens: int = 16384,
        tier_step_tokens: int = 16384,
        max_tiers: int = 8,
    ) -> None:
        if first_tier_tokens < 1 or tier_step_tokens < 1 or max_tiers < 0:
            raise ValueError("Prefix tier sizes must be positive and max_tiers non-negative")
        self.first_tier_tokens = first_tier_tokens
        self.tier_step_tokens = tier_step_tokens
        self.max_tiers = max_tiers

    def tier(self, position: int) -> int:
        if position >= UNKNOWN_POSITION:
            return UNKNOWN_POSITION
        if position <= self.first_tier_tokens:
            return 0
        return 1 + (position - self.first_tier_tokens - 1) // self.tier_step_tokens

    def score(
        self, position: int, rank: int, key: OffloadKey, is_dead: bool
    ) -> tuple[int, int, int, OffloadKey]:
        tier = self.tier(position)
        if is_dead:
            return (0, tier, rank, key)
        if self.max_tiers and tier >= self.max_tiers:
            return (2, 0, rank, key)
        return (1, tier, rank, key)
