# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tier-agnostic core for session-based prefix-pruning eviction.

Both the CPU primary tier (``PrefixPruningOnSessionsCachePolicy``) and the
storage tier (``StorageEvictionIndex``) share the same eviction order:

* every chunk of a session predicted dead, still in prefix-tier order
* live prefix tiers, low end-token tier first
* the LRU tail: chunks past ``max_tiers``, older free first

This module owns the two reusable pieces of that order: the
``SessionLivenessTracker`` think-time detector and the ``PrefixTierScorer``
that maps a chunk end-token position to its (bucket, tier) score. Callers
own chunk storage, recency ranks, and the eviction trigger.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from vllm.v1.kv_offload.base import OffloadKey, ReqContext

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


@dataclass
class SessionState:
    last_finish: float | None = None
    last_arrival: float | None = None
    n_gaps: int = 0
    ewa_gap: float = 0.0
    ewa_gap_sq: float = 0.0

    @property
    def std(self) -> float:
        var = max(0.0, self.ewa_gap_sq - self.ewa_gap * self.ewa_gap)
        return math.sqrt(var)


class SessionLivenessTracker:
    """Think-time detector that flags a session as dead once its idle gap
    exceeds ``max(session_dead_ttl_s, ewa + k·std)``.

    The gap is measured from a session's last finish to its next arrival. The
    first gap seeds the average. Fewer than ``session_min_gaps`` samples keep
    the static floor. A turn still in flight is not dead.
    ``session_dead_ttl_s=0`` disables the detector.
    """

    def __init__(
        self,
        session_dead_ttl_s: float = 200.0,
        session_ewa_alpha: float = 0.3,
        session_dead_k: float = 3.0,
        session_min_gaps: int = 2,
        session_detector: str = "ewa",
        session_quantile: float = 0.95,
        session_gap_window: int = 256,
    ) -> None:
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
        self.session_dead_ttl_s = session_dead_ttl_s
        self.session_ewa_alpha = session_ewa_alpha
        self.session_dead_k = session_dead_k
        self.session_min_gaps = session_min_gaps
        self.session_detector = session_detector
        self.session_quantile = session_quantile
        self.session_gap_window = session_gap_window

        self._sessions: dict[str, SessionState] = {}
        self._global_gaps: deque[float] = deque(maxlen=session_gap_window)

    @property
    def enabled(self) -> bool:
        return self.session_dead_ttl_s > 0

    def _observe_gap(self, state: SessionState, gap: float) -> None:
        self._global_gaps.append(gap)
        if state.n_gaps == 0:
            state.ewa_gap = gap
            state.ewa_gap_sq = gap * gap
        else:
            a = self.session_ewa_alpha
            state.ewa_gap = a * gap + (1.0 - a) * state.ewa_gap
            state.ewa_gap_sq = a * gap * gap + (1.0 - a) * state.ewa_gap_sq
        state.n_gaps += 1

    def on_arrival(self, sid: str | None, now: float) -> None:
        if not sid or not self.enabled:
            return
        state = self._sessions.setdefault(sid, SessionState())
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

    def on_finish(self, sid: str | None, now: float) -> None:
        if not sid or not self.enabled:
            return
        state = self._sessions.setdefault(sid, SessionState())
        state.last_finish = now

    def _threshold(self, state: SessionState) -> float:
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

    def dead_sessions(self, now: float) -> set[str]:
        if not self.enabled:
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

    def clear(self) -> None:
        self._sessions.clear()
        self._global_gaps.clear()


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
