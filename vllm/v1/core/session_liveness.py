# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session think-time detection shared by KV cache tiers."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


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
    """Predict idle sessions from per-session EWA or global gap quantiles."""

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
            raise ValueError(
                f"session_min_gaps must be >= 0, got {session_min_gaps}"
            )
        if session_detector not in ("ewa", "global"):
            raise ValueError(
                "session_detector must be 'ewa' or 'global', "
                f"got {session_detector!r}"
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
        self._global_gaps: deque[float] = deque(maxlen=session_gap_window)
        self._sessions: dict[str, SessionState] = {}

    @property
    def enabled(self) -> bool:
        return self.session_dead_ttl_s > 0

    def _observe_gap(self, state: SessionState, gap: float) -> None:
        self._global_gaps.append(gap)
        if state.n_gaps == 0:
            state.ewa_gap = gap
            state.ewa_gap_sq = gap * gap
        else:
            alpha = self.session_ewa_alpha
            state.ewa_gap = alpha * gap + (1.0 - alpha) * state.ewa_gap
            state.ewa_gap_sq = alpha * gap * gap + (1.0 - alpha) * state.ewa_gap_sq
        state.n_gaps += 1

    def on_arrival(self, session_id: str | None, now: float) -> None:
        if not session_id or not self.enabled:
            return
        state = self._sessions.setdefault(session_id, SessionState())
        same_turn = (
            state.last_arrival is not None
            and state.last_finish is not None
            and state.last_arrival > state.last_finish
        )
        if state.last_finish is not None and not same_turn:
            self._observe_gap(state, max(0.0, now - state.last_finish))
        state.last_arrival = now

    def on_finish(self, session_id: str | None, now: float) -> None:
        if not session_id or not self.enabled:
            return
        state = self._sessions.setdefault(session_id, SessionState())
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
        gaps = sorted(self._global_gaps)
        index = int((len(gaps) - 1) * self.session_quantile)
        return gaps[index]

    def dead_sessions(self, now: float) -> set[str]:
        if not self.enabled:
            return set()
        dead: set[str] = set()
        for session_id, state in self._sessions.items():
            if state.last_finish is None:
                continue
            if (
                state.last_arrival is not None
                and state.last_arrival > state.last_finish
            ):
                continue
            if now - state.last_finish >= self._threshold(state):
                dead.add(session_id)
        return dead

    def clear(self) -> None:
        self._sessions.clear()
        self._global_gaps.clear()
