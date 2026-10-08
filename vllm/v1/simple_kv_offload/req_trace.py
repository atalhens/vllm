# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional JSONL request trace for CPU offload (env VLLM_CPU_OFFLOAD_REQ_TRACE).

One compact object per line. No prompt text, no token ids.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

_lock = threading.Lock()


def enabled() -> bool:
    return bool(os.environ.get("VLLM_CPU_OFFLOAD_REQ_TRACE") or "")


def emit(event: str, **fields: Any) -> None:
    path = os.environ.get("VLLM_CPU_OFFLOAD_REQ_TRACE") or ""
    if not path:
        return
    rec: dict[str, Any] = {"t": time.time(), "event": event}
    rec.update(fields)
    line = json.dumps(rec, separators=(",", ":"), default=str) + "\n"
    try:
        with _lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except OSError:
        pass
