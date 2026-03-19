from __future__ import annotations

import os
import time


_ENABLED = str(os.getenv("MILP_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"}
_T0 = time.perf_counter()


def debug_log(message: str) -> None:
    if not _ENABLED:
        return
    dt = time.perf_counter() - _T0
    print(f"[MILP_DEBUG +{dt:8.3f}s] {message}", flush=True)
