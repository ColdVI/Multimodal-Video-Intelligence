from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@lru_cache(maxsize=2)
def readiness(profile: str):
    if os.getenv("RUN_FAZ8_INTEGRATION") != "1":
        pytest.skip("SKIPPED: set RUN_FAZ8_INTEGRATION=1 for live Faz 8 suites")
    from scripts.readiness_check import collect_readiness

    result = collect_readiness(profile)
    if not result["ready"]:
        reasons = "; ".join(
            f"{row['id']}: {row['detail']}" for row in result["checks"] if row["status"] != "PASS"
        )
        pytest.skip(f"SKIPPED: {profile} readiness missing - {reasons}")
    return result
