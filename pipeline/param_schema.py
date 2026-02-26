"""
pipeline/param_schema.py
------------------------

Defines the design variable schema for robot morphology and actuation.

Now extended to include:
    - 3 link lengths (subject to constraints)
    - Motor selection (robstride02 / robstride03)
    - Gear ratio (9 / 15 / 20)
DOF is fixed at 3 for this study.

Each generated param dict is validated before returning.
"""

from __future__ import annotations
from typing import Dict, Any, List
import random

# --------------------------------------------------------------
# 1. Design variable definitions
# --------------------------------------------------------------

MOTOR_TYPES = ["robstride02", "robstride03", "robstride06"]
GEAR_RATIOS = [1, 3, 5]

LINK_MIN = 0.1
LINK_COUNT = 3
LINK_SUM_MAX = 0.8

# Bound summary (for reference / GA)
BOUNDS = {
    "link_lengths": {
        "count": LINK_COUNT,
        "min": LINK_MIN,
        "sum_max": LINK_SUM_MAX,
    },
    "motor_type": {"choices": MOTOR_TYPES},
    "gear_ratio": {"choices": GEAR_RATIOS},
}


# --------------------------------------------------------------
# 2. Validation helpers
# --------------------------------------------------------------

def validate_params(params: Dict[str, Any]) -> bool:
    """Check all hard constraints."""
    try:
        Ls = params["link_lengths"]
        if len(Ls) != LINK_COUNT:
            return False
        if any(L < LINK_MIN for L in Ls):
            return False
        if sum(Ls) > LINK_SUM_MAX:
            return False
        if params["motor_type"] not in MOTOR_TYPES:
            return False
        if params["gear_ratio"] not in GEAR_RATIOS:
            return False
        return True
    except Exception:
        return False


# --------------------------------------------------------------
# 3. Sampling utilities
# --------------------------------------------------------------

def _sample_link_lengths() -> List[float]:
    """
    Randomly sample 3 link lengths satisfying:
        each >= LINK_MIN
        sum <= LINK_SUM_MAX
    Uses simple Dirichlet-like sampling.
    """
    budget = LINK_SUM_MAX - LINK_COUNT * LINK_MIN  # 0.5
    r = [random.random() + 1e-6 for _ in range(LINK_COUNT)]
    s = sum(r)
    xs = [budget * (ri / s) for ri in r]
    Ls = [LINK_MIN + xi for xi in xs]
    # Re-normalize to avoid floating drift
    if sum(Ls) > LINK_SUM_MAX:
        scale = (LINK_SUM_MAX - LINK_COUNT * LINK_MIN) / (
            sum(Ls) - LINK_COUNT * LINK_MIN + 1e-9
        )
        Ls = [LINK_MIN + (Li - LINK_MIN) * scale for Li in Ls]
    return Ls


# --------------------------------------------------------------
# 4. Parameter generation
# --------------------------------------------------------------

def make_params(seed: int | None = None) -> Dict[str, Any]:
    """Generate a random valid parameter dictionary."""
    if seed is not None:
        random.seed(seed)

    params: Dict[str, Any] = {
        "dof_per_leg": 3,  # fixed for now
        "link_lengths": _sample_link_lengths(),
        "motor_type": random.choice(MOTOR_TYPES),
        "gear_ratio": random.choice(GEAR_RATIOS),
    }

    assert validate_params(params), f"Invalid params: {params}"
    return params


# --------------------------------------------------------------
# 5. Self-test
# --------------------------------------------------------------

if __name__ == "__main__":
    print("=== Self-test: param_schema ===")
    for i in range(5):
        p = make_params(i)
        print(f"[Sample {i}] {p}")
        print(f"  Sum(link_lengths) = {sum(p['link_lengths']):.3f}")
        assert validate_params(p)
    print("All generated params valid ✅")
