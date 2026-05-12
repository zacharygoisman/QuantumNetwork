#analysis/metrics.py
"""
Rate, fidelity, and upper bound computations for paths and links.
"""

#ZHG
#2026.03.26
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

import math
from typing import Optional
import numpy as np


def rate_poly(x: float, y1: float, y2: float) -> float:
    """
    Raw per-link polynomial term:
        R(x) = x^2 + (2(y1+y2)+1)x + 4 y1 y2
    """
    a = 2.0 * (y1 + y2) + 1.0
    b = 4.0 * y1 * y2
    return x * x + a * x + b


def fidelity(x: float, y1: float, y2: float) -> float:
    """
    Link fidelity:
        F(x) = 0.25 * (1 + 3x / R(x))
    """
    r = rate_poly(x, y1, y2)
    if r <= 0.0:
        return float("-inf")
    return 0.25 * (1.0 + 3.0 * x / r)


def per_link_ub_value(
    y1: float,
    y2: float,
    f_min: float,
    x_one_channel_cap: Optional[float] = None,
    on_infeasible: str = "zero",
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Best per-link log-utility under the fidelity floor.

    This computes the largest admissible x satisfying:
        F(x) >= max(0.5, f_min)

    and then evaluates:
        log10(R(x))

    Args:
        y1, y2:
            Link noise parameters.
        f_min:
            Requested fidelity floor. Values below 0.5 are clamped to 0.5.
        x_one_channel_cap:
            Optional physical cap on x for the one-channel bound.
            If provided, the final x is min(x_fid_max, x_one_channel_cap).
        on_infeasible:
            "zero" -> return (0.0, None, None) when infeasible
            "none" -> return (None, None, None) when infeasible

    Returns:
        (ub_value, x_star, raw_rate)
        where:
            ub_value = log10(R(x_star))
            x_star   = chosen admissible x
            raw_rate = R(x_star)
    """
    f_eff = max(0.5, float(f_min))

    # F(x) >= f_eff  <=>  t x^2 + (tA - 3)x + tB <= 0
    # where:
    #   t = 4 f_eff - 1
    #   A = 2(y1+y2)+1
    #   B = 4 y1 y2
    t = 4.0 * f_eff - 1.0
    a = 2.0 * (y1 + y2) + 1.0
    b = 4.0 * y1 * y2

    disc = (t * a - 3.0) ** 2 - 4.0 * (t ** 2) * b
    if disc < 0.0:
        if on_infeasible == "none":
            return None, None, None
        return 0.0, None, None

    x_fid_max = (3.0 - t * a + math.sqrt(disc)) / (2.0 * t)
    if x_fid_max < 0.0:
        if on_infeasible == "none":
            return None, None, None
        return 0.0, None, None

    x_star = x_fid_max
    if x_one_channel_cap is not None:
        x_star = min(x_star, float(x_one_channel_cap))

    raw_rate = rate_poly(x_star, y1, y2)
    if raw_rate <= 0.0:
        if on_infeasible == "none":
            return None, None, None
        return 0.0, None, None

    ub_value = math.log10(raw_rate)
    return ub_value, x_star, raw_rate


def per_link_path_ub(f_min: float, y1: float, y2: float) -> float:
    """
    Convenience wrapper for a per-path upper bound with no extra physical x cap.
    Returns -inf if infeasible.
    """
    ub_value, _, _ = per_link_ub_value(
        y1=y1,
        y2=y2,
        f_min=f_min,
        x_one_channel_cap=None,
        on_infeasible="none",
    )
    return float(ub_value) if (ub_value is not None and np.isfinite(ub_value)) else float("-inf")


def compute_ub_max(y1: float, y2: float, f_min: float = 0.5) -> float:
    """
    Drop-in UB helper for the current pipeline.

    Returns the per-link upper bound in the same units as the current
    optimization objective:
        log10(R)

    Note:
        This is not the comparison-paper fitness expression.
        It is the bound consistent with the current solver objective.
    """
    return per_link_path_ub(f_min=f_min, y1=y1, y2=y2)