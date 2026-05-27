#allocation/allocator.py
"""
Allocate channels using APOPT via GEKKO to solve MINLP for each source independently and then combine results.

The MINLP only needs to find an integer allocation k_i for each link sharing a
source. Given any candidate k allocation, the optimal continuous parameter
mu* is determined analytically (closed-form upper root of the fidelity
equality, taken as the minimum across links). So for typical problem sizes
we can fully skip the GEKKO/APOPT subprocess by enumerating the small number
of valid k vectors (compositions of <=K channels into N parts with each k_i>=1).
GEKKO is kept as a fallback for unusually large search spaces.
"""

#ZHG
#2026.03.26
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

import math
import os
import contextlib
from collections import defaultdict
from functools import lru_cache

# Cache solved per-source sub-problems by their parameter signature so that
# repeating the same source group across combos does not redo the work.
_ALLOC_CACHE = {}
_ALLOC_CACHE_MAX = 8192

# Threshold for using the analytic enumeration fast path. C(K, N) gives the
# number of valid k vectors; this stays small for realistic configs.
_ENUM_MAX_CASES = 20000


def allocate_combo(combo, network, sources, cfg):
    by_source = defaultdict(list)

    # Group options by source
    for option in combo:
        by_source[option["source"]].append(option)

    total_utility = 0.0
    allocations = {}

    for s, opts in by_source.items():
        K = len(sources[s]["available_channels"])
        y1 = tuple(float(o["y1"]) for o in opts)
        y2 = tuple(float(o["y2"]) for o in opts)
        fidelity_limit = tuple(float(max(0.5, o.get("fidelity_limit", 0.5))) for o in opts)

        cache_key = (K, y1, y2, fidelity_limit)
        cached = _ALLOC_CACHE.get(cache_key)
        if cached is None:
            try:
                cached = _solve_source(K, fidelity_limit, y1, y2)
            except Exception:
                cached = "fallback"

            if cached == "infeasible":
                # Cache the negative result so we don't retry it.
                if len(_ALLOC_CACHE) < _ALLOC_CACHE_MAX:
                    _ALLOC_CACHE[cache_key] = "infeasible"
                return {"success": False}

            if cached is None or cached == "fallback":
                # Fall back to GEKKO/APOPT for any pathological case the
                # enumerator could not handle (search space too large).
                try:
                    _, obj, _, prelog_rates, optimal_mu, optimal_allocation = matt(
                        K,
                        list(fidelity_limit),
                        list(y1),
                        list(y2),
                        initial=0.001,
                        verbose=False,
                    )
                    cached = {
                        "obj": float(obj),
                        "mu": float(optimal_mu),
                        "k": [int(v) for v in optimal_allocation],
                        "prelog": [float(r) for r in prelog_rates],
                    }
                except Exception:
                    return {"success": False}

            if len(_ALLOC_CACHE) < _ALLOC_CACHE_MAX:
                _ALLOC_CACHE[cache_key] = cached
        elif cached == "infeasible":
            return {"success": False}

        total_utility += cached["obj"]
        for i, o in enumerate(opts):
            allocations[id(o)] = {
                "source": s,
                "link": o["link"],
                "link_idx": o["link_idx"],
                "k": int(cached["k"][i]),
                "mu": float(cached["mu"]),
                "prelog_rate": float(cached["prelog"][i]),
            }

    return {
        "success": True,
        "utility": float(total_utility),
        "allocation": allocations,
    }


def _mu_upper_for_link(y1_i, y2_i, F, k_i):
    """Closed-form upper root of the fidelity equality for a single link."""
    s = y1_i + y2_i
    p = y1_i * y2_i
    t = 4.0 * F - 1.0  # > 0 when F > 0.25
    A = t * (k_i ** 2)
    B = t * k_i * (2.0 * s + 1.0) - 3.0 * k_i
    C = 4.0 * t * p
    D = B * B - 4.0 * A * C
    if D < 0.0 or A <= 0.0:
        return None
    sqrtD = math.sqrt(D)
    mu_hi = (-B + sqrtD) / (2.0 * A)
    if mu_hi <= 1e-12:
        return None
    return mu_hi


def _expr(mu, k_i, y1_i, y2_i):
    return (
        mu * mu * (k_i * k_i)
        + mu * k_i * (2.0 * (y1_i + y2_i) + 1.0)
        + 4.0 * y1_i * y2_i
    )


def _solve_source(K, fidelity_limit, y1, y2):
    """
    Enumerate compositions of <=K channels into N parts (each >=1) and pick the
    allocation that maximizes sum_i log10(expr_i(mu*, k_i)).
    Returns:
        dict with obj/mu/k/prelog on success,
        "infeasible" if no valid k vector exists (e.g. N > K),
        "fallback" if the search space exceeds the enumeration threshold.
    """
    N = len(y1)

    if N == 0:
        return {"obj": 0.0, "mu": 1e-9, "k": [], "prelog": []}

    if N > K:
        return "infeasible"

    if N == 1:
        # Forced k_1 = 1 (matches existing behavior in matt()).
        k_i = 1
        mu_hi = _mu_upper_for_link(y1[0], y2[0], fidelity_limit[0], k_i)
        if mu_hi is None:
            return "infeasible"
        e = _expr(mu_hi, k_i, y1[0], y2[0])
        return {
            "obj": math.log10(e),
            "mu": mu_hi,
            "k": [k_i],
            "prelog": [e],
        }

    # Estimate the number of compositions: C(K, N).
    cases = _binom(K, N)
    if cases > _ENUM_MAX_CASES:
        return "fallback"

    # Precompute mu_upper(i, k) for k in 1..K. Since increasing k_i with
    # fidelity F generally tightens the mu bound, we don't need to evaluate
    # k_i values that drive mu* below 0.
    mu_table = [
        [_mu_upper_for_link(y1[i], y2[i], fidelity_limit[i], k) for k in range(K + 1)]
        for i in range(N)
    ]

    best_obj = -math.inf
    best_k = None
    best_mu = None
    best_prelog = None

    # Iterate compositions of total in [N..K] into N positive parts via
    # nested generation. We use a simple recursive walker to keep memory
    # constant even when the exhaustive set is enumerated.
    stack_k = [0] * N

    def recurse(idx, remaining):
        nonlocal best_obj, best_k, best_mu, best_prelog
        if idx == N - 1:
            # Last link must take whatever leaves sum<=K and >=1
            for kv in range(1, remaining + 1):
                stack_k[idx] = kv
                _evaluate(stack_k)
            return
        # idx < N-1: need to leave at least (N-1-idx) for remaining links
        upper = remaining - (N - 1 - idx)
        for kv in range(1, upper + 1):
            stack_k[idx] = kv
            recurse(idx + 1, remaining - kv)

    def _evaluate(k_vec):
        nonlocal best_obj, best_k, best_mu, best_prelog
        # Compute mu* = min over links of mu_upper(i, k_i)
        mu_star = math.inf
        for i in range(N):
            mu_i = mu_table[i][k_vec[i]]
            if mu_i is None:
                return
            if mu_i < mu_star:
                mu_star = mu_i
        if mu_star <= 0.0:
            return
        # Compute objective
        obj = 0.0
        prelog = [0.0] * N
        for i in range(N):
            e = _expr(mu_star, k_vec[i], y1[i], y2[i])
            if e <= 0.0:
                return
            prelog[i] = e
            obj += math.log10(e)
        if obj > best_obj:
            best_obj = obj
            best_k = list(k_vec)
            best_mu = mu_star
            best_prelog = prelog

    recurse(0, K)

    if best_k is None:
        return "infeasible"

    return {
        "obj": best_obj,
        "mu": best_mu,
        "k": best_k,
        "prelog": best_prelog,
    }


@lru_cache(maxsize=4096)
def _binom(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    num = 1
    den = 1
    for i in range(k):
        num *= (n - i)
        den *= (i + 1)
    return num // den


def matt(K, fidelity_limit, y1, y2, initial, per_link_k_cap=None, verbose=True):
    """Solves the MINLP for a single source's allocation problem using GEKKO with the APOPT solver.

    Kept as a fallback path for the rare case where the analytic enumerator
    cannot handle the search space.
    """
    from gekko import GEKKO  # imported lazily so the fast path avoids the cost

    N_links = len(y1)

    m = GEKKO(remote=False)
    m.options.IMODE = 3
    m.options.SOLVER = 1
    m.solver_options = [
        'minlp_maximum_iterations 500',
        'minlp_gap_tol 1e-3',
        'minlp_branch_method 1',
        'nlp_maximum_iterations 1000',
        'minlp_max_iter_with_int_sol 50',
        'minlp_integer_tol 1e-2',
        'minlp_as_nlp 0',
    ]

    mu_init = max(initial, 0.01)
    mu = m.Var(value=mu_init, lb=1e-9)

    if N_links == 1:
        k_vars = [m.Var(value=1, integer=True, lb=1, ub=1)]
        m.Equation(sum(k_vars) == 1)
    else:
        ub_each = K if per_link_k_cap is None else min(K, int(per_link_k_cap))
        total_quality = sum(1.0 / (y1[i] + y2[i] + 1e-9) for i in range(N_links))
        k_init = []
        for i in range(N_links):
            quality = 1.0 / (y1[i] + y2[i] + 1e-9)
            k_guess = max(1, min(ub_each, int(K * quality / total_quality)))
            k_init.append(k_guess)
        k_vars = [m.Var(value=k_init[i], integer=True, lb=1, ub=ub_each) for i in range(N_links)]
        m.Equation(sum(k_vars) <= K)

    for i in range(N_links):
        expr = mu**2 * k_vars[i]**2 + mu * k_vars[i] * (2 * (y1[i] + y2[i]) + 1) + 4 * y1[i] * y2[i]
        m.Equation(0.25 * (1 + (3 * mu * k_vars[i]) / expr) >= fidelity_limit[i])

    obj = 0
    for i in range(N_links):
        expr = mu**2 * k_vars[i]**2 + mu * k_vars[i] * (2 * (y1[i] + y2[i]) + 1) + 4 * y1[i] * y2[i]
        obj += m.log10(expr)
    m.Obj(-obj)

    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            m.solve(disp=False)

    ub_each = K if per_link_k_cap is None else min(K, int(per_link_k_cap))
    raw_k = [kv.value[0] for kv in k_vars]
    k_int = [int(round(v)) for v in raw_k]
    k_int = [min(max(1, k), ub_each) for k in k_int]

    if N_links > 1:
        total = sum(k_int)
        if total > K:
            round_up_amt = [k_int[i] - raw_k[i] for i in range(N_links)]
            while total > K:
                candidates = [(i, round_up_amt[i]) for i in range(N_links) if k_int[i] > 1]
                if not candidates:
                    break
                i_star = max(candidates, key=lambda t: t[1])[0]
                k_int[i_star] -= 1
                total -= 1

    def mu_upper_for_link(i, k_i):
        F = fidelity_limit[i]
        s = y1[i] + y2[i]
        p = y1[i] * y2[i]
        t = 4.0 * F - 1.0
        A = t * (k_i ** 2)
        B = t * k_i * (2.0 * s + 1.0) - 3.0 * k_i
        C = 4.0 * t * p
        D = B * B - 4.0 * A * C
        if D <= 0.0:
            return max(float(mu.value[0]), 1e-9)
        sqrtD = math.sqrt(D)
        mu_hi = (-B + sqrtD) / (2.0 * A)
        return max(mu_hi, 1e-9)

    mu_star = min(mu_upper_for_link(i, k_int[i]) for i in range(N_links)) if N_links > 0 else float(mu.value[0])

    optimal_mu = mu_star
    optimal_allocation = k_int[:]

    prelog_rates = []
    objective_value = 0.0
    for i in range(N_links):
        ki = optimal_allocation[i]
        expr_val = (optimal_mu**2 * (ki**2) + optimal_mu * ki * (2 * (y1[i] + y2[i]) + 1) + 4 * y1[i] * y2[i])
        prelog_rates.append(expr_val)
        objective_value += math.log10(expr_val)

    if verbose:
        print("Calculated Objective Value (log10):", objective_value)
    return k_vars, objective_value, mu, prelog_rates, optimal_mu, optimal_allocation
