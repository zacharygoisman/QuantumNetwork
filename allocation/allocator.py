#allocation/allocator.py
"""
Allocate channels using APOPT via GEKKO to solve MINLP for each source
independently and then combine results.

Per-source allocation problems are cached by their parameter signature so that
combos which share the same source group don't re-invoke the APOPT subprocess.
"""

#ZHG
#2026.03.26
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

import math
import os
import contextlib
from collections import defaultdict

from gekko import GEKKO

# Cache solved per-source sub-problems by their parameter signature so repeating
# the same source group across combos does not redo the APOPT solve.
_ALLOC_CACHE = {}
_ALLOC_CACHE_MAX = 8192
_EXHAUSTIVE_K_COMBINATION_LIMIT = 10_000


class _StaticVar:
    """Minimal GEKKO-var-like wrapper for exhaustive solutions."""

    def __init__(self, value):
        self.value = [value]


def _all_k_vectors(num_channels, num_links, per_link_k_cap=None):
    """Yield every positive integer k-vector with sum(k) <= num_channels."""
    if num_links <= 0 or num_channels < num_links:
        return

    ub_each = num_channels if per_link_k_cap is None else min(num_channels, int(per_link_k_cap))
    if ub_each < 1:
        return

    current = [1] * num_links

    def rec(index, remaining):
        if index == num_links - 1:
            for last in range(1, min(remaining, ub_each) + 1):
                current[index] = last
                yield list(current)
            return

        links_after_this = num_links - index - 1
        max_here = min(remaining - links_after_this, ub_each)
        for value in range(1, max_here + 1):
            current[index] = value
            yield from rec(index + 1, remaining - value)

    yield from rec(0, num_channels)


def _mu_interval_for_link(k_i, y1_i, y2_i, f_i, disc_tol=1e-15):
    """Return [mu_low, mu_high] where link fidelity is feasible for fixed k_i."""
    if k_i <= 0:
        return None
    if f_i <= 0.25:
        return (0.0, float("inf"))

    s = y1_i + y2_i
    p = y1_i * y2_i
    t = 4.0 * f_i - 1.0
    a = t * (k_i ** 2)
    b = t * k_i * (2.0 * s + 1.0) - 3.0 * k_i
    c = 4.0 * t * p
    disc = b * b - 4.0 * a * c

    if disc < -disc_tol:
        return None

    disc = max(0.0, disc)
    sqrt_disc = math.sqrt(disc)
    r1 = (-b - sqrt_disc) / (2.0 * a)
    r2 = (-b + sqrt_disc) / (2.0 * a)
    lo = max(0.0, min(r1, r2))
    hi = max(r1, r2)
    if hi < 0.0:
        return None
    return (lo, hi)


def _solve_exhaustive_allocation(K, fidelity_limit, y1, y2, per_link_k_cap=None):
    """Solve a source allocation exactly by enumerating integer k vectors."""
    N_links = len(y1)
    best_obj = float("-inf")
    best_mu = None
    best_k = None
    best_prelog = None
    tol = 1e-10

    for k in _all_k_vectors(K, N_links, per_link_k_cap=per_link_k_cap):
        mu_lb = 0.0
        mu_ub = float("inf")
        feasible = True

        for i in range(N_links):
            interval = _mu_interval_for_link(k[i], y1[i], y2[i], fidelity_limit[i])
            if interval is None:
                feasible = False
                break
            mu_lb = max(mu_lb, interval[0])
            mu_ub = min(mu_ub, interval[1])
            if mu_lb > mu_ub + tol:
                feasible = False
                break

        if not feasible or not math.isfinite(mu_ub):
            continue

        mu_star = max(mu_ub, 1e-15)
        prelog = []
        obj = 0.0
        for i in range(N_links):
            expr_val = (
                mu_star**2 * (k[i] ** 2)
                + mu_star * k[i] * (2 * (y1[i] + y2[i]) + 1)
                + 4 * y1[i] * y2[i]
            )
            if expr_val <= 0.0:
                feasible = False
                break
            prelog.append(expr_val)
            obj += math.log10(expr_val)

        if not feasible:
            continue

        if obj > best_obj + tol:
            best_obj = obj
            best_mu = mu_ub
            best_k = list(k)
            best_prelog = prelog

    if best_k is None or best_mu is None or best_prelog is None:
        raise RuntimeError("Infeasible allocation in exhaustive search")

    return best_obj, best_mu, best_k, best_prelog


def allocate_combo(combo, network, sources, cfg):
    """Allocate channels for a combo by solving each source's MINLP independently
    (with caching) and summing the per-source utilities. Returns a dict with
    success, total utility and a per-option allocation map, or {"success": False}
    if any source sub-problem is infeasible."""
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
        fidelity_limit = tuple(
            float(max(0.5, o.get("fidelity_limit", 0.5))) for o in opts
        )

        cache_key = (K, y1, y2, fidelity_limit)
        cached = _ALLOC_CACHE.get(cache_key)

        if cached == "infeasible":
            return {"success": False}

        if cached is None:
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
                if len(_ALLOC_CACHE) < _ALLOC_CACHE_MAX:
                    _ALLOC_CACHE[cache_key] = "infeasible"
                return {"success": False}

            if len(_ALLOC_CACHE) < _ALLOC_CACHE_MAX:
                _ALLOC_CACHE[cache_key] = cached

        for i, o in enumerate(opts):
            # `prelog` is the loss-normalized ("reduced") rate returned by the
            # MINLP. Convert to the physical log10 rate by subtracting the
            # end-to-end path loss (dB -> factor of 10) and the source-rate
            # normalization tau. We accumulate log10-rates so the total
            # utility is sum(log10 R_i), i.e. log10 of the product of rates.
            reduced_rate = float(cached["prelog"][i])
            reduced_log_rate = math.log10(reduced_rate)

            physical_log_rate = (
                reduced_log_rate
                - float(o["total_loss"]) / 10.0
                - math.log10(float(cfg.tau))
            )

            total_utility += physical_log_rate

            allocations[id(o)] = {
                "source": s,
                "link": o["link"],
                "link_idx": o["link_idx"],
                "k": int(cached["k"][i]),
                "mu": float(cached["mu"]),
                "prelog_rate": reduced_rate,
                # Physical (loss-and-tau-corrected) log10 rate for this link;
                # downstream reporting compares this against link_ub.
                "link_utility": physical_log_rate,
            }

    return {
        "success": True,
        "utility": float(total_utility),
        "allocation": allocations,
    }


def matt(K, fidelity_limit, y1, y2, initial, per_link_k_cap=None, verbose=True, solver_mode="auto"):
    """
    Solve one source-specific EFA allocation problem.

    solver_mode:
        "auto"       -> exhaustive when C(K,L) < 1e4, APOPT otherwise
        "exhaustive" -> force exact integer enumeration
        "apopt"      -> force GEKKO/APOPT

    The production pipeline calls matt() without solver_mode, so its behavior
    remains the hybrid policy.  Validation scripts can force both solvers on
    the same problem instance for an apples-to-apples comparison.
    """
    N_links = len(y1)
    k_combos = math.comb(K, N_links) if K >= N_links and N_links >= 0 else 0

    solver_mode = str(solver_mode).strip().lower()
    if solver_mode not in {"auto", "exhaustive", "apopt"}:
        raise ValueError(
            "solver_mode must be one of: 'auto', 'exhaustive', 'apopt'."
        )

    use_exhaustive = (
        solver_mode == "exhaustive"
        or (
            solver_mode == "auto"
            and k_combos < _EXHAUSTIVE_K_COMBINATION_LIMIT
        )
    )

    if use_exhaustive:
        objective_value, optimal_mu, optimal_allocation, prelog_rates = _solve_exhaustive_allocation(
            K,
            fidelity_limit,
            y1,
            y2,
            per_link_k_cap=per_link_k_cap,
        )
        if verbose:
            print(
                f"EFA solver=exhaustive, C(K,L)={k_combos:,}, "
                f"objective={objective_value}"
            )
        return (
            [_StaticVar(k) for k in optimal_allocation],
            objective_value,
            _StaticVar(optimal_mu),
            prelog_rates,
            optimal_mu,
            optimal_allocation,
        )

    # Create GEKKO model
    m = GEKKO(remote=False)
    m.options.IMODE = 3
    m.options.SOLVER = 1
    # APOPT options tuned for fast convergence on small MINLPs.
    m.solver_options = [
        'minlp_maximum_iterations 200',     # cap branches; small problems converge fast
        'minlp_max_iter_with_int_sol 25',   # stop quickly once an integer solution is found
        'minlp_gap_tol 1e-3',               # 0.1% optimality gap is plenty for log10 obj
        'minlp_branch_method 1',            # depth-first branching
        'minlp_integer_tol 1e-2',
        'minlp_as_nlp 0',
        'nlp_maximum_iterations 500',
    ]

    # Decision variable: mu (continuous, positive)
    mu_init = max(initial, 0.01)
    mu = m.Var(value=mu_init, lb=1e-9)

    if N_links == 1:
        # Only one link: skip the integer search and force k = 1.
        k_vars = [m.Var(value=1, integer=True, lb=1, ub=1)]
        m.Equation(sum(k_vars) == 1)
    else:
        ub_each = K if per_link_k_cap is None else min(K, int(per_link_k_cap))
        # Warm start: distribute channels proportionally to link "quality"
        total_quality = sum(1.0 / (y1[i] + y2[i] + 1e-9) for i in range(N_links))
        k_init = []
        for i in range(N_links):
            quality = 1.0 / (y1[i] + y2[i] + 1e-9)
            k_guess = max(1, min(ub_each, int(K * quality / total_quality)))
            k_init.append(k_guess)
        k_vars = [
            m.Var(value=k_init[i], integer=True, lb=1, ub=ub_each)
            for i in range(N_links)
        ]
        m.Equation(sum(k_vars) <= K)

    # Fidelity constraints per link
    for i in range(N_links):
        expr = mu**2 * k_vars[i]**2 + mu * k_vars[i] * (2 * (y1[i] + y2[i]) + 1) + 4 * y1[i] * y2[i]
        m.Equation(0.25 * (1 + (3 * mu * k_vars[i]) / expr) >= fidelity_limit[i])

    # Objective: maximize sum log10(expr_i)
    obj = 0
    for i in range(N_links):
        expr = mu**2 * k_vars[i]**2 + mu * k_vars[i] * (2 * (y1[i] + y2[i]) + 1) + 4 * y1[i] * y2[i]
        obj += m.log10(expr)
    m.Obj(-obj)

    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            m.solve(disp=False)

    # Sanitize k and compute mu* analytically from the rounded integer solution.
    ub_each = K if per_link_k_cap is None else min(K, int(per_link_k_cap))
    raw_k = [kv.value[0] for kv in k_vars]
    k_int = [int(round(v)) for v in raw_k]
    k_int = [min(max(1, k), ub_each) for k in k_int]

    # Cheap repair to enforce sum(k) <= K after rounding
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

    # Closed-form mu* from fidelity equalities (upper root, then min across links)
    def mu_upper_for_link(i, k_i):
        """Closed-form upper mu root where link i meets its fidelity limit at k_i channels."""
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

    mu_star = (
        min(mu_upper_for_link(i, k_int[i]) for i in range(N_links))
        if N_links > 0
        else float(mu.value[0])
    )

    optimal_mu = mu_star
    optimal_allocation = k_int[:]

    # Compute objective and per-link expr with (mu*, k_int)
    prelog_rates = []
    objective_value = 0.0
    for i in range(N_links):
        ki = optimal_allocation[i]
        expr_val = (
            optimal_mu**2 * (ki ** 2)
            + optimal_mu * ki * (2 * (y1[i] + y2[i]) + 1)
            + 4 * y1[i] * y2[i]
        )
        prelog_rates.append(expr_val)
        objective_value += math.log10(expr_val)

    if verbose:
        print(
            f"EFA solver=APOPT, C(K,L)={k_combos:,}, "
            f"objective={objective_value}"
        )
    return k_vars, objective_value, mu, prelog_rates, optimal_mu, optimal_allocation
