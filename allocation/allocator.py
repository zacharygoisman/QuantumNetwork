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


def matt(K, fidelity_limit, y1, y2, initial, per_link_k_cap=None, verbose=True):
    """Solves the MINLP for a single source's allocation problem using GEKKO with the APOPT solver."""
    N_links = len(y1)

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
        print("Calculated Objective Value (log10):", objective_value)
    return k_vars, objective_value, mu, prelog_rates, optimal_mu, optimal_allocation
