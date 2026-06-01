"""
allocation/comparison.py
========================

Allocation-only comparison study.

Reproduces the hard-coded benchmark cases (Cases 1-4) and renders the matching
output figures. For each total-channel budget ``K`` the module:

1. Solves a per-link channel allocation with GEKKO (APOPT MINLP solver),
   optionally pruning low-capacity links when doing so improves utility.
2. Records the resulting fluxes, fidelities and normalized rates.
3. Saves the link-fidelity, channel-bar, rate-utility and combined figures
   into the ``outputs/`` directory.

Run directly to reproduce a case::

    python -m allocation.comparison
"""

import contextlib
import math
import os
import string
import time
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from gekko import GEKKO
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator
from scipy.optimize import brentq, fsolve
from comparisons import fidelity_rate_plot
from comparisons import channel_bar_plot
from comparisons import rate_bar_plot


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
FS = 28  # one readable size for all text

# Combined-figure export controls. Inkscape uses 96 px per inch by default.
COMBINED_WIDTH_PX = 1906.394
INKSCAPE_PX_PER_INCH = 96.0
COMBINED_FIG_WIDTH = COMBINED_WIDTH_PX / INKSCAPE_PX_PER_INCH
COMBINED_MARKER_SIZE = 150
COMBINED_MARKER_LINEWIDTH = 1.6
# Give the right-hand utility/allocation column more room inside the fixed SVG width.
COMBINED_RIGHT_WIDTH_BOOST = 1.20
# Make the two right-hand panels taller so the allocation legend has more vertical breathing room.
COMBINED_RIGHT_HEIGHT_BOOST = 1.12


def save_svg_exact(fig, path):
    """Save without tight bbox cropping so the combined SVG keeps its physical width."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches=None)


# =========================================================================== #
# Link physics & channel-count helpers
# =========================================================================== #
def _k_int_from_vars(k_vars, K):
    """Round/clip integer k and repair sum(k) <= K (no equation changes)."""
    raw = [kv.value[0] for kv in k_vars]
    k_int = [int(round(v)) for v in raw]
    k_int = [min(max(1, k), int(K)) for k in k_int]

    total = sum(k_int)
    if total > K and len(k_int) > 1:
        round_up_amt = [k_int[i] - raw[i] for i in range(len(k_int))]
        while total > K:
            candidates = [(i, round_up_amt[i]) for i in range(len(k_int)) if k_int[i] > 1]
            if not candidates:
                break
            i_star = max(candidates, key=lambda t: t[1])[0]
            k_int[i_star] -= 1
            total -= 1
    return k_int



def compute_fidelity_rate(tau, mu, y1_val, y2_val):
    """
    For a given link (with parameters y1 and y2), compute fidelity and rate at
    x = tau * mu. This version handles x=0 safely, including the special case
    expr(0)=0, by using the x->0 limit instead of direct division.
    """
    x = np.asarray(tau * mu, dtype=float)
    A = 2 * (y1_val + y2_val) + 1
    B = 4 * y1_val * y2_val

    expr = x**2 + A * x + B

    ratio = np.empty_like(x, dtype=float)
    mask_nonzero = expr != 0.0
    ratio[mask_nonzero] = 3.0 * x[mask_nonzero] / expr[mask_nonzero]

    # When expr == 0, this only occurs at x == 0 and B == 0.
    # Then x^2 + A x = x(x + A), so
    #   3x / expr = 3 / (x + A)  ->  3 / A  as x -> 0.
    ratio[~mask_nonzero] = 3.0 / A

    fidelity = 0.25 * (1.0 + ratio)

    with np.errstate(divide="ignore", invalid="ignore"):
        rate = expr * np.log2(2.0 * fidelity)

    # If expr is exactly zero, the physical rate limit is zero.
    rate = np.where(expr == 0.0, 0.0, rate)

    if np.isscalar(tau * mu):
        return float(fidelity), float(rate), float(x)
    return fidelity, rate, x



@lru_cache(maxsize=None)
def compute_R_max_full(y1_val, y2_val):
    """
    For a given link (with parameters y1 and y2), compute the maximum possible
    rate from the rate equation
         Rate(x) = [ x^2 + (2*(y1+y2)+1)*x + 4*y1*y2 ] * log2( 2F(x) )
    where
         F(x) = 0.25*(1 + 3*x / [ x^2 + (2*(y1+y2)+1)*x + 4*y1*y2 ])
    The function samples x in a fixed interval and returns the maximum value.
    """
    A = 2 * (y1_val + y2_val) + 1
    B = 4 * y1_val * y2_val

    xs = np.arange(1, 100000, dtype=float) / 10000.0  # x from 0.0001 to 10
    expr_vals = xs**2 + A * xs + B
    F_vals = 0.25 * (1.0 + 3.0 * xs / expr_vals)
    rates = expr_vals * np.log2(2.0 * F_vals)
    return float(np.max(rates))



# =========================================================================== #
# Allocation & optimization
# =========================================================================== #
def _solve_active_set(K, y1, y2, fidelity_limit, active_idx, initial):
    """Solve GEKKO for a subset of links (active_idx). Returns numeric allocations on active set."""
    y1_a = [y1[i] for i in active_idx]
    y2_a = [y2[i] for i in active_idx]
    f_a = [fidelity_limit[i] for i in active_idx]

    k_vars, raw_obj, mu, prelog_rates = matt_comparison_gecko(K, f_a, y1_a, y2_a, initial)

    k_int = _k_int_from_vars(k_vars, K)
    mu_val = float(mu.value[0]) if hasattr(mu, "value") and mu.value else float(mu)

    per_link_flux = [mu_val * ki for ki in k_int]

    return {
        "k_int": k_int,
        "mu": mu_val,
        "raw_obj": float(raw_obj),
        "prelog_rates": list(map(float, prelog_rates)),
        "per_link_flux": per_link_flux,
    }



def prune_links_for_K(K, y1, y2, fidelity_limit, initial, base_active_idx=None, removal_cost=1.0):
    """
    Iteratively remove the lowest-capacity link, re-solve, and keep the removal
    only if effective utility improves, where effective = raw_obj - cost*(#removed).
    Returns full-length arrays (aligned to original links) with removed links = 0.
    """
    L = len(y1)
    if base_active_idx is None:
        active = list(range(L))
    else:
        active = list(base_active_idx)

    flux_cap = [max_allowed_flux_x(y1[i], y2[i], fidelity_limit[i]) for i in range(L)]

    removed_count = 0
    cur = _solve_active_set(K, y1, y2, fidelity_limit, active, initial)
    cur_effective = cur["raw_obj"] - removal_cost * removed_count

    while len(active) > 1:
        remove_global = min(active, key=lambda idx: flux_cap[idx])
        cand_active = [idx for idx in active if idx != remove_global]
        cand = _solve_active_set(K, y1, y2, fidelity_limit, cand_active, initial)
        cand_effective = cand["raw_obj"] - removal_cost * (removed_count + 1)

        if cand_effective > cur_effective:
            active = cand_active
            removed_count += 1
            cur = cand
            cur_effective = cand_effective
        else:
            break

    alloc_full = np.zeros(L, dtype=float)
    rates_full = np.zeros(L, dtype=float)

    for local_i, global_i in enumerate(active):
        alloc_full[global_i] = cur["k_int"][local_i]
        rates_full[global_i] = cur["prelog_rates"][local_i]

    return {
        "active_idx": active,
        "alloc_full": alloc_full,
        "rates_full": rates_full,
        "mu": cur["mu"],
        "raw_obj": cur["raw_obj"],
        "effective_obj": cur_effective,
        "removed_count": removed_count,
    }



def matt_comparison_gecko(K, fidelity_limit, y1, y2, initial):
    N_links = len(y1)

    R_max_list = []
    for i in range(N_links):
        R_max = compute_R_max_full(y1[i], y2[i])
        if R_max == 0:
            R_max = -1e-6
        R_max_list.append(R_max)

    m = GEKKO(remote=False)
    m.options.IMODE = 3
    m.options.SOLVER = 1
    m.solver_options = [
        'minlp_maximum_iterations 1000',
        'minlp_gap 0.0001',
        'minlp_branch_method 2',
        'nlp_maximum_iterations 2000',
    ]

    mu = m.Var(value=float(initial), lb=1e-9)

    if N_links == 1:
        k_vars = [m.Var(value=1, integer=True, lb=1, ub=1)]
        m.Equation(sum(k_vars) == 1)
    else:
        start_val = max(1, K // N_links)
        k_vars = [m.Var(value=start_val, integer=True, lb=1, ub=K) for _ in range(N_links)]
        m.Equation(sum(k_vars) <= K)

    for i in range(N_links):
        x = mu * k_vars[i]
        expr_i = x**2 + (2 * (y1[i] + y2[i]) + 1) * x + 4 * y1[i] * y2[i]
        F = 0.25 * (1 + 3 * x / expr_i)
        m.Equation(F >= fidelity_limit[i])

    obj = 0
    for i in range(N_links):
        x = mu * k_vars[i]
        expr_i = x**2 + (2 * (y1[i] + y2[i]) + 1) * x + 4 * y1[i] * y2[i]
        F = 0.25 * (1 + 3 * x / expr_i)
        rate_expr = expr_i * (m.log(2 * F) / math.log(2))
        normalized_rate = rate_expr / R_max_list[i]
        obj += normalized_rate
    m.Obj(-obj)

    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            m.solve(disp=False)

    raw_k = [kv.value[0] for kv in k_vars]
    k_int = [int(round(v)) for v in raw_k]
    k_int = [min(max(1, k), int(K)) for k in k_int]

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

    def _mu_upper_for_link(i, k_i):
        F = float(fidelity_limit[i])
        s = float(y1[i] + y2[i])
        p = float(y1[i] * y2[i])
        t = 4.0 * F - 1.0
        if t <= 0.0 or k_i <= 0:
            return max(float(mu.value[0]), 1e-9)
        A = t * (k_i ** 2)
        B = t * k_i * (2.0 * s + 1.0) - 3.0 * k_i
        C = 4.0 * t * p
        D = B * B - 4.0 * A * C
        if D <= 0.0:
            return max(float(mu.value[0]), 1e-9)
        sqrtD = math.sqrt(D)
        mu_hi = (-B + sqrtD) / (2.0 * A)
        return max(mu_hi, 1e-9)

    mu_star = min(_mu_upper_for_link(i, k_int[i]) for i in range(N_links)) if N_links > 0 else float(mu.value[0])

    try:
        mu.value = [float(mu_star)]
    except Exception:
        pass

    prelog_rates = []
    objective_value = 0.0
    for i in range(N_links):
        x_val = float(mu_star) * float(k_int[i])
        expr_val = x_val**2 + (2.0 * (y1[i] + y2[i]) + 1.0) * x_val + 4.0 * y1[i] * y2[i]
        F_val = 0.25 * (1.0 + 3.0 * x_val / expr_val)
        rate_val = expr_val * (math.log(2.0 * F_val) / math.log(2.0))
        normalized_rate_val = rate_val / R_max_list[i]
        objective_value += normalized_rate_val
        prelog_rates.append(normalized_rate_val)

    return k_vars, objective_value, mu, prelog_rates


# =========================================================================== #
# Flux-capacity helper
# =========================================================================== #
@lru_cache(maxsize=None)
def max_allowed_flux_x(y1, y2, f_th, x_hi=1e6):
    """
    Smallest "total possible flux capacity" criterion:
    returns the LARGEST x>=0 such that fidelity(x) == f_th (upper crossing).
    If f_th <= 0.25, fidelity constraint is inactive -> capacity = +inf.
    """
    f_th = float(f_th)
    if f_th <= 0.25:
        return float('inf')

    y1 = float(y1)
    y2 = float(y2)
    A = 2.0 * (y1 + y2) + 1.0
    B = 4.0 * y1 * y2

    def fidelity(x):
        expr = x * x + A * x + B
        return 0.25 * (1.0 + 3.0 * x / expr)

    def g(x):
        return fidelity(x) - f_th

    xs = np.logspace(-12, np.log10(x_hi), 600)
    vals = np.array([g(x) for x in xs])

    brackets = []
    for i in range(len(xs) - 1):
        if np.isnan(vals[i]) or np.isnan(vals[i + 1]):
            continue
        if vals[i] == 0:
            brackets.append((xs[i], xs[i]))
        elif vals[i] * vals[i + 1] < 0:
            brackets.append((xs[i], xs[i + 1]))

    if not brackets:
        for guess in [1e-6, 1e-3, 1e-1, 1.0, 10.0, 1e3]:
            try:
                root, = fsolve(lambda x: g(x), guess, maxfev=2000)
                if root > 0:
                    return float(root)
            except Exception:
                pass
        return float('inf')

    roots = []
    for a, b in brackets:
        if a == b:
            roots.append(a)
        else:
            try:
                roots.append(brentq(g, a, b, maxiter=2000))
            except Exception:
                pass

    return float(max(roots)) if roots else float('inf')



# =========================================================================== #
# Orchestration
# =========================================================================== #
def run_plots(k_list, fidelity_limit, tau, y1, y2, text, initial=0.001, skip=False):
    """
    Solve every channel budget in ``k_list`` and render the comparison figures.

    Args:
        k_list: source channel budgets (K) to evaluate.
        fidelity_limit: per-link fidelity lower bounds.
        tau: coincidence time in seconds.
        y1: y1 parameter for every link.
        y2: y2 parameter for every link.
        text: label used to differentiate the output plots (e.g. "Case 4").
        initial: initial mu value for the APOPT solver.
        skip: when True, drop the last link for the K=12 and K=24 budgets.
    """
    k_vars_array = []
    mu_array = []
    objective_array = []

    print(f"\n=== {text} ===")
    case_t0 = time.perf_counter()
    perK_times = []

    for k in k_list:
        t0 = time.perf_counter()

        if skip and k in [12, 24]:
            base_active = list(range(len(y1) - 1))
        else:
            base_active = list(range(len(y1)))

        result = prune_links_for_K(
            K=k,
            y1=y1,
            y2=y2,
            fidelity_limit=fidelity_limit,
            initial=initial,
            base_active_idx=base_active,
            removal_cost=1.0,
        )

        dt = time.perf_counter() - t0
        perK_times.append(dt)

        alloc_full = result['alloc_full']
        mu_val = result['mu']
        per_link_flux = (mu_val * alloc_full).tolist()
        total_flux = float(np.sum(mu_val * alloc_full))

        k_vars_array.append(alloc_full)
        mu_array.append(mu_val)
        objective_array.append(result['effective_obj'])

        print(
            f"{text} | K={int(k):>3} | raw_utility={result['raw_obj']:.6f} | "
            f"effective_utility={result['effective_obj']:.6f} | removed={result['removed_count']} | "
            f"mu={mu_val:.6g} | total_flux=∑(μ·k)={total_flux:.6g} | "
            f"per_link_flux={per_link_flux} | time={dt*1000:.1f} ms"
        )

    case_dt = time.perf_counter() - case_t0
    print(f"{text} | total_time={case_dt:.3f}s (per-K avg={np.mean(perK_times)*1000:.1f} ms)")

    fidelity_rate_plot(k_list, k_vars_array, mu_array, tau, y1, y2, fidelity_limit, text)
    channel_bar_plot(k_list, k_vars_array, text)
    rate_bar_plot(k_list, k_vars_array, objective_array, mu_array, tau, y1, y2, fidelity_limit, text)

    # Combined figure: link fidelity on the left, rate-utility + channel bars
    # stacked on the right (saved as comp_combined_{case}). This is a "nice to
    # have" extra, so a failure here should not abort the rest of the run.
    try:
        combined_plot(k_list, k_vars_array, objective_array, mu_array, tau, y1, y2, fidelity_limit, text)
    except Exception:
        pass



def get_case(case_num):
    """Return the hard-coded parameters for benchmark case 1, 2, 3 or 4."""
    tau = 1e-9

    if case_num == 1:
        return {
            'k_list': [5, 10, 20, 40],
            'y1': [0, 0.04, 0, 0.11, 0.15],
            'y2': [0, 0.007, 0.125, 0.019, 0.025],
            'fidelity_limit': np.repeat(0, 5),
            'tau': tau,
            'text': 'Case 1',
            'initial': 0.001,
            'skip': False,
        }

    if case_num == 2:
        return {
            'k_list': [5, 10, 20, 40],
            'y1': [0, 0.04, 0, 0.11, 0.15],
            'y2': [0, 0.007, 0.125, 0.019, 0.025],
            'fidelity_limit': np.repeat(0.7, 5),
            'tau': tau,
            'text': 'Case 2',
            'initial': 0.001,
            'skip': False,
        }

    if case_num == 3:
        return {
            'k_list': [5, 10, 20, 40],
            'y1': [0, 0.0034, 0.0104, 0.0179, 0],
            'y2': [0, 0.006, 0.0018, 0.0031, 0.0515],
            'fidelity_limit': np.repeat(0.9, 5),
            'tau': tau,
            'text': 'Case 3',
            'initial': 0.001,
            'skip': False,
        }

    if case_num == 4:
        return {
            'k_list': [12, 24, 48, 96],
            'y1': [0, 0.0034, 0, 0.0299, 0.0385, 0.0625, 0.0733, 0, 0.1106, 0.125, 0.1489, 0],
            'y2': [0, 0.0006, 0.0357, 0.0051, 0.0066, 0.0107, 0.0126, 0.1818, 0.019, 0.0214, 0.0256, 0.2979],
            'fidelity_limit': np.repeat(0.7, 12),
            'tau': tau,
            'text': 'Case 4',
            'initial': 0.001,
            'skip': False,
        }

    raise ValueError('case_num must be 1, 2, 3, or 4')



def run_case(case_num):
    """Look up ``case_num`` and run the full plotting pipeline for it."""
    case = get_case(case_num)
    run_plots(
        case['k_list'],
        case['fidelity_limit'],
        case['tau'],
        case['y1'],
        case['y2'],
        case['text'],
        initial=case['initial'],
        skip=case['skip'],
    )



def main():
    # Set to 1, 2, 3 or 4 to reproduce the corresponding benchmark case.
    run_case(4)


if __name__ == '__main__':
    main()
