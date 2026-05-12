#allocation/allocator.py
"""
Allocate channels using APOPT via GEKKO to solve MINLP for each source independently and then combine results.
"""

#ZHG
#2026.03.26
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from gekko import GEKKO
import numpy as np
import math
import os
import contextlib    
from collections import defaultdict

def allocate_combo(combo, network, sources, cfg):
    by_source = defaultdict(list)

    #Group options by source
    for option in combo:
        by_source[option["source"]].append(option)

    total_utility = 0
    allocations = {}

    #Run allocation for each source independently, then combine results
    for s, opts in by_source.items():
        #Obtain parameters for this source's options
        K = len(sources[s]["available_channels"])
        y1 = [o["y1"] for o in opts]
        y2 = [o["y2"] for o in opts]
        fidelity_limit = [float(max(0.5, o.get("fidelity_limit", 0.5))) for o in opts]

        try: #Run allocation for this source using APOPT
            k_alloc, obj, mu, prelog_rates, optimal_mu, optimal_allocation = matt(
                K,
                fidelity_limit,
                y1,
                y2,
                initial=0.001,
                verbose=False
            )

            total_utility += float(obj)

            #Store allocation results for each option from this source
            for i, o in enumerate(opts):
                allocations[id(o)] = {
                    "source": s,
                    "link": o["link"],
                    "link_idx": o["link_idx"],
                    "k": int(optimal_allocation[i]),
                    "mu": float(optimal_mu),
                    "prelog_rate": float(prelog_rates[i]),
                }

        #Failure to find solution for a source should invalidate the entire combo, so we can return early here.
        except Exception:
            return {"success": False}

    #Return combined results across all sources
    return {
        "success": True,
        "utility": float(total_utility),
        "allocation": allocations
    }

def matt(K, fidelity_limit, y1, y2, initial, per_link_k_cap=None, verbose = True):
    """Solves the MINLP for a single source's allocation problem using GEKKO with the APOPT solver. Also JACKED"""
    N_links = len(y1)

    #Create GEKKO model
    m = GEKKO(remote=False)
    m.options.IMODE = 3
    m.options.SOLVER = 1
    m.solver_options = [
        'minlp_maximum_iterations 1000',
        'minlp_gap_tol 1e-4',
        'minlp_branch_method 2',
        'nlp_maximum_iterations 2000'
    ]

    #Decision variable: mu (continuous, positive)
    mu = m.Var(value=initial, lb=1e-9)

    if N_links == 1: #If only one link, we can skip the integer variable and just set k=1
        k_vars = [m.Var(value=1, integer=True, lb=1, ub=1)]
        m.Equation(sum(k_vars) == 1)
    else: #Multiple links: need integer k variables and sum(k) <= K constraint
        ub_each = K if per_link_k_cap is None else min(K, int(per_link_k_cap))
        k_vars = [m.Var(value=K//N_links, integer=True, lb=1, ub=ub_each) for i in range(N_links)]
        m.Equation(sum(k_vars) <= K)

    #Fidelity constraints per link
    for i in range(N_links):
        expr = mu**2 * k_vars[i]**2 + mu * k_vars[i]*(2*(y1[i]+y2[i]) + 1) + 4*y1[i]*y2[i]
        m.Equation(0.25*(1 + (3*mu*k_vars[i]) / expr) >= fidelity_limit[i])

    #Objective: maximize sum log10(expr_i)
    obj = 0
    for i in range(N_links):
        expr = mu**2 * k_vars[i]**2 + mu * k_vars[i]*(2*(y1[i]+y2[i]) + 1) + 4*y1[i]*y2[i]
        obj += m.log10(expr)
    m.Obj(-obj)

    #Solve MINLP
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            m.solve(disp=False)

    #sanitize k and compute μ analytically
    #Round & clip integers
    ub_each = K if per_link_k_cap is None else min(K, int(per_link_k_cap))
    raw_k = [kv.value[0] for kv in k_vars]
    k_int = [int(round(v)) for v in raw_k]
    k_int = [min(max(1, k), ub_each) for k in k_int]

    #Cheap repair to enforce sum(k) <= K after rounding
    if N_links > 1:
        total = sum(k_int)
        if total > K:
            #Prefer to decrement those rounded up the most
            round_up_amt = [k_int[i] - raw_k[i] for i in range(N_links)]
            while total > K:
                candidates = [(i, round_up_amt[i]) for i in range(N_links) if k_int[i] > 1]
                if not candidates:
                    break
                i_star = max(candidates, key=lambda t: t[1])[0]
                k_int[i_star] -= 1
                total -= 1

    #Closed-form μ* from fidelity equalities (use the UPPER root; then take min across links)
    def mu_upper_for_link(i, k_i):
        F = fidelity_limit[i]
        s = y1[i] + y2[i]
        p = y1[i] * y2[i]
        t = 4.0*F - 1.0  #> 0
        A = t * (k_i**2)
        B = t * k_i * (2.0*s + 1.0) - 3.0 * k_i
        C = 4.0 * t * p
        #Discriminant
        D = B*B - 4.0*A*C
        if D <= 0.0:
            #numerically tight; fall back to current mu value to be safe
            return max(float(mu.value[0]), 1e-9)
        sqrtD = math.sqrt(D)
        #Upper root
        mu_hi = (-B + sqrtD) / (2.0 * A)
        return max(mu_hi, 1e-9)

    mu_star = min(mu_upper_for_link(i, k_int[i]) for i in range(N_links)) if N_links > 0 else float(mu.value[0])

    optimal_mu = mu_star
    optimal_allocation = k_int[:]

    #Compute objective and per-link expr with (μ*, k_int)
    prelog_rates = []
    objective_value = 0.0
    for i in range(N_links):
        ki = optimal_allocation[i]
        expr_val = (optimal_mu**2 * (ki**2) + optimal_mu * ki * (2*(y1[i]+y2[i]) + 1) + 4*y1[i]*y2[i])
        prelog_rates.append(expr_val)
        objective_value += math.log10(expr_val)

    if verbose:
        print("Calculated Objective Value (log10):", objective_value)
    return k_vars, objective_value, mu, prelog_rates, optimal_mu, optimal_allocation