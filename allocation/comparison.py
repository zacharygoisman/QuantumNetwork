# comparison.py
# Recreated to preserve the same allocation-only functionality as compare.py,
# with the same hard-coded cases and the same output plots.

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

FS = 18  # one readable size for all text


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



def fidelity_rate_plot(k_list, k_vars_array, objective_value_array, mu_determined,
                       tau, y1_array, y2_array, fidelity_limit, text):
    case_num = text[-1]
    xlimit = {'1': 1.2, '2': 0.8, '3': 0.2, '4': 0.7}.get(case_num, 1.0)

    l = len(y1_array)

    if l == 5:
        rows, cols = 3, 2
        legend_mode = 'panel'
        legend_tile_index = 1 * cols + 1
        fig_size = (6.1, 7.0)
        wspace, hspace = 0.35, 0.35
        bottom_margin = 0.08
    elif l == 12:
        rows, cols = 4, 3
        legend_mode = 'figure'
        legend_tile_index = None
        fig_size = (11.0, 8.6)
        wspace, hspace = 0.30, 0.30
        bottom_margin = 0.08
    else:
        cols = 2 if l <= 6 else 3
        rows = (l + cols - 1) // cols
        legend_mode = 'figure'
        legend_tile_index = None
        fig_size = (11, 8)
        wspace, hspace = 0.30, 0.30
        bottom_margin = 0.08

    fig, axs = plt.subplots(rows, cols, figsize=fig_size)
    fig.subplots_adjust(wspace=wspace, hspace=hspace, bottom=bottom_margin)
    if rows * cols == 1:
        axs = np.array([[axs]])
    elif rows == 1:
        axs = np.array([axs])
    elif cols == 1:
        axs = axs[:, None]

    matlab_blue = '#0072BD'
    matlab_orang = '#D95319'

    line_handles, line_labels = [], []
    proxy_hline = Line2D([0], [0], color='k', linestyle='dotted', linewidth=0.8,
                         label='Fidelity threshold')
    proxy_vline = Line2D([0], [0], color='k', linestyle='--', linewidth=0.8,
                         label='Max allowed flux')

    mu_total = 1.2 / tau
    mu_channel_array = [mu_determined[i] / tau for i in range(len(k_list))]
    max_rate_list = [compute_R_max_full(y1_array[i], y2_array[i]) for i in range(l)]
    max_flux_list = [max_allowed_flux_x(y1_array[i], y2_array[i], fidelity_limit[i]) for i in range(l)]

    floored_ratio_array = []
    for i in range(len(k_list)):
        arr = np.asarray(k_vars_array[i], dtype=float)
        if len(arr) < l:
            arr = np.pad(arr, (0, l - len(arr)), mode='constant', constant_values=0.0)
        elif len(arr) > l:
            arr = arr[:l]
        floored_ratio_array.append(arr)

    base_markers = ['o', '^', 's', 'd', 'x', 'v', 'P', '*']

    def pick_marker(i):
        return base_markers[i % len(base_markers)]

    letters = string.ascii_uppercase
    link_labels = [letters[2 * i] + letters[2 * i + 1] for i in range(l)]

    x_ticks = np.linspace(0, xlimit, 3)
    y_ticks = [0.0, 0.5, 1.0]

    for link_idx in range(l):
        plot_idx = link_idx if legend_mode != 'panel' else (link_idx if link_idx < legend_tile_index else link_idx + 1)
        r, c = divmod(plot_idx, cols)
        ax = axs[r, c]

        ax.minorticks_off()
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.xaxis.set_major_locator(FixedLocator(x_ticks))
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.tick_params(axis='both', which='major', direction='in', top=False, right=False,
                       bottom=True, left=True, labelsize=FS)

        show_y = (c == 0)
        show_x = (r == rows - 1)
        ax.tick_params(axis='y', labelleft=show_y, labelright=False)
        ax.tick_params(axis='x', labelbottom=show_x, labeltop=False)

        y1, y2 = y1_array[link_idx], y2_array[link_idx]
        mu = np.linspace(1e-6, mu_total, 1000)
        fidelity, rate, flux = compute_fidelity_rate(tau, mu, y1, y2)
        max_rate = max_rate_list[link_idx]
        rate = rate / max_rate

        line1, = ax.plot(tau * mu, fidelity, label='Fidelity', c=matlab_blue)
        line2, = ax.plot(tau * mu, rate, label=r'EBR/EBR$_\text{max}$', c=matlab_orang)

        if not line_handles:
            line_handles.extend([line1, line2, proxy_hline, proxy_vline])
            line_labels.extend([
                'Fidelity',
                r'EBR/EBR$_\text{max}$',
                'Fidelity Threshold',
                'Max Allowed Flux',
            ])

        for ki in range(len(k_list)):
            alloc = floored_ratio_array[ki][link_idx]
            marker = pick_marker(ki)

            if alloc <= 0:
                f_flux = 0.0
                f_fid, f_rate, _ = compute_fidelity_rate(tau, 0.0, y1, y2)
            else:
                f_mu = mu_channel_array[ki] * alloc
                f_fid, f_rate, f_flux = compute_fidelity_rate(tau, f_mu, y1, y2)

            f_rate = f_rate / max_rate

            sc1 = ax.scatter(f_flux, f_fid, marker=marker,
                             facecolors='none', edgecolors=matlab_blue,
                             linewidths=1.2, clip_on=False, zorder=3)
            ax.scatter(f_flux, f_rate, marker=marker,
                       facecolors='none', edgecolors=matlab_orang,
                       linewidths=1.2, clip_on=False, zorder=3)

            if len(line_labels) < 4 + len(k_list):
                line_handles.append(sc1)
                line_labels.append(f'K={k_list[ki]} (APOPT)')

        fid_min = fidelity_limit[link_idx]
        ax.axhline(fid_min, color='k', linestyle='dotted', linewidth=0.8)

        max_flux = max_flux_list[link_idx]
        if np.isfinite(max_flux):
            ax.axvline(max_flux, color='k', linestyle='--', linewidth=0.8)

        ax.set_xlim([0, xlimit])
        ax.set_ylim([0, 1])
        ax.text(0.95, 0.05, link_labels[link_idx], transform=ax.transAxes,
                ha='right', va='bottom', fontsize=FS, fontweight='bold')

    if legend_mode == 'panel':
        r, c = divmod(legend_tile_index, cols)
        ax_leg = axs[r, c]
        ax_leg.axis('off')
        ax_leg.legend(line_handles, line_labels, loc='center',
                      frameon=True, fancybox=True, framealpha=1.0,
                      edgecolor='0.3', fontsize=FS)
    else:
        fig.legend(handles=line_handles, labels=line_labels,
                   loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4,
                   frameon=True, fancybox=True, framealpha=1.0,
                   edgecolor='0.3', fontsize=FS - 2,
                   columnspacing=0.7, handletextpad=0.4,
                   handlelength=1.4, borderpad=0.3, labelspacing=0.3)

    fig.text(0.5, 0.02, r'Dimensionless Flux $\mu K_\ell \tau$', ha='center', fontsize=FS)
    plt.savefig(f'outputs/comp_link_fidelity_{case_num}.svg', dpi=300)
    plt.close()



def channel_bar_plot(channel_numbers, k_vars_array, text):
    case_num = text[-1]
    channel_str = [str(c) for c in channel_numbers]
    n_bars = len(channel_numbers)

    gem = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
           '#77AC30', '#4DBEEE', '#A2142F', '#003BFF', '#017501',
           '#FF0000', '#B526FF', '#FF00FF', '#000000']

    def cycle_from(idx, n):
        return [gem[(idx + i) % len(gem)] for i in range(n)]

    labels = ['Link AB', 'Link CD', 'Link EF', 'Link GH', 'Link IJ',
              'Link KL', 'Link MN', 'Link OP', 'Link QR', 'Link ST',
              'Link UV', 'Link WX', 'Link YZ']

    rows = []
    max_links = 0
    for kv in k_vars_array:
        if isinstance(kv, (list, tuple, np.ndarray)) and len(kv) > 0 and hasattr(kv[0], 'value'):
            row = [int(round(v.value[0])) for v in kv]
        else:
            row = list(np.asarray(kv, dtype=float))
        rows.append(row)
        max_links = max(max_links, len(row))

    alloc_matrix = np.zeros((n_bars, max_links), dtype=float)
    for r, row in enumerate(rows):
        alloc_matrix[r, :len(row)] = row

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.minorticks_off()
    ax.grid(True, axis='y', linestyle='--', linewidth=0.6, color='0.85')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=FS, direction='in',
                   length=4, width=1.0, top=False, right=False, bottom=True, left=True)

    total_per_bar = alloc_matrix.sum(axis=1)
    null_link = np.clip(np.asarray(channel_numbers) - total_per_bar, 0, None)
    bottom = np.zeros(n_bars, dtype=float)
    ax.bar(channel_str, null_link, bottom=bottom,
           color=gem[0], edgecolor='black', linewidth=0.7, label='Unassigned')
    bottom += null_link

    link_colors = cycle_from(1, max_links)
    for i in range(max_links):
        heights = alloc_matrix[:, i]
        ax.bar(channel_str, heights, bottom=bottom,
               color=link_colors[i], edgecolor='black', linewidth=0.7,
               label=labels[i] if i < len(labels) else f'Link {i + 1}')
        bottom += heights

    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    ax.set_ylabel(r'Allocated Channels $K_\ell$', fontsize=FS)
    ax.set_xlabel(r'$Total Channels $K$', fontsize=FS)

    leg_cols = 2 if max_links >= 12 else 1
    ax.legend(loc='upper left', bbox_to_anchor=(-0.00, 1.1), fontsize=FS,
              frameon=True, fancybox=True, framealpha=1.0,
              edgecolor='0.3', ncol=leg_cols)

    plt.tight_layout()
    plt.savefig(f'outputs/comp_channel_barplot_{case_num}.svg', dpi=300)
    plt.close()



def rate_bar_plot(channel_numbers, k_vars_array, objective_value,
                  mu_determined, tau, y1_array, y2_array, fidelity_limit, text):
    case_num = str(text)[-1]
    ylimit_map = {
        '1': [4.9, 5.001],
        '2': [2.0, 4.0],
        '3': [0.2, 1.0],
        '4': [3.0, 8.0],
    }
    ylimit = ylimit_map.get(case_num, None)

    matlab_blue = '#0072BD'

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.minorticks_off()
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('Utility', fontsize=FS)

    channel_str = [str(c) for c in channel_numbers]
    bars = ax.bar(channel_str, objective_value,
                  color=matlab_blue, edgecolor='black', linewidth=0.8)

    comparison_map = {
        '1': [0, 0, 0, 0],
        '2': [0, 0, 0, 0],
        '3': [0, 0, 0, 0],
        '4': [3.98, 5.13, 5.52, 6.51],
    }
    comparison_lines = comparison_map[case_num]

    for rect, y_line in zip(bars, comparison_lines):
        x0 = rect.get_x()
        w = rect.get_width()
        ax.hlines(y_line, x0, x0 + w, colors='red', linewidth=2)

    max_total_utility = 0.0
    for (y1, y2, f_th) in zip(y1_array, y2_array, fidelity_limit):
        A = 2 * (y1 + y2) + 1
        B = 4 * y1 * y2
        R_max_i = compute_R_max_full(y1, y2)
        if f_th <= 0:
            normalized_i = 1.0
        else:
            def fidelity_eq(x):
                return 0.25 * (1 + 3 * x / (x * x + A * x + B)) - f_th
            x_thresh, = fsolve(fidelity_eq, 1.0)
            expr_val = x_thresh**2 + A * x_thresh + B
            rate_at_thresh = expr_val * math.log2(2 * f_th)
            normalized_i = rate_at_thresh / R_max_i
        max_total_utility += normalized_i

    ax.axhline(max_total_utility, color='k', linestyle='--', label='Max Total Utility')

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='red', linewidth=2))
    labels.append('GA Utility')
    ax.legend(handles=handles, labels=labels, loc='upper left', fontsize=FS)

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(axis='both', which='major', direction='in', top=False, right=False,
                   bottom=True, left=True)
    ax.tick_params(axis='x', labelbottom=False, labeltop=False)
    ax.tick_params(axis='y', labelsize=FS, labelright=False)

    if ylimit is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        try:
            ax.set_ylim(float(ylimit[0]), float(ylimit[1]))
            # Show only three y-ticks at 3, 5.5 and 8 for the comp rate utility plot
            ax.set_yticks([3.0, 5.5, 8.0])
            ax.set_yticklabels(["3", "5.5", "8"])
        except Exception:
            pass

    fig.savefig(f'outputs/comp_rate_utility_{case_num}.svg', dpi=300, bbox_inches='tight')
    plt.close(fig)


def combined_plot(channel_numbers, k_vars_array, objective_value,
                  mu_determined, tau, y1_array, y2_array, fidelity_limit, text):
    case_num = str(text)[-1]
    l = len(y1_array)

    matlab_blue = '#0072BD'
    matlab_orang = '#D95319'
    legend_fs = FS

    if l == 5:
        left_fig_w, left_fig_h = 6.1, 7.0
        rows, cols = 3, 2
        left_wspace, left_hspace = 0.18, 0.14
    elif l == 12:
        left_fig_w, left_fig_h = 11.0, 8.6
        rows, cols = 4, 3
        left_wspace, left_hspace = 0.14, 0.14
    else:
        left_fig_w, left_fig_h = 11.0, 8.0
        cols = 2 if l <= 6 else 3
        rows = (l + cols - 1) // cols
        left_wspace, left_hspace = 0.14, 0.14

    right_fig_w = 6.4
    single_right_h = 5.2
    right_hspace_frac = 0.12
    right_fig_h = single_right_h * (2 + right_hspace_frac)

    fig_width = left_fig_w + right_fig_w
    fig_height = max(left_fig_h, right_fig_h)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)

    outer = fig.add_gridspec(
        1, 2,
        width_ratios=[left_fig_w, right_fig_w],
        wspace=0.12,
        left=0.055,
        right=0.985,
        bottom=0.085,
        top=0.91,
    )

    # ---------------- Left column ----------------
    left_outer = outer[0, 0].subgridspec(
        2, 1,
        height_ratios=[0.10, 0.90],
        hspace=0.02
    )
    ax_left_leg = fig.add_subplot(left_outer[0, 0])
    ax_left_leg.axis('off')

    left_sub = left_outer[1, 0].subgridspec(
        rows, cols,
        wspace=left_wspace,
        hspace=left_hspace
    )

    axs = np.empty((rows, cols), dtype=object)
    for r in range(rows):
        for c in range(cols):
            axs[r, c] = fig.add_subplot(left_sub[r, c])

    line_handles, line_labels = [], []
    proxy_hline = Line2D([0], [0], color='k', linestyle='dotted', linewidth=0.8,
                         label='Fidelity Threshold')
    proxy_vline = Line2D([0], [0], color='k', linestyle='--', linewidth=0.8,
                         label='Max Allowed Flux')

    mu_total = 1.2 / tau
    mu_channel_array = [mu_determined[i] / tau for i in range(len(channel_numbers))]
    max_rate_list = [compute_R_max_full(y1_array[i], y2_array[i]) for i in range(l)]
    max_flux_list = [max_allowed_flux_x(y1_array[i], y2_array[i], fidelity_limit[i]) for i in range(l)]

    floored_ratio_array = []
    for i in range(len(channel_numbers)):
        arr = np.asarray(k_vars_array[i], dtype=float)
        if len(arr) < l:
            arr = np.pad(arr, (0, l - len(arr)), mode='constant', constant_values=0.0)
        elif len(arr) > l:
            arr = arr[:l]
        floored_ratio_array.append(arr)

    base_markers = ['o', '^', 's', 'd', 'x', 'v', 'P', '*']

    def pick_marker(i):
        return base_markers[i % len(base_markers)]

    letters = string.ascii_uppercase
    link_labels = [letters[2 * i] + letters[2 * i + 1] for i in range(l)]

    xlimit = {'1': 1.2, '2': 0.8, '3': 0.2, '4': 0.7}.get(case_num, 1.0)
    x_ticks = np.linspace(0, xlimit, 3)
    y_ticks = [0.0, 0.5, 1.0]

    # Explicit x tick labels helps avoid 0.00/0.70 collisions looking worse than needed
    x_ticklabels = [f"{x:.2f}" for x in x_ticks]

    marker_size = 80  # larger markers for APOPT points

    for link_idx in range(l):
        r, c = divmod(link_idx, cols)
        ax = axs[r, c]

        ax.minorticks_off()
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.xaxis.set_major_locator(FixedLocator(x_ticks))
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        ax.set_xticklabels(x_ticklabels)

        # Push edge tick labels inward so they do not collide at subplot boundaries
        xtlbls = ax.get_xticklabels()
        if len(xtlbls) >= 3:
            xtlbls[0].set_ha('left')    # move 0.00 to the right
            xtlbls[1].set_ha('center')
            xtlbls[-1].set_ha('right')  # move 0.70 / right-edge label to the left
            xtlbls[0].set_x(0.05)     # was effectively 0.00; move 0.00 slightly right
            xtlbls[-1].set_x(0.95)    # was effectively 1.00; move 0.70 slightly left

        ax.tick_params(
            axis='both',
            which='major',
            direction='in',
            top=False,
            right=False,
            bottom=True,
            left=True,
            labelsize=FS,
            pad=6,          # keep numbers from crowding axis corners
            length=4
        )

        ax.tick_params(axis='y', labelleft=(c == 0), labelright=False)
        ax.tick_params(axis='x', labelbottom=(r == rows - 1), labeltop=False)

        y1, y2 = y1_array[link_idx], y2_array[link_idx]
        mu = np.linspace(1e-6, mu_total, 1000)
        fidelity, rate, flux = compute_fidelity_rate(tau, mu, y1, y2)
        max_rate = max_rate_list[link_idx]
        rate = rate / max_rate

        line1, = ax.plot(tau * mu, fidelity, c=matlab_blue)
        line2, = ax.plot(tau * mu, rate, c=matlab_orang)

        if not line_handles:
            line_handles.extend([line1, line2, proxy_hline, proxy_vline])
            line_labels.extend([
                'Fidelity',
                r'EBR/EBR$_\text{max}$',
                'Fidelity Threshold',
                'Max Allowed Flux',
            ])

        for ki in range(len(channel_numbers)):
            alloc = floored_ratio_array[ki][link_idx]
            marker = pick_marker(ki)

            if alloc <= 0:
                f_flux = 0.0
                f_fid, f_rate, _ = compute_fidelity_rate(tau, 0.0, y1, y2)
            else:
                f_mu = mu_channel_array[ki] * alloc
                f_fid, f_rate, f_flux = compute_fidelity_rate(tau, f_mu, y1, y2)

            f_rate = f_rate / max_rate

            sc1 = ax.scatter(
                f_flux, f_fid,
                marker=marker,
                s=marker_size,
                facecolors='none',
                edgecolors=matlab_blue,
                linewidths=1.6,
                clip_on=False,
                zorder=3
            )
            ax.scatter(
                f_flux, f_rate,
                marker=marker,
                s=marker_size,
                facecolors='none',
                edgecolors=matlab_orang,
                linewidths=1.6,
                clip_on=False,
                zorder=3
            )

            if len(line_labels) < 4 + len(channel_numbers):
                line_handles.append(sc1)
                line_labels.append(f'K={channel_numbers[ki]} (APOPT)')

        ax.axhline(fidelity_limit[link_idx], color='k', linestyle='dotted', linewidth=0.8)

        max_flux = max_flux_list[link_idx]
        if np.isfinite(max_flux):
            ax.axvline(max_flux, color='k', linestyle='--', linewidth=0.8)

        ax.set_xlim([0, xlimit])
        ax.set_ylim([0, 1])
        ax.text(0.95, 0.05, link_labels[link_idx], transform=ax.transAxes,
                ha='right', va='bottom', fontsize=FS, fontweight='bold')

    for idx in range(l, rows * cols):
        r, c = divmod(idx, cols)
        axs[r, c].axis('off')

    left_boxes = [ax.get_position() for ax in axs.flat if ax.get_visible()]
    x0_left = min(b.x0 for b in left_boxes)
    y0_left = min(b.y0 for b in left_boxes)
    x1_left = max(b.x1 for b in left_boxes)
    xmid_left = 0.5 * (x0_left + x1_left)

    fig.text(xmid_left, y0_left - 0.045,
             r'Dimensionless Flux $\mu K_\ell \tau$',
             ha='center', va='center', fontsize=FS)

    n_leg = len(line_labels)
    ncol_left_leg = int(np.ceil(n_leg / 2.0))
    ax_left_leg.legend(
        handles=line_handles,
        labels=line_labels,
        loc='center',
        ncol=ncol_left_leg,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor='0.3',
        fontsize=legend_fs,
        columnspacing=0.8,
        handletextpad=0.4,
        handlelength=1.4,
        borderpad=0.3,
        labelspacing=0.3,
    )

    # ---------------- Right column ----------------
    right_sub = outer[0, 1].subgridspec(
        2, 1,
        height_ratios=[1, 1],
        hspace=right_hspace_frac
    )

    ax_rate = fig.add_subplot(right_sub[0, 0])
    ax_chan = fig.add_subplot(right_sub[1, 0])

    # ---- rate utility plot ----
    ax_rate.minorticks_off()
    ax_rate.set_ylabel('Utility', fontsize=FS)

    channel_str = [str(c) for c in channel_numbers]
    bars = ax_rate.bar(channel_str, objective_value,
                       color=matlab_blue, edgecolor='black', linewidth=0.8)

    comparison_map = {
        '1': [0, 0, 0, 0],
        '2': [0, 0, 0, 0],
        '3': [0, 0, 0, 0],
        '4': [3.98, 5.13, 5.52, 6.51],
    }
    for rect, y_line in zip(bars, comparison_map.get(case_num, [])):
        x0 = rect.get_x()
        w = rect.get_width()
        ax_rate.hlines(y_line, x0, x0 + w, colors='red', linewidth=2)

    max_total_utility = 0.0
    for (y1, y2, f_th) in zip(y1_array, y2_array, fidelity_limit):
        A = 2 * (y1 + y2) + 1
        B = 4 * y1 * y2
        R_max_i = compute_R_max_full(y1, y2)
        if f_th <= 0:
            normalized_i = 1.0
        else:
            def fidelity_eq(x):
                return 0.25 * (1 + 3 * x / (x * x + A * x + B)) - f_th
            x_thresh, = fsolve(fidelity_eq, 1.0)
            expr_val = x_thresh**2 + A * x_thresh + B
            rate_at_thresh = expr_val * math.log2(2 * f_th)
            normalized_i = rate_at_thresh / R_max_i
        max_total_utility += normalized_i

    ax_rate.axhline(max_total_utility, color='k', linestyle='--', label='Max Total Utility')
    handles, labels = ax_rate.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='red', linewidth=2))
    labels.append('GA Utility')
    ax_rate.legend(handles=handles, labels=labels,
                   loc='upper left', fontsize=legend_fs,
                   frameon=True, fancybox=True, framealpha=1.0,
                   edgecolor='0.3')

    ax_rate.spines['top'].set_visible(True)
    ax_rate.spines['right'].set_visible(True)
    ax_rate.tick_params(axis='both', which='major', direction='in',
                        top=False, right=False, bottom=True, left=True,
                        labelsize=FS, pad=6, length=4)
    ax_rate.tick_params(axis='x', labelbottom=False)
    ax_rate.tick_params(axis='y', labelsize=FS)

    ylimit_map = {
        '1': [4.9, 5.001],
        '2': [2.0, 4.0],
        '3': [0.2, 1.0],
        '4': [3.0, 8.0],
    }
    ylimit = ylimit_map.get(case_num, None)
    if ylimit is not None:
        ax_rate.set_ylim(*ylimit)
        if case_num == '4':
            ax_rate.set_yticks([3.0, 5.5, 8.0])
            ax_rate.set_yticklabels(["3", "5.5", "8"])

    # ---- channel bar plot ----
    ax_chan.minorticks_off()
    ax_chan.grid(True, axis='y', linestyle='--', linewidth=0.6, color='0.85')
    ax_chan.set_axisbelow(True)
    ax_chan.tick_params(axis='both', which='major',
                        labelsize=FS, direction='in',
                        length=4, width=1.0, top=False, right=False,
                        bottom=True, left=True, pad=6)

    n_bars = len(channel_numbers)
    rows_data = []
    max_links = 0
    for kv in k_vars_array:
        if isinstance(kv, (list, tuple, np.ndarray)) and len(kv) > 0 and hasattr(kv[0], 'value'):
            row = [int(round(v.value[0])) for v in kv]
        else:
            row = list(np.asarray(kv, dtype=float))
        rows_data.append(row)
        max_links = max(max_links, len(row))

    alloc_matrix = np.zeros((n_bars, max_links), dtype=float)
    for r, row in enumerate(rows_data):
        alloc_matrix[r, :len(row)] = row

    total_per_bar = alloc_matrix.sum(axis=1)
    null_link = np.clip(np.asarray(channel_numbers) - total_per_bar, 0, None)
    bottom = np.zeros(n_bars, dtype=float)

    gem = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
           '#77AC30', '#4DBEEE', '#A2142F', '#003BFF', '#017501',
           '#FF0000', '#B526FF', '#FF00FF', '#000000']

    def cycle_from(idx, n):
        return [gem[(idx + i) % len(gem)] for i in range(n)]

    bc0 = ax_chan.bar(channel_str, null_link, bottom=bottom,
                      color=gem[0], edgecolor='black', linewidth=0.7, label='Unassigned')
    bottom += null_link

    labels_links = ['Link AB', 'Link CD', 'Link EF', 'Link GH', 'Link IJ',
                    'Link KL', 'Link MN', 'Link OP', 'Link QR', 'Link ST',
                    'Link UV', 'Link WX', 'Link YZ']
    link_colors = cycle_from(1, max_links)

    bar_handles_map = {'Unassigned': bc0[0]}

    for i in range(max_links):
        lbl = labels_links[i] if i < len(labels_links) else f'Link {i + 1}'
        bc = ax_chan.bar(channel_str, alloc_matrix[:, i], bottom=bottom,
                         color=link_colors[i], edgecolor='black', linewidth=0.7, label=lbl)
        bottom += alloc_matrix[:, i]
        bar_handles_map[lbl] = bc[0]

    ax_chan.set_ylabel(r'No. of Channels $K_\ell$', fontsize=FS)
    ax_chan.set_xlabel(r'$K_\ell$', fontsize=FS)
    ax_chan.set_ylim(0, 96)
    ax_chan.set_yticks([0, 48, 96])
    ax_chan.set_yticklabels(['0', '48', '96'])

    # Exact requested order
    left_col_labels = [
        'Unassigned', 'Link AB', 'Link CD', 'Link EF',
        'Link GH', 'Link IJ', 'Link KL', 'Link MN'
    ]
    right_col_labels = [
        'Link OP', 'Link QR', 'Link ST', 'Link UV', 'Link WX'
    ]

    left_col_labels = [lbl for lbl in left_col_labels if lbl in bar_handles_map]
    right_col_labels = [lbl for lbl in right_col_labels if lbl in bar_handles_map]

    left_col_handles = [bar_handles_map[lbl] for lbl in left_col_labels]
    right_col_handles = [bar_handles_map[lbl] for lbl in right_col_labels]

    # Build the left column legend first
    leg_left = ax_chan.legend(
        left_col_handles,
        left_col_labels,
        loc='upper left',
        bbox_to_anchor=(0.015, 0.995),
        fontsize=legend_fs,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor='0.3',
        ncol=1,
        borderpad=0.3,
        labelspacing=0.25,
        handlelength=1.3,
        handletextpad=0.4,
    )

    # Add it back so a second legend can coexist
    ax_chan.add_artist(leg_left)

    # Second column legend, placed just to the right of the first
    ax_chan.legend(
        right_col_handles,
        right_col_labels,
        loc='upper left',
        bbox_to_anchor=(0.43, 0.995),
        fontsize=legend_fs,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor='0.3',
        ncol=1,
        borderpad=0.3,
        labelspacing=0.25,
        handlelength=1.3,
        handletextpad=0.4,
    )

    fig.savefig(f'outputs/comp_combined_{case_num}.svg', dpi=300)
    plt.close(fig)

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



def run_plots(k_list, fidelity_limit, tau, y1, y2, text, initial=0.001, skip=False):
    """
    Inputs:
    k_list is an array of source channel amounts
    fidelity_limit is an array of fidelity lower bounds for each link
    tau is the coincidence time in seconds
    y1 is the whole y1 array for all links
    y2 is the whole y2 array for all links
    text is a string that differentiates the output plots
    initial is the initial condition mu value that the APOPT solver uses
    """
    k_vars_array = []
    rate_array = []
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
        rate_array.append(result['rates_full'])
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

    fidelity_rate_plot(k_list, k_vars_array, rate_array, mu_array, tau, y1, y2, fidelity_limit, text)
    channel_bar_plot(k_list, k_vars_array, text)
    rate_bar_plot(k_list, k_vars_array, objective_array, mu_array, tau, y1, y2, fidelity_limit, text)
    # Also create a combined figure that places link fidelity on the left and
    # the rate utility + channel bar stacked on the right (saved as comp_combined_{case}).
    try:
        combined_plot(k_list, k_vars_array, objective_array, mu_array, tau, y1, y2, fidelity_limit, text)
    except Exception:
        # Don't crash the run if combined plotting fails for some reason.
        pass



def get_case(case_num):
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
    # Change this to 1, 2, 3, or 4 to reproduce the exact old cases.
    run_case(4)


if __name__ == '__main__':
    main()
