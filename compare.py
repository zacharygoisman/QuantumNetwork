#compare.py
#Takes all functions for comparing to Alnas 2022, using the same utility function

#Packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import fsolve
from gekko import GEKKO
import math
import string
import os
import contextlib

import allocate_channels

def compute_fidelity_rate(tau, mu, y1_val, y2_val):
    """
    For a given link (with parameters y1 and y2), compute the maximum possible
    rate from the rate equation
         Rate(x) = [ x^2 + (2*(y1+y2)+1)*x + 4*y1*y2 ] * log2( F(x) )
    where
         F(x) = 0.25*(1 + 3*x / [ x^2 + (2*(y1+y2)+1)*x + 4*y1*y2 ])
    The function samples x in a fixed interval and returns the maximum value.
    """
    x = tau*mu
    A = 2*(y1_val+y2_val) + 1
    B = 4*y1_val*y2_val
    def expr(x):
         return x**2 + A*x + B
    def fidelity(x):
         return 0.25*(1 + 3*x/expr(x))
    def rate(x):
         # Using math.log2 for base-2 logarithm.
         return expr(x) * np.log2(2*fidelity(x))
    return fidelity(x), rate(x), x

def compute_R_max_full(y1_val, y2_val):
    """
    For a given link (with parameters y1 and y2), compute the maximum possible
    rate from the rate equation
         Rate(x) = [ x^2 + (2*(y1+y2)+1)*x + 4*y1*y2 ] * log2( 2F(x) )
    where
         F(x) = 0.25*(1 + 3*x / [ x^2 + (2*(y1+y2)+1)*x + 4*y1*y2 ])
    The function samples x in a fixed interval and returns the maximum value.
    """
    A = 2*(y1_val+y2_val) + 1
    B = 4*y1_val*y2_val
    def expr(x):
         return x**2 + A*x + B
    def F(x):
         return 0.25*(1 + 3*x/expr(x))
    def rate(x):
         # Using math.log2 for base-2 logarithm.
         return expr(x) * math.log2(2*F(x))
    # Sample x from a small positive value to a reasonable upper bound.
    xs = [i/10000 for i in range(1, 100000)]  # x from 0.0001 to 10
    rates = [rate(x) for x in xs]
    # The maximum possible rate (remember: with F<1, these rates are negative,
    # and the maximum is the one closest to zero)
    max_rate = max(rates)
    return max_rate

def matt_comparison_gecko(K, fidelity_limit, y1, y2, initial):
    N_links = len(y1)
    
    # --------------------------------------------------------
    # 1) Precompute the maximum possible rate for each link.
    # For link i, compute:
    #   R_max = max_{x>=0} { Rate(x) }
    # where Rate(x) = expr(x) * log2(F(x)) and
    #   expr(x) = x^2 + (2*(y1+y2)+1)*x + 4*y1*y2,
    #   F(x) = 0.25*(1 + 3*x/expr(x)).
    # --------------------------------------------------------
    R_max_list = []
    for i in range(N_links):
        R_max = compute_R_max_full(y1[i], y2[i])
        # If by chance R_max is zero, clamp to a small negative number (rates are negative).
        if R_max == 0:
            R_max = -1e-6
        R_max_list.append(R_max)
        #print(R_max)
    
    # --------------------------------------------------------
    # 2) Build the GEKKO model.
    # --------------------------------------------------------
    m = GEKKO(remote=False)
    m.options.IMODE = 3   # steady-state optimization
    m.options.SOLVER = 1  # use APOPT (supports MINLP)
    m.solver_options = [
    'minlp_maximum_iterations 1000',
    'minlp_gap 0.0001',
    'minlp_branch_method 2',
    'nlp_maximum_iterations 2000'
]
    # Decision variable: mu (continuous, positive)
    mu = m.Var(value=0.001, lb=1e-9)
    
    # Decision variables: k_i (integer channels, at least 1, up to K)
    k_vars = [m.Var(value=K//N_links, integer=True, lb=1, ub=K) for _ in range(N_links)]
    
    # Total channels allocated must not exceed K
    m.Equation(sum(k_vars) <= K)
    
    # Fidelity constraints for each link:
    # For each link, we define:
    #   x = mu * k_i,
    #   expr = x^2 + (2*(y1+y2)+1)*x + 4*y1*y2,
    #   F = 0.25*(1 + 3*x/expr),
    # and we enforce F >= fidelity_limit.
    for i in range(N_links):
        x = mu * k_vars[i]
        expr_i = x**2 + (2*(y1[i]+y2[i]) + 1)*x + 4*y1[i]*y2[i]
        F = 0.25*(1 + 3*x/expr_i)
        m.Equation(F >= fidelity_limit[i])
    
    # --------------------------------------------------------
    # 3) Define the objective.
    # For each link, compute the instantaneous rate:
    #   Rate = expr * log2(F)
    # and then the normalized rate is defined as:
    #   Normalized Rate = R_max / Rate.
    # (Because rates are negative, the best (least negative) rate equals R_max,
    # giving a normalized value of 1; worse rates yield a ratio below 1.)
    # We maximize the sum of normalized rates by minimizing its negative.
    # --------------------------------------------------------
    obj = 0
    for i in range(N_links):
        x = mu * k_vars[i]
        expr_i = x**2 + (2*(y1[i]+y2[i]) + 1)*x + 4*y1[i]*y2[i]
        F = 0.25*(1 + 3*x/expr_i)
        rate_expr = expr_i * (m.log(2*F)/math.log(2))
        # Normalized rate: using the precomputed R_max_list[i]
        normalized_rate = rate_expr / R_max_list[i]
        obj += normalized_rate
    m.Obj(-obj)
    
    # --------------------------------------------------------
    # 4) Solve the MINLP.
    # --------------------------------------------------------
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            m.solve(disp=False)
    
    # Output the solution:
    # print("Optimal mu =", mu.value[0])
    # print("Optimal channel allocation (k_i):")
    # for i in range(N_links):
    #     print(f" Link {i+1}: k = {int(round(k_vars[i].value[0]))}")
    
    # For reference, compute the final normalized objective value:
    prelog_rates = []
    objective_value = 0
    for i in range(N_links):
        x_val = mu.value[0] * k_vars[i].value[0]
        expr_val = x_val**2 + (2*(y1[i]+y2[i]) + 1)*x_val + 4*y1[i]*y2[i]
        F_val = 0.25*(1 + 3*x_val/expr_val)
        rate_val = expr_val * (math.log(2*F_val)/math.log(2))
        normalized_rate_val = rate_val / R_max_list[i]
        # print(i,': ',normalized_rate_val)
        objective_value += normalized_rate_val
        prelog_rates.append(normalized_rate_val)
    
    return k_vars, objective_value, mu, prelog_rates

#Fidelity and rate plot
def fidelity_rate_plot(k_list, k_vars_array, objective_value_array, mu_determined,
                       tau, y1_array, y2_array, fidelity_limit, text):
    case_num = text[-1]
    xlimit = {'1': 1.2, '2': 0.8, '3': 0.2, '4': 0.7}.get(case_num, 1.0)

    l = len(y1_array)

    # --- layout rules (unchanged) ---
    if l == 5:
        rows, cols = 3, 2
        legend_mode = 'panel'
        legend_tile_index = 1*cols + 1
        fig_size = (6.1, 7.0)
        wspace, hspace = 0.35, 0.35
        bottom_margin = 0.06
    elif l == 12:
        rows, cols = 4, 3
        legend_mode = 'figure'
        legend_tile_index = None
        fig_size = (11.0, 8.6)
        wspace, hspace = 0.35, 0.40
        bottom_margin = 0.06
    else:
        cols = 2 if l <= 6 else 3
        rows = (l + cols - 1) // cols
        legend_mode = 'figure'
        legend_tile_index = None
        fig_size = (11, 8)
        wspace, hspace = 0.35, 0.40
        bottom_margin = 0.06

    fig, axs = plt.subplots(rows, cols, figsize=fig_size)
    fig.subplots_adjust(wspace=wspace, hspace=hspace, bottom=bottom_margin)
    if rows*cols == 1:
        axs = np.array([[axs]])
    elif rows == 1:
        axs = np.array([axs])
    elif cols == 1:
        axs = axs[:, None]

    # Colors
    matlab_blue  = '#0072BD'
    matlab_orang = '#D95319'

    line_handles, line_labels = [], []
    proxy_hline = Line2D([0], [0], color='k', linestyle='dotted', linewidth=0.8,
                         label='Fidelity threshold')
    proxy_vline = Line2D([0], [0], color='k', linestyle='--', linewidth=0.8,
                         label='Max allowed flux')

    mu_total = 1.2 / tau
    mu_channel_array = [mu_determined[i] / tau for i in range(len(k_list))]

    floored_ratio_array = []
    for i in range(len(k_list)):
        arr = np.concatenate(k_vars_array[i])   # length may be 11 for K=12/24
        if len(arr) < l:
            arr = np.pad(arr, (0, l - len(arr)), mode='constant', constant_values=0.0)
        elif len(arr) > l:
            arr = arr[:l]
        floored_ratio_array.append(arr)

    base_markers = ['o', '^', 's', 'd', 'x', 'v', 'P', '*']  # circle, triangle, square, diamond, ...
    def pick_marker(i): return base_markers[i % len(base_markers)]

    letters = string.ascii_uppercase
    link_labels = [letters[2*i] + letters[2*i+1] for i in range(l)]

    for link_idx in range(l):
        r, c = divmod(link_idx if legend_mode != 'panel' else
                      (link_idx if link_idx < legend_tile_index else link_idx+1), cols)
        ax = axs[r, c]

        ax.minorticks_on()
        ax.tick_params(axis='both', which='both',
                       direction='in', top=True, right=True, labelsize=12)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

        y1, y2 = y1_array[link_idx], y2_array[link_idx]
        mu = np.linspace(1e-6, mu_total, 1000)
        fidelity, rate, flux = compute_fidelity_rate(tau, mu, y1, y2)
        max_rate = compute_R_max_full(y1, y2)
        rate = rate / max_rate

        line1, = ax.plot(tau * mu, fidelity, label='Fidelity (Model)', c=matlab_blue)
        line2, = ax.plot(tau * mu, rate,     label=r'$\mathcal{R}/\mathcal{R}_{\max}$ (Model)', c=matlab_orang)
        if not line_handles:
            line_handles.extend([line1, line2, proxy_hline, proxy_vline])
            line_labels.extend(['Fidelity (Model)',
                                r'$\mathcal{R}/\mathcal{R}_{\max}$ (Model)',
                                'Fidelity Threshold',
                                'Max Allowed Flux'])

        # Plot GA markers ONLY if allocation > 0 for that link and K
        for ki in range(len(k_list)):
            alloc = floored_ratio_array[ki][link_idx]
            if alloc <= 0:
                continue
            f_mu = mu_channel_array[ki] * alloc
            f_fid, f_rate, f_flux = compute_fidelity_rate(tau, f_mu, y1, y2)

            f_rate = f_rate / max_rate
            m = pick_marker(ki)
            sc1 = ax.scatter(f_flux, f_fid, marker=m,
                             facecolors='none', edgecolors=matlab_blue, linewidths=1.2,
                             clip_on=False, zorder=3)
            ax.scatter(f_flux, f_rate, marker=m,
                       facecolors='none', edgecolors=matlab_orang, linewidths=1.2,
                       clip_on=False, zorder=3)
            if len(line_labels) < 4 + len(k_list):
                line_handles.append(sc1)
                line_labels.append(f'K={k_list[ki]} (APOPT)')

            #print(f_flux/alloc)

        fid_min = fidelity_limit[link_idx]
        ax.axhline(fid_min, color='k', linestyle='dotted', linewidth=0.8)
        def max_flux_equation(x):
            return 0.25 * (1 + 3 * x / (x**2 + (2*y1 + 2*y2 + 1)*x + 4*y1*y2)) - fid_min
        try:
            max_flux = fsolve(max_flux_equation, 1.0)
            ax.axvline(max_flux, color='k', linestyle='--', linewidth=0.8)
        except Exception:
            pass

        ax.set_xlim([0, xlimit])
        ax.set_ylim([0, 1])

        ax.text(0.95, 0.05, link_labels[link_idx],
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=12, fontweight='bold')

    # Legends
    if legend_mode == 'panel':
        r, c = divmod(legend_tile_index, cols)
        ax_leg = axs[r, c]
        ax_leg.axis('off')
        ax_leg.legend(line_handles, line_labels, loc='center',
                      frameon=True, fancybox=True, framealpha=1.0,
                      edgecolor='0.3', fontsize=12)
    else:
        fig.legend(handles=line_handles, labels=line_labels,
                   loc='upper center', bbox_to_anchor=(0.5, 0.995),
                   ncol=4, frameon=True, fancybox=True, framealpha=1.0,
                   edgecolor='0.3', fontsize=12, columnspacing=1.2, handletextpad=0.8)

    fig.text(0.5, 0.005, 'Dimensionless Flux x', ha='center', fontsize=14)
    plt.savefig(f'outputs/comp_link_fidelity_{case_num}.svg', dpi=300)
    plt.close()


def channel_bar_plot(channel_numbers, k_vars_array, text):
    case_num = text[-1]
    channel_str = [str(c) for c in channel_numbers]
    n_bars = len(channel_numbers)

    # MATLAB R2014b "gem" palette (cycled to cover any count)
    gem = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
           '#77AC30', '#4DBEEE', '#A2142F', '#003BFF', '#017501', 
           '#FF0000', '#B526FF', '#FF00FF', '#000000']  # 13 colors

    def cycle_from(idx, n):
        return [gem[(idx + i) % len(gem)] for i in range(n)]
    
    # Legend labels (extend if you ever have more)
    labels = ['Link AB','Link CD','Link EF','Link GH','Link IJ',
              'Link KL','Link MN','Link OP','Link QR','Link ST',
              'Link UV','Link WX','Link YZ']

    # ---- Convert ragged GEKKO outputs to a dense numeric matrix [n_bars x L] ----
    numeric_rows = []
    max_links = 0
    for kv in k_vars_array:  # kv is a list of GEKKO Vars for one K
        row = [int(round(v.value[0])) for v in kv]
        numeric_rows.append(row)
        if len(row) > max_links:
            max_links = len(row)

    # Pad rows with zeros on the right to length max_links
    alloc_matrix = np.zeros((n_bars, max_links), dtype=float)
    for r, row in enumerate(numeric_rows):
        alloc_matrix[r, :len(row)] = row

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Grid and ticks
    ax.grid(True, axis='y', linestyle='--', linewidth=0.6, color='0.85')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=18, direction='in', length=4, width=1.0)
    ax.tick_params(top=True, right=True, which='both', direction='in')

    # Null Link (residual capacity) – uses gem[0] (blue)
    total_per_bar = alloc_matrix.sum(axis=1)
    null_link = np.clip(np.asarray(channel_numbers) - total_per_bar, 0, None)
    bottom = np.zeros(n_bars, dtype=float)
    ax.bar(channel_str, null_link, bottom=bottom,
           color=gem[0], edgecolor='black', linewidth=0.7, label='Null Link')
    bottom += null_link

    # Link stacks – start palette at gem[1] (orange), cycle as needed
    link_colors = cycle_from(1, max_links)
    for i in range(max_links):
        heights = alloc_matrix[:, i]
        ax.bar(channel_str, heights, bottom=bottom,
               color=link_colors[i], edgecolor='black', linewidth=0.7,
               label=labels[i] if i < len(labels) else f'Link {i+1}')
        bottom += heights

    # Labels & legend
    ax.set_ylabel('No. of Channels', fontsize=18)
    ax.set_xlabel('K', fontsize=18)

    leg_cols = 2 if max_links >= 12 else 1
    ax.legend(loc='upper left', fontsize=12,
              frameon=True, fancybox=True, framealpha=1.0,
              edgecolor='0.3', ncol=leg_cols)

    plt.tight_layout()
    plt.savefig(f'outputs/comp_channel_barplot_{case_num}.svg', dpi=300)
    plt.close()


def rate_bar_plot(channel_numbers, k_vars_array, objective_value,
                  mu_determined, tau, y1_array, y2_array, fidelity_limit, text):
    # case → y-limits
    case_num = str(text)[-1]
    ylimit_map = {
        '1': [4.9, 5.001],
        '2': [2.0, 4.0],
        '3': [0.2, 1.0],
        '4': [3.0, 8.0],
    }
    ylimit = ylimit_map.get(case_num, None)

    matlab_blue = '#0072BD'  # MATLAB default blue

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # remove title / x label; keep y label
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('Fitness', fontsize=18)

    # bars: MATLAB blue, thin black border
    channel_str = [str(c) for c in channel_numbers]
    ax.bar(channel_str, objective_value,
           color=matlab_blue, edgecolor='black', linewidth=0.8)
    #print(objective_value)

    # compute “best possible” horizontal line
    max_total_fitness = 0.0
    for (y1, y2, f_th) in zip(y1_array, y2_array, fidelity_limit):
        A = 2*(y1 + y2) + 1
        B = 4*y1*y2
        R_max_i = compute_R_max_full(y1, y2)
        if f_th <= 0:
            normalized_i = 1.0
        else:
            def fidelity_eq(x):
                return 0.25*(1 + 3*x/(x*x + A*x + B)) - f_th
            x_thresh, = fsolve(fidelity_eq, 1.0)
            expr_val       = x_thresh**2 + A*x_thresh + B
            rate_at_thresh = expr_val * math.log2(2*f_th)
            normalized_i   = rate_at_thresh / R_max_i
        max_total_fitness += normalized_i

    ax.axhline(max_total_fitness, color='k', linestyle='--', label='Max Total Fitness')

    # ticks: inside and on all four sides; hide x numbers; enlarge y numbers
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.tick_params(axis='x', which='both', labelbottom=False)  # hide x tick labels
    ax.tick_params(axis='y', labelsize=18)                     # bigger y numbers
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    if ylimit is not None:
        ax.set_ylim(ylimit)

    ax.legend(loc='upper left', fontsize=15)

    fig.savefig(f'outputs/comp_rate_utility_{case_num}.svg', dpi=300, bbox_inches='tight')
    plt.close(fig)


#Function to trigger all calculations and plots for each case
def run_plots(k_list, fidelity_limit, tau, y1, y2, text, initial = 0.001, skip = False):
    '''
    Inputs:
    k_list is an array of source channel amounts
    fidelity_limit is an array of fidelity lower bounds for each link
    tau is the coincidence time in seconds
    y1 is the whole y1 array for all links
    y2 is the whole y2 array for all links
    text is a string that differentiates the output plots
    initial is the initial condition mu value that the APOPT solver uses
    '''
    k_vars_array = []
    rate_array = []
    mu_array = []
    objective_array = []
    for k in k_list:
        if skip:
            if k in [12, 24]: #Remove last link for case 4 on the first two like in the paper #TODO update this with latest APOPT
                k_vars, objective_value, mu, prelog_rates = matt_comparison_gecko(k, fidelity_limit[:-1], y1[:-1], y2[:-1], initial)
            else:
                k_vars, objective_value, mu, prelog_rates = matt_comparison_gecko(k, fidelity_limit, y1, y2, initial)
        else:
            k_vars, objective_value, mu, prelog_rates = matt_comparison_gecko(k, fidelity_limit, y1, y2, initial)
        k_vars_array.append(k_vars)
        rate_array.append(prelog_rates)
        mu_array.append(mu.value[0])
        objective_array.append(objective_value)
    fidelity_rate_plot(k_list, k_vars_array, rate_array, mu_array, tau, y1, y2, fidelity_limit, text)
    channel_bar_plot(k_list, k_vars_array, text)
    rate_bar_plot(k_list, k_vars_array, objective_array, mu_array, tau, y1, y2, fidelity_limit, text)

def main():
    tau = 1E-9

    #Case 1
    k_list = [5, 10, 20, 40]
    y1 = [0,0.04,0,0.11,0.15]
    y2 = [0,0.007,0.125,0.019,0.025]
    fidelity_limit = np.repeat(0, len(y1))
    text = 'Case 1'
    run_plots(k_list, fidelity_limit, tau, y1, y2, text, initial = 0.001)

    #Case 2
    k_list = [5, 10, 20, 40]
    y1 = [0,0.04,0,0.11,0.15]
    y2 = [0,0.007,0.125,0.019,0.025]
    fidelity_limit = np.repeat(0.7, len(y1))
    text = 'Case 2'
    run_plots(k_list, fidelity_limit, tau, y1, y2, text, initial = 0.001)

    #Case 3
    k_list = [5, 10, 20, 40]
    y1 = [0,0.0034,0.0104,0.0179,0]
    y2 = [0,0.006,0.0018,0.0031,0.0515]
    fidelity_limit = np.repeat(0.9, len(y1))
    text = 'Case 3'
    run_plots(k_list, fidelity_limit, tau, y1, y2, text, initial = 0.001)

    #Case 4
    k_list = [12, 24, 48, 96]
    y1 = [0, 0.0034, 0, 0.0299, 0.0385, 0.0625, 0.0733, 0, 0.1106, 0.125, 0.1489, 0]
    y2 = [0, 0.0006, 0.0357, 0.0051, 0.0066, 0.0107, 0.0126, 0.1818, 0.019, 0.0214, 0.0256, 0.2979]
    fidelity_limit = np.repeat(0.7, len(y1))
    text = 'Case 4'
    run_plots(k_list, fidelity_limit, tau, y1, y2, text, initial = 0.001, skip = True) #skip allows us to skip the last link for K = 12 and 24 like in the paper


if __name__ == "__main__":
    main()