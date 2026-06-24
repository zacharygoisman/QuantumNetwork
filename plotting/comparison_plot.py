#plotting for comparisons
import math
import string

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator
from scipy.optimize import fsolve


def fidelity_rate_plot(k_list, k_vars_array, mu_determined,
                       tau, y1_array, y2_array, fidelity_limit, text):
    # Deferred to avoid circular import with allocation.comparison.
    from allocation.comparison import (
        FS,
        compute_fidelity_rate,
        compute_R_max_full,
        max_allowed_flux_x,
    )
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
        line2, = ax.plot(tau * mu, rate, label='Utility', c=matlab_orang)

        if not line_handles:
            line_handles.extend([line1, line2, proxy_hline, proxy_vline])
            line_labels.extend([
                'Fidelity',
                'Utility',
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
    from allocation.comparison import FS

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
    ax.set_xlabel(r'Total Channels $K$', fontsize=FS)

    leg_cols = 2 if max_links >= 12 else 1
    ax.legend(loc='upper left', bbox_to_anchor=(-0.00, 1.1), fontsize=FS,
              frameon=True, fancybox=True, framealpha=1.0,
              edgecolor='0.3', ncol=leg_cols)

    plt.tight_layout()
    plt.savefig(f'outputs/comp_channel_barplot_{case_num}.svg', dpi=300)
    plt.close()



def rate_bar_plot(channel_numbers, k_vars_array, objective_value,
                  mu_determined, tau, y1_array, y2_array, fidelity_limit, text):
    from allocation.comparison import FS, compute_R_max_full

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
    from allocation.comparison import (
        FS,
        COMBINED_FIG_WIDTH,
        COMBINED_MARKER_LINEWIDTH,
        COMBINED_MARKER_SIZE,
        COMBINED_RIGHT_HEIGHT_BOOST,
        COMBINED_RIGHT_WIDTH_BOOST,
        compute_fidelity_rate,
        compute_R_max_full,
        max_allowed_flux_x,
        save_svg_exact,
    )

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

    right_fig_w = 6.4 * COMBINED_RIGHT_WIDTH_BOOST
    single_right_h = 5.2
    right_hspace_frac = 0.12
    right_fig_h = single_right_h * (2 + right_hspace_frac)

    # Scale only the combined figure to the requested Inkscape width.
    # The standalone/separate plots keep their original sizes and save behavior.
    # The right-column boost reallocates more of the fixed width to the two bar plots.
    base_fig_width = left_fig_w + right_fig_w
    combined_scale = COMBINED_FIG_WIDTH / base_fig_width
    left_fig_w *= combined_scale
    left_fig_h *= combined_scale
    right_fig_w *= combined_scale
    single_right_h *= combined_scale * COMBINED_RIGHT_HEIGHT_BOOST
    right_fig_h = single_right_h * (2 + right_hspace_frac)

    fig_width = left_fig_w + right_fig_w
    fig_height = max(left_fig_h, right_fig_h)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)

    outer = fig.add_gridspec(
        1, 2,
        width_ratios=[left_fig_w, right_fig_w],
        wspace=0.09,
        left=0.055,
        right=0.985,
        bottom=0.085,
        top=0.91,
    )

    # ---------------- Left column ----------------
    left_outer = outer[0, 0].subgridspec(
        2, 1,
        height_ratios=[0.14, 0.86],
        hspace=0.03
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

    # Use explicit proxy handles for the combined-figure legend.
    # This avoids SVG/viewer quirks where handles copied from plotted axes can appear missing.
    proxy_fidelity = Line2D([0], [0], color=matlab_blue, linewidth=2.0, label='Fidelity')
    proxy_ebr = Line2D([0], [0], color=matlab_orang, linewidth=2.0,
                       label='Utility')
    proxy_hline = Line2D([0], [0], color='k', linestyle='dotted', linewidth=1.2,
                         label='Fidelity Threshold')
    proxy_vline = Line2D([0], [0], color='k', linestyle='--', linewidth=1.2,
                         label='Max Allowed Flux')
    line_handles = [proxy_fidelity, proxy_ebr, proxy_hline, proxy_vline]
    line_labels = [
        'Fidelity',
        'Utility',
        'Fidelity Threshold',
        'Max Allowed Flux',
    ]

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

    marker_size = COMBINED_MARKER_SIZE  # larger markers for APOPT points

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
                linewidths=COMBINED_MARKER_LINEWIDTH,
                clip_on=False,
                zorder=3
            )
            ax.scatter(
                f_flux, f_rate,
                marker=marker,
                s=marker_size,
                facecolors='none',
                edgecolors=matlab_orang,
                linewidths=COMBINED_MARKER_LINEWIDTH,
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
        handlelength=1.8,
        borderpad=0.35,
        labelspacing=0.35,
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

    ax_chan.set_ylabel(r'Allocated Channels $K_\ell$', fontsize=FS)
    ax_chan.set_xlabel(r'Total Channels $K$', fontsize=FS)
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
        bbox_to_anchor=(0.50, 0.995),
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

    save_svg_exact(fig, f'outputs/comp_combined_{case_num}.svg')
    plt.close(fig)