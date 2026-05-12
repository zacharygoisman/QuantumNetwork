from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np

from .base import (
    GEM,
    apply_plot_style,
    bar_link_label,
    entity_mathtext,
    gem_colors,
    option_link_label,
    per_link_log_utility,
    safe_float,
)
from .network import _layout, _node_color_map
from .utility import _ordered_link_keys, _best_result_to_source_rows, source_allocation


# ----------------------------
# Shared text sizes
# ----------------------------
TEXT_SIZE = 22
LEGEND_SIZE = TEXT_SIZE
NETWORK_NODE_LABEL_SIZE = TEXT_SIZE
NETWORK_EDGE_LABEL_SIZE = TEXT_SIZE
UTILITY_TICK_SIZE = TEXT_SIZE
UTILITY_LABEL_SIZE = TEXT_SIZE
SOURCE_LABEL_SIZE = TEXT_SIZE


def _fig_to_array(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    arr = mpimg.imread(buf)
    buf.close()
    plt.close(fig)
    return arr


def _set_consistent_plot_text():
    apply_plot_style()
    mpl.rcParams.update({
        "axes.titlesize": TEXT_SIZE,
        "axes.labelsize": TEXT_SIZE,
        "xtick.labelsize": TEXT_SIZE,
        "ytick.labelsize": TEXT_SIZE,
        "legend.fontsize": LEGEND_SIZE,
    })


def _make_network_fig(network: nx.Graph, best_result: dict[str, Any] | None):
    if best_result is None:
        return None

    _set_consistent_plot_text()
    fig, ax = plt.subplots(figsize=(13.5, 12.2))   # was (13.5, 7.2)

    combo = list(best_result.get("combo", []))
    pos = dict(_layout(network))

    # Stretch vertically to reduce overlap
    y_stretch = 1.35
    pos = {
        node: (xy[0], xy[1] * y_stretch)
        for node, xy in pos.items()
    }

    nx.draw_networkx_nodes(
        network,
        pos,
        ax=ax,
        node_size=2700,
        node_color=_node_color_map(network),
        edgecolors="black",
        linewidths=1.4,
    )

    nx.draw_networkx_edges(
        network,
        pos,
        ax=ax,
        width=1.8,
        edge_color="#646464",
        alpha=1.0,
        connectionstyle="arc3,rad=0.03",
    )

    nx.draw_networkx_labels(
        network,
        pos,
        labels={n: entity_mathtext(str(n)) for n in network.nodes()},
        ax=ax,
        font_size=NETWORK_NODE_LABEL_SIZE,
        font_weight="bold",
    )

    colors = gem_colors(len(combo), start=1)

    for idx, option in enumerate(combo):
        color = colors[idx]
        p1 = option.get("path1", [])
        p2 = option.get("path2", [])
        edges1 = list(zip(p1, p1[1:]))
        edges2 = list(zip(p2, p2[1:]))

        nx.draw_networkx_edges(
            network,
            pos,
            edgelist=edges1 + edges2,
            ax=ax,
            width=4.6,
            edge_color=color,
            alpha=0.85,
            connectionstyle="arc3,rad=0.03",
        )

    edge_labels = {
        (u, v): f"{safe_float(data.get('loss', 1.0), 1.0):.2f}"
        for u, v, data in network.edges(data=True)
        if u != v
    }

    texts = nx.draw_networkx_edge_labels(
        network,
        pos,
        edge_labels=edge_labels,
        ax=ax,
        font_size=NETWORK_EDGE_LABEL_SIZE,
        rotate=True,
        label_pos=0.5,
    )
    for text in texts.values():
        text.set_zorder(1)
        text.set_path_effects([pe.withStroke(linewidth=4, foreground="white")])

    ax.set_axis_off()
    fig.tight_layout()
    return fig


def _make_link_utility_fig(best_result: dict[str, Any] | None):
    if best_result is None:
        return None

    combo = list(best_result.get("combo", []))
    allocation = best_result.get("allocation", {}) or {}

    rows = []
    for idx, opt in enumerate(combo):
        alloc = allocation.get(id(opt), {}) if allocation else {}
        if not alloc and "allocation" in opt:
            alloc = opt["allocation"]

        link = opt.get("link") or opt.get("users") or option_link_label(opt)
        link_key = None
        if isinstance(link, (tuple, list)) and len(link) == 2:
            link_key = (str(link[0]), str(link[1]))

        rows.append({
            "link": link,
            "link_key": link_key,
            "combo_index": idx,
            "link_utility": per_link_log_utility(opt, alloc),
            "link_ub": safe_float(opt.get("link_ub"), float("nan")),
        })

    desired_order = _ordered_link_keys(best_result)
    if desired_order:
        rank = {k: i for i, k in enumerate(desired_order)}
        rows.sort(key=lambda r: rank.get(r.get("link_key"), 10**9))

    links = [r["link"] for r in rows]
    ub = np.array([safe_float(r.get("link_ub")) for r in rows], dtype=float)
    ac = np.array([safe_float(r.get("link_utility")) for r in rows], dtype=float)

    finite_mask = np.isfinite(ub) | np.isfinite(ac)
    if not np.any(finite_mask):
        return None

    _set_consistent_plot_text()
    fig, ax = plt.subplots(figsize=(10.2, 4.8))

    x = np.arange(len(links), dtype=float)
    w = 0.42

    finite_vals = np.concatenate([ub[np.isfinite(ub)], ac[np.isfinite(ac)]])
    data_min = float(np.min(finite_vals))
    data_max = float(np.max(finite_vals))

    y_bottom = min(0.0, data_min)
    y_top = max(0.0, data_max)

    if data_min < 0.0:
        y_bottom -= 1.0

    ub_bottoms = np.full_like(ub, y_bottom, dtype=float)
    ac_bottoms = np.full_like(ac, y_bottom, dtype=float)

    ub_heights = np.where(np.isfinite(ub), ub - y_bottom, np.nan)
    ac_heights = np.where(np.isfinite(ac), ac - y_bottom, np.nan)

    ax.bar(
        x - w / 2,
        ub_heights,
        width=w,
        bottom=ub_bottoms,
        facecolor="white",
        edgecolor="black",
        linewidth=1.6,
        label="Upper bound utility",
    )

    ax.bar(
        x + w / 2,
        ac_heights,
        width=w,
        bottom=ac_bottoms,
        color=GEM[0],
        edgecolor="black",
        linewidth=1.3,
        label="Actual utility",
    )

    pad = 0.03 * max(1.0, y_top - y_bottom)
    ax.set_ylim(y_bottom - pad, y_top + pad)

    yticks = np.linspace(y_bottom, y_top, 3)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.2f}" for t in yticks], fontsize=UTILITY_TICK_SIZE)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [bar_link_label(t) for t in links],
        ha="center",
        fontsize=UTILITY_TICK_SIZE,
    )
    ax.set_ylabel("Utility (log₁₀ units)", fontsize=UTILITY_LABEL_SIZE)

    ax.tick_params(axis="x", which="both", bottom=False, top=False, length=0, pad=10)
    ax.tick_params(axis="y", which="both", left=True, right=False, length=7, width=1.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.margins(x=0.12)
    ax.legend(fontsize=LEGEND_SIZE, frameon=True)

    fig.tight_layout()
    return fig


def _make_source_allocation_fig(
    best_result: dict[str, Any] | None,
    sources: dict[str, Any],
    **kwargs,
):
    previous_best_results = _best_result_to_source_rows(best_result)
    if not previous_best_results and not sources:
        return None

    link_order = _ordered_link_keys(best_result)

    _set_consistent_plot_text()
    fig, _ = source_allocation(
        previous_best_results=previous_best_results,
        sources=sources,
        link_order=link_order,
        height=1.25,
        axes_height=1.35,
        base_width=3.0,
        gap=0.45,
        cap_lw=1.8,
        bar_lw=1.6,
        **kwargs,
    )

    for ax in fig.axes:
        for txt in ax.texts:
            txt.set_fontsize(SOURCE_LABEL_SIZE)

        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize(LEGEND_SIZE)

    fig.tight_layout()
    return fig


def plot_combined_solution(
    network: nx.Graph,
    best_result: dict[str, Any] | None,
    sources: dict[str, Any],
    outdir: str | Path = "outputs",
    filename: str = "combined_solution.svg",
    source_alloc_kwargs: dict[str, Any] | None = None,
):
    if best_result is None:
        return None

    source_alloc_kwargs = source_alloc_kwargs or {}

    net_fig = _make_network_fig(network, best_result)
    util_fig = _make_link_utility_fig(best_result)
    src_fig = _make_source_allocation_fig(best_result, sources, **source_alloc_kwargs)

    if net_fig is None or util_fig is None or src_fig is None:
        for fig in (net_fig, util_fig, src_fig):
            if fig is not None:
                plt.close(fig)
        return None

    net_img = _fig_to_array(net_fig)
    util_img = _fig_to_array(util_fig)
    src_img = _fig_to_array(src_fig)

    _set_consistent_plot_text()
    fig = plt.figure(figsize=(18.0, 8.2))
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[net_img.shape[1], max(util_img.shape[1], src_img.shape[1])],
        height_ratios=[util_img.shape[0], src_img.shape[0]],
        wspace=0.04,
        hspace=0.08,
    )

    ax_net = fig.add_subplot(gs[:, 0])
    ax_util = fig.add_subplot(gs[0, 1])
    ax_src = fig.add_subplot(gs[1, 1])

    ax_net.imshow(net_img)
    ax_util.imshow(util_img)
    ax_src.imshow(src_img)

    for ax in (ax_net, ax_util, ax_src):
        ax.axis("off")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path