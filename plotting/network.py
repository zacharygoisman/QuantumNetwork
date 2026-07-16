# plotting/network.py
"""
General network plot.

Draws an arbitrary network with a spring layout and overlays the chosen
link paths from ``best_result`` in colored strokes. This is the
non-ring counterpart to :mod:`plotting.network_ring`.
"""

# ZHG
# 2026.03.26
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.patheffects as pe
import networkx as nx

from .base import (
    PLOT_FONT_SIZE,
    bar_link_label,
    entity_mathtext,
    format_float,
    gem_colors,
    make_figure,
    option_link_label,
    path_edges,
    path_loss,
    safe_float,
    save_figure,
    sorted_options_by_link,
    display_node_label,
    display_link_label,
    display_source_label,
    display_bar_link_label,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _node_color_map(network: nx.Graph) -> list[str]:
    """Light blue for source nodes, light coral for user nodes."""
    colors: list[str] = []
    for node, data in network.nodes(data=True):
        if data.get("node_type") == "source" or str(node).startswith("S"):
            colors.append("lightblue")
        else:
            colors.append("lightcoral")
    return colors


def _layout(network: nx.Graph):
    """Cached spring layout with a deterministic seed for reproducibility."""
    if "pos" in network.graph:
        return network.graph["pos"]

    pos = nx.spring_layout(network, seed=0, k=1.3, iterations=200)
    network.graph["pos"] = pos
    return pos


# --------------------------------------------------------------------------- #
# Public function
# --------------------------------------------------------------------------- #

def plot_network_solution(
    network: nx.Graph,
    best_result: dict[str, Any] | None,
    outdir: str | Path = "outputs",
    filename: str = "network_plot.svg",
):
    """Draw the network with the chosen link paths overlaid; save as SVG."""
    if best_result is None:
        return None

    combo = sorted_options_by_link(best_result.get("combo", []))
    allocation = best_result.get("allocation", {}) or {}

    fig, ax = make_figure(figsize=network.graph.get("figsize", (11, 9)))
    pos = _layout(network)

    nx.draw_networkx_nodes(
        network, pos, ax=ax,
        node_size=network.graph.get("node_size", 2000),
        node_color=_node_color_map(network),
        edgecolors="black",
        linewidths=1.0,
    )
    nx.draw_networkx_edges(
        network, pos, ax=ax,
        width=1.15,
        edge_color="#646464",   # darker than lightgray
        alpha=1.0,
    )
    nx.draw_networkx_labels(
        network, pos,
        labels={
            n: entity_mathtext(display_node_label(network, n))
            for n in network.nodes()
        },
        ax=ax,
        font_size=PLOT_FONT_SIZE,
        font_weight="bold",
    )

    colors = gem_colors(len(combo), start=1)
    legend_handles = []

    for idx, option in enumerate(combo):
        color = colors[idx]
        p1, p2 = option.get("path1", []), option.get("path2", [])
        edges = list(zip(p1, p1[1:])) + list(zip(p2, p2[1:]))

        chosen = allocation.get(id(option), {}) if allocation else {}
        if not chosen and "allocation" in option:
            chosen = option["allocation"]

        k = chosen.get("k")
        mu = safe_float(chosen.get("mu"), float("nan"))

        nx.draw_networkx_edges(
            network, pos,
            edgelist=edges,
            ax=ax,
            width=3.0,
            edge_color=color,
            alpha=0.75,
        )

        raw_link = option.get("link") or option.get("users") or option_link_label(option)
        link_label = display_bar_link_label(network, raw_link)
        label = f"Link {link_label}"
        if k is not None:
            label += f"  (k={k}"
            if mu == mu:  # NaN-safe finite check
                label += f", μ={format_float(mu, '.3g')}"
            label += ")"

        handle = ax.plot([], [], color=color, lw=3.5, label=label)[0]
        legend_handles.append(handle)

    if not network.graph.get("suppress_edge_labels", False):
        edge_labels = {
            (u, v): format_float(safe_float(data.get("loss", 1.0), 1.0), ".2f")
            for u, v, data in network.edges(data=True)
            if u != v
        }
        texts = nx.draw_networkx_edge_labels(
            network, pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=PLOT_FONT_SIZE,
            rotate=True,
            label_pos=0.5,
        )
        for text in texts.values():
            text.set_zorder(1)
            text.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    # Reserved for per-link source annotation; left in place for future use.
    for idx, option in enumerate(combo):
        color = colors[idx]
        p1 = option.get("path1", [])
        p2 = option.get("path2", [])
        s = option.get("source")
        # if s in pos and p1 and p2:
        #     x, y = pos[s]
        #     loss_sum = path_loss(network, p1) + path_loss(network, p2)
        #     txt = f"{option_link_label(option)}\nloss={loss_sum:.2f}"
        #     ax.text(
        #         x, y + 0.05, txt,
        #         fontsize=9, ha="center", va="bottom", color=color,
        #         path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        #     )

    if legend_handles:
        ax.legend(loc="best", frameon=True, fontsize=PLOT_FONT_SIZE)

    if network.graph.get("equal_aspect", False):
        ax.set_aspect("equal", adjustable="box")

        pad_x = network.graph.get("pad_x", 0.5)
        pad_y = network.graph.get("pad_y", 0.5)

        xs = [xy[0] for xy in pos.values()]
        ys = [xy[1] for xy in pos.values()]

        x_min, x_max = min(xs) - pad_x, max(xs) + pad_x
        y_min, y_max = min(ys) - pad_y, max(ys) + pad_y

        # Optional fixed data scale. Smaller number = larger network on canvas.
        data_units_per_inch = network.graph.get("data_units_per_inch", None)

        if data_units_per_inch is not None:
            x_mid = 0.5 * (x_min + x_max)
            y_mid = 0.5 * (y_min + y_max)

            fig_w, fig_h = fig.get_size_inches()

            half_x = 0.5 * fig_w * data_units_per_inch
            half_y = 0.5 * fig_h * data_units_per_inch

            ax.set_xlim(x_mid - half_x, x_mid + half_x)
            ax.set_ylim(y_mid - half_y, y_mid + half_y)
        else:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

    ax.set_axis_off()
    return save_figure(
        fig,
        outdir,
        filename,
        bbox_inches=network.graph.get("bbox_inches", "tight"),
    )