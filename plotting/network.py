#data/network.py
"""
Plot network topology
"""

#ZHG
#2026.03.26
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from pathlib import Path
from typing import Any

import matplotlib.patheffects as pe
import networkx as nx

from .base import (
    entity_mathtext,
    gem_colors,
    make_figure,
    option_link_label,
    path_edges,
    path_loss,
    safe_float,
    save_figure,
)


def _node_color_map(network: nx.Graph) -> list[str]:
    colors = []
    for node, data in network.nodes(data=True):
        if data.get("node_type") == "source" or str(node).startswith("S"):
            colors.append("lightblue")
        else:
            colors.append("lightcoral")
    return colors


def _layout(network: nx.Graph):
    if "pos" in network.graph:
        return network.graph["pos"]

    pos = nx.spring_layout(network, seed=0, k=1.3, iterations=200)
    network.graph["pos"] = pos
    return pos


def plot_network_solution(
    network: nx.Graph,
    best_result: dict[str, Any] | None,
    outdir: str | Path = "outputs",
    filename: str = "network_plot.svg",
):
    if best_result is None:
        return None

    combo = list(best_result.get("combo", []))
    allocation = best_result.get("allocation", {}) or {}

    fig, ax = make_figure(figsize=(12, 6))
    pos = _layout(network)

    nx.draw_networkx_nodes(
        network,
        pos,
        ax=ax,
        node_size=2000,
        node_color=_node_color_map(network),
        edgecolors="black",
        linewidths=1.0,
    )

    nx.draw_networkx_edges(
        network,
        pos,
        ax=ax,
        width=1.15,
        edge_color="#646464",   # darker than lightgray
        alpha=1.0,
    )

    nx.draw_networkx_labels(
        network,
        pos,
        labels={n: entity_mathtext(str(n)) for n in network.nodes()},
        ax=ax,
        font_size=22,
        font_weight="bold",
    )

    colors = gem_colors(len(combo), start=1)
    legend_handles = []

    for idx, option in enumerate(combo):
        color = colors[idx]
        p1 = option.get("path1", [])
        p2 = option.get("path2", [])
        edges1 = list(zip(p1, p1[1:]))
        edges2 = list(zip(p2, p2[1:]))

        chosen = allocation.get(id(option), {}) if allocation else {}
        if not chosen and "allocation" in option:
            chosen = option["allocation"]

        k = chosen.get("k")
        mu = safe_float(chosen.get("mu"), float("nan"))

        nx.draw_networkx_edges(
            network,
            pos,
            edgelist=edges1 + edges2,
            ax=ax,
            width=3.0,
            edge_color=color,
            alpha=0.75,
        )

        label = f"Link {option_link_label(option)}"
        if k is not None:
            label += f"  (k={k}"
            if mu == mu:
                label += f", μ={mu:.3g}"
            label += ")"

        handle = ax.plot([], [], color=color, lw=3.5, label=label)[0]
        legend_handles.append(handle)

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
        font_size=13,
        rotate=True,
        label_pos=0.5,
    )
    for text in texts.values():
        text.set_zorder(1)
        text.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    for idx, option in enumerate(combo):
        color = colors[idx]
        p1 = option.get("path1", [])
        p2 = option.get("path2", [])
        s = option.get("source")
        # if s in pos and p1 and p2:
        #     x, y = pos[s]
        #     loss_sum = path_loss(network, p1) + path_loss(network, p2)
            # txt = f"{option_link_label(option)}\nloss={loss_sum:.2f}"
            # ax.text(
            #     x,
            #     y + 0.05,
            #     txt,
            #     fontsize=9,
            #     ha="center",
            #     va="bottom",
            #     color=color,
            #     path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            # )

    if legend_handles:
        ax.legend(loc="best", frameon=True, fontsize=16)

    ax.set_axis_off()
    #ax.set_title("")
    return save_figure(fig, outdir, filename)