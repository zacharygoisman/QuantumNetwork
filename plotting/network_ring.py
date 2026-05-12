#data/network.py
"""
Plot network topology
"""

#ZHG
#2026.03.26
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from pathlib import Path
from typing import Any
import math
import random

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
    # If positions already provided, use them.
    if "pos" in network.graph:
        return network.graph["pos"]

    # Identify source nodes (by explicit node_type == 'source' or name starting with 'S')
    sources = [n for n, d in network.nodes(data=True) if d.get("node_type") == "source" or str(n).startswith("S")]
    if not sources:
        # Fallback to spring layout for arbitrary graphs
        pos = nx.spring_layout(network, seed=0, k=1.3, iterations=200)
        network.graph["pos"] = pos
        return pos

    # Deterministic ordering for reproducible polygon layout
    sources = sorted(sources, key=lambda x: str(x))
    n_sources = len(sources)

    # Place sources on a regular polygon (circle) centered at origin
    R = 1.0
    pos: dict[Any, tuple[float, float]] = {}
    for i, s in enumerate(sources):
        theta = 2.0 * math.pi * i / n_sources
        pos[s] = (R * math.cos(theta), R * math.sin(theta))

    # Attach users: direct neighbors of each source that are not sources
    users_by_source: dict[Any, list[Any]] = {s: [] for s in sources}
    others = set(network.nodes()) - set(sources)
    for s in sources:
        for nbr in network.neighbors(s):
            if nbr in others:
                users_by_source[s].append(nbr)

    # Place users on a primary ring; if there are more than 5 users for a source,
    # place the extras on an outer secondary ring to avoid crowding.
    user_radius = 2.0
    ring_capacity = 5
    ring_gap = 0.9
    for i, s in enumerate(sources):
        theta = 2.0 * math.pi * i / n_sources
        users = users_by_source.get(s, [])
        k = len(users)
        if k == 0:
            continue
        # Spread users in a small arc centered at the source angle
        max_arc = min(math.pi * 0.9, 2.0 * math.pi / max(1, n_sources))
        arc = min(max_arc, math.pi * 0.6) if k > 1 else 0.0

        inner_count = min(k, ring_capacity)
        outer_count = k - inner_count

        # compute inner angles
        if inner_count == 1:
            inner_angles = [theta]
        else:
            inner_angles = [theta - arc / 2.0 + j * (arc / (inner_count - 1)) for j in range(inner_count)]

        # compute outer angles (for extras)
        outer_angles = []
        if outer_count > 0:
            if outer_count == 1:
                outer_angles = [theta]
            else:
                outer_angles = [theta - arc / 2.0 + j * (arc / (outer_count - 1)) for j in range(outer_count)]

        # assign positions: first inner_count users to inner ring, remaining to outer ring
        for idx, u in enumerate(users):
            if idx < inner_count:
                ang = inner_angles[idx]
                r = user_radius
            else:
                ang = outer_angles[idx - inner_count]
                r = user_radius + ring_gap
            pos[u] = (r * math.cos(ang), r * math.sin(ang))

    # Collision avoidance: if users from different sources are too close, push them outward
    # Build polar coords for users (r, ang)
    user_polar: dict[Any, tuple[float, float]] = {}
    for s in sources:
        for u in users_by_source.get(s, []):
            x, y = pos.get(u, (0.0, 0.0))
            r = math.hypot(x, y)
            ang = math.atan2(y, x)
            user_polar[u] = (r, ang)

    # Resolve collisions iteratively
    min_sep = 0.35
    delta_r = 0.12
    delta_ang = 0.12
    max_iters = 10
    for _it in range(max_iters):
        moved = False
        users = list(user_polar.keys())
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                u = users[i]
                v = users[j]
                # only consider users from different sources
                src_u = next((s for s, ul in users_by_source.items() if u in ul), None)
                src_v = next((s for s, ul in users_by_source.items() if v in ul), None)
                if src_u is None or src_v is None or src_u == src_v:
                    continue
                ru, au = user_polar[u]
                rv, av = user_polar[v]
                xu, yu = ru * math.cos(au), ru * math.sin(au)
                xv, yv = rv * math.cos(av), rv * math.sin(av)
                dist = math.hypot(xu - xv, yu - yv)
                if dist < min_sep:
                    # If angles are (nearly) identical, separate by angle; otherwise push outward
                    if abs(au - av) < 1e-3:
                        user_polar[u] = (ru + delta_r, au - delta_ang)
                        user_polar[v] = (rv + delta_r, av + delta_ang)
                    else:
                        user_polar[u] = (ru + delta_r, au - delta_ang)
                        user_polar[v] = (rv + delta_r, av + delta_ang)
                    moved = True
        if not moved:
            break

    # Write adjusted polar coords back to positions
    for u, (r, ang) in user_polar.items():
        pos[u] = (r * math.cos(ang), r * math.sin(ang))

    # Any remaining nodes: place with a small spring layout and scale/shift
    remaining = [n for n in network.nodes() if n not in pos]
    if remaining:
        subG = network.subgraph(remaining).copy()
        if len(subG) > 0:
            sub_pos = nx.spring_layout(subG, seed=0, k=0.8, iterations=200)
            for n, p in sub_pos.items():
                subx, suby = p
                pos[n] = (1.2 * subx, 1.2 * suby)

    # Assign default losses where missing:
    # - source-source (inner ring) edges: 5-10 dB
    # - source-user edges: 1-5 dB
    rnd = random.Random(0)
    for u, v, data in network.edges(data=True):
        if data.get("loss") is not None:
            continue
        u_is_source = u in sources
        v_is_source = v in sources
        if u_is_source and v_is_source:
            data["loss"] = float(rnd.uniform(5.0, 10.0))
        elif (u_is_source and not v_is_source) or (v_is_source and not u_is_source):
            data["loss"] = float(rnd.uniform(1.0, 5.0))

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

    # Make the plot taller to give more vertical room for separated users
    fig, ax = make_figure(figsize=(12, 9))
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