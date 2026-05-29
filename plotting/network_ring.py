# plotting/network_ring.py
"""
Ring-topology network plot.

Draws the network with sources arranged on a regular polygon and their
attached users on outer ring(s), then overlays the chosen link paths from
``best_result`` in colored strokes.
"""

# ZHG
# 2026.03.26
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

from __future__ import annotations

import math
import random
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
)


# --------------------------------------------------------------------------- #
# Layout constants
# --------------------------------------------------------------------------- #

# Source polygon radius and user-ring geometry.
_SOURCE_RING_RADIUS = 1.0
_USER_RING_RADIUS = 2.0
_USER_RING_CAPACITY = 5      # users per source on the inner ring before spillover
_USER_RING_GAP = 0.9         # extra radius for users that overflow to the outer ring

# Collision-avoidance parameters used to nudge users from different sources apart.
_USER_MIN_SEP = 0.35
_USER_DELTA_R = 0.12
_USER_DELTA_ANG = 0.12
_USER_MAX_ITERS = 10

# Default loss ranges (dB) when a graph edge has no explicit ``loss`` attribute.
_LOSS_SOURCE_TO_SOURCE = (5.0, 10.0)
_LOSS_SOURCE_TO_USER = (1.0, 5.0)


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
    """
    Compute and cache positions for every node in ``network``.

    Sources are placed on a regular polygon. Each source's user neighbors are
    placed in an arc on a primary ring; spillover users go to a secondary
    outer ring. Users belonging to different sources are then nudged apart
    until they are at least :data:`_USER_MIN_SEP` apart, or until the
    iteration cap is reached. Any unconnected stragglers are positioned with
    a small spring layout.
    """
    # Reuse cached positions when available.
    if "pos" in network.graph:
        return network.graph["pos"]

    # Identify source nodes (by explicit node_type == 'source' or name starting with 'S').
    sources = [
        n for n, d in network.nodes(data=True)
        if d.get("node_type") == "source" or str(n).startswith("S")
    ]

    # No sources -> fall back to spring layout for arbitrary graphs.
    if not sources:
        pos = nx.spring_layout(network, seed=0, k=1.3, iterations=200)
        network.graph["pos"] = pos
        return pos

    # Deterministic ordering for reproducible polygon layout.
    sources = sorted(sources, key=lambda x: str(x))
    n_sources = len(sources)

    # 1. Place sources on the polygon.
    pos: dict[Any, tuple[float, float]] = {}
    for i, s in enumerate(sources):
        theta = 2.0 * math.pi * i / n_sources
        pos[s] = (
            _SOURCE_RING_RADIUS * math.cos(theta),
            _SOURCE_RING_RADIUS * math.sin(theta),
        )

    # 2. Group users by their parent source (direct non-source neighbors).
    others = set(network.nodes()) - set(sources)
    users_by_source: dict[Any, list[Any]] = {s: [] for s in sources}
    for s in sources:
        for nbr in network.neighbors(s):
            if nbr in others:
                users_by_source[s].append(nbr)

    # 3. Place users on the inner ring; spill the extras to the outer ring.
    for i, s in enumerate(sources):
        theta = 2.0 * math.pi * i / n_sources
        users = users_by_source.get(s, [])
        k = len(users)
        if k == 0:
            continue

        # Spread users in a small arc centered on the source angle.
        max_arc = min(math.pi * 0.9, 2.0 * math.pi / max(1, n_sources))
        arc = min(max_arc, math.pi * 0.6) if k > 1 else 0.0

        inner_count = min(k, _USER_RING_CAPACITY)
        outer_count = k - inner_count

        if inner_count == 1:
            inner_angles = [theta]
        else:
            inner_angles = [
                theta - arc / 2.0 + j * (arc / (inner_count - 1))
                for j in range(inner_count)
            ]

        outer_angles: list[float] = []
        if outer_count > 0:
            if outer_count == 1:
                outer_angles = [theta]
            else:
                outer_angles = [
                    theta - arc / 2.0 + j * (arc / (outer_count - 1))
                    for j in range(outer_count)
                ]

        # Inner ring first, overflow to the outer ring.
        for idx, u in enumerate(users):
            if idx < inner_count:
                ang = inner_angles[idx]
                r = _USER_RING_RADIUS
            else:
                ang = outer_angles[idx - inner_count]
                r = _USER_RING_RADIUS + _USER_RING_GAP
            pos[u] = (r * math.cos(ang), r * math.sin(ang))

    # 4. Resolve cross-source user collisions iteratively in polar coordinates.
    user_polar: dict[Any, tuple[float, float]] = {}
    for s in sources:
        for u in users_by_source.get(s, []):
            x, y = pos.get(u, (0.0, 0.0))
            user_polar[u] = (math.hypot(x, y), math.atan2(y, x))

    for _it in range(_USER_MAX_ITERS):
        moved = False
        users_list = list(user_polar.keys())
        for i in range(len(users_list)):
            for j in range(i + 1, len(users_list)):
                u, v = users_list[i], users_list[j]
                # Only consider users from different sources.
                src_u = next((s for s, ul in users_by_source.items() if u in ul), None)
                src_v = next((s for s, ul in users_by_source.items() if v in ul), None)
                if src_u is None or src_v is None or src_u == src_v:
                    continue

                ru, au = user_polar[u]
                rv, av = user_polar[v]
                xu, yu = ru * math.cos(au), ru * math.sin(au)
                xv, yv = rv * math.cos(av), rv * math.sin(av)
                dist = math.hypot(xu - xv, yu - yv)

                if dist < _USER_MIN_SEP:
                    # Push both users outward and nudge their angles apart.
                    user_polar[u] = (ru + _USER_DELTA_R, au - _USER_DELTA_ANG)
                    user_polar[v] = (rv + _USER_DELTA_R, av + _USER_DELTA_ANG)
                    moved = True
        if not moved:
            break

    # Write the adjusted polar coordinates back to the position dict.
    for u, (r, ang) in user_polar.items():
        pos[u] = (r * math.cos(ang), r * math.sin(ang))

    # 5. Anything we still haven't placed: small spring layout, slightly scaled.
    remaining = [n for n in network.nodes() if n not in pos]
    if remaining:
        subG = network.subgraph(remaining).copy()
        if len(subG) > 0:
            sub_pos = nx.spring_layout(subG, seed=0, k=0.8, iterations=200)
            for n, (subx, suby) in sub_pos.items():
                pos[n] = (1.2 * subx, 1.2 * suby)

    # 6. Default per-edge losses where missing:
    #    source-source edges 5-10 dB, source-user edges 1-5 dB.
    rnd = random.Random(0)
    for u, v, data in network.edges(data=True):
        if data.get("loss") is not None:
            continue
        u_is_source = u in sources
        v_is_source = v in sources
        if u_is_source and v_is_source:
            data["loss"] = float(rnd.uniform(*_LOSS_SOURCE_TO_SOURCE))
        elif u_is_source != v_is_source:
            data["loss"] = float(rnd.uniform(*_LOSS_SOURCE_TO_USER))

    network.graph["pos"] = pos
    return pos


# --------------------------------------------------------------------------- #
# Public function
# --------------------------------------------------------------------------- #

def plot_network_solution_ring(
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

    # Slightly taller than the standard plot so users separated onto outer
    # rings still have vertical room.
    fig, ax = make_figure(figsize=(12, 9))
    pos = _layout(network)

    nx.draw_networkx_nodes(
        network, pos, ax=ax,
        node_size=2000,
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
        labels={n: entity_mathtext(str(n)) for n in network.nodes()},
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

        link_label = bar_link_label(
            option.get("link") or option.get("users") or option_link_label(option)
        )
        label = f"Link {link_label}"
        if k is not None:
            label += f"  (k={k}"
            if mu == mu:  # NaN-safe finite check
                label += f", μ={format_float(mu, '.3g')}"
            label += ")"

        handle = ax.plot([], [], color=color, lw=3.5, label=label)[0]
        legend_handles.append(handle)

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

    ax.set_axis_off()
    return save_figure(fig, outdir, filename)
