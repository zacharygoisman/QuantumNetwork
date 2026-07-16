"""
network_manhattan.py

Manhattan ILEC plotting wrapper for the QuantumNetwork repository.

Purpose
-------
This file lets you keep the original plot_network_solution(...) function
unchanged while giving the Manhattan ILEC topology a custom schematic layout.

Recommended location
--------------------
Put this file here:

    QuantumNetwork/plotting/network_manhattan.py

Then in pipeline/runner.py, import:

    from plotting.network_manhattan import plot_manhattan

and call:

    elif cfg.topology == "custom" and network.number_of_nodes() == 17 and network.number_of_edges() > 80:
        plot_manhattan(network, best, outdir=outdir)

or, if you added cfg.topology_name = "manhattan_ilec":

    elif getattr(cfg, "topology_name", "") == "manhattan_ilec":
        plot_manhattan(network, best, outdir=outdir)

Notes
-----
This is a schematic layout, not a true geographic/GIS layout. It is designed
to resemble the long, narrow shape of Manhattan while preserving your original
plotting style, font sizes, colors, labels, edge labels, legends, and selected
lightpath drawing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import networkx as nx

# Prefer relative imports when this file lives inside the plotting/ package.
try:
    from .network import plot_network_solution
except Exception:  # pragma: no cover
    from plotting.network import plot_network_solution


Position = Dict[Any, Tuple[float, float]]

BALI_LABEL_MAP = {
    # Sources
    "S1": r"s_B",
    "S2": r"s_M",
    "S3": r"s_N",

    # Users
    "U1":  r"u_A",
    "U2":  r"u_C",
    "U3":  r"u_D",
    "U4":  r"u_E",
    "U5":  r"u_F",
    "U6":  r"u_G",
    "U7":  r"u_H",
    "U8":  r"u_I",
    "U9":  r"u_J",
    "U10": r"u_K",
    "U11": r"u_L",
    "U12": r"u_O",
    "U13": r"u_P",
    "U14": r"u_Q",
}


def apply_manhattan_label_map(cfg):
    """Attach Manhattan display labels to a config object for non-network plots."""
    try:
        cfg.node_label_map = BALI_LABEL_MAP
    except Exception:
        pass
    return cfg

def scale_pos(pos: Position, factor_x: float = 1.0, factor_y: float = 1.0) -> Position:
    """
    Scale the layout around its center.

    factor_x > 1 stretches horizontally.
    factor_y > 1 stretches vertically.
    """
    xs = [x for x, y in pos.values()]
    ys = [y for x, y in pos.values()]

    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    return {
        node: (
            cx + factor_x * (x - cx),
            cy + factor_y * (y - cy),
        )
        for node, (x, y) in pos.items()
    }

def manhattan_ilec_schematic_pos():
    """
    Manhattan-shaped schematic layout based on the Bali et al. topology figure.

    This is not a geographic/GIS layout. It is a manually designed schematic
    chosen to resemble the published Manhattan ILEC topology image while
    keeping the internal QuantumNetwork node names.

    Sources:
        S1 = B
        S2 = M
        S3 = N

    Users:
        U1  = A
        U2  = C
        U3  = D
        U4  = E
        U5  = F
        U6  = G
        U7  = H
        U8  = I
        U9  = J
        U10 = K
        U11 = L
        U12 = O
        U13 = P
        U14 = Q
    """

    # Coordinates chosen to mimic the uploaded Bali figure.
    # x increases roughly eastward/rightward.
    # y increases roughly northward/upward.
    base_pos = {
        # Sources
        "S1": (9.82, 0.90),  # B
        "S2": (4.19, 1.44),  # M
        "S3": (3.91, 0.11),  # N

        # Users
        "U1":  (10.17, 0.37),  # A
        "U2":  (8.94, 0.97),   # C
        "U3":  (8.21, 0.63),   # D
        "U4":  (7.71, 0.09),   # E
        "U5":  (6.79, 0.13),   # F
        "U6":  (7.30, 1.15),   # G
        "U7":  (6.41, 1.20),   # H
        "U8":  (5.82, 0.83),   # I
        "U9":  (6.08, 0.15),   # J
        "U10": (5.09, 0.16),   # K
        "U11": (5.50, 1.56),   # L
        "U12": (2.87, 0.20),   # O
        "U13": (2.25, 1.30),   # P
        "U14": (0.11, 1.42),   # Q
    }

    return scale_pos(base_pos, factor_x=1.8, factor_y=2.4)


def manhattan_ilec_original_label_pos() -> Position:
    """
    Fallback schematic layout keyed by the original ILEC labels A-Q.

    Note: these coordinates use a portrait (tall/narrow) orientation and are
    NOT the same as :func:`manhattan_ilec_schematic_pos`, which is landscape.
    Only used if the graph nodes are literally A, B, ..., Q instead of the
    QuantumNetwork-style S1/U1 mapping.
    """
    return {
        "A": (-0.72, 0.00),
        "B": (-0.52, 0.45),
        "C": (-0.87, 1.05),
        "D": (-0.25, 1.25),
        "E": (-0.68, 1.85),
        "F": (-0.43, 2.55),
        "G": (-0.98, 2.70),
        "H": (-0.70, 3.35),
        "I": (-0.30, 3.55),
        "J": (-0.55, 4.10),
        "K": (-0.10, 4.75),
        "L": (-0.78, 4.95),
        "M": (-0.45, 5.65),
        "N": (0.05, 6.20),
        "O": (0.15, 6.95),
        "P": (-1.08, 5.75),
        "Q": (-0.25, 7.60),
    }


def _layout_covers_graph(pos: Position, network: nx.Graph) -> bool:
    """Return True when every graph node has a coordinate."""
    return all(node in pos for node in network.nodes)


def _fallback_layout(network: nx.Graph) -> Position:
    """
    Fallback layout used if the graph does not match the expected Manhattan
    node naming. This should rarely be used for the suggested Manhattan preset.
    """
    return nx.spring_layout(network, seed=0, k=1.3, iterations=200)


def set_manhattan_pos(network: nx.Graph) -> Position:
    """
    Cache a Manhattan-shaped schematic layout in network.graph["pos"].

    This is the key function used by plot_manhattan(...). It tries both the
    QuantumNetwork S/U node mapping and the original A-Q node labels.

    Returns
    -------
    dict
        Position dictionary assigned to network.graph["pos"].
    """
    # If positions are already cached, use them.
    if "pos" in network.graph:
        return network.graph["pos"]

    su_pos = manhattan_ilec_schematic_pos()
    if _layout_covers_graph(su_pos, network):
        pos = {node: su_pos[node] for node in network.nodes}
        network.graph["pos"] = pos
        return pos

    original_pos = manhattan_ilec_original_label_pos()
    if _layout_covers_graph(original_pos, network):
        pos = {node: original_pos[node] for node in network.nodes}
        network.graph["pos"] = pos
        return pos

    # Fallback preserves functionality even if node names differ.
    pos = _fallback_layout(network)
    network.graph["pos"] = pos
    return pos


def plot_manhattan(
    network: nx.Graph,
    best: dict[str, Any] | None,
    outdir: str | Path = "outputs",
):
    network.graph.pop("pos", None)

    set_manhattan_pos(network)

    network.graph["node_label_map"] = BALI_LABEL_MAP
    network.graph["suppress_edge_labels"] = True
    network.graph["equal_aspect"] = True

    # Larger saved canvas.
    network.graph["figsize"] = (22, 9)

    # Keep nodes/text the same screen size.
    network.graph["node_size"] = 2000

    # This controls actual physical network scale.
    # Smaller = larger network.
    network.graph["data_units_per_inch"] = 1.25

    network.graph["pad_x"] = 0.75
    network.graph["pad_y"] = 0.75

    # Prevent tight cropping from shrinking the canvas.
    network.graph["bbox_inches"] = None

    return plot_network_solution(network, best, outdir=outdir)


def is_likely_manhattan_ilec(network: nx.Graph) -> bool:
    """
    Convenience detector for runner.py if you do not add cfg.topology_name.

    The suggested Manhattan ILEC preset has:
        |V| = 17
        |E| = 110

    This detector uses a loose edge threshold to remain robust if you slightly
    prune/modify the graph.
    """
    return network.number_of_nodes() == 17 and network.number_of_edges() > 80


def manhattan_ilec_density(network: nx.Graph | None = None) -> float:
    """
    Return the graph density.

    If a network is provided, uses nx.density(network).
    If no network is provided, returns the density of the suggested ILEC graph:
        |V| = 17, |E| = 110, D = 110 / C(17, 2).
    """
    if network is not None:
        return nx.density(network)

    n_nodes = 17
    n_edges = 110
    return n_edges / (n_nodes * (n_nodes - 1) / 2)
