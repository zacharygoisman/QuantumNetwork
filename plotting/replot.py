# plotting/replot.py
"""
Regenerate the standard plots from a saved replot payload.

Run from the project root:

    python -m plotting.replot --payload outputs/replot_payload.json --outdir outputs_replot
"""

# ZHG
# 2026.03.26
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import networkx as nx

from data.save import load_json
from plotting.network import plot_network_solution
from plotting.utility import (
    plot_link_utility_bars,
    plot_source_allocation,
    plot_utility_comparison,
)


# --------------------------------------------------------------------------- #
# Payload -> NetworkX graph
# --------------------------------------------------------------------------- #

def _build_network_from_payload(payload: dict) -> nx.Graph:
    """Reconstruct the network (nodes, edges, cached positions) from a payload dict."""
    G = nx.Graph()

    for node in payload["network"]["nodes"]:
        G.add_node(node["id"], node_type=node.get("node_type"))

    for edge in payload["network"]["edges"]:
        G.add_edge(edge["u"], edge["v"], loss=float(edge["loss"]))

    pos = {
        node["id"]: tuple(node["pos"])
        for node in payload["network"]["nodes"]
        if node.get("pos") is not None
    }
    if pos:
        G.graph["pos"] = pos

    return G


# --------------------------------------------------------------------------- #
# Main entry points
# --------------------------------------------------------------------------- #

def replot_from_payload(payload_path: str | Path, outdir: str | Path = "outputs_replot") -> dict:
    """Load ``payload_path`` and regenerate every plot the pipeline produces."""
    payload = load_json(payload_path)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    network = _build_network_from_payload(payload)
    best = payload.get("best_result")
    results = payload.get("results_full", [])
    sources = payload.get("sources", {})

    cfg = SimpleNamespace(topology=payload.get("topology", "custom"))
    if payload.get("topology_name") == "manhattan_ilec" or payload.get("topology") == "manhattan":
        from plotting.network_manhattan import BALI_LABEL_MAP
        cfg.node_label_map = BALI_LABEL_MAP
        network.graph["node_label_map"] = BALI_LABEL_MAP

    if best:
        plot_network_solution(network, best, outdir=outdir)
        plot_link_utility_bars(cfg, best, outdir=outdir)
        plot_source_allocation(cfg, best, sources, outdir=outdir)

    if results:
        plot_utility_comparison(results, outdir=outdir)

    return {
        "network": network,
        "best": best,
        "results": results,
        "sources": sources,
        "outdir": outdir,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--payload",
        default="outputs/replot_payload.json",
        help="Path to saved replot payload JSON",
    )
    parser.add_argument(
        "--outdir",
        default="outputs_replot",
        help="Directory for regenerated plots",
    )
    args = parser.parse_args()

    replot_from_payload(args.payload, args.outdir)


if __name__ == "__main__":
    main()
