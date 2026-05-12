# plotting/replot.py
#cd 'C:\Users\zgois\OneDrive\Desktop\Stuff\Quantum\QuantumNetwork\locked_in_version'
#python -m plotting.replot --payload outputs/replot_payload.json --outdir outputs_replot
from pathlib import Path
import argparse
import networkx as nx

from data.save import load_json
from plotting.network import plot_network_solution
from plotting.utility import (
    plot_link_utility_bars,
    plot_utility_comparison,
    plot_source_allocation,
)
#from plotting.composite import plot_combined_solution

def _build_network_from_payload(payload):
    G = nx.Graph()

    for node in payload["network"]["nodes"]:
        node_id = node["id"]
        attrs = {"node_type": node.get("node_type")}
        G.add_node(node_id, **attrs)

    for edge in payload["network"]["edges"]:
        G.add_edge(edge["u"], edge["v"], loss=float(edge["loss"]))

    pos = {}
    for node in payload["network"]["nodes"]:
        if node.get("pos") is not None:
            pos[node["id"]] = tuple(node["pos"])

    if pos:
        G.graph["pos"] = pos

    return G


def replot_from_payload(payload_path, outdir="outputs_replot"):
    payload = load_json(payload_path)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    network = _build_network_from_payload(payload)
    best = payload.get("best_result")
    results = payload.get("results_full", [])
    sources = payload.get("sources", {})

    if best:
        plot_network_solution(network, best, outdir=outdir)
        plot_link_utility_bars(best, outdir=outdir)
        plot_source_allocation(best, sources, outdir=outdir)
        #plot_combined_solution(network, best, sources, outdir=outdir)

    if results:
        plot_utility_comparison(results, outdir=outdir)

    return {
        "network": network,
        "best": best,
        "results": results,
        "sources": sources,
        "outdir": outdir,
    }


def main():
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