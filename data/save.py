#data/save.py
"""
Save files and dataframes
"""

#ZHG
#2026.03.26
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

import json
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any


def ensure_output_dir(path="outputs"):
    """Create the output directory if needed and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _jsonable(obj: Any):
    """Recursively convert dataclasses, Paths, dicts and sequences into JSON-safe values."""
    if is_dataclass(obj):
        return _jsonable(asdict(obj))

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]

    return obj


def save_json(data, filepath):
    """Serialize data to JSON-safe form and write it to filepath (creating parent dirs)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2)


def load_json(filepath):
    """Read and parse a JSON file, returning the decoded object."""
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_df(df, filepath):
    """Write a DataFrame to CSV at filepath (creating parent dirs), without the index."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def serialize_network(network):
    """Convert a network graph into a JSON-safe {nodes, edges} dict with positions and losses."""
    pos = network.graph.get("pos", {})

    nodes = []
    for node, data in network.nodes(data=True):
        node_pos = pos.get(node)
        nodes.append({
            "id": str(node),
            "node_type": data.get("node_type"),
            "pos": [float(node_pos[0]), float(node_pos[1])] if node_pos is not None else None,
        })

    edges = []
    for u, v, data in network.edges(data=True):
        edges.append({
            "u": str(u),
            "v": str(v),
            "loss": float(data.get("loss", 0.0)),
        })

    return {
        "nodes": nodes,
        "edges": edges,
    }


def serialize_result(result):
    """Convert a single combo result into a JSON-safe dict, resolving each option's
    allocation/assignment from the result-level maps (keyed by id(opt) or index) or
    the option's own embedded data. Returns None when result is None."""
    if result is None:
        return None

    combo = list(result.get("combo", []))
    allocation = result.get("allocation", {}) or {}
    assignment = result.get("assignment", {}) or {}

    combo_out = []
    for i, opt in enumerate(combo):
        opt_copy = dict(opt)

        alloc = allocation.get(id(opt), {})
        if not alloc:
            alloc = allocation.get(i, {})
        if not alloc and "allocation" in opt_copy:
            alloc = opt_copy["allocation"]

        assign = assignment.get(i, {})
        if not assign and "assignment" in opt_copy:
            assign = opt_copy["assignment"]

        opt_copy["allocation"] = dict(alloc) if alloc else {}
        opt_copy["assignment"] = dict(assign) if assign else {}
        opt_copy["combo_index"] = i

        combo_out.append(opt_copy)

    return {
        "valid": bool(result.get("valid", False)),
        "reason": result.get("reason"),
        "utility": result.get("utility"),
        "combo_path_ub": result.get("combo_path_ub"),
        "combo_link_ub": result.get("combo_link_ub"),
        "combo": combo_out,
    }


def serialize_results(results):
    """Serialize a list of results, returning an empty list when results is falsy."""
    return [serialize_result(r) for r in (results or [])]


def build_replot_payload(
    *,
    cfg,
    network,
    sources,
    links,
    best_result,
    results_full,
    summary,
):
    """Assemble a single JSON-safe payload bundling the config, network, sources,
    links, best result, full results and summary for later replotting."""
    return {
        "cfg": _jsonable(cfg),
        "network": serialize_network(network),
        "sources": _jsonable(sources),
        "links": _jsonable(links),
        "best_result": serialize_result(best_result),
        "results_full": serialize_results(results_full),
        "summary": _jsonable(summary),
    }