#data/dataframe.py
"""
Convert results into pandas dataframes
"""

#ZHG
#2026.03.26
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

import pandas as pd


def _alloc_for_opt(result, opt):
    """Return the allocation dict for a combo option, falling back to opt's own."""
    alloc = result.get("allocation", {}).get(id(opt), {})
    if not alloc and "allocation" in opt:
        alloc = opt["allocation"]
    return alloc


def _assign_for_idx(result, i, opt):
    """Return the assignment dict for combo position i, falling back to opt's own."""
    assign = result.get("assignment", {}).get(i, {})
    if not assign and "assignment" in opt:
        assign = opt["assignment"]
    return assign


def combo_to_rows(result, combo_idx=None):
    """Flatten a single combo result into one row per link, returned as a DataFrame."""
    rows = []
    combo = result["combo"]

    for i, opt in enumerate(combo):
        alloc = _alloc_for_opt(result, opt)
        assign = _assign_for_idx(result, i, opt)

        rows.append({
            "combo_idx": combo_idx,
            "combo_link_pos": i,
            "valid": result.get("valid", False),
            "reason": result.get("reason"),
            "combo_utility": result.get("utility"),
            "combo_path_ub": result.get("combo_path_ub"),
            "combo_link_ub": result.get("combo_link_ub"),

            "link_idx": opt.get("link_idx"),
            "u1": opt["link"][0],
            "u2": opt["link"][1],
            "source": opt["source"],

            "path1": "->".join(opt["path1"]),
            "path2": "->".join(opt["path2"]),

            "y1": opt["y1"],
            "y2": opt["y2"],
            "dark_count_1": opt.get("dark_count_1"),
            "dark_count_2": opt.get("dark_count_2"),
            "fidelity_limit": opt.get("fidelity_limit"),

            "total_loss": opt.get("total_loss"),
            "path_ub": opt.get("path_ub"),
            "link_ub": opt.get("link_ub"),

            "allocated_k": alloc.get("k"),
            "mu": alloc.get("mu"),
            "prelog_rate": alloc.get("prelog_rate"),
            "link_utility": alloc.get("link_utility"),

            "swap": assign.get("swap"),
            "channels": assign.get("channels"),
            "to_u1": assign.get("to_u1"),
            "to_u2": assign.get("to_u2"),
        })

    return pd.DataFrame(rows)


def results_summary_df(results):
    """Build a one-row-per-combo summary DataFrame across all results."""
    rows = []
    for idx, r in enumerate(results):
        combo_idx = r.get("combo_idx", idx)
        combo = r.get("combo", [])

        sources_used = sorted({opt.get("source") for opt in combo})
        total_loss = sum(float(opt.get("total_loss", 0.0)) for opt in combo)
        total_path_ub = sum(float(opt.get("path_ub", 0.0)) for opt in combo)
        total_link_ub = sum(float(opt.get("link_ub", 0.0)) for opt in combo)

        combo_signature = " | ".join(
            f"{opt['link'][0]}-{opt['link'][1]}:{opt['source']}"
            for opt in combo
        )

        rows.append({
            "combo_idx": combo_idx,
            "valid": r.get("valid", False),
            "reason": r.get("reason"),
            "utility": r.get("utility"),
            "combo_path_ub": r.get("combo_path_ub"),
            "combo_link_ub": r.get("combo_link_ub"),
            "n_links": len(combo),
            "n_unique_sources": len(sources_used),
            "sources_used": ",".join(map(str, sources_used)),
            "sum_total_loss": total_loss,
            "sum_path_ub_from_links": total_path_ub,
            "sum_link_ub_from_links": total_link_ub,
            "combo_signature": combo_signature,
        })

    return pd.DataFrame(rows)


def all_results_link_rows_df(results):
    """Concatenate the per-link rows of every combo into a single DataFrame."""
    frames = []
    for idx, r in enumerate(results):
        combo_idx = r.get("combo_idx", idx)
        frames.append(combo_to_rows(r, combo_idx=combo_idx))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)