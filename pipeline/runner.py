#pipeline/runner.py
"""
Runs through entire pipeline of building network, computing paths, generating combos, 
evaluating, and outputting results. This is the main entry point for running the routing 
and spectrum allocation code.
"""

#ZHG
#2026.03.20
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

import time
from network.builder import build_network
from routing.paths import build_path_options
from routing.combos import generate_combos
from pipeline.evaluator import evaluate_stream, select_best
from data.save import save_json, save_df, ensure_output_dir, build_replot_payload
from data.dataframe import results_summary_df, combo_to_rows, all_results_link_rows_df
from plotting.network import plot_network_solution
from plotting.network_ring import plot_network_solution_ring
from plotting.utility import (
    plot_link_utility_bars,
    plot_link_normalized_rate_bars,
    plot_utility_comparison,
    plot_source_allocation,
)
from plotting.composite import plot_paper_combined_solution_ring
from plotting.network_manhattan import plot_manhattan
from plotting.network_manhattan import BALI_LABEL_MAP
from analysis.metrics import per_link_ub_value


def run_pipeline(cfg):
    """Run the full pipeline (build, route, combo, evaluate, select, output) and
    return the list of evaluated results, or None if no feasible solution is found."""
    print("=== RUN PIPELINE ===")
    t0 = time.time()

    #1. Build network
    network, sources, links = build_network(cfg)
    if cfg.verbose:
        print(f"Network built in {time.time() - t0:.2f}s")

    #2. Routing
    paths = build_path_options(network, links, sources, cfg)
    if cfg.verbose:
        print(f"Paths computed in {time.time() - t0:.2f}s")

    #3. Generate combos
    total_possible_combos = ((cfg.n_paths_per_leg ** 2) * len(sources)) ** len(links)

    combos_to_test_cap = total_possible_combos
    if cfg.max_combos is not None:
        combos_to_test_cap = min(total_possible_combos, int(cfg.max_combos))

    print(f"Total possible combinations: {total_possible_combos}")
    print(f"Maximum combinations to evaluate: {combos_to_test_cap}")

    if cfg.use_upper_bound:
        print("Note: actual evaluated combinations may be lower because upper-bound pruning can skip combos.")

    combo_stream = generate_combos(paths, cfg)

    if cfg.verbose:
        print(f"Combos generated in {time.time() - t0:.2f}s")

    #4. Evaluate
    results = evaluate_stream(combo_stream, network, sources, cfg)
    if cfg.verbose:
        print(f"Evaluation completed in {time.time() - t0:.2f}s")

    #5. Select best
    first_feasible = next((r for r in results if r.get("valid")), None)
    # route/combo index for first feasible
    first_feasible_idx = None if first_feasible is None else first_feasible.get("combo_idx")
    first_feasible_time = None if first_feasible is None else first_feasible.get("elapsed_s")

    best = select_best(results)
    best_idx = None if best is None else best.get("combo_idx")
    best_reached_time = None if best is None else best.get("elapsed_s")

    # Print route numbers and time-to-best (elapsed since start of evaluation)
    if first_feasible is None:
        print("No first feasible (valid) combo found during evaluation.")
    else:
        if first_feasible_time is not None:
            print(f"First working combo index: {first_feasible_idx} (reached at {first_feasible_time:.3f}s into evaluation)")
        else:
            print(f"First working combo index: {first_feasible_idx}")

    if best is None:
        print("No feasible solution found (no best result).")
    else:
        if best_reached_time is not None:
            print(f"Best combo index: {best_idx} (reached at {best_reached_time:.3f}s into evaluation)")
            print(f"Time to reach best solution: {best_reached_time:.3f} s")
        else:
            # fallback: print total pipeline time as a coarse approximation
            print(f"Best combo index: {best_idx}")
            print(f"Best combo selected in {time.time() - t0:.2f}s")

    total_pipeline_time = time.time() - t0

    if best is None:
        print("No feasible solution found.")
        print(f"Total pipeline time: {total_pipeline_time:.3f} s")
        return None

    # 6. Summary metrics
    total_network_utility = float(best.get("utility", float("nan")))

    total_utility_upper_bound_limit = float(
        sum(float(opt.get("link_ub", 0.0)) for opt in best.get("combo", []))
    )

    alloc_map = best.get("allocation", {}) or {}
    per_link_pct = []

    for i, opt in enumerate(best.get("combo", [])):
        alloc = alloc_map.get(id(opt), {}) if alloc_map else {}
        if not alloc:
            alloc = alloc_map.get(i, {})
        if not alloc and "allocation" in opt:
            alloc = opt["allocation"]

        actual_log_rate = alloc.get("link_utility")
        best_log_rate = opt.get("link_ub")

        if actual_log_rate is not None and best_log_rate is not None:
            actual_log_rate = float(actual_log_rate)
            best_log_rate = float(best_log_rate)

            # 100 * R_actual / R_best_infinite
            per_link_pct.append(100.0 * 10.0 ** (actual_log_rate - best_log_rate))

    avg_link_utility_gap = (
        total_utility_upper_bound_limit - total_network_utility
    ) / len(links)

    avg_pct_of_link_upper_bound = (
        sum(per_link_pct) / len(per_link_pct) if per_link_pct else float("nan")
    )

    print("\n=== PIPELINE SUMMARY ===")
    print(f"Total pipeline time: {total_pipeline_time:.3f} s")
    print(f"Total network utility: {total_network_utility:.6f}")
    print(f"Total utility upper bound limit: {total_utility_upper_bound_limit:.6f}")
    print(
        "Average link utility gap: "
        f"{avg_link_utility_gap:.6e}"
    )
    print(
        "Average percent of best-path infinite-resource rate: "
        f"{avg_pct_of_link_upper_bound:.6f}%"
    )
    # 7. Output dir
    outdir = ensure_output_dir(cfg.output_directory)

    # 8. CSVs

    if results:
        save_df(
            results_summary_df(results),
            outdir / "all_results_summary.csv",
        )
        save_df(
            all_results_link_rows_df(results),
            outdir / "all_results_links.csv",
        )
        save_json(best, outdir / "best_result.json")

        best_links_df = combo_to_rows(best, combo_idx=None)
        best_links_df["normalized_rate"] = 10.0 ** (
            best_links_df["link_utility"] - best_links_df["link_ub"]
        )
        save_df(best_links_df, outdir / "best_links.csv")

    # 9. Plots first so node positions get cached into network.graph["pos"]
    if cfg.topology == "ring":
        plot_network_solution_ring(network, best, outdir=outdir)
        plot_paper_combined_solution_ring(
            network,
            best,
            sources,
            outdir="outputs_replot",
            filename="combined_solution_paper.pdf",
            width_inches=6.5,
            font_size=8.0,
            layout="stacked",
        )
    elif cfg.topology_name == "manhattan":
        cfg.node_label_map = BALI_LABEL_MAP
        plot_manhattan(network, best, outdir=outdir)
    else:
        plot_network_solution(network, best, outdir=outdir)

    if results:
        plot_link_utility_bars(cfg, best, outdir=outdir)
        plot_link_normalized_rate_bars(cfg, best, outdir=outdir)
        plot_utility_comparison(results, outdir=outdir)

    plot_source_allocation(cfg, best, sources, outdir=outdir)

    # 10. Save normalized replot payload
    summary = {
        "total_pipeline_time": total_pipeline_time,
        "total_network_utility": total_network_utility,
        "total_utility_upper_bound_limit": total_utility_upper_bound_limit,
        "avg_link_utility_gap": avg_link_utility_gap,
        "avg_pct_of_best_path_infinite_resource_rate": avg_pct_of_link_upper_bound,
    }

    payload = build_replot_payload(
        cfg=cfg,
        network=network,
        sources=sources,
        links=links,
        best_result=best,
        results_full=results,
        summary=summary,
    )

    save_json(payload, outdir / "replot_payload.json")

    # Write the normalized best result on its own for convenience.
    save_json(payload["best_result"], outdir / "best_result.json")

    return results