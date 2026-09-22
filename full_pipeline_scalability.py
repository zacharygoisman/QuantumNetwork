"""
Full-pipeline scalability / topology benchmark for the QuantumNetwork project.

Place this file in the TOP LEVEL of the repository, next to main.py.

Purpose
-------
This script is intended for the third reviewer-response figure.  Unlike the
first two validation figures, this is NOT an exhaustive-certification study.
It runs the production routing + APOPT allocation + CP-SAT scheduling pipeline
over ensembles of network topologies and records three quantities:

1. Total pipeline runtime.
2. Arithmetic mean of the individual link-rate percentages relative to the
   per-link infinite-resource upper bounds.
3. Number of routing combinations attempted before the first feasible
   end-to-end solution, or before the max-combos search limit is reached.

The three publication panels are generated separately, with matching size,
font, colors, and no titles:

    full_pipeline_runtime_vs_complexity.{png,pdf}
    full_pipeline_rate_vs_complexity.{png,pdf}
    full_pipeline_attempts_vs_complexity.{png,pdf}

Complexity metric
-----------------
The default x-axis is a NOMINAL combined algorithmic search-space proxy:

    C_combined = (S * N^2)^L * C(K, L)

where:
    S = number of sources
    N = Double-Yen paths retained per source-user leg
    L = number of requested entangled user pairs
    K = channels available per source

The first factor is the nominal routing-combination count used in the current
pipeline.  The second is the same EFA integer-allocation count used in the
standalone APOPT validation figure.

This product is a complexity PROXY, not an exact operation count for the full
pipeline.  In a multisource network, a route combination can distribute links
among sources, so APOPT actually solves separate source-specific EFA problems.
The CSV therefore also stores:

    routing_complexity
    efa_complexity
    combined_complexity
    actual_route_combos
    actual_combined_complexity
    number of nodes
    number of edges
    realized graph density

so the x-axis can be changed later without rerunning the simulations.

You can replot with one of three x metrics:

    --x-metric combined   -> (S*N^2)^L * C(K,L)          [default]
    --x-metric routing    -> (S*N^2)^L
    --x-metric actual     -> actual route combos * C(K,L)

Rate metric
-----------
For the first feasible full-pipeline solution,

    U = sum_l log10(R_l),

and the corresponding per-link infinite-resource upper bounds define

    U_UB = sum_l log10(R_l^UB).

The script reports

    100 * 10^[(U - U_UB) / L],

which is the achieved geometric-mean link rate expressed as a percentage of
the geometric mean of the corresponding per-link infinite-resource upper
bounds.  This is not a percentage of an exact global RSA optimum.

Attempts metric
---------------
The pipeline evaluator numbers routing combinations from zero.  The number of
attempts required to reach the first valid end-to-end solution is therefore

    first_valid_combo_idx + 1.

A "valid" result has passed both APOPT allocation and CP-SAT interference /
frequency scheduling.

For plotting, unsuccessful cases are included using the total number of
combinations evaluated (all failed attempts).

Topologies
----------
Supported directly by the current repository:
    dense
    ring
    star

Dense networks are repeated for every requested --density value.
Ring and star ignore density in the repository, so each is generated only once
per parameter tuple / repetition.

The repository's current star constructor connects users only to the first
source, so this script restricts star tests to S=1.  The current ring
constructor is most meaningful for S>=2, so S=1 ring cases are skipped.

Example small run
-----------------
    python full_pipeline_scalability_paper.py ^
        --topologies dense ring star ^
        --users 6 8 ^
        --sources 1 2 ^
        --links 1 2 3 ^
        --channels 10 20 ^
        --n-paths 2 3 ^
        --densities 0.15 0.35 ^
        --per-shape 2 ^
        --max-combos 200

Replot without rerunning:
    python full_pipeline_scalability_paper.py ^
        --records-csv outputs/full_pipeline_scalability/full_pipeline_results.csv

"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from config.config import Config
from network.builder import build_network
from pipeline.evaluator import evaluate_combo
from routing.combos import generate_combos
from routing.paths import build_path_options


# ============================================================================
# Publication style
# ============================================================================

PAPER_FIGSIZE = (6.5, 4.0)
PAPER_DPI = 600

FONT_SIZE = 16
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 16
LEGEND_SIZE = 14

RAW_MARKER_SIZE = 34
MEDIAN_MARKER_SIZE = 6
MEDIAN_LINE_WIDTH = 2.0
RAW_ALPHA = 0.34
GRID_ALPHA = 0.22

# Colorblind-friendly Okabe-Ito-inspired palette.
TOPOLOGY_STYLE = {
    "dense": {
        "color": "#0072B2",
        "marker": "o",
        "label": "Dense",
    },
    "ring": {
        "color": "#D55E00",
        "marker": "s",
        "label": "Ring",
    },
    "star": {
        "color": "#009E73",
        "marker": "^",
        "label": "Star",
    },
}

plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ============================================================================
# Data model
# ============================================================================

@dataclass(frozen=True)
class BenchmarkShape:
    topology: str
    density: float | None
    num_users: int
    num_sources: int
    num_links: int
    channels_per_source: int
    n_paths: int
    repetition: int
    network_seed: int


# ============================================================================
# Complexity metrics
# ============================================================================

def _routing_complexity(
    num_sources: int,
    n_paths: int,
    num_links: int,
) -> int:
    """
    Nominal number of Double-Yen routing combinations:

        (S * N^2)^L
    """
    return (
        num_sources
        * n_paths
        * n_paths
    ) ** num_links


def _efa_complexity(
    channels_per_source: int,
    num_links: int,
) -> int:
    """
    Same positive-integer EFA allocation count used in Figure 1:

        C(K, L)

    For this full-pipeline benchmark we require K >= L so this metric remains
    directly comparable to the standalone EFA validation metric.
    """
    if channels_per_source < num_links:
        raise ValueError(
            "This benchmark defines EFA complexity as C(K,L), "
            "so channels_per_source must be >= num_links."
        )

    return math.comb(
        channels_per_source,
        num_links,
    )


def _combined_complexity(
    num_sources: int,
    n_paths: int,
    num_links: int,
    channels_per_source: int,
) -> int:
    return (
        _routing_complexity(
            num_sources,
            n_paths,
            num_links,
        )
        * _efa_complexity(
            channels_per_source,
            num_links,
        )
    )


def _product(values) -> int:
    answer = 1

    for value in values:
        answer *= int(value)

    return answer


# ============================================================================
# Reproducible test-case construction
# ============================================================================

def _case_seed(
    base_seed: int,
    topology: str,
    density: float | None,
    num_users: int,
    num_sources: int,
    num_links: int,
    channels_per_source: int,
    n_paths: int,
    repetition: int,
) -> int:
    topology_code = {
        "dense": 11,
        "ring": 23,
        "star": 37,
    }[topology]

    density_code = (
        0
        if density is None
        else int(round(float(density) * 10_000))
    )

    return (
        int(base_seed)
        + topology_code * 100_000_007
        + density_code * 1_000_003
        + int(num_users) * 100_003
        + int(num_sources) * 10_007
        + int(num_links) * 1_009
        + int(channels_per_source) * 101
        + int(n_paths) * 11
        + int(repetition)
    )


def _make_config(
    shape: BenchmarkShape,
    max_combos: int,
    min_loss_db: float,
    max_loss_db: float,
    min_dark_count: float,
    max_dark_count: float,
    min_fidelity: float,
    max_fidelity: float,
    use_upper_bound: bool,
    use_best_first: bool,
) -> Config:
    """
    Construct one production-pipeline Config using the same physical ranges
    used by the validation scripts.
    """
    rng = random.Random(
        shape.network_seed + 31_415_927
    )

    dark_counts = [
        rng.uniform(
            min_dark_count,
            max_dark_count,
        )
        for _ in range(shape.num_users)
    ]

    fidelity_limits = [
        rng.uniform(
            min_fidelity,
            max_fidelity,
        )
        for _ in range(shape.num_links)
    ]

    density_value = (
        0.25
        if shape.density is None
        else float(shape.density)
    )

    # build_path_options truncates to combo_limit_per_link after generating
    # S*N^2 source/path-pair candidates.  Set the limit to the nominal maximum
    # so N, rather than an unrelated fixed cap, controls the routing study.
    combo_limit_per_link = (
        shape.num_sources
        * shape.n_paths
        * shape.n_paths
    )

    return Config(
        num_usr=shape.num_users,
        num_src=shape.num_sources,
        num_lnks=shape.num_links,
        topology=shape.topology,
        topology_name="random",
        density=density_value,
        loss_range=(
            min_loss_db,
            max_loss_db,
        ),
        num_channels=[
            shape.channels_per_source
            for _ in range(shape.num_sources)
        ],
        random_seed=shape.network_seed,
        require_disjoint_links=True,
        n_paths_per_leg=shape.n_paths,
        combo_limit_per_link=combo_limit_per_link,
        use_best_first=use_best_first,
        upper_bound_sort=False,
        use_diverse_paths=False,
        fidelity_limit=fidelity_limits,
        tau=1e-9,
        dark_count_rate=dark_counts,
        use_upper_bound=use_upper_bound,
        max_combos=max_combos,
        verbose=False,
        parallel=False,
    )


def _iter_shapes(
    topologies: list[str],
    densities: list[float],
    user_values: list[int],
    source_values: list[int],
    link_values: list[int],
    channel_values: list[int],
    n_path_values: list[int],
    per_shape: int,
    seed: int,
):
    """
    Yield all requested benchmark shapes while avoiding meaningless duplicate
    density sweeps for structured topologies.
    """
    for topology in topologies:

        density_values: list[float | None]

        if topology == "dense":
            density_values = [
                float(value)
                for value in densities
            ]
        else:
            density_values = [None]

        for density in density_values:
            for num_users in user_values:
                for num_sources in source_values:

                    # Current repository behavior:
                    #   star -> users connect only to source 1
                    #   ring with a single source degenerates to a self-loop
                    if (
                        topology == "star"
                        and num_sources != 1
                    ):
                        continue

                    if (
                        topology == "ring"
                        and num_sources < 2
                    ):
                        continue

                    for num_links in link_values:

                        if num_users < 2 * num_links:
                            continue

                        for channels_per_source in channel_values:

                            # Keeps C(K,L) well-defined and comparable to the
                            # standalone EFA figure.
                            if channels_per_source < num_links:
                                continue

                            for n_paths in n_path_values:
                                for repetition in range(
                                    1,
                                    per_shape + 1,
                                ):
                                    network_seed = _case_seed(
                                        base_seed=seed,
                                        topology=topology,
                                        density=density,
                                        num_users=num_users,
                                        num_sources=num_sources,
                                        num_links=num_links,
                                        channels_per_source=channels_per_source,
                                        n_paths=n_paths,
                                        repetition=repetition,
                                    )

                                    yield BenchmarkShape(
                                        topology=topology,
                                        density=density,
                                        num_users=num_users,
                                        num_sources=num_sources,
                                        num_links=num_links,
                                        channels_per_source=channels_per_source,
                                        n_paths=n_paths,
                                        repetition=repetition,
                                        network_seed=network_seed,
                                    )


# ============================================================================
# Full-pipeline metrics
# ============================================================================

def _first_feasible_metrics(
    results: list[dict],
) -> tuple[
    int | None,
    float | None,
]:
    """
    Return:
        attempts_to_first_feasible
        evaluation_time_to_first_feasible_s

    evaluate_stream assigns combo_idx=0 to the first yielded routing
    combination.  Before any feasible solution exists, upper-bound pruning
    cannot occur because best_utility starts at -infinity, so combo_idx + 1
    is also the number of full candidate attempts required to get the first
    feasible solution.
    """
    for result in results:
        if result.get("valid"):
            combo_idx = result.get(
                "combo_idx"
            )

            elapsed_s = result.get(
                "elapsed_s"
            )

            attempts = (
                None
                if combo_idx is None
                else int(combo_idx) + 1
            )

            elapsed = (
                None
                if elapsed_s is None
                else float(elapsed_s)
            )

            return attempts, elapsed

    return None, None


def _geometric_mean_rate_pct_upper_bound(
    best: dict | None,
    num_links: int,
) -> float | None:
    """
    Geometric-mean achieved link rate as a percentage of the corresponding
    per-link infinite-resource upper-bound rates.

    If

        U = sum_l log10(R_l)
        U_UB = sum_l log10(R_l^UB),

    then

        percentage = 100 * 10**((U - U_UB) / L).

    This is the uniform multiplicative rate-retention factor corresponding to
    the pipeline's sum-log utility.
    """
    if best is None or num_links <= 0:
        return None

    utility = best.get("utility")
    combo = best.get("combo", [])

    if utility is None or len(combo) != num_links:
        return None

    link_ubs = []

    for option in combo:
        link_ub = option.get("link_ub")

        if link_ub is None:
            return None

        link_ubs.append(float(link_ub))

    ub_total = sum(link_ubs)

    return (
        100.0
        * 10.0 ** (
            (
                float(utility)
                - float(ub_total)
            )
            / float(num_links)
        )
    )


def _realized_density(
    num_nodes: int,
    num_edges: int,
) -> float:
    if num_nodes <= 1:
        return 0.0

    possible = (
        num_nodes
        * (num_nodes - 1)
        / 2.0
    )

    return (
        float(num_edges)
        / possible
    )



def _evaluate_until_first_feasible(
    combo_stream,
    network,
    sources,
    cfg,
):
    """Evaluate routing combinations in order and stop at first feasible."""
    results = []
    t_start = time.perf_counter()

    for combo_idx, combo in enumerate(combo_stream):
        if (
            cfg.max_combos is not None
            and cfg.max_combos > 0
            and len(results) >= cfg.max_combos
        ):
            break

        result = evaluate_combo(
            combo,
            network,
            sources,
            cfg,
        )
        result["combo_idx"] = combo_idx
        result["elapsed_s"] = time.perf_counter() - t_start
        results.append(result)

        if result.get("valid"):
            return results, result

    return results, None


def run_one_case(
    shape: BenchmarkShape,
    max_combos: int,
    min_loss_db: float,
    max_loss_db: float,
    min_dark_count: float,
    max_dark_count: float,
    min_fidelity: float,
    max_fidelity: float,
    use_upper_bound: bool,
    use_best_first: bool,
) -> dict:
    """
    Run build -> Double Yen -> combo generation -> APOPT -> CP-SAT -> select
    best, while retaining the detailed scalability metrics needed for plotting.
    """
    routing_complexity = _routing_complexity(
        shape.num_sources,
        shape.n_paths,
        shape.num_links,
    )

    efa_complexity = _efa_complexity(
        shape.channels_per_source,
        shape.num_links,
    )

    combined_complexity = (
        routing_complexity
        * efa_complexity
    )

    case_id = (
        f"{shape.topology}_"
        f"U{shape.num_users}_"
        f"S{shape.num_sources}_"
        f"L{shape.num_links}_"
        f"K{shape.channels_per_source}_"
        f"N{shape.n_paths}_"
        f"D{('NA' if shape.density is None else shape.density)}_"
        f"R{shape.repetition}"
    )

    cfg = _make_config(
        shape=shape,
        max_combos=max_combos,
        min_loss_db=min_loss_db,
        max_loss_db=max_loss_db,
        min_dark_count=min_dark_count,
        max_dark_count=max_dark_count,
        min_fidelity=min_fidelity,
        max_fidelity=max_fidelity,
        use_upper_bound=use_upper_bound,
        use_best_first=use_best_first,
    )

    pipeline_start = time.perf_counter()

    try:
        # ------------------------------------------------------------------
        # Network
        # ------------------------------------------------------------------
        t0 = time.perf_counter()

        network, sources, links = build_network(
            cfg
        )

        t1 = time.perf_counter()

        num_nodes = int(
            network.number_of_nodes()
        )

        num_edges = int(
            network.number_of_edges()
        )

        realized_density = _realized_density(
            num_nodes,
            num_edges,
        )

        # ------------------------------------------------------------------
        # Double Yen candidate paths
        # ------------------------------------------------------------------
        paths = build_path_options(
            network,
            links,
            sources,
            cfg,
        )

        t2 = time.perf_counter()

        path_candidate_counts = [
            len(options)
            for options in paths
        ]

        if any(
            count == 0
            for count in path_candidate_counts
        ):
            total_time = (
                time.perf_counter()
                - pipeline_start
            )

            return {
                "case_id": case_id,
                "status": "NO_ROUTING_CANDIDATE",
                "topology": shape.topology,
                "requested_density": shape.density,
                "realized_density": realized_density,
                "users": shape.num_users,
                "sources": shape.num_sources,
                "links": shape.num_links,
                "channels_per_source": shape.channels_per_source,
                "n_paths": shape.n_paths,
                "repetition": shape.repetition,
                "network_seed": shape.network_seed,
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "routing_complexity": routing_complexity,
                "efa_complexity": efa_complexity,
                "combined_complexity": combined_complexity,
                "actual_route_combos": 0,
                "actual_combined_complexity": 0,
                "path_candidate_counts": str(path_candidate_counts),
                "combos_returned": 0,
                "valid_combos": 0,
                "allocation_failures": 0,
                "contention_failures": 0,
                "ub_pruned": 0,
                "rejected_combos": 0,
                "evaluated_combos": 0,
                "attempts_to_first_feasible": None,
                "time_to_first_feasible_s": None,
                "best_combo_idx": None,
                "time_to_best_s": None,
                "best_utility": None,
                "upper_bound_utility": None,
                "rate_pct_upper_bound": None,
                "build_time_s": t1 - t0,
                "routing_time_s": t2 - t1,
                "evaluation_time_s": 0.0,
                "total_pipeline_time_s": total_time,
                "error": None,
            }

        actual_route_combos = _product(
            path_candidate_counts
        )

        actual_combined_complexity = (
            actual_route_combos
            * efa_complexity
        )

        # ------------------------------------------------------------------
        # Production-style combo evaluation:
        #   route -> Phase 2 allocation -> CP-SAT scheduling.
        # Stop immediately at the first feasible end-to-end RSA solution.
        # ------------------------------------------------------------------
        combo_stream = generate_combos(
            paths,
            cfg,
        )

        evaluation_start = time.perf_counter()

        results, solution = _evaluate_until_first_feasible(
            combo_stream,
            network,
            sources,
            cfg,
        )

        evaluation_end = time.perf_counter()

        total_pipeline_time = (
            time.perf_counter()
            - pipeline_start
        )

        # ------------------------------------------------------------------
        # Result accounting
        # ------------------------------------------------------------------
        evaluated_combos = len(results)

        valid_results = [
            result for result in results
            if result.get("valid")
        ]

        allocation_failures = sum(
            1 for result in results
            if result.get("reason") == "allocation_failed"
        )

        contention_failures = sum(
            1 for result in results
            if result.get("reason") == "contention_failed"
        )

        ub_pruned = sum(
            1 for result in results
            if result.get("reason") == "ub_pruned"
        )

        rejected_combos = sum(
            1 for result in results
            if not result.get("valid")
        )

        if solution is not None:
            attempts_to_first = evaluated_combos
            time_to_first = solution.get("elapsed_s")
            solution_combo_idx = solution.get("combo_idx")
            solution_utility = solution.get("utility")
            rate_pct = _geometric_mean_rate_pct_upper_bound(
                solution,
                shape.num_links,
            )
            status = "SOLVED"
        else:
            attempts_to_first = None
            time_to_first = None
            solution_combo_idx = None
            solution_utility = None
            rate_pct = None

            if actual_route_combos <= evaluated_combos:
                status = "NO_FEASIBLE_IN_CANDIDATE_SET"
            else:
                status = "NO_FEASIBLE_WITHIN_CAP"

        upper_bound_utility = None

        if solution is not None:
            combo = solution.get("combo", [])

            if (
                len(combo) == shape.num_links
                and all(
                    option.get("link_ub") is not None
                    for option in combo
                )
            ):
                upper_bound_utility = sum(
                    float(option["link_ub"])
                    for option in combo
                )

        return {
            "case_id": case_id,
            "status": status,
            "topology": shape.topology,
            "requested_density": shape.density,
            "realized_density": realized_density,
            "users": shape.num_users,
            "sources": shape.num_sources,
            "links": shape.num_links,
            "channels_per_source": shape.channels_per_source,
            "n_paths": shape.n_paths,
            "repetition": shape.repetition,
            "network_seed": shape.network_seed,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "routing_complexity": routing_complexity,
            "efa_complexity": efa_complexity,
            "combined_complexity": combined_complexity,
            "actual_route_combos": actual_route_combos,
            "actual_combined_complexity": (
                actual_combined_complexity
            ),
            "path_candidate_counts": str(
                path_candidate_counts
            ),
            "combos_returned": len(
                results
            ),
            "valid_combos": len(
                valid_results
            ),
            "allocation_failures": (
                allocation_failures
            ),
            "contention_failures": (
                contention_failures
            ),
            "ub_pruned": ub_pruned,
            "rejected_combos": rejected_combos,
            "evaluated_combos": evaluated_combos,
            "attempts_to_first_feasible": (
                attempts_to_first
            ),
            "time_to_first_feasible_s": (
                time_to_first
            ),
            # Backward-compatible field names; these now describe the
            # returned first-feasible production solution.
            "best_combo_idx": (
                None
                if solution_combo_idx is None
                else int(solution_combo_idx)
            ),
            "time_to_best_s": (
                None
                if time_to_first is None
                else float(time_to_first)
            ),
            "best_utility": (
                None
                if solution_utility is None
                else float(solution_utility)
            ),
            "upper_bound_utility": (
                upper_bound_utility
            ),
            "rate_pct_upper_bound": (
                rate_pct
            ),
            "build_time_s": (
                t1 - t0
            ),
            "routing_time_s": (
                t2 - t1
            ),
            "evaluation_time_s": (
                evaluation_end
                - evaluation_start
            ),
            "total_pipeline_time_s": (
                total_pipeline_time
            ),
            "error": None,
        }

    except Exception as exc:
        total_pipeline_time = (
            time.perf_counter()
            - pipeline_start
        )

        return {
            "case_id": case_id,
            "status": "ERROR",
            "topology": shape.topology,
            "requested_density": shape.density,
            "realized_density": None,
            "users": shape.num_users,
            "sources": shape.num_sources,
            "links": shape.num_links,
            "channels_per_source": shape.channels_per_source,
            "n_paths": shape.n_paths,
            "repetition": shape.repetition,
            "network_seed": shape.network_seed,
            "num_nodes": None,
            "num_edges": None,
            "routing_complexity": routing_complexity,
            "efa_complexity": efa_complexity,
            "combined_complexity": combined_complexity,
            "actual_route_combos": None,
            "actual_combined_complexity": None,
            "path_candidate_counts": None,
            "combos_returned": None,
            "valid_combos": None,
            "allocation_failures": None,
            "contention_failures": None,
            "ub_pruned": None,
            "rejected_combos": None,
            "evaluated_combos": None,
            "attempts_to_first_feasible": None,
            "time_to_first_feasible_s": None,
            "best_combo_idx": None,
            "time_to_best_s": None,
            "best_utility": None,
            "upper_bound_utility": None,
            "rate_pct_upper_bound": None,
            "build_time_s": None,
            "routing_time_s": None,
            "evaluation_time_s": None,
            "total_pipeline_time_s": total_pipeline_time,
            "error": str(exc),
        }


def run_benchmark(
    topologies: list[str],
    densities: list[float],
    user_values: list[int],
    source_values: list[int],
    link_values: list[int],
    channel_values: list[int],
    n_path_values: list[int],
    per_shape: int,
    seed: int,
    max_combos: int,
    min_loss_db: float,
    max_loss_db: float,
    min_dark_count: float,
    max_dark_count: float,
    min_fidelity: float,
    max_fidelity: float,
    use_upper_bound: bool,
    use_best_first: bool,
) -> list[dict]:

    shapes = list(
        _iter_shapes(
            topologies=topologies,
            densities=densities,
            user_values=user_values,
            source_values=source_values,
            link_values=link_values,
            channel_values=channel_values,
            n_path_values=n_path_values,
            per_shape=per_shape,
            seed=seed,
        )
    )

    records: list[dict] = []

    print(
        f"Running {len(shapes)} "
        f"full-pipeline benchmark cases.\n"
    )

    for index, shape in enumerate(
        shapes,
        start=1,
    ):
        density_text = (
            "-"
            if shape.density is None
            else f"{shape.density:.3f}"
        )

        print(
            f"[{index}/{len(shapes)}] "
            f"{shape.topology}: "
            f"U={shape.num_users}, "
            f"S={shape.num_sources}, "
            f"L={shape.num_links}, "
            f"K={shape.channels_per_source}, "
            f"N={shape.n_paths}, "
            f"density={density_text}, "
            f"rep={shape.repetition}"
        )

        record = run_one_case(
            shape=shape,
            max_combos=max_combos,
            min_loss_db=min_loss_db,
            max_loss_db=max_loss_db,
            min_dark_count=min_dark_count,
            max_dark_count=max_dark_count,
            min_fidelity=min_fidelity,
            max_fidelity=max_fidelity,
            use_upper_bound=use_upper_bound,
            use_best_first=use_best_first,
        )

        records.append(
            record
        )

        rate_text = (
            "NA"
            if record.get(
                "rate_pct_upper_bound"
            ) is None
            else (
                f"{record['rate_pct_upper_bound']:.3f}%"
            )
        )

        attempts_text = (
            "NA"
            if record.get(
                "attempts_to_first_feasible"
            ) is None
            else str(
                record[
                    "attempts_to_first_feasible"
                ]
            )
        )

        print(
            f"    status={record['status']}, "
            f"time={record['total_pipeline_time_s']:.3f}s, "
            f"rate={rate_text}, "
            f"first_feasible_attempts={attempts_text}"
        )

    return records


# ============================================================================
# CSV save / load
# ============================================================================

CSV_FIELDS = [
    "case_id",
    "status",
    "topology",
    "requested_density",
    "realized_density",
    "users",
    "sources",
    "links",
    "channels_per_source",
    "n_paths",
    "repetition",
    "network_seed",
    "num_nodes",
    "num_edges",
    "routing_complexity",
    "efa_complexity",
    "combined_complexity",
    "actual_route_combos",
    "actual_combined_complexity",
    "path_candidate_counts",
    "combos_returned",
    "valid_combos",
    "allocation_failures",
    "contention_failures",
    "ub_pruned",
    "rejected_combos",
    "evaluated_combos",
    "attempts_to_first_feasible",
    "time_to_first_feasible_s",
    "best_combo_idx",
    "time_to_best_s",
    "best_utility",
    "upper_bound_utility",
    "rate_pct_upper_bound",
    "build_time_s",
    "routing_time_s",
    "evaluation_time_s",
    "total_pipeline_time_s",
    "error",
]


def save_records_csv(
    records: list[dict],
    outpath: Path,
) -> None:

    outpath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with outpath.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    key: record.get(key)
                    for key in CSV_FIELDS
                }
            )

    print(
        f"\nSaved benchmark CSV to {outpath}"
    )


INT_FIELDS = {
    "users",
    "sources",
    "links",
    "channels_per_source",
    "n_paths",
    "repetition",
    "network_seed",
    "num_nodes",
    "num_edges",
    "routing_complexity",
    "efa_complexity",
    "combined_complexity",
    "actual_route_combos",
    "actual_combined_complexity",
    "combos_returned",
    "valid_combos",
    "allocation_failures",
    "contention_failures",
    "ub_pruned",
    "rejected_combos",
    "evaluated_combos",
    "attempts_to_first_feasible",
    "best_combo_idx",
}

FLOAT_FIELDS = {
    "requested_density",
    "realized_density",
    "time_to_first_feasible_s",
    "time_to_best_s",
    "best_utility",
    "upper_bound_utility",
    "rate_pct_upper_bound",
    "build_time_s",
    "routing_time_s",
    "evaluation_time_s",
    "total_pipeline_time_s",
}


def _parse_optional_int(
    value: str | None,
) -> int | None:
    if value is None:
        return None

    text = value.strip()

    if text == "":
        return None

    return int(text)


def _parse_optional_float(
    value: str | None,
) -> float | None:
    if value is None:
        return None

    text = value.strip()

    if text == "":
        return None

    return float(text)


def load_records_csv(
    csv_path: Path,
) -> list[dict]:

    records: list[dict] = []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(
            handle
        )

        for row in reader:
            record = dict(
                row
            )

            for key in INT_FIELDS:
                record[key] = (
                    _parse_optional_int(
                        row.get(key)
                    )
                )

            for key in FLOAT_FIELDS:
                record[key] = (
                    _parse_optional_float(
                        row.get(key)
                    )
                )

            # Backfill the combined metrics if a CSV is edited / produced by
            # an earlier version that still has the component quantities.
            if (
                record.get(
                    "combined_complexity"
                ) is None
                and record.get(
                    "routing_complexity"
                ) is not None
                and record.get(
                    "efa_complexity"
                ) is not None
            ):
                record[
                    "combined_complexity"
                ] = (
                    int(
                        record[
                            "routing_complexity"
                        ]
                    )
                    * int(
                        record[
                            "efa_complexity"
                        ]
                    )
                )

            if (
                record.get(
                    "actual_combined_complexity"
                ) is None
                and record.get(
                    "actual_route_combos"
                ) is not None
                and record.get(
                    "efa_complexity"
                ) is not None
            ):
                record[
                    "actual_combined_complexity"
                ] = (
                    int(
                        record[
                            "actual_route_combos"
                        ]
                    )
                    * int(
                        record[
                            "efa_complexity"
                        ]
                    )
                )

            records.append(
                record
            )

    return records


# ============================================================================
# Plotting
# ============================================================================

def _x_key_and_label(
    x_metric: str,
) -> tuple[str, str]:

    if x_metric == "combined":
        return (
            "combined_complexity",
            r"Nominal pipeline complexity, "
            r"$(SN^2)^L \binom{K}{L}$",
        )

    if x_metric == "routing":
        return (
            "routing_complexity",
            r"Nominal routing combinations, "
            r"$(SN^2)^L$",
        )

    if x_metric == "actual":
        return (
            "actual_combined_complexity",
            r"Actual candidate complexity, "
            r"$N_{\mathrm{route}} \binom{K}{L}$",
        )

    raise ValueError(
        f"Unknown x metric: {x_metric}"
    )


def _paper_axis_format(
    ax,
) -> None:

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_SIZE,
        width=1.2,
        length=5,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        width=1.0,
        length=3,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(
            1.2
        )

    ax.grid(
        True,
        which="major",
        linewidth=0.8,
        alpha=GRID_ALPHA,
    )


def _safe_save_figure(
    fig,
    outdir: Path,
    stem: str,
) -> None:

    # Resolve the output directory to an absolute path before saving.
    # This avoids Windows/Pillow issues with some relative Path objects.
    outdir = Path(outdir).expanduser().resolve()

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for suffix in (
        ".png",
        ".pdf",
    ):
        path = (
            outdir
            / f"{stem}{suffix}"
        )

        print(f"Saving figure to: {path}")

        fig.savefig(
            str(path),
            dpi=PAPER_DPI,
            bbox_inches="tight",
        )


def _valid_records(
    records: list[dict],
    x_key: str,
    y_key: str,
    positive_y: bool = False,
) -> list[dict]:

    output = []

    for record in records:

        x = record.get(
            x_key
        )

        y = record.get(
            y_key
        )

        if x is None or y is None:
            continue

        try:
            x_value = float(
                x
            )

            y_value = float(
                y
            )
        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            continue

        if (
            not math.isfinite(x_value)
            or not math.isfinite(y_value)
            or x_value <= 0.0
        ):
            continue

        if positive_y and y_value <= 0.0:
            continue

        output.append(
            record
        )

    return output


def _attempts_for_plot(
    record: dict,
) -> float | None:
    """Actual route attempts made before success or termination."""
    if record.get("status") == "SOLVED":
        value = record.get("attempts_to_first_feasible")
    else:
        value = record.get("evaluated_combos")

    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(value) or value <= 0.0:
        return None

    return value


def _group_medians(
    records: list[dict],
    x_key: str,
    y_key: str,
    topology: str,
) -> tuple[list[float], list[float]]:

    grouped: dict[
        int,
        list[float],
    ] = {}

    for record in records:

        if (
            record.get("topology")
            != topology
        ):
            continue

        x = record.get(
            x_key
        )

        y = record.get(
            y_key
        )

        if x is None or y is None:
            continue

        grouped.setdefault(
            int(x),
            [],
        ).append(
            float(y)
        )

    xs = sorted(
        grouped
    )

    ys = [
        float(
            statistics.median(
                grouped[x]
            )
        )
        for x in xs
    ]

    return (
        [float(x) for x in xs],
        ys,
    )


def _scatter_and_median_by_topology(
    ax,
    records: list[dict],
    x_key: str,
    y_key: str,
) -> None:

    for topology in (
        "dense",
        "ring",
        "star",
    ):

        style = TOPOLOGY_STYLE[
            topology
        ]

        subset = [
            record
            for record in records
            if record.get("topology")
            == topology
        ]

        if not subset:
            continue

        ax.scatter(
            [
                float(
                    record[x_key]
                )
                for record in subset
            ],
            [
                float(
                    record[y_key]
                )
                for record in subset
            ],
            s=RAW_MARKER_SIZE,
            alpha=RAW_ALPHA,
            color=style["color"],
            marker=style["marker"],
            edgecolors="none",
            zorder=2,
        )

        xs, ys = _group_medians(
            subset,
            x_key=x_key,
            y_key=y_key,
            topology=topology,
        )

        if xs:
            ax.plot(
                xs,
                ys,
                color=style["color"],
                marker=style["marker"],
                markersize=MEDIAN_MARKER_SIZE,
                linewidth=MEDIAN_LINE_WIDTH,
                label=style["label"],
                zorder=3,
            )


def plot_runtime_panel(
    records: list[dict],
    outdir: Path,
    x_metric: str,
) -> None:

    x_key, x_label = (
        _x_key_and_label(
            x_metric
        )
    )

    valid = _valid_records(
        records,
        x_key=x_key,
        y_key="total_pipeline_time_s",
        positive_y=True,
    )

    if not valid:
        print(
            "No valid records for runtime plot."
        )
        return

    fig, ax = plt.subplots(
        figsize=PAPER_FIGSIZE
    )

    _scatter_and_median_by_topology(
        ax,
        valid,
        x_key=x_key,
        y_key="total_pipeline_time_s",
    )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        x_label
    )

    ax.set_ylabel(
        "Total pipeline runtime (s)"
    )

    _paper_axis_format(
        ax
    )

    ax.legend(
        loc="best",
        frameon=False,
    )

    fig.tight_layout()

    _safe_save_figure(
        fig,
        outdir,
        "full_pipeline_runtime_vs_complexity",
    )

    plt.close(
        fig
    )


def plot_rate_panel(
    records: list[dict],
    outdir: Path,
    x_metric: str,
) -> None:

    x_key, x_label = (
        _x_key_and_label(
            x_metric
        )
    )

    valid = _valid_records(
        records,
        x_key=x_key,
        y_key="rate_pct_upper_bound",
        positive_y=True,
    )

    if not valid:
        print(
            "No valid records for rate plot."
        )
        return

    fig, ax = plt.subplots(
        figsize=PAPER_FIGSIZE
    )

    _scatter_and_median_by_topology(
        ax,
        valid,
        x_key=x_key,
        y_key="rate_pct_upper_bound",
    )

    ax.axhline(
        100.0,
        linestyle="--",
        linewidth=1.8,
        color="#555555",
        zorder=1,
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        x_label
    )

    ax.set_ylabel(
        "Geometric-mean link rate\n"
        "(% of infinite-resource upper bound)"
    )

    rate_values = [
        float(record["rate_pct_upper_bound"])
        for record in valid
    ]

    minimum_rate = min(rate_values)
    maximum_rate = max(rate_values)

    lower = max(
        0.0,
        math.floor(
            (minimum_rate - 2.0) / 5.0
        ) * 5.0,
    )

    if lower >= 95.0:
        lower = 95.0

    upper = max(
        101.0,
        math.ceil(
            (maximum_rate + 1.0) / 5.0
        ) * 5.0,
    )

    ax.set_ylim(
        lower,
        upper,
    )

    _paper_axis_format(
        ax
    )

    ax.legend(
        loc="lower left",
        frameon=False,
    )

    fig.tight_layout()

    _safe_save_figure(
        fig,
        outdir,
        "full_pipeline_rate_vs_complexity",
    )

    plt.close(
        fig
    )


def plot_attempts_panel(
    records: list[dict],
    outdir: Path,
    x_metric: str,
) -> None:

    x_key, x_label = (
        _x_key_and_label(
            x_metric
        )
    )

    valid = []
    for record in records:
        x = record.get(
            x_key
        )
        y_value = _attempts_for_plot(
            record
        )
        if x is None or y_value is None:
            continue
        try:
            x_value = float(
                x
            )
        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            continue
        if (
            not math.isfinite(x_value)
            or x_value <= 0.0
        ):
            continue

        updated = dict(record)
        updated["attempts_plot_value"] = y_value
        valid.append(updated)

    if not valid:
        print(
            "No valid records for attempts plot."
        )
        return

    fig, ax = plt.subplots(
        figsize=PAPER_FIGSIZE
    )

    _scatter_and_median_by_topology(
        ax,
        valid,
        x_key=x_key,
        y_key="attempts_plot_value",
    )

    ax.set_xscale(
        "log"
    )

    # symlog preserves zero-failure cases while still accommodating
    # configurations spanning orders of magnitude.
    ax.set_yscale(
        "symlog",
        linthresh=1.0,
    )

    ax.set_xlabel(
        x_label
    )

    ax.set_ylabel(
        "Route configurations attempted\n"
        "(to first success or search limit)"
    )

    _paper_axis_format(
        ax
    )

    ax.legend(
        loc="best",
        frameon=False,
    )

    fig.tight_layout()

    _safe_save_figure(
        fig,
        outdir,
        "full_pipeline_attempts_vs_complexity",
    )

    plt.close(
        fig
    )


def plot_paper_results(
    records: list[dict],
    outdir: Path,
    x_metric: str,
) -> None:

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_runtime_panel(
        records,
        outdir,
        x_metric,
    )

    plot_rate_panel(
        records,
        outdir,
        x_metric,
    )

    plot_attempts_panel(
        records,
        outdir,
        x_metric,
    )

    print(
        f"Saved paper figures to {outdir}"
    )


# ============================================================================
# Console summary
# ============================================================================

def _percentile(
    values: list[float],
    q: float,
) -> float:

    ordered = sorted(
        float(value)
        for value in values
    )

    if not ordered:
        raise ValueError(
            "Cannot compute percentile of empty sequence."
        )

    if len(ordered) == 1:
        return ordered[0]

    position = (
        len(ordered) - 1
    ) * q

    lo = math.floor(
        position
    )

    hi = math.ceil(
        position
    )

    if lo == hi:
        return ordered[lo]

    fraction = (
        position - lo
    )

    return (
        ordered[lo]
        * (1.0 - fraction)
        + ordered[hi]
        * fraction
    )


def print_summary(
    records: list[dict],
) -> None:

    print(
        "\n=== Full-pipeline scalability summary ==="
    )

    print(
        f"Total cases: {len(records)}"
    )

    status_counts: dict[
        str,
        int,
    ] = {}

    for record in records:
        status = str(
            record.get(
                "status",
                "UNKNOWN",
            )
        )

        status_counts[
            status
        ] = (
            status_counts.get(
                status,
                0,
            )
            + 1
        )

    for status in sorted(
        status_counts
    ):
        print(
            f"  {status}: "
            f"{status_counts[status]}"
        )

    solved = [
        record
        for record in records
        if record.get(
            "status"
        ) == "SOLVED"
    ]

    if not solved:
        return

    runtime = [
        float(
            record[
                "total_pipeline_time_s"
            ]
        )
        for record in solved
        if record.get(
            "total_pipeline_time_s"
        ) is not None
    ]

    rates = [
        float(
            record[
                "rate_pct_upper_bound"
            ]
        )
        for record in solved
        if record.get(
            "rate_pct_upper_bound"
        ) is not None
    ]

    attempts = [
        float(
            record[
                "attempts_to_first_feasible"
            ]
        )
        for record in solved
        if record.get(
            "attempts_to_first_feasible"
        ) is not None
    ]

    if runtime:
        print(
            f"Median runtime: "
            f"{statistics.median(runtime):.6f} s"
        )

        print(
            f"95th percentile runtime: "
            f"{_percentile(runtime, 0.95):.6f} s"
        )

        print(
            f"Maximum runtime: "
            f"{max(runtime):.6f} s"
        )

    if rates:
        print(
            f"Median average-link-rate percentage: "
            f"{statistics.median(rates):.6f}% "
            f"of upper bound"
        )

        print(
            f"5th percentile average-link-rate percentage: "
            f"{_percentile(rates, 0.05):.6f}%"
        )

        print(
            f"Worst average-link-rate percentage: "
            f"{min(rates):.6f}%"
        )

    if attempts:
        print(
            f"Median attempts to first feasible: "
            f"{statistics.median(attempts):.1f}"
        )

        print(
            f"95th percentile attempts: "
            f"{_percentile(attempts, 0.95):.1f}"
        )

        print(
            f"Maximum attempts: "
            f"{max(attempts):.0f}"
        )


# ============================================================================
# CLI
# ============================================================================

def _normalize_path_string(
    text: str,
) -> str:

    cleaned = text.strip()

    if (
        len(cleaned) >= 2
        and cleaned[0] == cleaned[-1]
        and cleaned[0] in (
            "'",
            '"',
        )
    ):
        cleaned = cleaned[
            1:-1
        ].strip()

    return cleaned


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run full QuantumNetwork pipeline scalability / "
            "topology experiments and create the three-paper-panel figure."
        )
    )

    parser.add_argument(
        "--topologies",
        nargs="+",
        choices=[
            "dense",
            "ring",
            #"star",
        ],
        default=[
            "dense",
            "ring",
            #"star",
        ],
    )

    parser.add_argument(
        "--densities",
        type=float,
        nargs="+",
        default=[
            0.15,
            0.35,
            0.60,
        ],
        help=(
            "Dense-topology density values. "
            "Ignored for ring/star."
        ),
    )

    parser.add_argument(
        "--users",
        type=int,
        nargs="+",
        default=[
            5,
            10,
            20,
            100,
        ],
    )

    parser.add_argument(
        "--sources",
        type=int,
        nargs="+",
        default=[
            1,2,3,5,10,20
        ],
    )

    parser.add_argument(
        "--links",
        type=int,
        nargs="+",
        default=[
            1,2,3,5,10,20
        ],
    )

    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[
            5,10,20,100
        ],
        help=(
            "Available positive channel indices per source."
        ),
    )

    parser.add_argument(
        "--n-paths",
        type=int,
        nargs="+",
        default=[
            1,
            2,
            3,
        ],
        help=(
            "Double-Yen paths retained per source-user leg."
        ),
    )

    parser.add_argument(
        "--per-shape",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=41,
    )

    parser.add_argument(
        "--max-combos",
        type=int,
        default=10000,
        help=(
            "Maximum routing combinations attempted per network. "
            "The run stops earlier when the first feasible RSA solution is found."
        ),
    )

    parser.add_argument(
        "--loss-min-db",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--loss-max-db",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--dark-count-min",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--dark-count-max",
        type=float,
        default=1000.0,
    )

    parser.add_argument(
        "--fidelity-min",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--fidelity-max",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--no-upper-bound",
        action="store_true",
        help=(
            "Disable utility upper-bound pruning during combo evaluation."
        ),
    )

    parser.add_argument(
        "--brute-force-order",
        action="store_true",
        help=(
            "Use Cartesian-product combo order instead of the "
            "production best-first routing-combination order."
        ),
    )

    parser.add_argument(
        "--x-metric",
        choices=[
            "combined",
            # "routing",
            #"actual",
        ],
        default="combined",
        help=(
            "X-axis metric for all three plots. "
            "'combined' uses (S*N^2)^L*C(K,L); "
            "'routing' uses (S*N^2)^L; "
            "'actual' uses actual candidate combos*C(K,L)."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "outputs/"
            "full_pipeline_scalability"
        ),
    )

    parser.add_argument(
        "--records-csv",
        type=str,
        default=None,
        help=(
            "Existing full_pipeline_results.csv. "
            "When supplied, skip simulation and only regenerate plots."
        ),
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # Basic validation
    # ----------------------------------------------------------------------
    if args.per_shape <= 0:
        raise ValueError(
            "--per-shape must be positive."
        )

    if args.max_combos <= 0:
        raise ValueError(
            "--max-combos must be positive."
        )

    if any(
        value <= 0
        for value in args.users
    ):
        raise ValueError(
            "All --users values must be positive."
        )

    if any(
        value <= 0
        for value in args.sources
    ):
        raise ValueError(
            "All --sources values must be positive."
        )

    if any(
        value <= 0
        for value in args.links
    ):
        raise ValueError(
            "All --links values must be positive."
        )

    if any(
        value <= 0
        for value in args.channels
    ):
        raise ValueError(
            "All --channels values must be positive."
        )

    if any(
        value <= 0
        for value in args.n_paths
    ):
        raise ValueError(
            "All --n-paths values must be positive."
        )

    if any(
        not 0.0 <= value <= 1.0
        for value in args.densities
    ):
        raise ValueError(
            "All --densities values must lie in [0,1]."
        )

    output_text = _normalize_path_string(
        args.output_dir
    )

    if not output_text:
        raise ValueError(
            "--output-dir cannot be empty."
        )

    outdir = Path(
        output_text
    ).expanduser().resolve()

    # ----------------------------------------------------------------------
    # CSV-only plotting
    # ----------------------------------------------------------------------
    if args.records_csv is not None:

        csv_text = _normalize_path_string(
            args.records_csv
        )

        csv_path = Path(
            csv_text
        ).expanduser().resolve()

        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {csv_path}"
            )

        records = load_records_csv(
            csv_path
        )

        print_summary(
            records
        )

        plot_paper_results(
            records,
            outdir=outdir,
            x_metric=args.x_metric,
        )

        print(
            f"Regenerated plots from {csv_path}"
        )

        return

    # ----------------------------------------------------------------------
    # Simulation
    # ----------------------------------------------------------------------
    records = run_benchmark(
        topologies=args.topologies,
        densities=args.densities,
        user_values=args.users,
        source_values=args.sources,
        link_values=args.links,
        channel_values=args.channels,
        n_path_values=args.n_paths,
        per_shape=args.per_shape,
        seed=args.seed,
        max_combos=args.max_combos,
        min_loss_db=args.loss_min_db,
        max_loss_db=args.loss_max_db,
        min_dark_count=args.dark_count_min,
        max_dark_count=args.dark_count_max,
        min_fidelity=args.fidelity_min,
        max_fidelity=args.fidelity_max,
        use_upper_bound=(
            not args.no_upper_bound
        ),
        use_best_first=(
            not args.brute_force_order
        ),
    )

    csv_path = (
        outdir
        / "full_pipeline_results.csv"
    )

    save_records_csv(
        records,
        csv_path,
    )

    print_summary(
        records
    )

    plot_paper_results(
        records,
        outdir=outdir,
        x_metric=args.x_metric,
    )


if __name__ == "__main__":
    main()
