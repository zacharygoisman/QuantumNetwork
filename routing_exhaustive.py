"""
Publication benchmark for Double Yen routing versus exhaustive routing.

Place this file in the TOP LEVEL of the QuantumNetwork repository, next to
main.py.

Purpose
-------
This benchmark isolates the routing stage as much as possible while using the
same downstream EFA allocator for both methods:

1. Build a random connected quantum-network instance.
2. Double Yen:
   - For every entangled user pair and every source, generate the N
     lowest-loss simple paths from the source to each user using the repository
     routing code.
   - Pair the two legs, giving up to S*N^2 candidates per entangled link.
   - Exhaustively evaluate the cross-link combinations formed from that
     Double-Yen candidate set.
3. Exact routing reference:
   - Enumerate ALL simple source-to-user paths for every source/user leg.
   - Form every source/path-pair candidate for each entangled link.
   - Exhaustively evaluate the resulting cross-link routing combinations.
4. Score every routing combination with the SAME production EFA allocator.
5. Require the SAME CP-SAT spectrum-scheduling feasibility check for both methods.
6. Compare:
      arithmetic mean of the individual link-rate ratios for Double Yen
      relative to exhaustive routing, matching the main pipeline convention;
   and
      wall-clock runtime for Double Yen versus exhaustive routing.

Important interpretation
------------------------
APOPT does NOT itself choose the source.  A routing candidate already specifies
its source.  Source choice is therefore part of the routing-combination search.
APOPT is used only to allocate EFA resources after a routing/source combination
has been chosen.

The exhaustive routing result enumerates all simple paths and all cross-link
routing combinations subject to the same downstream EFA and CP-SAT feasibility
checks used for Double Yen.  Figure 1 separately validates the EFA solver; this
figure tests the loss caused by restricting routing to Double Yen's N-shortest
path candidate set under actual spectrum-contention constraints.

Paper figure
------------
The script produces two panels with the same style and size as the EFA figure:

  double_yen_rate_vs_complexity.{png,pdf}
      y = average individual-link rate relative to exhaustive routing (%)

  double_yen_runtime_vs_complexity.{png,pdf}
      y = Double Yen / exhaustive routing wall-clock time (s)

The common x-axis is the ACTUAL number of exhaustive routing combinations
that were evaluated for that network.

where
      S = number of sources,
      N = number of Yen paths retained per source-user leg,
      L = number of requested entangled user pairs.

The actual Double-Yen and exhaustive combination counts are also saved to CSV.
This matters because a particular graph can contain fewer than N simple paths
for some source-user legs, and the exhaustive all-simple-path space is generally
larger than the nominal Double-Yen space.

Safety limits
-------------
All-simple-path enumeration is combinatorial.  The script therefore skips an
instance rather than silently truncating it if either:
  - a single exact source-user leg exceeds --max-exact-paths-per-leg, or
  - the exact cross-link combination count exceeds --max-exact-combos.

A skipped instance is NOT treated as an exact result.

Example
-------
Run a small validation sweep:

    python double_yen_exhaustive_paper.py ^
        --users 6 ^
        --sources 1 2 3 ^
        --links 1 2 ^
        --n-paths 1 2 3 4 ^
        --per-shape 5

Replot later without rerunning optimization:

    python double_yen_exhaustive_paper.py ^
        --records-csv outputs/double_yen_exhaustive/double_yen_validation_results.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from allocation import allocator as allocator_module
from allocation.allocator import allocate_combo
from scheduling import scheduler as scheduler_module
from scheduling.scheduler import check_interference
from analysis.metrics import compute_ub_max
from config.config import Config
from network.builder import build_network
from routing.pairing import compute_path_loss, compute_y
from routing.paths import build_path_options


# ============================================================================
# Publication plot style
# ============================================================================

# Keep these identical to efa_exhaustive_paper.py so the two figures match.
PAPER_FIGSIZE = (6.5, 4.0)   # inches
PAPER_DPI = 600

FONT_SIZE = 16
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 16
LEGEND_SIZE = 15

MARKER_SIZE = 34
TREND_MARKER_SIZE = 6
TREND_LINE_WIDTH = 2.0
REFERENCE_LINE_WIDTH = 1.8

# Okabe-Ito-style colorblind-friendly choices.
COLOR_YEN = "#0072B2"         # blue
COLOR_EXACT = "#D55E00"       # vermillion
COLOR_REFERENCE = "#555555"   # dark gray

RAW_ALPHA = 0.32
GRID_ALPHA = 0.22

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
# Data classes
# ============================================================================

@dataclass(frozen=True)
class NetworkShape:
    num_users: int
    num_sources: int
    num_links: int
    repetition: int
    network_seed: int


@dataclass
class RoutingResult:
    status: str
    utility: float | None
    elapsed_s: float
    best_combo: tuple | None
    num_link_candidates: tuple[int, ...]
    num_combos: int
    path_generation_s: float
    combo_evaluation_s: float
    error: str | None = None
    best_link_utilities: tuple[float, ...] | None = None


class ExactSearchTooLarge(RuntimeError):
    pass


# ============================================================================
# Small helpers
# ============================================================================

def _product(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _geometric_mean_rate_retention_pct(
    heuristic_utility: float | None,
    exact_utility: float | None,
    num_links: int,
) -> float | None:
    """
    Geometric-mean link-rate retention relative to exhaustive routing.

        U = sum_l log10(R_l)

    Therefore

        100 * 10**((U_Yen - U_exact) / L)

    is the percentage of the exhaustive solution's geometric-mean link rate
    retained by Double Yen.  No clipping is applied.
    """
    if (
        heuristic_utility is None
        or exact_utility is None
        or num_links <= 0
    ):
        return None

    return (
        100.0
        * 10.0 ** (
            (
                float(heuristic_utility)
                - float(exact_utility)
            )
            / float(num_links)
        )
    )


def _clear_allocator_cache() -> None:
    """
    Prevent one routing method from benefiting from allocation or scheduling
    results cached while timing the other method.

    Caching remains enabled WITHIN each method, matching production behavior.
    """
    cache = getattr(allocator_module, "_ALLOC_CACHE", None)
    if isinstance(cache, dict):
        cache.clear()

    sched_cache = getattr(
        scheduler_module,
        "_SCHED_CACHE",
        None,
    )
    if isinstance(sched_cache, dict):
        sched_cache.clear()


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of an empty sequence.")

    vals = sorted(float(v) for v in values)

    if len(vals) == 1:
        return vals[0]

    position = (len(vals) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)

    if lo == hi:
        return vals[lo]

    fraction = position - lo
    return vals[lo] * (1.0 - fraction) + vals[hi] * fraction


def _safe_save_figure(fig, outpath: Path) -> None:
    """
    Save using an absolute native Windows path.  This avoids the
    Windows/OneDrive/Pillow Errno 22 issue seen with relative output paths.
    """
    target = Path(outpath).expanduser()

    if not target.is_absolute():
        target = Path.cwd() / target

    target = target.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.savefig(
            str(target),
            dpi=PAPER_DPI,
            bbox_inches="tight",
        )
    except OSError as exc:
        fallback_dir = Path.cwd() / "double_yen_plot_fallback"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback = (fallback_dir / target.name).resolve(strict=False)

        print(f"Warning: could not save figure to {target}: {exc}")
        print(f"Retrying in {fallback_dir.resolve(strict=False)}")

        fig.savefig(
            str(fallback),
            dpi=PAPER_DPI,
            bbox_inches="tight",
        )


# ============================================================================
# Test-network construction
# ============================================================================

def _make_case_config(
    shape: NetworkShape,
    n_paths: int,
    channels_per_source: int,
    density: float,
    loss_min_db: float,
    loss_max_db: float,
    min_fidelity: float,
    max_fidelity: float,
    min_dark_count: float,
    max_dark_count: float,
) -> Config:
    """
    Construct a Config compatible with the current QuantumNetwork repository.

    The network builder seeds its own random topology / edge-loss generation
    from cfg.random_seed.  Physics arrays are separately generated from the
    same case seed for reproducibility.
    """
    if shape.num_users < 2 * shape.num_links:
        raise ValueError(
            f"Need at least 2*L users for disjoint links: "
            f"U={shape.num_users}, L={shape.num_links}."
        )

    rng = random.Random(shape.network_seed + 1_000_003)

    dark_counts = [
        rng.uniform(min_dark_count, max_dark_count)
        for _ in range(shape.num_users)
    ]

    fidelity_limits = [
        rng.uniform(min_fidelity, max_fidelity)
        for _ in range(shape.num_links)
    ]

    # Avoid build_path_options truncating the S*N^2 candidate set.
    combo_limit_per_link = (
        shape.num_sources * n_paths * n_paths
    )

    return Config(
        num_usr=shape.num_users,
        num_src=shape.num_sources,
        num_lnks=shape.num_links,
        topology="dense",
        density=density,
        loss_range=(loss_min_db, loss_max_db),
        num_channels=[
            channels_per_source
            for _ in range(shape.num_sources)
        ],
        random_seed=shape.network_seed,
        require_disjoint_links=True,
        n_paths_per_leg=n_paths,
        combo_limit_per_link=combo_limit_per_link,
        use_best_first=False,
        upper_bound_sort=False,
        use_diverse_paths=False,
        fidelity_limit=fidelity_limits,
        tau=1e-9,
        dark_count_rate=dark_counts,
        use_upper_bound=False,
        max_combos=None,
        verbose=False,
        parallel=False,
    )


# ============================================================================
# Routing candidate generation
# ============================================================================

def _yen_link_options(
    network,
    links,
    sources,
    cfg,
) -> list[list[dict]]:
    """
    Use the repository's actual Double-Yen path-building implementation.

    build_path_options calls Yen / NetworkX shortest_simple_paths independently
    for both legs and every source, then pairs the retained leg paths.
    """
    return build_path_options(
        network,
        links,
        sources,
        cfg,
    )


def _all_simple_paths_checked(
    network,
    source: str,
    target: str,
    max_paths: int,
) -> list[list[str]]:
    """
    Enumerate ALL simple source-target paths.

    If the number exceeds max_paths, raise instead of truncating.  That keeps
    every retained "exact" result genuinely exhaustive.
    """
    paths: list[list[str]] = []

    try:
        generator = nx.all_simple_paths(
            network,
            source=source,
            target=target,
        )

        for path in generator:
            paths.append(list(path))

            if len(paths) > max_paths:
                raise ExactSearchTooLarge(
                    f"More than {max_paths} simple paths on leg "
                    f"{source}->{target}."
                )

    except nx.NetworkXNoPath:
        return []

    return paths


def _exact_link_options(
    network,
    links,
    sources,
    cfg,
    max_paths_per_leg: int,
) -> list[list[dict]]:
    """
    Build exact routing candidates using all simple paths on each source-user
    leg, while preserving the option structure expected by allocate_combo().
    """
    user_nodes = [
        str(node)
        for node, data in network.nodes(data=True)
        if data.get("node_type") == "user"
        or str(node).startswith("U")
    ]

    user_nodes = sorted(
        user_nodes,
        key=lambda text: int(
            "".join(ch for ch in text if ch.isdigit()) or 0
        ),
    )

    if len(user_nodes) != len(cfg.dark_count_rate):
        raise ValueError(
            "dark_count_rate must match the number of user nodes."
        )

    dark_count_map = {
        user: float(cfg.dark_count_rate[i])
        for i, user in enumerate(user_nodes)
    }

    all_link_options: list[list[dict]] = []

    for link_idx, (u1, u2) in enumerate(links):
        candidates: list[dict] = []

        f_req = float(
            max(0.5, cfg.fidelity_limit[link_idx])
        )

        d_u1 = dark_count_map[str(u1)]
        d_u2 = dark_count_map[str(u2)]

        for source in sources:
            paths_u1 = _all_simple_paths_checked(
                network,
                str(source),
                str(u1),
                max_paths=max_paths_per_leg,
            )

            paths_u2 = _all_simple_paths_checked(
                network,
                str(source),
                str(u2),
                max_paths=max_paths_per_leg,
            )

            for path1 in paths_u1:
                loss1 = compute_path_loss(
                    network,
                    path1,
                )

                y1 = compute_y(
                    loss1,
                    cfg.tau,
                    d_u1,
                )

                for path2 in paths_u2:
                    loss2 = compute_path_loss(
                        network,
                        path2,
                    )

                    y2 = compute_y(
                        loss2,
                        cfg.tau,
                        d_u2,
                    )

                    path_ub = compute_ub_max(
                        y1,
                        y2,
                        f_min=f_req,
                    )

                    candidates.append(
                        {
                            "link": (u1, u2),
                            "link_idx": link_idx,
                            "source": source,
                            "users": (u1, u2),
                            "path1": path1,
                            "path2": path2,
                            "y1": y1,
                            "y2": y2,
                            "dark_count_1": d_u1,
                            "dark_count_2": d_u2,
                            "fidelity_limit": f_req,
                            "total_loss": loss1 + loss2,
                            "path_ub": path_ub,
                        }
                    )

        if not candidates:
            all_link_options.append([])
            continue

        link_ub = max(
            candidate["path_ub"]
            for candidate in candidates
        )

        for candidate in candidates:
            candidate["link_ub"] = link_ub

        # Match the normal total-loss ordering used by build_path_options.
        candidates.sort(
            key=lambda candidate: candidate["total_loss"]
        )

        all_link_options.append(candidates)

    return all_link_options


# ============================================================================
# Combination scoring
# ============================================================================

def _best_apopt_routing_combo(
    all_link_options: list[list[dict]],
    network,
    sources,
    cfg,
    max_combos: int | None,
) -> tuple[
    str,
    float | None,
    tuple | None,
    int,
    str | None,
    tuple[float, ...] | None,
]:
    """
    Exhaustively evaluate all cross-link routing combinations in a supplied
    candidate set using the production EFA allocator.

    Both candidate sets are passed through the same production EFA allocator
    and CP-SAT scheduler.  This makes alternative paths matter when the
    shortest-path choice creates shared-edge spectral contention.

    In addition to the total utility, retain the selected solution's physical
    per-link log10 rates.  These are needed to reproduce the main pipeline's
    arithmetic-mean per-link percentage metric.
    """
    candidate_counts = [
        len(options)
        for options in all_link_options
    ]

    if any(count == 0 for count in candidate_counts):
        return (
            "no_routes",
            None,
            None,
            0,
            "At least one entangled link has no routing candidates.",
            None,
        )

    num_combos = _product(candidate_counts)

    if (
        max_combos is not None
        and num_combos > max_combos
    ):
        return (
            "too_large",
            None,
            None,
            num_combos,
            (
                f"Routing combination count {num_combos} exceeds "
                f"limit {max_combos}."
            ),
            None,
        )

    best_utility = float("-inf")
    best_combo = None
    best_link_utilities = None

    for combo in itertools.product(*all_link_options):
        alloc_result = allocate_combo(
            combo,
            network,
            sources,
            cfg,
        )

        if not alloc_result.get("success"):
            continue

        # Route alternatives are only meaningful when shared-edge spectrum
        # contention is included.  Apply the same production CP-SAT
        # scheduling feasibility test to both Double Yen and exhaustive
        # routing candidates.
        sched_result = check_interference(
            combo,
            alloc_result,
            sources,
        )

        if not sched_result.get("success"):
            continue

        utility = float(
            alloc_result["utility"]
        )

        if utility > best_utility:
            # Extract physical log10 rate in requested-link order.
            ordered = [None] * len(combo)

            allocation_map = (
                alloc_result.get("allocation", {})
                or {}
            )

            valid_link_rates = True

            for option in combo:
                link_idx = int(option["link_idx"])
                alloc = allocation_map.get(
                    id(option),
                    {},
                )
                link_utility = alloc.get(
                    "link_utility"
                )

                if link_utility is None:
                    valid_link_rates = False
                    break

                ordered[link_idx] = float(
                    link_utility
                )

            if (
                not valid_link_rates
                or any(value is None for value in ordered)
            ):
                continue

            best_utility = utility
            best_combo = tuple(combo)
            best_link_utilities = tuple(
                float(value)
                for value in ordered
            )

    if best_combo is None:
        return (
            "infeasible",
            None,
            None,
            num_combos,
            "No routing combination produced a feasible EFA allocation.",
            None,
        )

    return (
        "solved",
        best_utility,
        best_combo,
        num_combos,
        None,
        best_link_utilities,
    )


def solve_double_yen(
    network,
    links,
    sources,
    cfg,
    max_yen_combos: int | None,
) -> RoutingResult:
    """
    Generate Double-Yen candidates and exhaustively find the best APOPT-scored
    routing combination within that restricted candidate set.
    """
    start = time.perf_counter()

    generation_start = time.perf_counter()

    try:
        options = _yen_link_options(
            network,
            links,
            sources,
            cfg,
        )
    except Exception as exc:
        return RoutingResult(
            status="error",
            utility=None,
            elapsed_s=time.perf_counter() - start,
            best_combo=None,
            num_link_candidates=(),
            num_combos=0,
            path_generation_s=time.perf_counter() - generation_start,
            combo_evaluation_s=0.0,
            error=str(exc),
        )

    generation_end = time.perf_counter()

    _clear_allocator_cache()

    eval_start = time.perf_counter()

    (
        status,
        utility,
        best_combo,
        num_combos,
        error,
        best_link_utilities,
    ) = _best_apopt_routing_combo(
        options,
        network,
        sources,
        cfg,
        max_combos=max_yen_combos,
    )

    eval_end = time.perf_counter()

    return RoutingResult(
        status=status,
        utility=utility,
        elapsed_s=eval_end - start,
        best_combo=best_combo,
        num_link_candidates=tuple(
            len(options_for_link)
            for options_for_link in options
        ),
        num_combos=num_combos,
        path_generation_s=generation_end - generation_start,
        combo_evaluation_s=eval_end - eval_start,
        error=error,
        best_link_utilities=best_link_utilities,
    )


def solve_exact_routing(
    network,
    links,
    sources,
    cfg,
    max_paths_per_leg: int,
    max_exact_combos: int,
) -> RoutingResult:
    """
    Enumerate all simple paths and all cross-link routing combinations.
    """
    start = time.perf_counter()
    generation_start = time.perf_counter()

    try:
        options = _exact_link_options(
            network,
            links,
            sources,
            cfg,
            max_paths_per_leg=max_paths_per_leg,
        )

    except ExactSearchTooLarge as exc:
        return RoutingResult(
            status="too_large",
            utility=None,
            elapsed_s=time.perf_counter() - start,
            best_combo=None,
            num_link_candidates=(),
            num_combos=0,
            path_generation_s=time.perf_counter() - generation_start,
            combo_evaluation_s=0.0,
            error=str(exc),
        )

    except Exception as exc:
        return RoutingResult(
            status="error",
            utility=None,
            elapsed_s=time.perf_counter() - start,
            best_combo=None,
            num_link_candidates=(),
            num_combos=0,
            path_generation_s=time.perf_counter() - generation_start,
            combo_evaluation_s=0.0,
            error=str(exc),
        )

    generation_end = time.perf_counter()

    candidate_counts = tuple(
        len(options_for_link)
        for options_for_link in options
    )

    if any(count == 0 for count in candidate_counts):
        return RoutingResult(
            status="no_routes",
            utility=None,
            elapsed_s=time.perf_counter() - start,
            best_combo=None,
            num_link_candidates=candidate_counts,
            num_combos=0,
            path_generation_s=generation_end - generation_start,
            combo_evaluation_s=0.0,
            error="At least one entangled link has no exact routing candidate.",
        )

    num_combos = _product(candidate_counts)

    if num_combos > max_exact_combos:
        return RoutingResult(
            status="too_large",
            utility=None,
            elapsed_s=time.perf_counter() - start,
            best_combo=None,
            num_link_candidates=candidate_counts,
            num_combos=num_combos,
            path_generation_s=generation_end - generation_start,
            combo_evaluation_s=0.0,
            error=(
                f"Exact routing combinations {num_combos} exceed "
                f"--max-exact-combos={max_exact_combos}."
            ),
        )

    _clear_allocator_cache()

    eval_start = time.perf_counter()

    (
        status,
        utility,
        best_combo,
        num_combos,
        error,
        best_link_utilities,
    ) = _best_apopt_routing_combo(
        options,
        network,
        sources,
        cfg,
        max_combos=max_exact_combos,
    )

    eval_end = time.perf_counter()

    return RoutingResult(
        status=status,
        utility=utility,
        elapsed_s=eval_end - start,
        best_combo=best_combo,
        num_link_candidates=candidate_counts,
        num_combos=num_combos,
        path_generation_s=generation_end - generation_start,
        combo_evaluation_s=eval_end - eval_start,
        error=error,
        best_link_utilities=best_link_utilities,
    )


# ============================================================================
# Experiment driver
# ============================================================================

def _shape_seed(
    base_seed: int,
    num_users: int,
    num_sources: int,
    num_links: int,
    repetition: int,
) -> int:
    """
    Deterministic, stable seed for one physical network instance.

    N is intentionally NOT included.  This means all N values for a given
    (U,S,L,repetition) are tested on exactly the same network, links, losses,
    dark counts and fidelity thresholds.
    """
    return (
        int(base_seed)
        + 1_000_003 * int(num_users)
        + 10_007 * int(num_sources)
        + 101 * int(num_links)
        + int(repetition)
    )


def run_experiment(
    user_values: list[int],
    source_values: list[int],
    link_values: list[int],
    n_path_values: list[int],
    per_shape: int,
    seed: int,
    channels_per_source: int,
    density: float,
    loss_min_db: float,
    loss_max_db: float,
    min_fidelity: float,
    max_fidelity: float,
    min_dark_count: float,
    max_dark_count: float,
    max_exact_paths_per_leg: int,
    max_exact_combos: int,
    max_yen_combos: int | None,
) -> list[dict]:
    """
    Run exact routing once per physical network and Double Yen once per N.

    Reusing the exact result across N values saves substantial runtime without
    changing the comparison, because exhaustive all-simple-path routing is
    independent of N.
    """
    records: list[dict] = []

    total_network_shapes = sum(
        1
        for u in user_values
        for s in source_values
        for l in link_values
        if u >= 2 * l
        for _ in range(per_shape)
    )

    network_counter = 0

    for num_users in user_values:
        for num_sources in source_values:
            for num_links in link_values:

                if num_users < 2 * num_links:
                    continue

                for repetition in range(1, per_shape + 1):
                    network_counter += 1

                    network_seed = _shape_seed(
                        seed,
                        num_users,
                        num_sources,
                        num_links,
                        repetition,
                    )

                    shape = NetworkShape(
                        num_users=num_users,
                        num_sources=num_sources,
                        num_links=num_links,
                        repetition=repetition,
                        network_seed=network_seed,
                    )

                    # Use the largest requested N only for config construction
                    # of the physical network; N does not affect build_network().
                    n_for_base = max(n_path_values)

                    base_cfg = _make_case_config(
                        shape=shape,
                        n_paths=n_for_base,
                        channels_per_source=channels_per_source,
                        density=density,
                        loss_min_db=loss_min_db,
                        loss_max_db=loss_max_db,
                        min_fidelity=min_fidelity,
                        max_fidelity=max_fidelity,
                        min_dark_count=min_dark_count,
                        max_dark_count=max_dark_count,
                    )

                    print(
                        f"\n[{network_counter}/{total_network_shapes}] "
                        f"Network U={num_users}, S={num_sources}, "
                        f"L={num_links}, rep={repetition}, "
                        f"seed={network_seed}"
                    )

                    try:
                        network, sources, links = build_network(
                            base_cfg
                        )
                    except Exception as exc:
                        print(
                            f"  NETWORK_ERROR: {exc}"
                        )

                        for n_paths in n_path_values:
                            nominal_complexity = (
                                (num_sources * n_paths * n_paths)
                                ** num_links
                            )

                            records.append(
                                {
                                    "case_id": (
                                        f"U{num_users}_S{num_sources}_"
                                        f"L{num_links}_N{n_paths}_"
                                        f"R{repetition}"
                                    ),
                                    "users": num_users,
                                    "sources": num_sources,
                                    "links": num_links,
                                    "n_paths": n_paths,
                                    "repetition": repetition,
                                    "network_seed": network_seed,
                                    "nominal_complexity": nominal_complexity,
                                    "yen_status": "network_error",
                                    "exact_status": "network_error",
                                    "rate_retention_pct": None,
                                    "yen_utility": None,
                                    "exact_utility": None,
                                    "yen_time_s": None,
                                    "exact_time_s": None,
                                    "yen_path_generation_s": None,
                                    "exact_path_generation_s": None,
                                    "yen_combo_evaluation_s": None,
                                    "exact_combo_evaluation_s": None,
                                    "actual_yen_combos": None,
                                    "actual_exact_combos": None,
                                    "yen_link_candidate_counts": None,
                                    "exact_link_candidate_counts": None,
                                    "error": str(exc),
                                }
                            )

                        continue

                    # --------------------------------------------------------
                    # Exact routing reference: independent of N.
                    # --------------------------------------------------------
                    exact_cfg = base_cfg.copy_with(
                        n_paths_per_leg=n_for_base,
                        combo_limit_per_link=(
                            num_sources * n_for_base * n_for_base
                        ),
                    )

                    exact = solve_exact_routing(
                        network=network,
                        links=links,
                        sources=sources,
                        cfg=exact_cfg,
                        max_paths_per_leg=max_exact_paths_per_leg,
                        max_exact_combos=max_exact_combos,
                    )

                    print(
                        f"  Exact: status={exact.status}, "
                        f"combos={exact.num_combos:,}, "
                        f"time={exact.elapsed_s:.3f}s"
                    )

                    # --------------------------------------------------------
                    # Double Yen for each N on the SAME physical network.
                    # --------------------------------------------------------
                    for n_paths in n_path_values:
                        nominal_complexity = (
                            (num_sources * n_paths * n_paths)
                            ** num_links
                        )

                        yen_cfg = base_cfg.copy_with(
                            n_paths_per_leg=n_paths,
                            combo_limit_per_link=(
                                num_sources * n_paths * n_paths
                            ),
                        )

                        yen = solve_double_yen(
                            network=network,
                            links=links,
                            sources=sources,
                            cfg=yen_cfg,
                            max_yen_combos=max_yen_combos,
                        )

                        retention_pct = None
                        comparison_status = None

                        if (
                            exact.status == "solved"
                            and yen.status == "solved"
                        ):
                            retention_pct = (
                                _geometric_mean_rate_retention_pct(
                                    heuristic_utility=yen.utility,
                                    exact_utility=exact.utility,
                                    num_links=num_links,
                                )
                            )

                            if retention_pct is None:
                                comparison_status = "RATE_METRIC_ERROR"
                            else:
                                comparison_status = "COMPARABLE"

                        elif exact.status == "too_large":
                            comparison_status = "EXACT_TOO_LARGE"

                        elif exact.status != "solved":
                            comparison_status = (
                                f"EXACT_{exact.status.upper()}"
                            )

                        else:
                            comparison_status = (
                                f"YEN_{yen.status.upper()}"
                            )

                        case_id = (
                            f"U{num_users}_S{num_sources}_"
                            f"L{num_links}_N{n_paths}_"
                            f"R{repetition}"
                        )

                        record = {
                            "case_id": case_id,
                            "users": num_users,
                            "sources": num_sources,
                            "links": num_links,
                            "n_paths": n_paths,
                            "repetition": repetition,
                            "network_seed": network_seed,
                            "nominal_complexity": nominal_complexity,
                            "comparison_status": comparison_status,
                            "yen_status": yen.status,
                            "exact_status": exact.status,
                            "rate_retention_pct": retention_pct,
                            "yen_utility": yen.utility,
                            "exact_utility": exact.utility,
                            "yen_link_utilities": (
                                None
                                if yen.best_link_utilities is None
                                else repr(list(yen.best_link_utilities))
                            ),
                            "exact_link_utilities": (
                                None
                                if exact.best_link_utilities is None
                                else repr(list(exact.best_link_utilities))
                            ),
                            "yen_time_s": yen.elapsed_s,
                            "exact_time_s": (
                                exact.elapsed_s
                                if exact.status == "solved"
                                else None
                            ),
                            "yen_path_generation_s": (
                                yen.path_generation_s
                            ),
                            "exact_path_generation_s": (
                                exact.path_generation_s
                                if exact.status == "solved"
                                else None
                            ),
                            "yen_combo_evaluation_s": (
                                yen.combo_evaluation_s
                            ),
                            "exact_combo_evaluation_s": (
                                exact.combo_evaluation_s
                                if exact.status == "solved"
                                else None
                            ),
                            "actual_yen_combos": (
                                yen.num_combos
                            ),
                            "actual_exact_combos": (
                                exact.num_combos
                            ),
                            "yen_link_candidate_counts": (
                                str(yen.num_link_candidates)
                            ),
                            "exact_link_candidate_counts": (
                                str(exact.num_link_candidates)
                            ),
                            "error": (
                                exact.error
                                if exact.status != "solved"
                                else yen.error
                            ),
                        }

                        records.append(record)

                        retention_text = (
                            "NA"
                            if retention_pct is None
                            else f"{retention_pct:.4f}%"
                        )

                        print(
                            f"  N={n_paths}: "
                            f"nominal={(nominal_complexity):,}, "
                            f"yen_combos={yen.num_combos:,}, "
                            f"status={comparison_status}, "
                            f"rate={retention_text}, "
                            f"yen_time={yen.elapsed_s:.3f}s"
                        )

    return records


# ============================================================================
# CSV save / load
# ============================================================================

CSV_FIELDS = [
    "case_id",
    "users",
    "sources",
    "links",
    "n_paths",
    "repetition",
    "network_seed",
    "nominal_complexity",
    "comparison_status",
    "yen_status",
    "exact_status",
    "rate_retention_pct",
    "yen_utility",
    "exact_utility",
    "yen_link_utilities",
    "exact_link_utilities",
    "yen_time_s",
    "exact_time_s",
    "yen_path_generation_s",
    "exact_path_generation_s",
    "yen_combo_evaluation_s",
    "exact_combo_evaluation_s",
    "actual_yen_combos",
    "actual_exact_combos",
    "yen_link_candidate_counts",
    "exact_link_candidate_counts",
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
                    field: record.get(field)
                    for field in CSV_FIELDS
                }
            )

    print(
        f"\nSaved raw benchmark data to {outpath}"
    )


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None

    text = value.strip()

    if text == "":
        return None

    return float(text)


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None

    text = value.strip()

    if text == "":
        return None

    return int(text)


def load_records_csv(
    csv_path: Path,
) -> list[dict]:
    records: list[dict] = []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            record = dict(row)

            for key in (
                "users",
                "sources",
                "links",
                "n_paths",
                "repetition",
                "network_seed",
                "nominal_complexity",
                "actual_yen_combos",
                "actual_exact_combos",
            ):
                record[key] = _optional_int(
                    row.get(key)
                )

            for key in (
                "rate_retention_pct",
                "yen_utility",
                "exact_utility",
                "yen_time_s",
                "exact_time_s",
                "yen_path_generation_s",
                "exact_path_generation_s",
                "yen_combo_evaluation_s",
                "exact_combo_evaluation_s",
            ):
                record[key] = _optional_float(
                    row.get(key)
                )

            # Do not reconstruct the new arithmetic per-link metric from
            # total utilities: the per-link information is required. New CSVs
            # written by this script already store rate_retention_pct directly.

            records.append(record)

    return records


# ============================================================================
# Publication plotting
# ============================================================================

def _group_by_complexity(
    records: list[dict],
    y_key: str,
) -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = {}

    for record in records:
        x = record.get("actual_exact_combos")
        y = record.get(y_key)

        if x is None or y is None:
            continue

        x = int(x)
        y = float(y)

        if x <= 0 or not math.isfinite(y):
            continue

        grouped.setdefault(x, []).append(y)

    return grouped


def _paper_axis_format(ax) -> None:
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
        spine.set_linewidth(1.2)

    ax.grid(
        True,
        which="major",
        alpha=GRID_ALPHA,
        linewidth=0.8,
    )


def plot_rate_panel(
    records: list[dict],
    outdir: Path,
) -> None:
    """
    Top paper panel.

    Each faint point is one network realization.
    The blue connected markers are medians at each exhaustive-search size.
    The horizontal dashed line is the exhaustive-routing reference (100%).
    """
    valid = [
        record
        for record in records
        if record.get("rate_retention_pct") is not None
        and record.get("actual_exact_combos") is not None
        and int(record["actual_exact_combos"]) > 0
    ]

    if not valid:
        print(
            "No comparable rate records available for rate plot."
        )
        return

    fig, ax = plt.subplots(
        figsize=PAPER_FIGSIZE,
    )

    x_raw = [
        int(record["actual_exact_combos"])
        for record in valid
    ]

    y_raw = [
        float(record["rate_retention_pct"])
        for record in valid
    ]

    ax.scatter(
        x_raw,
        y_raw,
        s=MARKER_SIZE,
        alpha=RAW_ALPHA,
        color=COLOR_YEN,
        edgecolors="none",
        label="Individual networks",
        zorder=2,
    )

    grouped = _group_by_complexity(
        valid,
        "rate_retention_pct",
    )

    xs = sorted(grouped)
    ys = [
        _median(grouped[x])
        for x in xs
    ]

    ax.plot(
        xs,
        ys,
        marker="o",
        markersize=TREND_MARKER_SIZE,
        linewidth=TREND_LINE_WIDTH,
        color=COLOR_YEN,
        label="Double Yen median",
        zorder=3,
    )

    ax.axhline(
        100.0,
        linestyle="--",
        linewidth=REFERENCE_LINE_WIDTH,
        color=COLOR_REFERENCE,
        label="Exhaustive routing",
        zorder=1,
    )

    ax.set_xscale("log")

    ax.set_xlabel(
        "Exhaustive route configurations evaluated"
    )

    ax.set_ylabel(
        "Geometric-mean link rate\n(% of exhaustive routing)"
    )

    # No title: panel labels / caption should be handled in the manuscript.
    minimum = min(y_raw)
    maximum = max(y_raw)

    lower = max(
        0.0,
        math.floor((minimum - 2.0) / 5.0) * 5.0,
    )

    if lower >= 95.0:
        lower = 95.0

    upper = max(
        101.0,
        math.ceil((maximum + 1.0) / 5.0) * 5.0,
    )

    ax.set_ylim(
        lower,
        upper,
    )

    _paper_axis_format(ax)

    ax.legend(
        loc="lower left",
        frameon=False,
    )

    fig.tight_layout()

    _safe_save_figure(
        fig,
        outdir / "double_yen_rate_vs_complexity.png",
    )

    _safe_save_figure(
        fig,
        outdir / "double_yen_rate_vs_complexity.pdf",
    )

    plt.close(fig)


def plot_runtime_panel(
    records: list[dict],
    outdir: Path,
) -> None:
    """
    Bottom paper panel.

    Individual timings are shown faintly and medians at each nominal
    complexity are connected.
    """
    yen_valid = [
        record
        for record in records
        if record.get("yen_time_s") is not None
        and record.get("actual_exact_combos") is not None
        and float(record["yen_time_s"]) > 0.0
        and int(record["actual_exact_combos"]) > 0
    ]

    exact_valid = [
        record
        for record in records
        if record.get("exact_time_s") is not None
        and record.get("actual_exact_combos") is not None
        and float(record["exact_time_s"]) > 0.0
        and int(record["actual_exact_combos"]) > 0
    ]

    if not yen_valid and not exact_valid:
        print(
            "No runtime records available for runtime plot."
        )
        return

    fig, ax = plt.subplots(
        figsize=PAPER_FIGSIZE,
    )

    if yen_valid:
        ax.scatter(
            [
                int(record["actual_exact_combos"])
                for record in yen_valid
            ],
            [
                float(record["yen_time_s"])
                for record in yen_valid
            ],
            s=MARKER_SIZE,
            alpha=RAW_ALPHA,
            color=COLOR_YEN,
            edgecolors="none",
            zorder=2,
        )

        grouped_yen = _group_by_complexity(
            yen_valid,
            "yen_time_s",
        )

        xs_yen = sorted(grouped_yen)

        ax.plot(
            xs_yen,
            [
                _median(grouped_yen[x])
                for x in xs_yen
            ],
            marker="o",
            markersize=TREND_MARKER_SIZE,
            linewidth=TREND_LINE_WIDTH,
            color=COLOR_YEN,
            label="Double Yen",
            zorder=3,
        )

    if exact_valid:
        ax.scatter(
            [
                int(record["actual_exact_combos"])
                for record in exact_valid
            ],
            [
                float(record["exact_time_s"])
                for record in exact_valid
            ],
            s=MARKER_SIZE,
            alpha=RAW_ALPHA,
            color=COLOR_EXACT,
            edgecolors="none",
            zorder=2,
        )

        grouped_exact = _group_by_complexity(
            exact_valid,
            "exact_time_s",
        )

        xs_exact = sorted(grouped_exact)

        ax.plot(
            xs_exact,
            [
                _median(grouped_exact[x])
                for x in xs_exact
            ],
            marker="s",
            markersize=TREND_MARKER_SIZE,
            linewidth=TREND_LINE_WIDTH,
            color=COLOR_EXACT,
            label="Exhaustive routing",
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        "Exhaustive route configurations evaluated"
    )

    ax.set_ylabel(
        "Runtime (s)"
    )

    # No title.
    _paper_axis_format(ax)

    ax.legend(
        loc="best",
        frameon=False,
    )

    fig.tight_layout()

    _safe_save_figure(
        fig,
        outdir / "double_yen_runtime_vs_complexity.png",
    )

    _safe_save_figure(
        fig,
        outdir / "double_yen_runtime_vs_complexity.pdf",
    )

    plt.close(fig)


def plot_paper_results(
    records: list[dict],
    outdir: Path,
) -> None:
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_rate_panel(
        records,
        outdir,
    )

    plot_runtime_panel(
        records,
        outdir,
    )

    print(
        f"Saved paper plots to {outdir}"
    )


# ============================================================================
# Console summary
# ============================================================================

def print_summary(
    records: list[dict],
) -> None:
    comparable = [
        record
        for record in records
        if record.get("rate_retention_pct") is not None
    ]

    print("\n=== Double Yen validation summary ===")
    print(
        f"Total records: {len(records)}"
    )
    print(
        f"Comparable exact/Double-Yen records: {len(comparable)}"
    )

    if comparable:
        rates = [
            float(record["rate_retention_pct"])
            for record in comparable
        ]

        shortfalls = [
            100.0 - rate
            for rate in rates
        ]

        print(
            f"Median retained geometric-mean rate: "
            f"{statistics.median(rates):.6f}%"
        )

        print(
            f"Mean retained geometric-mean rate: "
            f"{statistics.mean(rates):.6f}%"
        )

        print(
            f"5th percentile retained rate: "
            f"{_percentile(rates, 0.05):.6f}%"
        )

        print(
            f"Worst retained rate: "
            f"{min(rates):.6f}%"
        )

        print(
            f"Median rate shortfall: "
            f"{statistics.median(shortfalls):.6f}%"
        )

        print(
            f"Worst rate shortfall: "
            f"{max(shortfalls):.6f}%"
        )

    statuses: dict[str, int] = {}

    for record in records:
        status = str(
            record.get("comparison_status", "UNKNOWN")
        )

        statuses[status] = (
            statuses.get(status, 0) + 1
        )

    print("\nStatuses:")

    for status in sorted(statuses):
        print(
            f"  {status}: {statuses[status]}"
        )


# ============================================================================
# CLI
# ============================================================================

def _normalize_path_string(
    path_text: str,
) -> str:
    cleaned = path_text.strip()

    if (
        len(cleaned) >= 2
        and cleaned[0] == cleaned[-1]
        and cleaned[0] in ("'", '"')
    ):
        cleaned = cleaned[1:-1].strip()

    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Double Yen routing against exhaustive "
            "all-simple-path routing using APOPT EFA scoring."
        )
    )

    parser.add_argument(
        "--users",
        type=int,
        nargs="+",
        default=[4, 6],
        help=(
            "Numbers of user nodes to test. "
            "Need at least 2*L for disjoint requested links."
        ),
    )

    parser.add_argument(
        "--sources",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Numbers of source nodes to test.",
    )

    parser.add_argument(
        "--links",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Numbers of requested entangled user-pair links.",
    )

    parser.add_argument(
        "--n-paths",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help=(
            "Double-Yen N: number of lowest-loss paths retained "
            "per source-user leg."
        ),
    )

    parser.add_argument(
        "--per-shape",
        type=int,
        default=10,
        help=(
            "Independent random network realizations for each "
            "(U,S,L) physical shape."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=41,
    )

    parser.add_argument(
        "--channels-per-source",
        type=int,
        default=20,
        help=(
            "Frequency-bin pairs available at each source for the "
            "APOPT EFA evaluation."
        ),
    )

    parser.add_argument(
        "--density",
        type=float,
        default=0.2,
        help=(
            "Density parameter passed to the repository's dense "
            "random topology generator."
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
        "--max-exact-paths-per-leg",
        type=int,
        default=100,
        help=(
            "Safety cutoff. If an exact source-user leg has MORE "
            "than this many simple paths, skip the network instead "
            "of truncating the exact reference."
        ),
    )

    parser.add_argument(
        "--max-exact-combos",
        type=int,
        default=50000,
        help=(
            "Maximum exact cross-link routing combinations to "
            "evaluate for one physical network."
        ),
    )

    parser.add_argument(
        "--max-yen-combos",
        type=int,
        default=100_000,
        help=(
            "Maximum Double-Yen cross-link combinations to evaluate. "
            "Use a sufficiently high value so the full Double-Yen "
            "candidate set is searched."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/double_yen_exhaustive",
    )

    parser.add_argument(
        "--records-csv",
        type=str,
        default=None,
        help=(
            "Replot an existing double_yen_validation_results.csv "
            "without rebuilding networks or rerunning APOPT."
        ),
    )

    args = parser.parse_args()

    outdir_text = _normalize_path_string(
        args.output_dir
    )

    if not outdir_text:
        raise ValueError(
            "--output-dir cannot be empty."
        )

    outdir = Path(
        outdir_text
    ).expanduser()

    # ----------------------------------------------------------------------
    # CSV-only replot mode
    # ----------------------------------------------------------------------
    if args.records_csv is not None:
        csv_text = _normalize_path_string(
            args.records_csv
        )

        csv_path = Path(
            csv_text
        ).expanduser()

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
            outdir,
        )

        print(
            f"Regenerated plots from {csv_path}"
        )

        return

    # ----------------------------------------------------------------------
    # Input validation
    # ----------------------------------------------------------------------
    if args.per_shape <= 0:
        raise ValueError(
            "--per-shape must be positive."
        )

    if any(value <= 0 for value in args.users):
        raise ValueError(
            "All --users values must be positive."
        )

    if any(value <= 0 for value in args.sources):
        raise ValueError(
            "All --sources values must be positive."
        )

    if any(value <= 0 for value in args.links):
        raise ValueError(
            "All --links values must be positive."
        )

    if any(value <= 0 for value in args.n_paths):
        raise ValueError(
            "All --n-paths values must be positive."
        )

    if args.channels_per_source <= 0:
        raise ValueError(
            "--channels-per-source must be positive."
        )

    if not (0.0 <= args.density <= 1.0):
        raise ValueError(
            "--density must lie between 0 and 1."
        )

    if args.max_exact_paths_per_leg <= 0:
        raise ValueError(
            "--max-exact-paths-per-leg must be positive."
        )

    if args.max_exact_combos <= 0:
        raise ValueError(
            "--max-exact-combos must be positive."
        )

    if args.max_yen_combos <= 0:
        raise ValueError(
            "--max-yen-combos must be positive."
        )

    # ----------------------------------------------------------------------
    # Run benchmark
    # ----------------------------------------------------------------------
    records = run_experiment(
        user_values=args.users,
        source_values=args.sources,
        link_values=args.links,
        n_path_values=args.n_paths,
        per_shape=args.per_shape,
        seed=args.seed,
        channels_per_source=args.channels_per_source,
        density=args.density,
        loss_min_db=args.loss_min_db,
        loss_max_db=args.loss_max_db,
        min_fidelity=args.fidelity_min,
        max_fidelity=args.fidelity_max,
        min_dark_count=args.dark_count_min,
        max_dark_count=args.dark_count_max,
        max_exact_paths_per_leg=args.max_exact_paths_per_leg,
        max_exact_combos=args.max_exact_combos,
        max_yen_combos=args.max_yen_combos,
    )

    csv_path = (
        outdir
        / "double_yen_validation_results.csv"
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
        outdir,
    )


if __name__ == "__main__":
    main()
