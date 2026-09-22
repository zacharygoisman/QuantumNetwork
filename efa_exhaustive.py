"""
Compare APOPT (GEKKO) against an exact exhaustive validator for the
entangled flux allocation (EFA) MINLP used in the main pipeline.

The APOPT side reuses allocation.allocator.matt directly.

The exact side:
  1. Enumerates every positive integer channel-allocation vector k satisfying
         sum(k_i) <= K.
  2. For each k, solves the continuous source-flux variable mu analytically
     from the intersection of the fidelity-feasible intervals.
  3. Evaluates the same log-utility objective used by allocation.allocator.matt.
  4. Keeps the best feasible allocation.

The script also:
  - checks that the exhaustive k-vector generator has exactly C(K, L) unique
    vectors;
  - distinguishes APOPT solver errors from mathematical infeasibility;
  - samples physically meaningful loss / dark-count / fidelity parameters;
  - computes the production-pipeline-style arithmetic mean of individual link-rate percentages relative to exhaustive EFA;
  - saves all individual trial results to CSV;
  - produces two publication-ready panels with identical dimensions:
      (a) average individual-link rate relative to exhaustive EFA, and
      (b) APOPT versus exhaustive runtime;
  - uses the number of integer EFA allocations C(K, L) as the common
    search-space complexity axis.
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
from typing import Iterable

import matplotlib.pyplot as plt

from allocation.allocator import (
    _EXHAUSTIVE_K_COMBINATION_LIMIT,
    matt,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestCase:
    case_id: str
    num_channels: int
    num_links: int
    y1: tuple[float, ...]
    y2: tuple[float, ...]
    fidelity_limit: tuple[float, ...]


@dataclass
class SolveResult:
    # "solved", "infeasible", "error", or "not_run"
    status: str
    objective: float | None
    mu: float | None
    k: list[int] | None
    elapsed_s: float
    error: str | None = None
    k_vectors_checked: int = 0

    @property
    def solved(self) -> bool:
        return self.status == "solved"


# ---------------------------------------------------------------------------
# Physics / objective functions
# ---------------------------------------------------------------------------

def _expr(
    mu: float,
    k_i: int,
    y1_i: float,
    y2_i: float,
) -> float:
    """
    Dimensionless factor inside the rate / utility expression.

    For x_i = mu * k_i:
        expr_i = x_i^2
                 + x_i * [2(y1_i + y2_i) + 1]
                 + 4 y1_i y2_i
    """
    return (
        mu * mu * (k_i ** 2)
        + mu * k_i * (2.0 * (y1_i + y2_i) + 1.0)
        + 4.0 * y1_i * y2_i
    )


def _objective(
    mu: float,
    k: list[int],
    y1: tuple[float, ...],
    y2: tuple[float, ...],
) -> float:
    """
    Same dimensionless log-utility form used by allocation.allocator.matt:

        U = sum_i log10(expr_i)

    Any per-link multiplicative physical constants that are independent of
    the allocation cancel when comparing APOPT and exact solutions on the
    same fixed set of links.
    """
    total = 0.0

    for i in range(len(k)):
        value = _expr(mu, k[i], y1[i], y2[i])

        if value <= 0.0:
            return float("-inf")

        total += math.log10(value)

    return total


def _fidelity(
    mu: float,
    k_i: int,
    y1_i: float,
    y2_i: float,
) -> float:
    """
    Fidelity model:

        F = 1/4 * [1 + 3(mu k) / expr(mu, k)]
    """
    expr = _expr(mu, k_i, y1_i, y2_i)

    if expr <= 0.0:
        return 0.25

    return 0.25 * (1.0 + 3.0 * mu * k_i / expr)


def _mu_interval_for_link(
    k_i: int,
    y1_i: float,
    y2_i: float,
    f_i: float,
    disc_tol: float = 1e-15,
) -> tuple[float, float] | None:
    """
    Return the fidelity-feasible interval [mu_low, mu_high] for one link
    with a fixed integer channel allocation k_i.

    Starting from:
        F_i(mu, k_i) >= f_i

    and defining:
        t = 4 f_i - 1,

    the constraint becomes:
        a mu^2 + b mu + c <= 0,

    where
        a = t k_i^2
        b = t k_i [2(y1_i + y2_i) + 1] - 3 k_i
        c = 4 t y1_i y2_i.

    For f_i > 1/4, a > 0, so feasibility lies between the real roots.
    """
    if k_i <= 0:
        return None

    if f_i <= 0.25:
        return (0.0, float("inf"))

    s = y1_i + y2_i
    p = y1_i * y2_i
    t = 4.0 * f_i - 1.0

    a = t * (k_i ** 2)
    b = t * k_i * (2.0 * s + 1.0) - 3.0 * k_i
    c = 4.0 * t * p

    disc = b * b - 4.0 * a * c

    # Allow tiny negative values caused by floating-point roundoff.
    if disc < -disc_tol:
        return None

    disc = max(0.0, disc)

    sqrt_disc = math.sqrt(disc)
    r1 = (-b - sqrt_disc) / (2.0 * a)
    r2 = (-b + sqrt_disc) / (2.0 * a)

    lo = min(r1, r2)
    hi = max(r1, r2)

    if hi < 0.0:
        return None

    lo = max(0.0, lo)

    return (lo, hi)


# ---------------------------------------------------------------------------
# Exact integer-allocation enumeration
# ---------------------------------------------------------------------------

def _all_k_vectors(
    num_channels: int,
    num_links: int,
) -> Iterable[list[int]]:
    """
    Yield every positive integer vector k satisfying:

        k_i >= 1
        sum_i k_i <= num_channels

    exactly once.

    The number of such vectors is:

        C(num_channels, num_links)
    """
    if num_links <= 0:
        return

    if num_channels < num_links:
        return

    current = [1] * num_links

    def rec(index: int, remaining: int) -> Iterable[list[int]]:
        if index == num_links - 1:
            # Final link may receive 1 through all remaining channels.
            # This naturally includes all total allocations <= num_channels.
            for last in range(1, remaining + 1):
                current[index] = last
                yield list(current)
            return

        # Leave at least 1 channel for every remaining link.
        links_after_this = num_links - index - 1
        max_here = remaining - links_after_this

        for value in range(1, max_here + 1):
            current[index] = value
            yield from rec(index + 1, remaining - value)

    yield from rec(0, num_channels)


def _validate_k_generator(
    channel_values: list[int],
    link_values: list[int],
) -> None:
    """
    Verify the exhaustive enumerator before trusting the benchmark.
    """
    for num_links in link_values:
        for num_channels in channel_values:
            if num_channels < num_links:
                continue

            vectors = list(_all_k_vectors(num_channels, num_links))
            unique_vectors = {tuple(v) for v in vectors}

            expected = math.comb(num_channels, num_links)

            if len(vectors) != expected:
                raise RuntimeError(
                    f"k-vector generator failed for L={num_links}, "
                    f"K={num_channels}: generated {len(vectors)} vectors, "
                    f"expected {expected}."
                )

            if len(unique_vectors) != expected:
                raise RuntimeError(
                    f"k-vector generator produced duplicates for "
                    f"L={num_links}, K={num_channels}: "
                    f"{len(unique_vectors)} unique vectors vs "
                    f"{expected} expected."
                )


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def solve_exact(
    case: TestCase,
    tol: float = 1e-10,
    fidelity_tol: float = 1e-8,
) -> SolveResult:
    """
    Globally solve the small EFA instance by exhaustive enumeration of k.

    For each fixed k, the objective is monotonically increasing with mu
    for mu >= 0. Therefore the best continuous solution is the largest mu
    in the intersection of all fidelity-feasible intervals.
    """
    start = time.perf_counter()

    best_obj = float("-inf")
    best_mu = None
    best_k = None

    k_vectors_checked = 0

    for k in _all_k_vectors(case.num_channels, case.num_links):
        k_vectors_checked += 1

        mu_lb = 0.0
        mu_ub = float("inf")
        feasible_k = True

        for i in range(case.num_links):
            interval = _mu_interval_for_link(
                k[i],
                case.y1[i],
                case.y2[i],
                case.fidelity_limit[i],
            )

            if interval is None:
                feasible_k = False
                break

            mu_lb = max(mu_lb, interval[0])
            mu_ub = min(mu_ub, interval[1])

            if mu_lb > mu_ub + tol:
                feasible_k = False
                break

        if not feasible_k:
            continue

        if not math.isfinite(mu_ub):
            continue

        # Objective increases with mu, so the optimal continuous value for
        # this fixed integer allocation is the upper feasible boundary.
        mu_star = mu_ub

        # Directly verify the fidelity inequalities.
        fidelity_ok = True

        for i in range(case.num_links):
            f_actual = _fidelity(
                mu_star,
                k[i],
                case.y1[i],
                case.y2[i],
            )

            if f_actual < case.fidelity_limit[i] - fidelity_tol:
                fidelity_ok = False
                break

        if not fidelity_ok:
            continue

        mu_eval = max(mu_star, 1e-15)

        obj = _objective(
            mu_eval,
            k,
            case.y1,
            case.y2,
        )

        if obj > best_obj + tol:
            best_obj = obj
            best_mu = mu_star
            best_k = list(k)

    elapsed_s = time.perf_counter() - start

    expected_count = math.comb(
        case.num_channels,
        case.num_links,
    )

    if k_vectors_checked != expected_count:
        return SolveResult(
            status="error",
            objective=None,
            mu=None,
            k=None,
            elapsed_s=elapsed_s,
            error=(
                f"Exact enumerator checked {k_vectors_checked} vectors, "
                f"but expected {expected_count}."
            ),
            k_vectors_checked=k_vectors_checked,
        )

    if best_k is None or best_mu is None:
        return SolveResult(
            status="infeasible",
            objective=None,
            mu=None,
            k=None,
            elapsed_s=elapsed_s,
            error=None,
            k_vectors_checked=k_vectors_checked,
        )

    return SolveResult(
        status="solved",
        objective=best_obj,
        mu=best_mu,
        k=best_k,
        elapsed_s=elapsed_s,
        error=None,
        k_vectors_checked=k_vectors_checked,
    )


def solve_apopt(
    case: TestCase,
    initial: float,
) -> SolveResult:
    """
    Run the production APOPT / GEKKO allocation routine.

    An exception is treated as a solver error, not as mathematical
    infeasibility.
    """
    start = time.perf_counter()

    try:
        (
            _,
            objective_value,
            _,
            _,
            optimal_mu,
            optimal_allocation,
        ) = matt(
            case.num_channels,
            list(case.fidelity_limit),
            list(case.y1),
            list(case.y2),
            initial=initial,
            verbose=False,
            solver_mode="apopt",
        )

    except Exception as exc:
        return SolveResult(
            status="error",
            objective=None,
            mu=None,
            k=None,
            elapsed_s=time.perf_counter() - start,
            error=str(exc),
        )

    return SolveResult(
        status="solved",
        objective=float(objective_value),
        mu=float(optimal_mu),
        k=[int(v) for v in optimal_allocation],
        elapsed_s=time.perf_counter() - start,
        error=None,
    )


# ---------------------------------------------------------------------------
# Random physically meaningful test-case generation
# ---------------------------------------------------------------------------

def _max_fidelity(
    y1_i: float,
    y2_i: float,
) -> float:
    """
    Maximum possible fidelity as a function of x = mu*k.

    F(x) = 1/4 * [1 + 3x / (x^2 + a x + b)]

    The x-dependent ratio is maximized at:
        x_peak = sqrt(b) = sqrt(4 y1 y2).
    """
    a = 2.0 * (y1_i + y2_i) + 1.0
    b = 4.0 * y1_i * y2_i

    x_peak = math.sqrt(max(0.0, b))
    expr = x_peak * x_peak + a * x_peak + b

    if expr <= 0.0:
        return 0.25

    return 0.25 * (1.0 + 3.0 * x_peak / expr)


def _sample_physical_link(
    rng: random.Random,
    tau: float,
    min_loss_db: float,
    max_loss_db: float,
    min_dark_count: float,
    max_dark_count: float,
    min_fidelity: float,
    max_fidelity: float,
    fidelity_margin: float = 1e-4,
    max_tries: int = 1000,
) -> tuple[float, float, float]:
    """
    Generate one physically meaningful random link.

        eta = 10^(-loss_db/10)
        y   = tau * d / eta

    Rejection-sample until a fidelity threshold in the requested range is
    physically achievable by that link.
    """
    for _ in range(max_tries):
        loss1_db = rng.uniform(
            min_loss_db,
            max_loss_db,
        )
        loss2_db = rng.uniform(
            min_loss_db,
            max_loss_db,
        )

        eta1 = 10.0 ** (-loss1_db / 10.0)
        eta2 = 10.0 ** (-loss2_db / 10.0)

        d1 = rng.uniform(
            min_dark_count,
            max_dark_count,
        )
        d2 = rng.uniform(
            min_dark_count,
            max_dark_count,
        )

        y1_i = tau * d1 / eta1
        y2_i = tau * d2 / eta2

        f_max = _max_fidelity(
            y1_i,
            y2_i,
        )

        f_hi = min(
            max_fidelity,
            f_max - fidelity_margin,
        )

        if f_hi < min_fidelity:
            continue

        f_i = rng.uniform(
            min_fidelity,
            f_hi,
        )

        return y1_i, y2_i, f_i

    raise RuntimeError(
        "Unable to generate a physically feasible random link with the "
        "requested parameter ranges."
    )


def make_test_cases(
    channel_values: list[int],
    link_values: list[int],
    per_shape: int,
    seed: int,
    tau: float = 1e-9,
    min_loss_db: float = 1.0,
    max_loss_db: float = 10.0,
    min_dark_count: float = 100.0,
    max_dark_count: float = 1000.0,
    min_fidelity: float = 0.90,
    max_fidelity: float = 0.95,
) -> list[TestCase]:
    """
    Build per_shape independent random instances for every valid (L, K)
    problem shape.
    """
    rng = random.Random(seed)

    cases: list[TestCase] = []
    case_counter = 1

    for num_links in link_values:
        for num_channels in channel_values:

            if num_channels < num_links:
                continue

            for rep in range(per_shape):

                y1: list[float] = []
                y2: list[float] = []
                fidelity: list[float] = []

                for _ in range(num_links):
                    y1_i, y2_i, f_i = _sample_physical_link(
                        rng=rng,
                        tau=tau,
                        min_loss_db=min_loss_db,
                        max_loss_db=max_loss_db,
                        min_dark_count=min_dark_count,
                        max_dark_count=max_dark_count,
                        min_fidelity=min_fidelity,
                        max_fidelity=max_fidelity,
                    )

                    y1.append(y1_i)
                    y2.append(y2_i)
                    fidelity.append(f_i)

                case_id = (
                    f"L{num_links}_K{num_channels}_"
                    f"R{rep + 1}_C{case_counter}"
                )

                cases.append(
                    TestCase(
                        case_id=case_id,
                        num_channels=num_channels,
                        num_links=num_links,
                        y1=tuple(y1),
                        y2=tuple(y2),
                        fidelity_limit=tuple(fidelity),
                    )
                )

                case_counter += 1

    return cases


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def _solution_link_rates(
    case: TestCase,
    result: SolveResult,
) -> tuple[float, ...] | None:
    """
    Reconstruct each link's rate factor for a solved EFA result.

    The EFA benchmark keeps routing/loss/tau fixed between APOPT and exhaustive,
    so all per-link physical multiplicative constants cancel in the ratio.
    Therefore comparing these rate factors is exactly equivalent to comparing
    the physical link rates used by the production pipeline.
    """
    if (
        not result.solved
        or result.mu is None
        or result.k is None
        or len(result.k) != case.num_links
    ):
        return None

    return tuple(
        _expr(
            float(result.mu),
            int(result.k[i]),
            case.y1[i],
            case.y2[i],
        )
        for i in range(case.num_links)
    )


def _pipeline_style_rate_pct_exact(
    case: TestCase,
    apopt: SolveResult,
    exact: SolveResult,
) -> float | None:
    """
    Match the production pipeline's aggregation rule, but use the exhaustive
    EFA solution as the per-link reference instead of the infinite-resource UB.

    Production pipeline:
        p_i = 100 * 10**(u_i - u_i,reference)
        reported = arithmetic mean_i(p_i)

    Here:
        reference = exhaustive EFA result for the same exact test instance.

    Thus:
        p_i = 100 * R_i,APOPT / R_i,exact

    No clipping at 100% is applied.
    """
    apopt_rates = _solution_link_rates(case, apopt)
    exact_rates = _solution_link_rates(case, exact)

    if apopt_rates is None or exact_rates is None:
        return None

    percentages = []

    for r_apopt, r_exact in zip(apopt_rates, exact_rates):
        if r_apopt <= 0.0 or r_exact <= 0.0:
            return None

        # Written in log-rate form to mirror pipeline/runner.py exactly.
        u_apopt = math.log10(r_apopt)
        u_exact = math.log10(r_exact)

        percentages.append(
            100.0 * 10.0 ** (u_apopt - u_exact)
        )

    return (
        sum(percentages) / len(percentages)
        if percentages
        else None
    )


def _percentile(
    values: list[float],
    percentile: float,
) -> float:
    """Simple linear-interpolation percentile."""
    if not values:
        raise ValueError("Cannot compute percentile of an empty list.")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


# ---------------------------------------------------------------------------
# Publication plot helpers
# ---------------------------------------------------------------------------

# These constants intentionally apply the same dimensions and typography to
# both panels so they can be stacked cleanly as parts (a) and (b) in LaTeX.
PAPER_FIGSIZE = (6.5, 4.0)          # inches
PAPER_FONT_SIZE = 16
PAPER_DPI = 600
PAPER_MARKER_SIZE = 46
PAPER_LINE_WIDTH = 2.2

# Colorblind-friendly Okabe-Ito inspired colors.
COLOR_APOPT = "#0072B2"            # blue
COLOR_EXACT = "#D55E00"            # vermillion
COLOR_REFERENCE = "#555555"        # dark gray

plt.rcParams.update({
    "font.size": PAPER_FONT_SIZE,
    "axes.labelsize": PAPER_FONT_SIZE,
    "xtick.labelsize": PAPER_FONT_SIZE,
    "ytick.labelsize": PAPER_FONT_SIZE,
    "legend.fontsize": PAPER_FONT_SIZE,
    "axes.linewidth": 1.2,
    "lines.linewidth": PAPER_LINE_WIDTH,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _safe_save_figure(
    fig,
    outpath: Path,
    dpi: int = PAPER_DPI,
) -> None:
    """
    Save both a high-resolution PNG and vector PDF.

    Windows/Pillow can occasionally reject a relative pathlib path with
    OSError [Errno 22] in OneDrive-backed working directories.  Resolve the
    destination first and pass an absolute native string to Matplotlib.
    """
    target = Path(outpath).expanduser()

    # Make the destination absolute before handing it to Pillow/Matplotlib.
    if not target.is_absolute():
        target = Path.cwd() / target

    target = target.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)

    stem = target.with_suffix("")
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")

    try:
        fig.savefig(
            str(png_path),
            dpi=dpi,
            format="png",
        )
        fig.savefig(
            str(pdf_path),
            format="pdf",
        )

    except OSError as exc:
        # Do not throw away a completed benchmark merely because the plotting
        # backend rejected the requested Windows path.  Retry in a simple
        # local folder and report exactly where the figures went.
        fallback_dir = (
            Path.cwd()
            / "efa_plot_fallback"
        )
        fallback_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        fallback_png = (
            fallback_dir
            / png_path.name
        )
        fallback_pdf = (
            fallback_dir
            / pdf_path.name
        )

        print(
            f"Warning: could not save figure to {png_path!s}: {exc}"
        )
        print(
            f"Retrying in {fallback_dir.resolve(strict=False)!s}"
        )

        fig.savefig(
            str(
                fallback_png.resolve(
                    strict=False
                )
            ),
            dpi=dpi,
            format="png",
        )
        fig.savefig(
            str(
                fallback_pdf.resolve(
                    strict=False
                )
            ),
            format="pdf",
        )


def _style_publication_axis(ax) -> None:
    """Apply the same visual styling to every paper panel."""
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=PAPER_FONT_SIZE,
        width=1.2,
        length=6,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=1.0,
        length=3,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.8,
        alpha=0.28,
    )


def _complexity_xlim(records: list[dict]) -> tuple[float, float] | None:
    """
    Return common log-scale x limits for both panels using all positive
    C(K,L) values in the records.
    """
    xs = [
        float(r["num_allocations"])
        for r in records
        if r.get("num_allocations") is not None
        and float(r["num_allocations"]) > 0.0
    ]

    if not xs:
        return None

    xmin = min(xs)
    xmax = max(xs)

    if xmin == xmax:
        return xmin / 1.5, xmax * 1.5

    log_min = math.log10(xmin)
    log_max = math.log10(xmax)
    pad = 0.05 * (log_max - log_min)

    return (
        10.0 ** (log_min - pad),
        10.0 ** (log_max + pad),
    )


def _median_by_complexity(
    records: list[dict],
    y_key: str,
    scale: float = 1.0,
) -> tuple[list[float], list[float]]:
    """
    Compute one median y value for every distinct integer-allocation count.

    Individual trials are still shown as scatter points; this median line is
    only a visual guide through repeated random realizations.
    """
    grouped: dict[int, list[float]] = {}

    for record in records:
        x = record.get("num_allocations")
        y = record.get(y_key)

        if x is None or y is None:
            continue

        x = int(x)
        y = float(y) * scale

        if x <= 0 or not math.isfinite(y):
            continue

        grouped.setdefault(x, []).append(y)

    xs = sorted(grouped)
    ys = [statistics.median(grouped[x]) for x in xs]

    return [float(x) for x in xs], ys


def _plot_rate_retention_paper(
    records: list[dict],
    outpath: Path,
    common_xlim: tuple[float, float] | None,
) -> None:
    """
    Paper panel (a): arithmetic mean of the individual APOPT/exhaustive
    link-rate percentages, matching the production pipeline aggregation rule.

    100% means the mean individual-link rate ratio is 100%. Each translucent
    point is one random problem instance; the solid blue line is the median at
    each distinct C(K,L).
    """
    valid = [
        r
        for r in records
        if r.get("num_allocations") is not None
        and r.get("avg_rate_pct_exact") is not None
        and float(r["num_allocations"]) > 0.0
        and math.isfinite(float(r["avg_rate_pct_exact"]))
    ]

    if not valid:
        print("Warning: no rate-retention data available for the paper plot.")
        return

    x = [float(r["num_allocations"]) for r in valid]
    y = [float(r["avg_rate_pct_exact"]) for r in valid]

    median_x, median_y = _median_by_complexity(
        valid,
        "avg_rate_pct_exact",
    )

    fig, ax = plt.subplots(
        figsize=PAPER_FIGSIZE,
        constrained_layout=True,
    )

    # Individual random realizations.
    ax.scatter(
        x,
        y,
        s=PAPER_MARKER_SIZE,
        alpha=0.38,
        color=COLOR_APOPT,
        edgecolors="none",
        zorder=2,
    )

    # Median performance at each search-space size.
    ax.plot(
        median_x,
        median_y,
        marker="o",
        markersize=7,
        color=COLOR_APOPT,
        linewidth=PAPER_LINE_WIDTH,
        label="APOPT median",
        zorder=3,
    )

    # The exact exhaustive optimum is, by definition, 100%.
    ax.axhline(
        100.0,
        color=COLOR_REFERENCE,
        linestyle="--",
        linewidth=1.8,
        label="Exhaustive reference",
        zorder=1,
    )

    ax.axvline(
        _EXHAUSTIVE_K_COMBINATION_LIMIT,
        color=COLOR_REFERENCE,
        linestyle=":",
        linewidth=1.5,
        label=r"Production switch ($10^4$)",
        zorder=1,
    )

    ax.set_xscale("log")
    ax.set_xlabel(
        r"Number of integer EFA allocations, $\binom{K}{L}$"
    )
    ax.set_ylabel(
        "Average individual link rate\n(% of exhaustive rate)"
    )

    if common_xlim is not None:
        ax.set_xlim(*common_xlim)

    # Always include the exact 100% reference while keeping enough vertical
    # resolution to show small deviations from optimality.
    ymin = min(y)
    ymax = max(y)
    spread = max(1.0, ymax - ymin)
    vertical_margin = max(0.5, 0.08 * spread)
    ax.set_ylim(
        max(0.0, ymin - vertical_margin),
        max(100.5, ymax + vertical_margin),
    )

    _style_publication_axis(ax)
    ax.legend(frameon=False, loc="lower left")

    _safe_save_figure(fig, outpath)
    plt.close(fig)


def _plot_runtime_paper(
    records: list[dict],
    outpath: Path,
    common_xlim: tuple[float, float] | None,
) -> None:
    """
    Paper panel (b): runtime of APOPT and exact exhaustive EFA validation as a
    function of the integer EFA search-space size C(K,L).

    Runtime is displayed in seconds on a log scale. Individual trials are
    translucent; solid lines show the median at each distinct complexity.
    """
    valid = [
        r
        for r in records
        if r.get("num_allocations") is not None
        and r.get("apopt_time_ms") is not None
        and r.get("exact_time_ms") is not None
        and float(r["num_allocations"]) > 0.0
        and float(r["apopt_time_ms"]) > 0.0
        and float(r["exact_time_ms"]) > 0.0
    ]

    if not valid:
        print("Warning: no runtime data available for the paper plot.")
        return

    x = [float(r["num_allocations"]) for r in valid]
    apopt_s = [float(r["apopt_time_ms"]) / 1000.0 for r in valid]
    exact_s = [float(r["exact_time_ms"]) / 1000.0 for r in valid]

    apopt_median_x, apopt_median_y = _median_by_complexity(
        valid,
        "apopt_time_ms",
        scale=1.0 / 1000.0,
    )
    exact_median_x, exact_median_y = _median_by_complexity(
        valid,
        "exact_time_ms",
        scale=1.0 / 1000.0,
    )

    fig, ax = plt.subplots(
        figsize=PAPER_FIGSIZE,
        constrained_layout=True,
    )

    ax.scatter(
        x,
        apopt_s,
        s=PAPER_MARKER_SIZE,
        alpha=0.36,
        color=COLOR_APOPT,
        edgecolors="none",
        zorder=2,
    )
    ax.scatter(
        x,
        exact_s,
        s=PAPER_MARKER_SIZE,
        alpha=0.36,
        color=COLOR_EXACT,
        marker="s",
        edgecolors="none",
        zorder=2,
    )

    ax.plot(
        apopt_median_x,
        apopt_median_y,
        marker="o",
        markersize=7,
        color=COLOR_APOPT,
        linewidth=PAPER_LINE_WIDTH,
        label="APOPT",
        zorder=3,
    )
    ax.plot(
        exact_median_x,
        exact_median_y,
        marker="s",
        markersize=7,
        color=COLOR_EXACT,
        linewidth=PAPER_LINE_WIDTH,
        label="Exhaustive",
        zorder=3,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.axvline(
        _EXHAUSTIVE_K_COMBINATION_LIMIT,
        color=COLOR_REFERENCE,
        linestyle=":",
        linewidth=1.5,
        label=r"Production switch ($10^4$)",
        zorder=1,
    )

    ax.set_xlabel(
        r"Number of integer EFA allocations, $\binom{K}{L}$"
    )
    ax.set_ylabel("Runtime (s)")

    if common_xlim is not None:
        ax.set_xlim(*common_xlim)

    _style_publication_axis(ax)
    ax.legend(frameon=False, loc="best")

    _safe_save_figure(fig, outpath)
    plt.close(fig)


def plot_comparison_results(
    records: list[dict],
    outdir: Path,
) -> None:
    """
    Generate the two publication panels for the APOPT / exact EFA comparison.

    Both panels have:
      - identical physical dimensions;
      - identical font sizes;
      - identical x-axis limits and log scaling;
      - no title;
      - PNG and vector PDF output.

    Output files:
      efa_rate_vs_complexity.png/.pdf
      efa_runtime_vs_complexity.png/.pdf
    """
    outdir.mkdir(parents=True, exist_ok=True)

    common_xlim = _complexity_xlim(records)

    _plot_rate_retention_paper(
        records=records,
        outpath=outdir / "efa_rate_vs_complexity.png",
        common_xlim=common_xlim,
    )

    _plot_runtime_paper(
        records=records,
        outpath=outdir / "efa_runtime_vs_complexity.png",
        common_xlim=common_xlim,
    )

    print(
        "Saved publication plots to "
        f"{outdir} (PNG and PDF versions)."
    )


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_records_csv(
    records: list[dict],
    outpath: Path,
) -> None:
    """
    Save every individual trial for later statistics / publication plots.
    """
    if not records:
        return

    outpath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "case_id",
        "channels",
        "links",
        "num_allocations",
        "comparison_status",
        "matched_exact",
        "gap",
        "avg_rate_pct_exact",
        "objective_rate_pct_exact",
        "rate_retention_pct",
        "rate_shortfall_pct",
        "exact_objective",
        "apopt_objective",
        "exact_mu",
        "apopt_mu",
        "exact_k",
        "apopt_k",
        "exact_link_rates",
        "apopt_link_rates",
        "apopt_time_ms",
        "exact_time_ms",
        "exact_k_vectors_checked",
        "exact_error",
        "apopt_error",
    ]

    with outpath.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow({
                key: record.get(key)
                for key in fieldnames
            })

    print(f"Saved raw results to {outpath}")


def _normalize_path_string(
    path_text: str,
) -> str:
    cleaned = path_text.strip()
    if len(cleaned) >= 2 and (
        (cleaned[0] == '"' and cleaned[-1] == '"')
        or (cleaned[0] == "'" and cleaned[-1] == "'")
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned


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


def _parse_optional_bool(
    value: str | None,
) -> bool | None:
    if value is None:
        return None
    text = value.strip().lower()
    if text == "":
        return None
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def load_records_csv(
    csv_path: Path,
) -> list[dict]:
    """
    Load previously saved exhaustive-comparison records from CSV for plotting.
    """
    records: list[dict] = []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record = {
                "case_id": row.get("case_id"),
                "channels": _parse_optional_int(row.get("channels")),
                "links": _parse_optional_int(row.get("links")),
                "num_allocations": _parse_optional_int(row.get("num_allocations")),
                "comparison_status": row.get("comparison_status"),
                "matched_exact": _parse_optional_bool(row.get("matched_exact")),
                "gap": _parse_optional_float(row.get("gap")),
                "avg_rate_pct_exact": _parse_optional_float(row.get("avg_rate_pct_exact")),
                "objective_rate_pct_exact": _parse_optional_float(row.get("objective_rate_pct_exact")),
                "rate_retention_pct": _parse_optional_float(row.get("rate_retention_pct")),
                "rate_shortfall_pct": _parse_optional_float(row.get("rate_shortfall_pct")),
                "exact_objective": _parse_optional_float(row.get("exact_objective")),
                "apopt_objective": _parse_optional_float(row.get("apopt_objective")),
                "exact_mu": _parse_optional_float(row.get("exact_mu")),
                "apopt_mu": _parse_optional_float(row.get("apopt_mu")),
                "exact_k": row.get("exact_k"),
                "apopt_k": row.get("apopt_k"),
                "exact_link_rates": row.get("exact_link_rates"),
                "apopt_link_rates": row.get("apopt_link_rates"),
                "apopt_time_ms": _parse_optional_float(row.get("apopt_time_ms")),
                "exact_time_ms": _parse_optional_float(row.get("exact_time_ms")),
                "exact_k_vectors_checked": _parse_optional_int(row.get("exact_k_vectors_checked")),
                "exact_error": row.get("exact_error"),
                "apopt_error": row.get("apopt_error"),
            }
            # New CSVs store the pipeline-style arithmetic mean directly.
            # The alias supports CSVs written by the immediately preceding
            # version of this benchmark.
            if record.get("avg_rate_pct_exact") is None:
                record["avg_rate_pct_exact"] = record.get("rate_retention_pct")
            if record.get("rate_retention_pct") is None:
                record["rate_retention_pct"] = record.get("avg_rate_pct_exact")


            records.append(record)

    return records


# ---------------------------------------------------------------------------
# Case comparison
# ---------------------------------------------------------------------------

def compare_cases(
    cases: list[TestCase],
    initial: float,
    tol: float,
) -> list[dict]:
    """
    Run BOTH solvers on every tractable test case.

    There is deliberately no hybrid mixing in this validation figure:
        - exact exhaustive EFA is always run;
        - APOPT is always forced, even below the production 1e4 threshold.

    This allows the runtime crossover itself to justify the production switch.
    """
    records: list[dict] = []

    apopt_errors = 0
    exact_errors = 0
    exact_infeasible = 0
    solved_pairs = 0

    total = len(cases)

    print(
        f"Running BOTH forced APOPT and exhaustive EFA on {total} cases. "
        f"Production switch threshold = {_EXHAUSTIVE_K_COMBINATION_LIMIT:,} allocations.\n"
    )

    for case in cases:
        num_allocations = math.comb(
            case.num_channels,
            case.num_links,
        )

        exact = solve_exact(case)
        apopt = solve_apopt(
            case,
            initial=initial,
        )

        comparison_status = "UNKNOWN"

        if exact.status == "error":
            comparison_status = "EXACT_ERROR"
            exact_errors += 1
        elif exact.status == "infeasible":
            comparison_status = "EXACT_INFEASIBLE"
            exact_infeasible += 1
        elif apopt.status == "error":
            comparison_status = "APOPT_ERROR"
            apopt_errors += 1
        elif not apopt.solved:
            comparison_status = "APOPT_NOT_SOLVED"
        else:
            comparison_status = "BOTH_SOLVED"
            solved_pairs += 1

        gap = None
        matched_exact = None
        avg_rate_pct_exact = None
        objective_rate_pct_exact = None
        rate_shortfall_pct = None

        if exact.solved and apopt.solved:
            gap = (
                float(exact.objective)
                - float(apopt.objective)
            )

            matched_exact = (
                abs(gap) <= tol
            )

            # Pipeline-style arithmetic mean of individual link ratios.
            avg_rate_pct_exact = (
                _pipeline_style_rate_pct_exact(
                    case,
                    apopt,
                    exact,
                )
            )

            # Objective-aligned comparison.  Since U=sum(log10 R_l), this is
            # the ratio of geometric-mean link rates and directly measures
            # retention of the objective being optimized.
            objective_rate_pct_exact = (
                100.0
                * 10.0 ** (
                    (
                        float(apopt.objective)
                        - float(exact.objective)
                    )
                    / float(case.num_links)
                )
            )

            if avg_rate_pct_exact is not None:
                rate_shortfall_pct = (
                    100.0 - avg_rate_pct_exact
                )

        exact_rates = _solution_link_rates(
            case,
            exact,
        )
        apopt_rates = _solution_link_rates(
            case,
            apopt,
        )

        record = {
            "case_id": case.case_id,
            "channels": case.num_channels,
            "links": case.num_links,
            "num_allocations": num_allocations,
            "comparison_status": comparison_status,
            "matched_exact": matched_exact,
            "gap": gap,
            # Primary paper quality metric.
            "avg_rate_pct_exact": avg_rate_pct_exact,
            # Keep this alias so existing lightweight replot code can be adapted
            # with minimal changes.
            "rate_retention_pct": avg_rate_pct_exact,
            "rate_shortfall_pct": rate_shortfall_pct,
            "exact_objective": exact.objective,
            "apopt_objective": apopt.objective,
            "exact_mu": exact.mu,
            "apopt_mu": apopt.mu,
            "exact_k": exact.k,
            "apopt_k": apopt.k,
            "exact_link_rates": (
                None if exact_rates is None else list(exact_rates)
            ),
            "apopt_link_rates": (
                None if apopt_rates is None else list(apopt_rates)
            ),
            "apopt_time_ms": 1000.0 * apopt.elapsed_s,
            "exact_time_ms": 1000.0 * exact.elapsed_s,
            "exact_k_vectors_checked": exact.k_vectors_checked,
            "exact_error": exact.error,
            "apopt_error": apopt.error,
        }

        records.append(record)

        rate_text = (
            "None"
            if avg_rate_pct_exact is None
            else f"{avg_rate_pct_exact:.6f}%"
        )

        print(
            f"{case.case_id}: "
            f"L={case.num_links}, "
            f"K={case.num_channels}, "
            f"C(K,L)={num_allocations:,}, "
            f"status={comparison_status}, "
            f"U_exact={None if exact.objective is None else round(exact.objective, 10)}, "
            f"U_APOPT={None if apopt.objective is None else round(apopt.objective, 10)}, "
            f"gap={None if gap is None else f'{gap:.3e}'}, "
            f"avg_rate_vs_exact={rate_text}, "
            f"t_APOPT={apopt.elapsed_s * 1000:.1f} ms, "
            f"t_exact={exact.elapsed_s * 1000:.1f} ms"
        )

    print("\n=== Summary ===")
    print(f"Total generated cases: {total}")
    print(f"Both solved: {solved_pairs}")
    print(f"APOPT errors: {apopt_errors}")
    print(f"Exact errors: {exact_errors}")
    print(f"Exact infeasible: {exact_infeasible}")
    print(
        f"Production crossover threshold: "
        f"{_EXHAUSTIVE_K_COMBINATION_LIMIT:,} allocations"
    )

    rates = [
        float(r["avg_rate_pct_exact"])
        for r in records
        if r.get("avg_rate_pct_exact") is not None
    ]

    if rates:
        print("\n=== Average individual-link rate relative to exhaustive ===")
        print(f"Mean: {statistics.mean(rates):.6f}%")
        print(f"Median: {statistics.median(rates):.6f}%")
        print(f"5th percentile: {_percentile(rates, 0.05):.6f}%")
        print(f"Minimum: {min(rates):.6f}%")
        print(f"Maximum: {max(rates):.6f}%")

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Compare APOPT EFA allocation against an "
            "exact exhaustive small-instance validator."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--per-shape",
        type=int,
        default=20,
        help=(
            "Number of independent random cases generated "
            "for each valid (number of links, number of "
            "channels) pair."
        ),
    )

    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[
            8, 12, 16, 20, 24
        ],
    )

    parser.add_argument(
        "--links",
        type=int,
        nargs="+",
        default=[
            2, 3, 4, 5, 6
        ],
    )

    parser.add_argument(
        "--initial",
        type=float,
        default=0.001,
        help=(
            "Initial mu value passed to "
            "allocation.allocator.matt."
        ),
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help=(
            "Objective tolerance used to classify "
            "an APOPT exact match."
        ),
    )

    parser.add_argument(
        "--plot-dir",
        type=str,
        default="outputs/efa_exhaustive",
        help=(
            "Directory in which plots and the CSV "
            "file are saved."
        ),
    )

    parser.add_argument(
        "--records-csv",
        type=str,
        default=None,
        help=(
            "Optional path to an existing efa_validation_results.csv. "
            "If provided, the script skips solving and regenerates all plots "
            "from this CSV."
        ),
    )

    args = parser.parse_args()

    outdir_text = _normalize_path_string(args.plot_dir)
    if outdir_text == "":
        raise ValueError("--plot-dir must not be empty.")
    outdir = Path(outdir_text).expanduser()

    if args.records_csv is not None:
        csv_text = _normalize_path_string(args.records_csv)
        if csv_text == "":
            raise ValueError("--records-csv must not be empty when provided.")
        csv_path = Path(csv_text).expanduser()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        records = load_records_csv(csv_path)
        if not records:
            raise ValueError(f"CSV has no data rows: {csv_path}")

        plot_comparison_results(
            records=records,
            outdir=outdir,
        )
        print(f"Regenerated plots from CSV: {csv_path}")
        return

    if args.per_shape <= 0:
        raise ValueError(
            "--per-shape must be positive."
        )

    if any(
        k <= 0
        for k in args.channels
    ):
        raise ValueError(
            "All --channels values must be positive."
        )

    if any(
        l <= 0
        for l in args.links
    ):
        raise ValueError(
            "All --links values must be positive."
        )

    # Verify exhaustive integer enumeration before
    # trusting any benchmark.
    _validate_k_generator(
        channel_values=args.channels,
        link_values=args.links,
    )

    cases = make_test_cases(
        channel_values=args.channels,
        link_values=args.links,
        per_shape=args.per_shape,
        seed=args.seed,
    )

    records = compare_cases(
        cases=cases,
        initial=args.initial,
        tol=args.tol,
    )

    save_records_csv(
        records=records,
        outpath=(
            outdir
            / "efa_validation_results.csv"
        ),
    )

    plot_comparison_results(
        records=records,
        outdir=outdir,
    )


if __name__ == "__main__":
    main()
