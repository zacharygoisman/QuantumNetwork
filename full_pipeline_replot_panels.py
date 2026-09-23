from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

from paper_panel_utils import (
    DEFAULT_STYLE,
    compute_log_limits,
    new_panel_figure,
    new_stacked_figure,
    save_figure,
    style_axis,
)

RAW_ALPHA = 0.30

# ------------------------------------------------------------------
# Fixed y-axis limits for the three paper panels.
# Edit these values directly if you want to change the displayed range.
# ------------------------------------------------------------------
RATE_YLIM = None
RUNTIME_YLIM = None
ATTEMPTS_YLIM = None

# ---------------------------------------------------------------
# Optional y-axis scaling trims used ONLY for choosing the displayed
# axis range.  The underlying scatter points are NOT removed.
#
# Example use:
# - RATE_TRIM_OUTLIER_SIDE = None     ignores one unusually low rate point
# - RUNTIME_TRIM_OUTLIER_SIDE = None ignores one unusually high runtime point
#
# Valid values: "low", "high", or None
# ---------------------------------------------------------------
RATE_TRIM_OUTLIER_SIDE = None
RUNTIME_TRIM_OUTLIER_SIDE = None
ATTEMPTS_TRIM_OUTLIER_SIDE = None

TOPOLOGY_STYLE = {
    "dense": {"color": "#0072B2", "marker": "o", "label": "Dense"},
    # "ring": {"color": "#D55E00", "marker": "s", "label": "Ring"},
    "star": {"color": "#009E73", "marker": "^", "label": "Star"},
}


def _f(v):
    if v is None or str(v).strip() == "":
        return None
    return float(v)


def _i(v):
    if v is None or str(v).strip() == "":
        return None
    return int(v)


INT_FIELDS = {
    "users", "sources", "links", "channels_per_source", "n_paths",
    "repetition", "network_seed", "num_nodes", "num_edges",
    "routing_complexity", "efa_complexity", "combined_complexity",
    "actual_route_combos", "actual_combined_complexity", "combos_returned",
    "valid_combos", "allocation_failures", "contention_failures", "ub_pruned",
    "attempts_to_first_feasible", "best_combo_idx",
    "rejected_combos", "evaluated_combos",
}

FLOAT_FIELDS = {
    "requested_density", "realized_density", "time_to_first_feasible_s",
    "time_to_best_s", "best_utility", "upper_bound_utility",
    "rate_pct_upper_bound", "build_time_s", "routing_time_s",
    "evaluation_time_s", "total_pipeline_time_s",
}


def load_records(path: Path):
    rows = []

    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = dict(row)

            for k in INT_FIELDS:
                r[k] = _i(row.get(k))

            for k in FLOAT_FIELDS:
                r[k] = _f(row.get(k))

            if (
                r.get("combined_complexity") is None
                and r.get("routing_complexity") is not None
                and r.get("efa_complexity") is not None
            ):
                r["combined_complexity"] = (
                    r["routing_complexity"]
                    * r["efa_complexity"]
                )

            if (
                r.get("actual_combined_complexity") is None
                and r.get("actual_route_combos") is not None
                and r.get("efa_complexity") is not None
            ):
                r["actual_combined_complexity"] = (
                    r["actual_route_combos"]
                    * r["efa_complexity"]
                )

            rows.append(r)

    return rows


def x_info(metric):
    if metric == "combined":
        return (
            "combined_complexity",
            r"Pipeline complexity, $(SN^2)^L \binom{K}{L}$",
        )

    if metric == "routing":
        return (
            "routing_complexity",
            r"Nominal routing combinations, $(SN^2)^L$",
        )

    if metric == "actual":
        return (
            "actual_combined_complexity",
            r"Actual candidate complexity, $N_{\mathrm{route}}\binom{K}{L}$",
        )

    raise ValueError(metric)


def valid(records, x_key, y_key, positive=False):
    out = []

    for r in records:
        x = r.get(x_key)
        y = r.get(y_key)

        if x is None or y is None:
            continue

        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError):
            continue

        if (
            not math.isfinite(x)
            or not math.isfinite(y)
            or x <= 0
        ):
            continue

        if positive and y <= 0:
            continue

        out.append(r)

    return out


def draw_topologies(ax, records, x_key, y_key):
    """
    Draw raw scatter points only.

    No median-by-x trend is calculated and no connecting line is drawn.
    """
    for topo in ("dense", "star"):
        style = TOPOLOGY_STYLE[topo]

        subset = [
            r for r in records
            if r.get("topology") == topo
        ]

        if not subset:
            continue

        ax.scatter(
            [float(r[x_key]) for r in subset],
            [float(r[y_key]) for r in subset],
            s=DEFAULT_STYLE.marker_size,
            alpha=RAW_ALPHA,
            color=style["color"],
            marker=style["marker"],
            edgecolors="none",
            label=style["label"],
            zorder=2,
        )



def _trimmed_display_ylim(
    values,
    *,
    trim_side=None,
    log_scale=False,
    lower_floor=None,
    linear_pad_fraction=0.08,
    log_pad_factor=1.15,
):
    """
    Choose a y-axis range using all finite values except ONE outlier, but keep
    every point in the plot itself.

    trim_side:
        None   -> use all values
        "low"  -> ignore the single smallest value when choosing limits
        "high" -> ignore the single largest value when choosing limits
    """
    vals = sorted(
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    )

    if not vals:
        return None

    kept = vals[:]

    if len(vals) >= 2:
        if trim_side == "low":
            kept = vals[1:]
        elif trim_side == "high":
            kept = vals[:-1]

    if not kept:
        kept = vals

    if log_scale:
        positive = [v for v in kept if v > 0.0]
        if not positive:
            return None

        ymin = min(positive)
        ymax = max(positive)

        lower = ymin / float(log_pad_factor)
        upper = ymax * float(log_pad_factor)

        if lower_floor is not None:
            lower = max(float(lower_floor), lower)

        return lower, upper

    ymin = min(kept)
    ymax = max(kept)

    span = ymax - ymin
    if span <= 0.0:
        span = max(abs(ymax) * 0.1, 1.0)

    lower = ymin - float(linear_pad_fraction) * span
    upper = ymax + float(linear_pad_fraction) * span

    if lower_floor is not None:
        lower = max(float(lower_floor), lower)

    return lower, upper



def draw_rate(ax, records, x_key, xlabel):
    v = valid(
        records,
        x_key,
        "rate_pct_upper_bound",
        positive=True,
    )

    print(
        f"Rate scatter points: {len(v)}"
    )

    if not v:
        raise ValueError(
            "No valid rate records."
        )

    draw_topologies(
        ax,
        v,
        x_key,
        "rate_pct_upper_bound",
    )

    ax.axhline(
        100.0,
        linestyle="--",
        linewidth=DEFAULT_STYLE.reference_line_width,
        color="#555555",
        zorder=1,
    )

    style_axis(
        ax,
        # xlabel=xlabel,
        # ylabel=(
        #     r"Geometric mean rate $\rho_\infty$ (%)"
        # ),
        xscale="log",
        yscale="linear",
    )

    # Remove x-axis label and numbers on the top geometric-mean panel,
    # while keeping the tick marks.
    ax.set_xlabel("")
    ax.tick_params(
        axis="x",
        which="both",
        bottom=True,
        top=False,
        labelbottom=False,
    )

    # Remove the plot grid.
    ax.grid(False, which="both", axis="both")

    vals = [
        r.get("rate_pct_upper_bound")
        for r in v
    ]
    auto_ylim = _trimmed_display_ylim(
        vals,
        trim_side=RATE_TRIM_OUTLIER_SIDE,
        log_scale=False,
        lower_floor=0.0,
    )
    if auto_ylim is not None:
        ax.set_ylim(
            *auto_ylim,
            auto=False,
        )


def draw_runtime(ax, records, x_key, xlabel):
    v = valid(
        records,
        x_key,
        "total_pipeline_time_s",
        positive=True,
    )

    if not v:
        raise ValueError(
            "No valid runtime records."
        )

    draw_topologies(
        ax,
        v,
        x_key,
        "total_pipeline_time_s",
    )

    style_axis(
        ax,
        # xlabel=xlabel,
        # ylabel="Runtime (s)",
        xscale="log",
        yscale="log",
    )

    # Remove the plot grid.
    ax.grid(False, which="both", axis="both")

    vals = [
        r.get("total_pipeline_time_s")
        for r in v
    ]
    auto_ylim = _trimmed_display_ylim(
        vals,
        trim_side=RUNTIME_TRIM_OUTLIER_SIDE,
        log_scale=True,
        lower_floor=1.0e-12,
    )
    if auto_ylim is not None:
        ax.set_ylim(
            *auto_ylim,
            auto=False,
        )


def draw_attempts(ax, records, x_key, xlabel):
    """
    Plot the number of routing configurations attempted before the first
    feasible end-to-end solution is found.

    This uses the CSV field attempts_to_first_feasible directly.
    Rows without a first-feasible solution are omitted from this panel.
    """
    v = valid(
        records,
        x_key,
        "attempts_to_first_feasible",
        positive=True,
    )

    print(
        f"First-feasible attempt scatter points: {len(v)}"
    )

    if not v:
        raise ValueError(
            "No valid attempts_to_first_feasible records."
        )

    draw_topologies(
        ax,
        v,
        x_key,
        "attempts_to_first_feasible",
    )

    style_axis(
        ax,
        # xlabel=xlabel,
        # ylabel=(
        #     "Attempted configurations"
        # ),
        xscale="log",
        yscale="linear",
    )

    # Remove the plot grid.
    ax.grid(False, which="both", axis="both")

    vals = [
        r.get("attempts_to_first_feasible")
        for r in v
    ]
    auto_ylim = _trimmed_display_ylim(
        vals,
        trim_side=ATTEMPTS_TRIM_OUTLIER_SIDE,
        log_scale=True,
        lower_floor=0.5,
    )
    if auto_ylim is not None:
        ax.set_ylim(
            *auto_ylim,
            auto=False,
        )

def print_summary_statistics(records):
    """
    Print the paper-summary statistics after topology filtering.

    rate_pct_upper_bound is already the geometric-mean link-rate retention
    for each network:
        rho_i = 100 * 10^[(U_i - U_inf,i) / L_i]

    To summarize the entire ensemble in one rate number, use an L-weighted
    geometric mean. This gives every individual entangled link equal weight
    rather than giving every network equal weight.
    """
    rate_rows = []

    for r in records:
        pct = r.get(
            "rate_pct_upper_bound"
        )

        links = r.get(
            "links"
        )

        if (
            pct is None
            or links is None
        ):
            continue

        pct = float(pct)
        links = int(links)

        if (
            not math.isfinite(pct)
            or pct <= 0.0
            or links <= 0
        ):
            continue

        rate_rows.append(
            (
                pct,
                links,
            )
        )

    runtime_values = []

    for r in records:
        runtime = r.get(
            "total_pipeline_time_s"
        )

        if runtime is None:
            continue

        runtime = float(runtime)

        if (
            math.isfinite(runtime)
            and runtime > 0.0
        ):
            runtime_values.append(
                runtime
            )

    print(
        "\n=== FULL-PIPELINE SUMMARY ==="
    )

    if rate_rows:
        total_links = sum(
            links
            for _, links in rate_rows
        )

        # Since each pct is itself a per-network geometric mean,
        # weighting log(pct/100) by that network's L reconstructs
        # the geometric mean across all represented links.
        weighted_log_ratio = sum(
            links
            * math.log(
                pct / 100.0
            )
            for pct, links
            in rate_rows
        ) / float(
            total_links
        )

        overall_rate_pct = (
            100.0
            * math.exp(
                weighted_log_ratio
            )
        )

        print(
            "Geometric-mean rate retention "
            "(all represented links): "
            f"{overall_rate_pct:.3f}%"
        )

        print(
            "Networks contributing rate data: "
            f"{len(rate_rows)}"
        )

        print(
            "Total links contributing rate data: "
            f"{total_links}"
        )
    else:
        print(
            "Geometric-mean rate retention: "
            "no valid rate records"
        )

    if runtime_values:
        median_runtime = statistics.median(
            runtime_values
        )

        maximum_runtime = max(
            runtime_values
        )

        print(
            "Median total pipeline runtime: "
            f"{median_runtime:.6f} s"
        )

        print(
            "Maximum total pipeline runtime: "
            f"{maximum_runtime:.6f} s"
        )

        print(
            "Networks contributing runtime data: "
            f"{len(runtime_values)}"
        )
    else:
        print(
            "Runtime statistics: "
            "no valid runtime records"
        )


def replot(
    csv_path: Path,
    outdir: Path,
    metric: str,
):
    records = load_records(
        csv_path
    )

    # Remove all ring-topology results.
    records = [
        r for r in records
        if r.get("topology") != "ring"
    ]

    # Print the values needed for the paper text.
    print_summary_statistics(
        records
    )

    x_key, xlabel = x_info(
        metric
    )

    xvals = [
        r[x_key]
        for r in records
        if (
            r.get(x_key) is not None
            and float(r[x_key]) > 0
        )
    ]

    xlim = compute_log_limits(
        xvals
    )

    fig, ax = new_panel_figure()

    draw_rate(
        ax,
        records,
        x_key,
        xlabel,
    )

    if xlim:
        ax.set_xlim(
            *xlim
        )

    if RATE_YLIM is not None:
        ax.set_ylim(
            *RATE_YLIM,
            auto=False,
        )

    print(
        "RATE YLIM:",
        ax.get_ylim(),
    )

    save_figure(
        fig,
        outdir,
        "full_pipeline_rate_panel",
    )

    fig, ax = new_panel_figure()

    draw_runtime(
        ax,
        records,
        x_key,
        xlabel,
    )

    if xlim:
        ax.set_xlim(
            *xlim
        )

    if RUNTIME_YLIM is not None:
        ax.set_ylim(
            *RUNTIME_YLIM,
            auto=False,
        )

    print(
        "RUNTIME YLIM:",
        ax.get_ylim(),
    )

    save_figure(
        fig,
        outdir,
        "full_pipeline_runtime_panel",
    )

    fig, ax = new_panel_figure()

    draw_attempts(
        ax,
        records,
        x_key,
        xlabel,
    )

    if xlim:
        ax.set_xlim(
            *xlim
        )

    if ATTEMPTS_YLIM is not None:
        ax.set_ylim(
            *ATTEMPTS_YLIM,
            auto=False,
        )

    print(
        "ATTEMPTS YLIM:",
        ax.get_ylim(),
    )

    save_figure(
        fig,
        outdir,
        "full_pipeline_attempts_panel",
    )

    fig, axes = new_stacked_figure(
        3,
        gap=0.022,
    )

    draw_rate(
        axes[0],
        records,
        x_key,
        xlabel,
    )

    draw_runtime(
        axes[1],
        records,
        x_key,
        xlabel,
    )

    draw_attempts(
        axes[2],
        records,
        x_key,
        xlabel,
    )

    if xlim:
        for ax in axes:
            ax.set_xlim(
                *xlim
            )

    # The stacked figure uses new Axes objects. Keep fixed ranges only for
    # panels that are NOT using trimmed auto-scaling.
    if RATE_YLIM is not None:
        axes[0].set_ylim(
            *RATE_YLIM,
            auto=False,
        )

    if RUNTIME_YLIM is not None:
        axes[1].set_ylim(
            *RUNTIME_YLIM,
            auto=False,
        )

    if ATTEMPTS_YLIM is not None:
        axes[2].set_ylim(
            *ATTEMPTS_YLIM,
            auto=False,
        )

    axes[0].set_xlabel("")
    axes[1].set_xlabel("")

    save_figure(
        fig,
        outdir,
        "full_pipeline_three_panel_figure",
    )

    print(
        f"Saved standardized full-pipeline panels to {outdir}"
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--records-csv",
        required=True,
    )

    p.add_argument(
        "--output-dir",
        default="outputs/full_pipeline_replot_panels",
    )

    p.add_argument(
        "--x-metric",
        choices=(
            "combined",
            "routing",
            "actual",
        ),
        default="combined",
    )

    a = p.parse_args()

    csv_path = Path(
        a.records_csv
    ).expanduser()

    if not csv_path.exists():
        raise FileNotFoundError(
            csv_path
        )

    replot(
        csv_path,
        Path(
            a.output_dir
        ).expanduser(),
        a.x_metric,
    )


if __name__ == "__main__":
    main()

# python .\full_pipeline_replot_panels.py `
#    --records-csv ".\outputs\full_pipeline_scalability\full_pipeline_results.csv" `
#    --output-dir ".\outputs\full_pipeline_scalability" `
