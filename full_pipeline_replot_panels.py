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
TOPOLOGY_STYLE = {
    "dense": {"color": "#0072B2", "marker": "o", "label": "Dense"},
    # "ring": {"color": "#D55E00", "marker": "s", "label": "Ring"},
    # "star": {"color": "#009E73", "marker": "^", "label": "Star"},
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

            if (r.get("combined_complexity") is None
                    and r.get("routing_complexity") is not None
                    and r.get("efa_complexity") is not None):
                r["combined_complexity"] = r["routing_complexity"] * r["efa_complexity"]

            if (r.get("actual_combined_complexity") is None
                    and r.get("actual_route_combos") is not None
                    and r.get("efa_complexity") is not None):
                r["actual_combined_complexity"] = r["actual_route_combos"] * r["efa_complexity"]

            rows.append(r)
    return rows


def x_info(metric):
    if metric == "combined":
        return "combined_complexity", r"Nominal pipeline complexity, $(SN^2)^L \binom {K}{L}$"
    if metric == "routing":
        return "routing_complexity", r"Nominal routing combinations, $(SN^2)^L$"
    if metric == "actual":
        return "actual_combined_complexity", r"Actual candidate complexity, $N_{\mathrm{route}} \binom {K}{L}$"
    raise ValueError(metric)


def valid(records, x_key, y_key, positive=False):
    out = []
    for r in records:
        x, y = r.get(x_key), r.get(y_key)
        if x is None or y is None:
            continue
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(x) or not math.isfinite(y) or x <= 0:
            continue
        if positive and y <= 0:
            continue
        out.append(r)
    return out


def grouped_median(records, x_key, y_key, topology):
    g = {}
    for r in records:
        if r.get("topology") != topology:
            continue
        x, y = r.get(x_key), r.get(y_key)
        if x is None or y is None:
            continue
        g.setdefault(int(x), []).append(float(y))
    xs = sorted(g)
    return xs, [statistics.median(g[x]) for x in xs]


def draw_topologies(ax, records, x_key, y_key):
    for topo in ("dense"):
        style = TOPOLOGY_STYLE[topo]
        subset = [r for r in records if r.get("topology") == topo]
        if not subset:
            continue
        ax.scatter([float(r[x_key]) for r in subset], [float(r[y_key]) for r in subset],
                   s=DEFAULT_STYLE.marker_size, alpha=RAW_ALPHA,
                   color=style["color"], marker=style["marker"],
                   edgecolors="none", zorder=2)
        xs, ys = grouped_median(subset, x_key, y_key, topo)
        if xs:
            ax.plot(xs, ys, color=style["color"], marker=style["marker"],
                    markersize=DEFAULT_STYLE.trend_marker_size,
                    linewidth=DEFAULT_STYLE.trend_line_width,
                    label=style["label"], zorder=3)


def draw_rate(ax, records, x_key, xlabel):
    v = valid(records, x_key, "rate_pct_upper_bound", positive=True)
    print(f"Rate scatter points: {len(v)}")
    if not v:
        raise ValueError("No valid rate records.")
    draw_topologies(ax, v, x_key, "rate_pct_upper_bound")
    ax.axhline(100.0, linestyle="--", linewidth=DEFAULT_STYLE.reference_line_width,
               color="#555555", zorder=1)
    style_axis(ax, xlabel=xlabel,
               ylabel="Geometric-mean link rate\n(% of infinite-resource upper bound)",
               xscale="log", yscale="linear")
    rate_values = [float(r["rate_pct_upper_bound"]) for r in v]
    ymin = min(rate_values)
    ymax = max(rate_values)
    lower = max(0.0, math.floor((ymin - 2.0) / 5.0) * 5.0)
    if lower >= 95.0:
        lower = 95.0
    upper = max(101.0, math.ceil((ymax + 1.0) / 5.0) * 5.0)
    ax.set_ylim(lower, upper)
    ax.legend(loc="lower left", frameon=False)


def draw_runtime(ax, records, x_key, xlabel):
    v = valid(records, x_key, "total_pipeline_time_s", positive=True)
    if not v:
        raise ValueError("No valid runtime records.")
    draw_topologies(ax, v, x_key, "total_pipeline_time_s")
    style_axis(ax, xlabel=xlabel, ylabel="Total pipeline runtime (s)",
               xscale="log", yscale="log")
    ax.legend(loc="best", frameon=False)


def draw_attempts(ax, records, x_key, xlabel):
    """
    Plot every non-valid route configuration returned by evaluate_stream.

    New CSVs contain rejected_combos directly.  For older CSVs, reconstruct it
    as combos_returned - valid_combos.
    """
    prepared = []

    for r in records:
        rejected = r.get("rejected_combos")

        if rejected is None:
            total = r.get("combos_returned")
            valid_count = r.get("valid_combos")

            if total is not None and valid_count is not None:
                rejected = int(total) - int(valid_count)

        if rejected is None:
            continue

        rr = dict(r)
        rr["rejected_plot"] = float(rejected)
        prepared.append(rr)

    v = valid(
        prepared,
        x_key,
        "rejected_plot",
        positive=False,
    )

    if not v:
        raise ValueError(
            "No valid rejected-combination records."
        )

    draw_topologies(
        ax,
        v,
        x_key,
        "rejected_plot",
    )

    style_axis(
        ax,
        xlabel=xlabel,
        ylabel=(
            "Rejected route configurations\n"
            "(all evaluated combinations)"
        ),
        xscale="log",
        yscale="symlog",
    )

    ax.set_yscale(
        "symlog",
        linthresh=1.0,
    )

    ax.legend(
        loc="best",
        frameon=False,
    )


def replot(csv_path: Path, outdir: Path, metric: str):
    records = load_records(csv_path)

    # Remove all ring-topology results
    records = [
        r for r in records
        if r.get("topology") != "ring"
    ]

    x_key, xlabel = x_info(metric)
    xvals = [r[x_key] for r in records if r.get(x_key) is not None and float(r[x_key]) > 0]
    xlim = compute_log_limits(xvals)

    fig, ax = new_panel_figure()
    draw_rate(ax, records, x_key, xlabel)
    if xlim: ax.set_xlim(*xlim)
    save_figure(fig, outdir, "full_pipeline_rate_panel")

    fig, ax = new_panel_figure()
    draw_runtime(ax, records, x_key, xlabel)
    if xlim: ax.set_xlim(*xlim)
    save_figure(fig, outdir, "full_pipeline_runtime_panel")

    fig, ax = new_panel_figure()
    draw_attempts(ax, records, x_key, xlabel)
    if xlim: ax.set_xlim(*xlim)
    save_figure(fig, outdir, "full_pipeline_attempts_panel")

    fig, axes = new_stacked_figure(3, gap=0.022)
    draw_rate(axes[0], records, x_key, xlabel)
    draw_runtime(axes[1], records, x_key, xlabel)
    draw_attempts(axes[2], records, x_key, xlabel)
    if xlim:
        for ax in axes: ax.set_xlim(*xlim)
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    save_figure(fig, outdir, "full_pipeline_three_panel_figure")
    print(f"Saved standardized full-pipeline panels to {outdir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records-csv", required=True)
    p.add_argument("--output-dir", default="outputs/full_pipeline_replot_panels")
    p.add_argument("--x-metric", choices=("combined", "routing", "actual"), default="combined")
    a = p.parse_args()
    csv_path = Path(a.records_csv).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    replot(csv_path, Path(a.output_dir).expanduser(), a.x_metric)


if __name__ == "__main__":
    main()

# python .\full_pipeline_replot_panels.py `
#    --records-csv ".\outputs\full_pipeline_scalability\full_pipeline_results.csv" `
#    --output-dir ".\outputs\full_pipeline_scalability" `
