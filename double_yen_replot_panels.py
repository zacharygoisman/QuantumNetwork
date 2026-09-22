"""
Replot Double-Yen validation using ONLY N=1 runs.

Uses:
    x = actual number of exhaustive routing combinations evaluated
    y = geometric-mean link-rate retention relative to exhaustive routing

Requires:
    paper_panel_utils.py
"""

from __future__ import annotations

import argparse
import csv
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

COLOR_YEN = "#0072B2"
COLOR_EXACT = "#D55E00"
COLOR_REFERENCE = "#555555"
RAW_ALPHA = 0.30


def _to_float(x):
    if x is None or str(x).strip() == "":
        return None
    return float(x)


def _to_int(x):
    if x is None or str(x).strip() == "":
        return None
    return int(x)


def load_records(path: Path):
    records = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = dict(row)

            for key in (
                "actual_exact_combos",
                "actual_yen_combos",
                "links",
                "n_paths",
            ):
                r[key] = _to_int(row.get(key))

            for key in (
                "rate_retention_pct",
                "yen_utility",
                "exact_utility",
                "yen_time_s",
                "exact_time_s",
            ):
                r[key] = _to_float(row.get(key))

            # Recompute the geometric-mean paper metric from the utilities.
            if (
                r.get("yen_utility") is not None
                and r.get("exact_utility") is not None
                and r.get("links") is not None
                and r["links"] > 0
            ):
                r["rate_retention_pct"] = (
                    100.0
                    * 10.0 ** (
                        (r["yen_utility"] - r["exact_utility"])
                        / float(r["links"])
                    )
                )

            records.append(r)

    return records


def _median_by_x(records, y_key):
    grouped = {}
    for r in records:
        x = r.get("actual_exact_combos")
        y = r.get(y_key)
        if x is None or y is None or x <= 0:
            continue
        grouped.setdefault(x, []).append(float(y))

    xs = sorted(grouped)
    return xs, [statistics.median(grouped[x]) for x in xs]


def _draw_rate(ax, records):
    valid = [
        r for r in records
        if r.get("actual_exact_combos") is not None
        and r.get("rate_retention_pct") is not None
        and r["actual_exact_combos"] > 0
    ]

    x = [r["actual_exact_combos"] for r in valid]
    y = [r["rate_retention_pct"] for r in valid]

    ax.scatter(
        x,
        y,
        s=DEFAULT_STYLE.marker_size,
        alpha=RAW_ALPHA,
        color=COLOR_YEN,
        edgecolors="none",
        zorder=2,
    )

    xs, ys = _median_by_x(valid, "rate_retention_pct")
    ax.plot(
        xs,
        ys,
        marker="o",
        markersize=DEFAULT_STYLE.trend_marker_size,
        linewidth=DEFAULT_STYLE.trend_line_width,
        color=COLOR_YEN,
        label="Double Yen, N=1",
        zorder=3,
    )

    ax.axhline(
        100.0,
        linestyle="--",
        linewidth=DEFAULT_STYLE.reference_line_width,
        color=COLOR_REFERENCE,
        label="Exhaustive routing",
        zorder=1,
    )

    style_axis(
        ax,
        xlabel="Exhaustive route configurations",
        ylabel=r"Geometric mean rate $\rho_{opt}$ (%)",
        xscale="log",
        yscale="linear",
    )

    # Fixed paper-panel range.  Leave a small gap above the 100% reference.
    ax.set_ylim(
        80.0,
        101.0,
    )

    # Remove the grid, but keep tick marks.
    ax.grid(False, which="both", axis="both")

    # The runtime panel below carries the shared x-axis values/title.
    # Keep x tick marks, but hide their numbers.
    ax.set_xlabel("")
    ax.tick_params(
        axis="x",
        which="both",
        bottom=True,
        top=False,
        labelbottom=False,
    )

    # Keep the y-axis numbers on the Double-Yen geometric-mean panel.
    ax.tick_params(
        axis="y",
        which="both",
        left=True,
        right=False,
        labelleft=True,
    )

    ax.legend(loc="best", frameon=False)


def _draw_runtime(ax, records):
    yen = [
        r for r in records
        if r.get("actual_exact_combos") is not None
        and r.get("yen_time_s") is not None
        and r["actual_exact_combos"] > 0
        and r["yen_time_s"] > 0
    ]

    exact = [
        r for r in records
        if r.get("actual_exact_combos") is not None
        and r.get("exact_time_s") is not None
        and r["actual_exact_combos"] > 0
        and r["exact_time_s"] > 0
    ]

    ax.scatter(
        [r["actual_exact_combos"] for r in yen],
        [r["yen_time_s"] for r in yen],
        s=DEFAULT_STYLE.marker_size,
        alpha=RAW_ALPHA,
        color=COLOR_YEN,
        edgecolors="none",
    )

    xs, ys = _median_by_x(yen, "yen_time_s")
    ax.plot(
        xs,
        ys,
        marker="o",
        markersize=DEFAULT_STYLE.trend_marker_size,
        linewidth=DEFAULT_STYLE.trend_line_width,
        color=COLOR_YEN,
        label="Double Yen, N=1",
    )

    ax.scatter(
        [r["actual_exact_combos"] for r in exact],
        [r["exact_time_s"] for r in exact],
        s=DEFAULT_STYLE.marker_size,
        alpha=RAW_ALPHA,
        color=COLOR_EXACT,
        edgecolors="none",
    )

    xs, ys = _median_by_x(exact, "exact_time_s")
    ax.plot(
        xs,
        ys,
        marker="s",
        markersize=DEFAULT_STYLE.trend_marker_size,
        linewidth=DEFAULT_STYLE.trend_line_width,
        color=COLOR_EXACT,
        label="Exhaustive routing",
    )

    style_axis(
        ax,
        xlabel="Exhaustive route configurations",
        ylabel="Runtime (s)",
        xscale="log",
        yscale="log",
    )

    ax.grid(False, which="both", axis="both")

    ax.legend(loc="best", frameon=False)


def replot(csv_path: Path, outdir: Path):
    all_records = load_records(csv_path)

    # Filter ONCE here so every panel and x-axis limit uses N=1 only.
    records = [
        r for r in all_records
        if r.get("n_paths") == 1
    ]

    if not records:
        raise ValueError(
            "No N=1 records found in the CSV (expected n_paths == 1)."
        )

    print(
        f"Loaded {len(all_records)} total rows; "
        f"plotting {len(records)} rows with N=1."
    )

    xvals = [
        r["actual_exact_combos"]
        for r in records
        if r.get("actual_exact_combos") is not None
        and r["actual_exact_combos"] > 0
    ]
    # Fixed shared log-scale x-axis for both panels.
    # A logarithmic axis cannot start at zero, so use 1 as the left bound.
    xlim = (1.0, 2.0e4)

    fig, ax = new_panel_figure()
    _draw_rate(ax, records)
    if xlim:
        ax.set_xlim(*xlim)
    save_figure(fig, outdir, "double_yen_n1_rate_panel")

    fig, ax = new_panel_figure()
    _draw_runtime(ax, records)
    if xlim:
        ax.set_xlim(*xlim)
    save_figure(fig, outdir, "double_yen_n1_runtime_panel")

    fig, axes = new_stacked_figure(2, gap=0.03)
    _draw_rate(axes[0], records)
    _draw_runtime(axes[1], records)
    if xlim:
        for ax in axes:
            ax.set_xlim(*xlim)
    axes[0].set_xlabel("")
    save_figure(fig, outdir, "double_yen_n1_two_panel_figure")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records-csv", required=True)
    p.add_argument(
        "--output-dir",
        default="outputs/double_yen_n1_replot_panels",
    )
    args = p.parse_args()
    replot(Path(args.records_csv), Path(args.output_dir))


if __name__ == "__main__":
    main()



# python .\double_yen_replot_panels.py `
#    --records-csv ".\outputs\double_yen_exhaustive\double_yen_validation_results.csv" `
#    --output-dir ".\outputs\double_yen_exhaustive"