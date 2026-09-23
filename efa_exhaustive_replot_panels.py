"""
Replot the EFA/APOPT validation figure with standardized paper-panel styling.

This is a plotting-first script:
- load an existing CSV
- regenerate publication-quality single-panel outputs
- also generate one stacked two-panel figure directly

Why use this script?
--------------------
It avoids the common alignment issues caused by tight_layout() and
bbox_inches="tight", and it centralizes the styling so the same visual
language can be reused in the Double Yen and full-pipeline figure scripts.

Outputs
-------
Separate panels:
    efa_rate_panel.png / .pdf
    efa_runtime_panel.png / .pdf

Stacked two-panel figure:
    efa_two_panel_figure.png / .pdf
"""

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

COLOR_APOPT = "#0072B2"
COLOR_EXACT = "#D55E00"
COLOR_REFERENCE = "#555555"

RAW_ALPHA = 0.30


def _optional_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def _optional_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return int(text)


def _load_records(csv_path: Path):
    records = []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            record = dict(row)

            for key in ("channels", "links", "num_allocations"):
                record[key] = _optional_int(row.get(key))

            for key in (
                "gap",
                "avg_rate_pct_exact",
                "objective_rate_pct_exact",
                "rate_retention_pct",
                "exact_objective",
                "apopt_objective",
                "apopt_time_ms",
                "exact_time_ms",
            ):
                record[key] = _optional_float(row.get(key))

            # Pipeline-style arithmetic metric: new CSVs store it explicitly.
            if record.get("avg_rate_pct_exact") is None:
                record["avg_rate_pct_exact"] = record.get(
                    "rate_retention_pct"
                )

            # Objective-aligned metric can be reconstructed exactly from the
            # two objective values:
            #
            #   100 * 10^[(U_APOPT - U_exact)/L]
            #
            # This makes the replotter compatible with older CSVs that were
            # written before objective_rate_pct_exact was added.
            if (
                record.get("objective_rate_pct_exact") is None
                and record.get("apopt_objective") is not None
                and record.get("exact_objective") is not None
                and record.get("links") is not None
                and int(record["links"]) > 0
            ):
                record["objective_rate_pct_exact"] = (
                    100.0
                    * 10.0 ** (
                        (
                            float(record["apopt_objective"])
                            - float(record["exact_objective"])
                        )
                        / float(record["links"])
                    )
                )

            records.append(record)

    return records


def _group_median(records, x_key, y_key):
    grouped = {}

    for record in records:
        x = record.get(x_key)
        y = record.get(y_key)

        if x is None or y is None:
            continue

        x = int(x)
        y = float(y)

        if not math.isfinite(y):
            continue

        grouped.setdefault(x, []).append(y)

    xs = sorted(grouped)
    ys = [statistics.median(grouped[x]) for x in xs]

    return xs, ys


def _valid_rate_records(
    records,
    rate_metric,
):
    """
    Keep only rows that actually contain the selected rate metric.

    This matters because older CSVs may contain the pipeline-style arithmetic
    metric but not the newer objective-aligned metric, and failed solver cases
    can legitimately have no rate comparison at all.
    """
    valid = []

    for record in records:
        x = record.get("num_allocations")
        y = record.get(rate_metric)

        if x is None or y is None:
            continue

        try:
            x = int(x)
            y = float(y)
        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            continue

        if x <= 0 or not math.isfinite(y):
            continue

        valid.append(record)

    return valid


def _valid_runtime_records(records):
    return [
        record
        for record in records
        if record.get("num_allocations") is not None
        and record["num_allocations"] > 0
        and (
            (record.get("apopt_time_ms") is not None and record["apopt_time_ms"] > 0.0)
            or
            (record.get("exact_time_ms") is not None and record["exact_time_ms"] > 0.0)
        )
    ]


def _draw_rate_panel(ax, records, rate_metric):
    valid = _valid_rate_records(records, rate_metric)

    if not valid:
        raise ValueError(
            f"No usable rows contain rate metric '{rate_metric}'. "
            "If this is an older CSV, make sure it contains "
            "exact_objective, apopt_objective, and links so the objective "
            "metric can be reconstructed."
        )

    x_raw = [int(r["num_allocations"]) for r in valid]
    y_raw = [float(r[rate_metric]) for r in valid]

    ax.scatter(
        x_raw,
        y_raw,
        s=DEFAULT_STYLE.marker_size,
        alpha=RAW_ALPHA,
        color=COLOR_APOPT,
        edgecolors="none",
        zorder=2,
    )

    xs, ys = _group_median(valid, "num_allocations", rate_metric)

    ax.plot(
        xs,
        ys,
        marker="o",
        markersize=DEFAULT_STYLE.trend_marker_size,
        linewidth=DEFAULT_STYLE.trend_line_width,
        color=COLOR_APOPT,
        label="APOPT",
        zorder=3,
    )

    ax.axhline(
        100.0,
        linestyle="--",
        linewidth=DEFAULT_STYLE.reference_line_width,
        color=COLOR_REFERENCE,
        label="Exhaustive reference",
        zorder=1,
    )

    ax.axvline(
        1.0e4,
        linestyle=":",
        linewidth=DEFAULT_STYLE.reference_line_width,
        color=COLOR_REFERENCE,
        label=r"Exact/APOPT threshold ($10^4$)",
        zorder=1,
    )

    style_axis(
        ax,
        xlabel=r"Number of integer EFA allocations, $\binom{K}{L}$",
        ylabel=(
            ""
            if rate_metric == "objective_rate_pct_exact"
            else "Average individual link rate\n(% of exhaustive rate)"
        ),
        xscale="log",
        yscale="linear",
    )

    # Match the Double-Yen geometric-mean panel exactly.
    ax.set_ylim(80.0, 101.0)

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

    # For the EFA geometric-mean panel, keep y tick marks but hide
    # the y-axis numbers.
    ax.tick_params(
        axis="y",
        which="both",
        left=True,
        right=False,
        labelleft=False,
    )

    ax.legend(loc="lower left", frameon=False)


def _draw_runtime_panel(ax, records):
    valid = _valid_runtime_records(records)

    apopt_valid = [r for r in valid if r.get("apopt_time_ms") is not None and r["apopt_time_ms"] > 0.0]
    exact_valid = [r for r in valid if r.get("exact_time_ms") is not None and r["exact_time_ms"] > 0.0]

    ax.scatter(
        [int(r["num_allocations"]) for r in apopt_valid],
        [float(r["apopt_time_ms"]) / 1000.0 for r in apopt_valid],
        s=DEFAULT_STYLE.marker_size,
        alpha=RAW_ALPHA,
        color=COLOR_APOPT,
        edgecolors="none",
        zorder=2,
    )

    xs_a, ys_a = _group_median(
        [{"num_allocations": r["num_allocations"], "value": float(r["apopt_time_ms"]) / 1000.0} for r in apopt_valid],
        "num_allocations",
        "value",
    )

    ax.plot(
        xs_a,
        ys_a,
        marker="o",
        markersize=DEFAULT_STYLE.trend_marker_size,
        linewidth=DEFAULT_STYLE.trend_line_width,
        color=COLOR_APOPT,
        label="APOPT",
        zorder=3,
    )

    ax.scatter(
        [int(r["num_allocations"]) for r in exact_valid],
        [float(r["exact_time_ms"]) / 1000.0 for r in exact_valid],
        s=DEFAULT_STYLE.marker_size,
        alpha=RAW_ALPHA,
        color=COLOR_EXACT,
        marker="s",
        edgecolors="none",
        zorder=2,
    )

    xs_e, ys_e = _group_median(
        [{"num_allocations": r["num_allocations"], "value": float(r["exact_time_ms"]) / 1000.0} for r in exact_valid],
        "num_allocations",
        "value",
    )

    ax.plot(
        xs_e,
        ys_e,
        marker="s",
        markersize=DEFAULT_STYLE.trend_marker_size,
        linewidth=DEFAULT_STYLE.trend_line_width,
        color=COLOR_EXACT,
        label="Exhaustive",
        zorder=3,
    )

    ax.axvline(
        1.0e4,
        linestyle=":",
        linewidth=DEFAULT_STYLE.reference_line_width,
        color=COLOR_REFERENCE,
        label=r"Exact/APOPT threshold ($10^4$)",
        zorder=1,
    )

    style_axis(
        ax,
        xlabel=r"Number of integer EFA allocations, $\binom{K}{L}$",
        ylabel="",
        xscale="log",
        yscale="log",
    )

    # Match the Double-Yen runtime panel exactly.
    ax.set_ylim(1.0e-4, 1.0e3)

    ax.grid(False, which="both", axis="both")

    ax.legend(loc="lower left", frameon=False)


def replot(csv_path: Path, outdir: Path, rate_metric: str):
    records = _load_records(csv_path)

    x_values = [
        r["num_allocations"]
        for r in records
        if r.get("num_allocations") is not None and r["num_allocations"] > 0
    ]

    xlim = compute_log_limits(x_values)

    fig_rate, ax_rate = new_panel_figure()
    _draw_rate_panel(ax_rate, records, rate_metric)
    if xlim is not None:
        ax_rate.set_xlim(*xlim)
    save_figure(fig_rate, outdir, "efa_rate_panel")

    fig_runtime, ax_runtime = new_panel_figure()
    _draw_runtime_panel(ax_runtime, records)
    if xlim is not None:
        ax_runtime.set_xlim(*xlim)
    save_figure(fig_runtime, outdir, "efa_runtime_panel")

    fig_stack, axes = new_stacked_figure(nrows=2, gap=0.03)
    ax1, ax2 = axes
    _draw_rate_panel(ax1, records, rate_metric)
    _draw_runtime_panel(ax2, records)
    if xlim is not None:
        ax1.set_xlim(*xlim)
        ax2.set_xlim(*xlim)
    ax1.set_xlabel("")
    save_figure(fig_stack, outdir, "efa_two_panel_figure")

    print(f"Saved standardized panels to {outdir}")


def main():
    parser = argparse.ArgumentParser(description="Replot EFA paper panels with standardized styling.")
    parser.add_argument("--records-csv", type=str, required=True, help="Path to efa_validation_results.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/efa_replot_panels")
    parser.add_argument(
        "--rate-metric",
        choices=["objective", "pipeline"],
        default="objective",
        help=(
            "objective: objective-aligned geometric-mean rate retention; "
            "pipeline: arithmetic mean of individual APOPT/exhaustive link-rate ratios."
        ),
    )
    args = parser.parse_args()

    csv_path = Path(args.records_csv).expanduser()
    outdir = Path(args.output_dir).expanduser()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rate_metric = (
        "objective_rate_pct_exact"
        if args.rate_metric == "objective"
        else "avg_rate_pct_exact"
    )

    replot(
        csv_path,
        outdir,
        rate_metric,
    )


if __name__ == "__main__":
    main()

# python .\efa_exhaustive_replot_panels.py `
#    --records-csv ".\outputs\efa_exhaustive\efa_validation_results.csv" `
#    --output-dir ".\outputs\efa_exhaustive"