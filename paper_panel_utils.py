"""
Reusable plotting utilities for paper-quality panel figures.

Goal
----
Keep all panel figures visually identical across scripts:
- same canvas size
- same actual axes box size
- same tick/label font sizes
- same mathtext font
- same line widths / grid / spine widths
- same export behavior

Most important design choice
----------------------------
Do NOT use tight_layout() or bbox_inches="tight" for standalone panels.
Those cause different amounts of trimming for different y-labels, legends,
and tick-label widths, which is why separate panels often fail to align
when stacked in a paper.

Instead this module creates a fixed axes rectangle inside a fixed-size
figure canvas, then saves without tight cropping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PanelStyle:
    figure_width: float = 6.5
    figure_height: float = 2.55   # less tall for paper
    left: float = 0.18
    right: float = 0.98
    bottom: float = 0.22
    top: float = 0.95

    font_family: str = "DejaVu Sans"
    math_fontset: str = "dejavusans"

    base_fontsize: float = 15
    axis_label_size: float = 15
    tick_label_size: float = 14
    legend_size: float = 14

    spine_width: float = 1.1
    grid_width: float = 0.75
    grid_alpha: float = 0.22
    tick_width_major: float = 1.1
    tick_width_minor: float = 0.95
    tick_length_major: float = 4.6
    tick_length_minor: float = 2.8

    marker_size: float = 30
    trend_marker_size: float = 5.5
    trend_line_width: float = 1.9
    reference_line_width: float = 1.5

    png_dpi: int = 600


DEFAULT_STYLE = PanelStyle()


def apply_global_style(style: PanelStyle = DEFAULT_STYLE) -> None:
    plt.rcParams.update(
        {
            "font.family": style.font_family,
            "font.size": style.base_fontsize,
            "axes.labelsize": style.axis_label_size,
            "xtick.labelsize": style.tick_label_size,
            "ytick.labelsize": style.tick_label_size,
            "legend.fontsize": style.legend_size,
            "mathtext.fontset": style.math_fontset,
            "text.usetex": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def new_panel_figure(style: PanelStyle = DEFAULT_STYLE):
    """
    Create a single panel with a fixed axes box.  This is the key to allowing
    seamless manual stacking later.
    """
    apply_global_style(style)

    fig = plt.figure(
        figsize=(style.figure_width, style.figure_height),
        facecolor="white",
    )

    ax = fig.add_axes(
        [
            style.left,
            style.bottom,
            style.right - style.left,
            style.top - style.bottom,
        ]
    )

    return fig, ax


def new_stacked_figure(
    nrows: int,
    style: PanelStyle = DEFAULT_STYLE,
    gap: float = 0.035,
):
    """
    Create a vertically stacked figure with identical axes widths / heights.
    """
    apply_global_style(style)

    total_height = nrows * style.figure_height

    fig = plt.figure(
        figsize=(style.figure_width, total_height),
        facecolor="white",
    )

    axes = []

    panel_h_frac = style.figure_height / total_height
    bottom_frac = style.bottom * panel_h_frac
    top_frac = style.top * panel_h_frac
    usable_panel_h = top_frac - bottom_frac

    current_top = 1.0

    for _ in range(nrows):
        top = current_top - (1.0 - top_frac)
        bottom = top - usable_panel_h

        ax = fig.add_axes(
            [
                style.left,
                bottom,
                style.right - style.left,
                usable_panel_h,
            ]
        )
        axes.append(ax)
        current_top = bottom - gap

    return fig, axes


def style_axis(
    ax,
    style: PanelStyle = DEFAULT_STYLE,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
):
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=style.tick_label_size,
        width=style.tick_width_major,
        length=style.tick_length_major,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        width=style.tick_width_minor,
        length=style.tick_length_minor,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(style.spine_width)

    ax.grid(
        True,
        which="major",
        linewidth=style.grid_width,
        alpha=style.grid_alpha,
    )


def save_figure(
    fig,
    outdir: str | Path,
    stem: str,
    style: PanelStyle = DEFAULT_STYLE,
) -> None:
    """
    Save to PNG and PDF WITHOUT tight cropping.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    png_path = outdir / f"{stem}.png"
    pdf_path = outdir / f"{stem}.svg"

    fig.savefig(
        png_path,
        dpi=style.png_dpi,
        facecolor="white",
        edgecolor="white",
    )

    fig.savefig(
        pdf_path,
        facecolor="white",
        edgecolor="white",
    )


def compute_log_limits(
    values,
    lower_pad_decades: float = 0.03,
    upper_pad_decades: float = 0.03,
):
    valid = [
        float(v)
        for v in values
        if v is not None and float(v) > 0.0 and math.isfinite(float(v))
    ]

    if not valid:
        return None

    log_min = math.log10(min(valid))
    log_max = math.log10(max(valid))

    if log_min == log_max:
        log_min -= 0.1
        log_max += 0.1

    return (
        10.0 ** (log_min - lower_pad_decades),
        10.0 ** (log_max + upper_pad_decades),
    )
