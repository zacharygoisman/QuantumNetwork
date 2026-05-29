# plotting/composite.py
"""
Publication-width composite figure for Overleaf.

Renders a single final-size vector figure containing the three project
panels in one canvas:

    1. ring network plot,
    2. per-link utility bar plot, and
    3. source allocation cap plot.

The saved figure is already at the target column width (default 6.5 in), so
it can be included in LaTeX with no Inkscape rescaling.

Layout selection is done through :class:`_PaperLayout`. Two presets are
provided:

* ``"stacked"`` — three rows, network on top.
* ``"side"`` — network on the left, utility and allocation stacked on the right.
"""

# ZHG
# 2026.03.26
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

from .base import (
    GEM,
    NULL_LINK_GRAY,
    bar_link_label,
    canonical_link_tuple,
    entity_mathtext,
    format_float,
    gem_colors,
    option_link_label,
    ordered_link_keys_from_options,
    per_link_log_utility,
    safe_float,
    safe_int,
    sorted_options_by_link,
)
from .network_ring import _layout as _ring_layout
from .network_ring import _node_color_map
from .utility import _best_result_to_source_rows


# --------------------------------------------------------------------------- #
# Publication defaults
# --------------------------------------------------------------------------- #

PAPER_WIDTH_IN = 6.5
PAPER_FONT_SIZE = 8.0
PAPER_SMALL_FONT_SIZE = 7.0
PAPER_LINEWIDTH = 0.55
PAPER_DPI = 300

# Cap-shape sampling defaults for the source-allocation panel.
_CAP_PLATEAU_FRAC = 0.55
_CAP_EDGE_POWER = 5.0
_CAP_CUTOFF_FRAC = 0.90
_CAP_PROFILE_SAMPLES = 160
_CAP_UCUT_SAMPLES = 2001


def _set_paper_style(font_size: float = PAPER_FONT_SIZE) -> None:
    """Apply the rcParams used by the publication composite figure."""
    small = max(5.5, font_size - 1.0)
    mpl.rcParams.update({
        "figure.dpi": PAPER_DPI,
        "savefig.dpi": PAPER_DPI,
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": small,
        "ytick.labelsize": small,
        "legend.fontsize": small,
        "mathtext.default": "regular",
        "axes.unicode_minus": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #

def _link_key_for_option(option: dict[str, Any]) -> tuple[str, str] | None:
    return canonical_link_tuple(option.get("link") or option.get("users"))


def _link_label_from_key(key: tuple[str, str]) -> str:
    return f"Link {key[0]}{key[1]}"


def _color_by_link(best_result: dict[str, Any] | None) -> dict[tuple[str, str], str]:
    """Stable color assignment keyed by canonical link tuple."""
    if best_result is None:
        return {}
    keys = ordered_link_keys_from_options(best_result.get("combo", []))
    return {key: color for key, color in zip(keys, gem_colors(len(keys), start=1))}


# --------------------------------------------------------------------------- #
# Network panel
# --------------------------------------------------------------------------- #

def _draw_network_panel(
    ax: Axes,
    network: nx.Graph,
    best_result: dict[str, Any] | None,
    *,
    font_size: float,
    edge_label_size: float,
    node_size: float,
    show_edge_labels: bool = True,
    y_stretch: float = 1.15,
) -> None:
    combo = sorted_options_by_link(best_result.get("combo", [])) if best_result else []
    color_by_link = _color_by_link(best_result)

    # Stretch the ring vertically so the panel uses the available height.
    pos = {node: (xy[0], xy[1] * y_stretch) for node, xy in dict(_ring_layout(network)).items()}

    nx.draw_networkx_nodes(
        network,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=_node_color_map(network),
        edgecolors="black",
        linewidths=0.55,
    )

    # Light gray background edges, then per-link colored overlays for the chosen paths.
    nx.draw_networkx_edges(
        network,
        pos,
        ax=ax,
        width=0.45,
        edge_color="#6A6A6A",
        alpha=1.0,
        connectionstyle="arc3,rad=0.03",
    )

    nx.draw_networkx_labels(
        network,
        pos,
        labels={n: entity_mathtext(str(n)) for n in network.nodes()},
        ax=ax,
        font_size=font_size,
        font_weight="bold",
    )

    for option in combo:
        key = _link_key_for_option(option)
        color = color_by_link.get(key, "black")
        p1, p2 = option.get("path1", []), option.get("path2", [])
        edges = list(zip(p1, p1[1:])) + list(zip(p2, p2[1:]))
        nx.draw_networkx_edges(
            network,
            pos,
            edgelist=edges,
            ax=ax,
            width=1.15,
            edge_color=color,
            alpha=0.82,
            connectionstyle="arc3,rad=0.03",
        )

    if show_edge_labels:
        edge_labels = {
            (u, v): format_float(safe_float(data.get("loss", 1.0), 1.0), ".2f")
            for u, v, data in network.edges(data=True)
            if u != v
        }
        texts = nx.draw_networkx_edge_labels(
            network,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=edge_label_size,
            rotate=True,
            label_pos=0.5,
        )
        for text in texts.values():
            text.set_zorder(4)
            text.set_path_effects([pe.withStroke(linewidth=1.4, foreground="white")])

    ax.set_axis_off()
    ax.margins(0.05)


# --------------------------------------------------------------------------- #
# Utility panel
# --------------------------------------------------------------------------- #

def _utility_rows(best_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Build per-link rows for the utility bar plot, ordered like the network panel."""
    if best_result is None:
        return []

    allocation = best_result.get("allocation", {}) or {}
    rows: list[dict[str, Any]] = []
    for idx, opt in enumerate(best_result.get("combo", [])):
        alloc = allocation.get(id(opt), {}) if allocation else {}
        if not alloc and "allocation" in opt:
            alloc = opt["allocation"]

        link = opt.get("link") or opt.get("users") or option_link_label(opt)
        rows.append({
            "link": link,
            "link_key": canonical_link_tuple(link),
            "combo_index": idx,
            "link_utility": per_link_log_utility(opt, alloc),
            "link_ub": safe_float(opt.get("link_ub"), float("nan")),
        })

    rank = {k: i for i, k in enumerate(ordered_link_keys_from_options(best_result.get("combo", [])))}
    rows.sort(key=lambda r: rank.get(r.get("link_key"), 10**9))
    return rows


def _draw_utility_panel(
    ax: Axes,
    best_result: dict[str, Any] | None,
    *,
    font_size: float,
    tick_size: float,
    show_legend: bool = True,
) -> None:
    rows = _utility_rows(best_result)
    if not rows:
        ax.axis("off")
        return

    links = [r["link"] for r in rows]
    ub = np.array([safe_float(r.get("link_ub")) for r in rows], dtype=float)
    ac = np.array([safe_float(r.get("link_utility")) for r in rows], dtype=float)

    finite_vals = np.concatenate([ub[np.isfinite(ub)], ac[np.isfinite(ac)]])
    if finite_vals.size == 0:
        ax.axis("off")
        return

    x = np.arange(len(links), dtype=float)
    width = 0.34
    data_min = float(np.min(finite_vals))
    data_max = float(np.max(finite_vals))
    y_bottom = min(0.0, data_min)
    y_top = max(0.0, data_max)
    if data_min < 0.0:
        y_bottom -= 0.35  # Reserve a little headroom below the lowest negative bar.

    # Bars are drawn relative to ``y_bottom`` so that the visual baseline matches
    # the lowest finite value rather than zero.
    ax.bar(
        x - width / 2,
        np.where(np.isfinite(ub), ub - y_bottom, np.nan),
        width=width,
        bottom=y_bottom,
        facecolor="white",
        edgecolor="black",
        linewidth=PAPER_LINEWIDTH,
        label="Upper bound",
    )
    ax.bar(
        x + width / 2,
        np.where(np.isfinite(ac), ac - y_bottom, np.nan),
        width=width,
        bottom=y_bottom,
        color=GEM[0],
        edgecolor="black",
        linewidth=PAPER_LINEWIDTH,
        label="Actual",
    )

    pad = 0.05 * max(1.0, y_top - y_bottom)
    ax.set_ylim(y_bottom - pad, y_top + pad)

    ax.set_xticks(x)
    rotate = 0 if len(links) <= 6 else 35
    ha = "center" if rotate == 0 else "right"
    ax.set_xticklabels(
        [bar_link_label(t) for t in links], rotation=rotate, ha=ha, fontsize=tick_size,
    )

    yticks = np.linspace(y_bottom, y_top, 3)
    ax.set_yticks(yticks)
    ax.set_yticklabels([format_float(t, ".2f") for t in yticks], fontsize=tick_size)
    ax.set_ylabel(r"$\log_{10}\mathcal{R}_{\ell}$", fontsize=font_size, labelpad=2)

    ax.tick_params(axis="x", which="both", bottom=False, top=False, length=0, pad=1)
    ax.tick_params(axis="y", which="both", left=True, right=False,
                   length=2.5, width=PAPER_LINEWIDTH, pad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(PAPER_LINEWIDTH)
    ax.spines["bottom"].set_linewidth(PAPER_LINEWIDTH)
    ax.margins(x=0.04)

    if show_legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.24),
            ncol=2,
            frameon=True,
            fontsize=max(5.5, font_size - 1.5),
            borderpad=0.25,
            handlelength=1.2,
            columnspacing=0.8,
        )


# --------------------------------------------------------------------------- #
# Source allocation panel
# --------------------------------------------------------------------------- #

def _profile_sinc2_flattop(u, plateau_frac=_CAP_PLATEAU_FRAC, lobes=0.0,
                           edge_power=_CAP_EDGE_POWER):
    """Flat-top profile with a raised-cosine taper, used to shape the cap envelope."""
    u_in = u
    u = np.atleast_1d(np.clip(u, 0.0, 1.0)).astype(float)
    p = float(np.clip(plateau_frac, 0.0, 0.99))
    y = np.empty_like(u)

    flat = u <= p
    y[flat] = 1.0

    edge = ~flat
    if np.any(edge):
        ue = (u[edge] - p) / (1.0 - p)
        taper = 0.5 * (1.0 + np.cos(np.pi * ue))
        y[edge] = taper ** max(1.0, edge_power)

    return y.item() if np.isscalar(u_in) else y


def _compute_ucut(cutoff_frac, *, plateau_frac, edge_power,
                  samples=_CAP_UCUT_SAMPLES) -> float:
    """Largest ``u`` for which the cap profile is still ``>= cutoff_frac``."""
    if cutoff_frac is None or cutoff_frac is True or cutoff_frac <= 0:
        return 1.0
    u = np.linspace(0.0, 1.0, samples)
    y = _profile_sinc2_flattop(u, plateau_frac=plateau_frac, edge_power=edge_power)
    idx = np.where(y >= float(cutoff_frac))[0]
    return float(u[idx[-1]]) if idx.size else 0.0


def _sample_half_cap(xc, half_w, height, side, *,
                     samples=_CAP_PROFILE_SAMPLES,
                     plateau_frac=_CAP_PLATEAU_FRAC,
                     edge_power=_CAP_EDGE_POWER) -> MplPath:
    """Sample one half (left or right) of the cap-shaped clip path."""
    u = np.linspace(0.0, 1.0, samples)
    yfrac = _profile_sinc2_flattop(u, plateau_frac=plateau_frac, edge_power=edge_power)
    y = height * yfrac

    if side == "right":
        x = xc + u * half_w
        verts = [(xc, 0.0), (xc, y[0]), *zip(x[1:], y[1:]), (xc + half_w, 0.0), (xc, 0.0)]
    else:
        x = xc - u * half_w
        verts = [(xc, 0.0), (xc, y[0]), *zip(x[1:], y[1:]), (xc - half_w, 0.0), (xc, 0.0)]

    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


def _source_sort_key(label: Any) -> tuple[int, str]:
    """Numeric-aware sort for source labels: ``S1 < S2 < S10``."""
    nums = re.findall(r"\d+", str(label))
    return (int(nums[0]) if nums else 10**9, str(label))


def _counts_for_source(entry: dict[str, Any], K: int) -> list[tuple[str, int]]:
    """Channel-count pairs ``(link_label, count)`` for one source, padded with Null Link."""
    pairs: list[tuple[str, int]] = []
    for lk, k in zip(entry.get("links", []), entry.get("channel_allocation", [])):
        key = canonical_link_tuple(lk) or (str(lk[0]), str(lk[1]))
        pairs.append((_link_label_from_key(key), max(0, safe_int(k))))

    used = sum(c for _, c in pairs)
    if used < K:
        pairs.append(("Null Link", K - used))
    return pairs


def _draw_source_allocation_panel(
    ax: Axes,
    best_result: dict[str, Any] | None,
    sources: dict[str, Any],
    *,
    font_size: float,
    legend_size: float,
    show_legend: bool = True,
    side: str = "right",
    height: float = 1.0,
    base_width: float = 0.86,
    gap: float = 0.30,
    plateau_frac: float = _CAP_PLATEAU_FRAC,
    edge_power: float = _CAP_EDGE_POWER,
    cutoff_frac: float = _CAP_CUTOFF_FRAC,
) -> None:
    rows = _best_result_to_source_rows(best_result)
    if not rows and not sources:
        ax.axis("off")
        return

    # Stable label order: ordered by canonical link, then Null Link last.
    link_order = ordered_link_keys_from_options(best_result.get("combo", [])) if best_result else []
    link_labels = [_link_label_from_key(k) for k in link_order]
    if "Null Link" not in link_labels:
        link_labels.append("Null Link")

    link_to_color = {
        label: color
        for label, color in zip(
            [x for x in link_labels if x != "Null Link"],
            gem_colors(len(link_labels), start=1),
        )
    }
    link_to_color["Null Link"] = NULL_LINK_GRAY

    src_names = sorted(list(sources.keys()), key=_source_sort_key)
    entry_by_source = {str(e["source"]): e for e in rows}

    centers = np.arange(len(src_names), dtype=float) * (base_width + gap)
    if centers.size == 0:
        ax.axis("off")
        return

    half_w = base_width / 2.0
    u_cut = _compute_ucut(cutoff_frac, plateau_frac=plateau_frac, edge_power=edge_power)
    used_set: set[str] = set()

    for i, sname in enumerate(src_names):
        entry = entry_by_source.get(
            str(sname),
            {"source": str(sname), "links": [], "channel_allocation": []},
        )
        K = len(sources[sname]["available_channels"])
        pairs = _counts_for_source(entry, K)

        labels: list[str] = []
        for label, count in pairs:
            labels.extend([label] * int(count))
        labels = labels[:K]
        used_set.update(labels)

        xc = centers[i]
        cap_path = _sample_half_cap(
            xc, half_w, height, side,
            plateau_frac=plateau_frac, edge_power=edge_power,
        )
        cap_patch = PathPatch(
            cap_path,
            facecolor="none",
            edgecolor="black",
            lw=PAPER_LINEWIDTH,
            antialiased=True,
            zorder=1,
        )
        ax.add_patch(cap_patch)

        if side == "right":
            x_left, x_right = xc, xc + u_cut * half_w
        else:
            x_left, x_right = xc - u_cut * half_w, xc
        label_x = 0.5 * (x_left + x_right)

        if labels:
            edges = np.linspace(x_left, x_right, len(labels) + 1)
            eps = half_w * 1e-4  # tiny overlap to suppress hairline gaps between rectangles
            for j, label in enumerate(labels):
                left = edges[j] - (eps if j != 0 else 0.0)
                right = edges[j + 1] + (eps if j != len(labels) - 1 else 0.0)
                rect = Rectangle(
                    (left, 0.0),
                    right - left,
                    height,
                    facecolor=link_to_color.get(label, NULL_LINK_GRAY),
                    edgecolor="black",
                    linewidth=PAPER_LINEWIDTH,
                    antialiased=True,
                    zorder=2,
                )
                rect.set_clip_path(cap_patch)
                ax.add_patch(rect)

        ax.text(
            label_x,
            -0.13 * height,
            entity_mathtext(str(sname)),
            ha="center",
            va="top",
            fontsize=font_size,
        )

    ax.set_xlim(centers[0] - half_w - 0.08, centers[-1] + half_w + 0.08)
    ax.set_ylim(-0.30 * height, 1.08 * height)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    if show_legend:
        # Show links that received any channels; place "Unassigned" first when present.
        used = [lab for lab in link_labels if lab in used_set and lab != "Null Link"]
        if "Null Link" in used_set:
            used = ["Null Link"] + used

        handles: list[Patch] = []
        for label in used:
            if label == "Null Link":
                handles.append(Patch(facecolor=link_to_color[label], edgecolor="black",
                                     label="Unassigned"))
            else:
                users = re.findall(r"(?i)[us](?:er)?\d+", label)
                if len(users) >= 2:
                    nice = f"Link {entity_mathtext(users[0])}{entity_mathtext(users[1])}"
                else:
                    nice = label
                handles.append(Patch(facecolor=link_to_color[label], edgecolor="black",
                                     label=nice))

        if handles:
            ncol = min(4, max(1, len(handles)))
            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.22),
                ncol=ncol,
                frameon=True,
                fontsize=legend_size,
                borderpad=0.25,
                handlelength=1.0,
                columnspacing=0.7,
            )


# --------------------------------------------------------------------------- #
# Layout configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _PaperLayout:
    """Per-layout panel geometry for the composite figure."""
    height_in: float
    node_size: float
    source_base_width: float
    source_gap: float
    network_y_stretch: float


_LAYOUTS: dict[str, _PaperLayout] = {
    "stacked": _PaperLayout(
        height_in=6.15,
        node_size=280,
        source_base_width=0.72,
        source_gap=0.23,
        network_y_stretch=1.16,
    ),
    "side": _PaperLayout(
        height_in=3.95,
        node_size=210,
        source_base_width=0.62,
        source_gap=0.20,
        network_y_stretch=1.28,
    ),
}


def _build_axes(fig, layout: str) -> tuple[Axes, Axes, Axes]:
    """Create the three panel axes according to the chosen layout."""
    if layout == "stacked":
        gs = GridSpec(3, 1, figure=fig, height_ratios=[3.25, 1.05, 1.05], hspace=0.18)
        return fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2, 0])

    gs = GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.35, 1.0],
        height_ratios=[1.05, 0.95],
        wspace=0.12, hspace=0.22,
    )
    return fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])


# --------------------------------------------------------------------------- #
# Public function
# --------------------------------------------------------------------------- #

def plot_paper_combined_solution_ring(
    network: nx.Graph,
    best_result: dict[str, Any] | None,
    sources: dict[str, Any],
    outdir: str | Path = "outputs",
    filename: str = "combined_solution_paper.pdf",
    *,
    width_inches: float = PAPER_WIDTH_IN,
    height_inches: float | None = None,
    font_size: float = PAPER_FONT_SIZE,
    layout: str = "stacked",
    show_network_edge_labels: bool = True,
    show_utility_legend: bool = True,
    show_source_legend: bool = True,
) -> Path | None:
    r"""
    Save a final-size composite figure for Overleaf.

    Recommended LaTeX usage for the default width::

        \includegraphics{combined_solution_paper.pdf}

    Since the figure is already 6.5 in wide, do not rescale it in Inkscape.
    If you use ``\includegraphics[width=\textwidth]{...}``, make sure
    ``\textwidth`` is also approximately 6.5 in; otherwise LaTeX will scale
    the fonts as well.

    Parameters
    ----------
    layout : {"stacked", "side"}
        Panel arrangement. ``"stacked"`` places network on top with the
        utility and source panels below. ``"side"`` places the network on
        the left and the other two stacked on the right.
    """
    if best_result is None:
        return None

    if layout not in _LAYOUTS:
        raise ValueError(f"layout must be one of {sorted(_LAYOUTS)}")

    _set_paper_style(font_size)
    tick_size = max(5.5, font_size - 1.0)
    legend_size = max(5.5, font_size - 1.2)
    edge_label_size = max(5.5, font_size - 1.5)

    cfg = _LAYOUTS[layout]
    fig_height = cfg.height_in if height_inches is None else height_inches
    fig = plt.figure(figsize=(width_inches, fig_height), constrained_layout=True)
    ax_net, ax_util, ax_src = _build_axes(fig, layout)

    _draw_network_panel(
        ax_net, network, best_result,
        font_size=font_size,
        edge_label_size=edge_label_size,
        node_size=cfg.node_size,
        show_edge_labels=show_network_edge_labels,
        y_stretch=cfg.network_y_stretch,
    )
    _draw_utility_panel(
        ax_util, best_result,
        font_size=font_size,
        tick_size=tick_size,
        show_legend=show_utility_legend,
    )
    _draw_source_allocation_panel(
        ax_src, best_result, sources,
        font_size=font_size,
        legend_size=legend_size,
        show_legend=show_source_legend,
        base_width=cfg.source_base_width,
        gap=cfg.source_gap,
    )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename

    # Do not pass bbox_inches="tight" here: keeping the exact canvas size is
    # what makes the figure predictable in Overleaf.
    fig.savefig(path, dpi=PAPER_DPI)
    plt.close(fig)
    return path
