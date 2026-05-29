# plotting/utility.py
"""
Per-link utility bars, scatter comparison and source-allocation cap plot.

This module produces the three standalone (non-composite) figures saved
during a normal pipeline run:

* :func:`plot_link_utility_bars` — paired upper-bound vs actual log-rate bars.
* :func:`plot_utility_comparison` — scatter plot of every evaluated combo.
* :func:`plot_source_allocation` — cap-shaped per-source channel allocation.

Plus :func:`fidelity_rate`, an analytical fidelity/rate plot used for
parameter exploration.
"""

# ZHG
# 2026.03.26
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

from .base import (
    GEM,
    NULL_LINK_GRAY,
    PLOT_LABEL_SIZE,
    PLOT_LEGEND_SIZE,
    PLOT_TICK_SIZE,
    bar_link_label,
    canonical_link_tuple,
    entity_mathtext,
    format_float,
    gem_colors,
    make_figure,
    option_link_label,
    ordered_link_keys_from_options,
    per_link_log_utility,
    safe_float,
    safe_int,
    save_figure,
)


# --------------------------------------------------------------------------- #
# Cap-profile sampling defaults (source allocation plot)
# --------------------------------------------------------------------------- #

_CAP_PLATEAU_FRAC = 0.55
_CAP_EDGE_POWER = 5.0
_CAP_CUTOFF_FRAC = 0.90
_CAP_PROFILE_SAMPLES = 240
_CAP_UCUT_SAMPLES = 2001


# --------------------------------------------------------------------------- #
# Shared row builders
# --------------------------------------------------------------------------- #

def _ordered_link_keys(best_result: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Canonical link order from a best-result combo."""
    if best_result is None:
        return []
    return ordered_link_keys_from_options(best_result.get("combo", []))


def _best_rows(best_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """One row per link in ``best_result.combo`` with utility / upper-bound fields."""
    if best_result is None:
        return []

    combo = list(best_result.get("combo", []))
    allocation = best_result.get("allocation", {}) or {}
    rows: list[dict[str, Any]] = []

    for idx, opt in enumerate(combo):
        alloc = allocation.get(id(opt), {}) if allocation else {}
        if not alloc and "allocation" in opt:
            alloc = opt["allocation"]

        link = opt.get("link") or opt.get("users") or option_link_label(opt)
        rows.append({
            "link": link,
            "link_key": canonical_link_tuple(link),
            "combo_index": idx,
            "link_utility": per_link_log_utility(opt, alloc),
            "path_ub": safe_float(opt.get("path_ub"), float("nan")),
            "link_ub": safe_float(opt.get("link_ub"), float("nan")),
            "k": alloc.get("k"),
            "mu": safe_float(alloc.get("mu"), float("nan")),
        })

    return rows

# --------------------------------------------------------------------------- #
# Public: link utility bars
# --------------------------------------------------------------------------- #
def plot_link_utility_bars(cfg,
    best_result: dict[str, Any] | None,
    outdir: str | Path = "outputs",
    filename: str = "link_utility_bars.svg",
):
    """Paired bar plot: upper-bound (white) vs actual (color) log-rate per link."""
    rows = _best_rows(best_result)
    if not rows:
        return None

    desired_order = _ordered_link_keys(best_result)
    if desired_order:
        rank = {k: i for i, k in enumerate(desired_order)}
        rows.sort(key=lambda r: rank.get(r.get("link_key"), 10**9))

    links = [r["link"] for r in rows]
    ub = np.array([safe_float(r.get("link_ub")) for r in rows], dtype=float)
    ac = np.array([safe_float(r.get("link_utility")) for r in rows], dtype=float)

    if not np.any(np.isfinite(ub) | np.isfinite(ac)):
        return None

    # Add extra horizontal spacing between groups when there are many links so
    # the labels stay directly under the bar pairs.
    spacing = 1.5 if len(links) > 10 else 1.0
    x = np.arange(len(links), dtype=float) * spacing
    w = 0.34

    if cfg.topology == "ring":
        fig, ax = make_figure(figsize=(20, 5))
    else:
        fig, ax = make_figure(figsize=(12, 3.8))

    finite_vals = np.concatenate([ub[np.isfinite(ub)], ac[np.isfinite(ac)]])
    data_min = float(np.min(finite_vals))
    data_max = float(np.max(finite_vals))

    y_bottom = min(0.0, data_min)
    y_top = max(0.0, data_max)
    if data_min < 0.0:
        y_bottom -= 1.0  # extra headroom below the lowest negative bar

    ub_heights = np.where(np.isfinite(ub), ub - y_bottom, np.nan)
    ac_heights = np.where(np.isfinite(ac), ac - y_bottom, np.nan)

    ax.bar(
        x - w / 2,
        ub_heights,
        width=w,
        bottom=y_bottom,
        facecolor="white",
        edgecolor="black",
        linewidth=1.3,
        label="Upper bound utility",
    )
    ax.bar(
        x + w / 2,
        ac_heights,
        width=w,
        bottom=y_bottom,
        color=GEM[0],
        edgecolor="black",
        linewidth=1.0,
        label="Actual utility",
    )

    pad = 0.02 * max(1.0, y_top - y_bottom)
    ax.set_ylim(y_bottom - pad, y_top + pad)

    yticks = np.linspace(y_bottom, y_top, 3)
    ax.set_yticks(yticks)
    ax.set_yticklabels([format_float(t, ".2f") for t in yticks], fontsize=PLOT_TICK_SIZE)
    ax.tick_params(axis="y", labelsize=PLOT_TICK_SIZE)

    ax.set_xticks(x)
    ax.set_xticklabels([bar_link_label(t) for t in links], ha="center", fontsize=PLOT_TICK_SIZE)
    try:
        fig.subplots_adjust(bottom=0.16)
    except Exception:
        pass
    ax.set_ylabel(r"$\log_{10}\mathcal{R}_{\ell}$", fontsize=PLOT_LABEL_SIZE)

    # No x-tick marks; keep y-tick marks.
    ax.tick_params(axis="x", which="both", bottom=False, top=False, length=0)
    ax.tick_params(axis="y", which="both", left=True, right=False, length=6, width=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.10)
    ax.legend(fontsize=PLOT_LEGEND_SIZE, frameon=True)

    return save_figure(fig, outdir, filename)


# --------------------------------------------------------------------------- #
# Public: utility comparison scatter
# --------------------------------------------------------------------------- #

def plot_utility_comparison(
    results: list[dict[str, Any]] | None,
    outdir: str | Path = "outputs",
    filename: str = "utility_comparison.svg",
):
    """Scatter every evaluated combo by index, colored by feasibility / pruning state."""
    if not results:
        return None

    x_ok, y_ok = [], []
    x_fail, y_fail = [], []
    x_pruned, y_pruned = [], []
    dashed_limit = float("-inf")

    for idx, result in enumerate(results, start=1):
        util = safe_float(result.get("utility"), float("nan"))
        ub_path = safe_float(result.get("combo_path_ub"), float("nan"))

        if np.isfinite(ub_path):
            dashed_limit = max(dashed_limit, ub_path)

        reason = result.get("reason", "")
        valid = bool(result.get("valid"))

        if valid and np.isfinite(util):
            x_ok.append(idx)
            y_ok.append(util)
            continue

        # Failed / pruned: prefer the path upper bound as the y value, fall back to util.
        y = ub_path if np.isfinite(ub_path) else util
        if not np.isfinite(y):
            continue

        if reason == "ub_pruned":
            x_pruned.append(idx)
            y_pruned.append(y)
        else:
            x_fail.append(idx)
            y_fail.append(y)

    if not (x_ok or x_fail or x_pruned):
        return None

    fig, ax = make_figure(figsize=(9, 5.5))

    if x_pruned:
        ax.scatter(x_pruned, y_pruned, marker="x", linewidth=1.2, s=22,
                   label="UB-pruned", color="gray")
    if x_fail:
        ax.scatter(x_fail, y_fail, marker="o", linewidth=1.0, s=18,
                   label="Failed", color="#de762dff")
    if x_ok:
        ax.scatter(x_ok, y_ok, marker="o", linewidth=1.0, s=18,
                   label="Feasible", color="#1a9d96ff")

    if np.isfinite(dashed_limit):
        eps = 1e-9 * max(1.0, abs(dashed_limit))
        ax.axhline(dashed_limit + eps, linestyle="--", linewidth=1.75,
                   label="Max possible", color="k")

    ax.set_xlabel("Path Combination", fontsize=PLOT_LABEL_SIZE)
    ax.set_ylabel(r"$\log_{10}\mathcal{R}_{\ell}$", fontsize=PLOT_LABEL_SIZE)
    ax.legend(loc="best")

    return save_figure(fig, outdir, filename)


# --------------------------------------------------------------------------- #
# Source allocation: best-result -> per-source row format
# --------------------------------------------------------------------------- #

def _best_result_to_source_rows(best_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """
    Convert the pipeline's best_result format into the source-allocation-style format.

    From::

        {
            "combo": [...],
            "allocation": { id(opt): {"k": ..., "mu": ...}, ... }
        }

    To::

        [
            {"source": "S0",
             "links": [("U1", "U4"), ...],
             "channel_allocation": [k1, ...]},
            ...
        ]
    """
    if best_result is None:
        return []

    combo = list(best_result.get("combo", []))
    allocation = best_result.get("allocation", {}) or {}

    grouped: dict[str, dict[str, Any]] = {}
    for opt in combo:
        source = str(opt.get("source"))
        link = opt.get("link") or opt.get("users")
        if not (isinstance(link, (tuple, list)) and len(link) == 2):
            continue

        alloc = allocation.get(id(opt), {}) if allocation else {}
        if not alloc and "allocation" in opt:
            alloc = opt["allocation"]
        k = safe_int(alloc.get("k"), 0)

        entry = grouped.setdefault(
            source,
            {"source": source, "links": [], "channel_allocation": []},
        )
        entry["links"].append(canonical_link_tuple(link) or (str(link[0]), str(link[1])))
        entry["channel_allocation"].append(k)

    return list(grouped.values())


# --------------------------------------------------------------------------- #
# Cap profiles
# --------------------------------------------------------------------------- #

def _profile_sinc2_flattop(u, plateau_frac=_CAP_PLATEAU_FRAC, lobes=0.0,
                           edge_power=_CAP_EDGE_POWER):
    """Flat-top profile with raised-cosine taper (and optional sinc^2 ripple)."""
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

        if lobes and lobes > 0:
            x = ue * np.pi * lobes
            with np.errstate(divide="ignore", invalid="ignore"):
                s = np.sin(x) / np.where(x == 0.0, 1.0, x)
            ripple = s * s
            denom = ripple[0] if ripple.size > 0 else 1.0
            ripple = ripple / (denom if denom != 0 else 1.0)
            taper = taper * (0.85 + 0.15 * ripple)

        y[edge] = taper ** max(1.0, edge_power)

    return y.item() if np.isscalar(u_in) else y


def _profile_value(u, profile="sinc2_flattop", **kw):
    """Scalar evaluation of any supported cap profile at ``u``."""
    if profile == "sinc2_flattop":
        return float(_profile_sinc2_flattop(
            u,
            plateau_frac=kw.get("plateau_frac", _CAP_PLATEAU_FRAC),
            lobes=kw.get("lobes", 0.0),
            edge_power=kw.get("edge_power", _CAP_EDGE_POWER),
        ))
    if profile == "linear":
        return float(1.0 - np.clip(u, 0.0, 1.0))
    if profile == "gaussian":
        edge_floor = kw.get("edge_floor", 0.02)
        sigma = 1.0 / np.sqrt(2.0 * np.log(1.0 / max(edge_floor, 1e-9)))
        return float(np.exp(-0.5 * (np.clip(u, 0.0, 1.0) / sigma) ** 2))
    raise ValueError("Unknown profile")


def _compute_ucut(cutoff_frac, profile, *, plateau_frac, edge_power, lobes,
                  samples=_CAP_UCUT_SAMPLES) -> float:
    """Largest ``u`` with profile value ``>= cutoff_frac``."""
    if cutoff_frac is None or cutoff_frac is True or cutoff_frac <= 0:
        return 1.0

    u = np.linspace(0.0, 1.0, samples)
    if profile == "sinc2_flattop":
        y = _profile_sinc2_flattop(u, plateau_frac=plateau_frac, lobes=lobes,
                                   edge_power=edge_power)
    elif profile == "linear":
        y = 1.0 - u
    elif profile == "gaussian":
        y = np.array([_profile_value(ui, "gaussian", edge_floor=0.02) for ui in u])
    else:
        raise ValueError("Unknown profile")

    idx = np.where(y >= float(cutoff_frac))[0]
    return float(u[idx[-1]]) if idx.size else 0.0


def _sample_half_cap(xc, half_w, height, side, profile="sinc2_flattop",
                     samples=_CAP_PROFILE_SAMPLES, **profile_kw) -> MplPath:
    """Sample one half (left or right) of the cap-shaped clip path."""
    u = np.linspace(0.0, 1.0, samples)

    if profile == "sinc2_flattop":
        yfrac = _profile_sinc2_flattop(u, **profile_kw)
    elif profile == "linear":
        yfrac = 1.0 - u
    elif profile == "gaussian":
        yfrac = np.array([_profile_value(ui, "gaussian", **profile_kw) for ui in u])
    else:
        raise ValueError("profile must be 'sinc2_flattop'|'linear'|'gaussian'")

    y = height * yfrac
    if side == "right":
        x = xc + u * half_w
        verts = [(xc, 0.0), (xc, y[0]), *zip(x[1:], y[1:]), (xc + half_w, 0.0), (xc, 0.0)]
    else:
        x = xc - u * half_w
        verts = [(xc, 0.0), (xc, y[0]), *zip(x[1:], y[1:]), (xc - half_w, 0.0), (xc, 0.0)]

    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


# --------------------------------------------------------------------------- #
# Source allocation: core renderer
# --------------------------------------------------------------------------- #

def source_allocation(
    previous_best_results,
    sources,
    freqs_by_link=None,
    *,
    link_order=None,
    manual_bins=None,
    side="right",
    profile="sinc2_flattop",
    plateau_frac=_CAP_PLATEAU_FRAC,
    edge_power=_CAP_EDGE_POWER,
    lobes=0.0,
    cutoff_frac=_CAP_CUTOFF_FRAC,
    base_width=2.55,
    height=1.0,
    gap=0.35,
    axes_height=1.0,
    samples=_CAP_PROFILE_SAMPLES,
    dpi=180,
    title="",
    bars_per_side=None,
    include_center=False,
    cap_lw=1.5,
    bar_lw=None,
):
    """Render the per-source channel-allocation cap plot. Returns ``(fig, ax)``."""
    if bar_lw is None:
        bar_lw = cap_lw

    # ----- nested helpers (closures over the call's parameters) ------------ #

    def _link_colors(labels):
        non_null = [lab for lab in labels if lab != "Null Link"]
        colors = {lab: c for lab, c in zip(non_null, gem_colors(len(non_null), start=1))}
        colors["Null Link"] = NULL_LINK_GRAY
        return colors

    def _counts_for_source(entry, K):
        """Channel-count pairs ``(label, count)`` for one source, padded with Null Link."""
        if freqs_by_link:
            link_to_source = {
                tuple(lk): e["source"]
                for e in previous_best_results
                for lk in e.get("links", [])
            }
            counts: dict[str, int] = {}
            for (u1, u2), (to_u1, to_u2) in freqs_by_link.items():
                if link_to_source.get((u1, u2)) != entry["source"]:
                    continue
                key = f"Link {u1}{u2}"
                counts[key] = counts.get(key, 0) + len(
                    set(abs(int(x)) for x in list(to_u1) + list(to_u2))
                )
            pairs = [
                (f"Link {u1}{u2}", int(counts.get(f"Link {u1}{u2}", 0)))
                for (u1, u2) in entry.get("links", [])
            ]
        else:
            pairs = [
                (f"Link {lk[0]}{lk[1]}", max(0, safe_int(k)))
                for lk, k in zip(entry.get("links", []), entry.get("channel_allocation", []))
            ]

        used = sum(c for _, c in pairs)
        if used < K:
            pairs.append(("Null Link", K - used))
        return pairs

    def _make_half_labels(K, pairs):
        """Split a sequence of labels symmetrically across the two cap halves."""
        k_right = int(np.ceil(K / 2))
        k_left = K - k_right
        right = ["Null Link"] * k_right
        left = ["Null Link"] * k_left

        seq: list[str] = []
        for lbl, c in pairs:
            seq.extend([lbl] * int(c))
        seq = seq[:K]

        p = 0
        if K % 2 == 1 and p < len(seq):
            right[0] = seq[p]
            p += 1

        layer = 0
        while p < len(seq) and layer < min(k_right - (K % 2), k_left):
            lbl = seq[p]
            p += 1
            right[(K % 2) + layer] = lbl
            left[layer] = lbl
            layer += 1

        while p < len(seq) and (K % 2) + layer < k_right:
            right[(K % 2) + layer] = seq[p]
            p += 1
            layer += 1

        return right, left

    def _force_bar_count(labels, *, half_side, K):
        """Trim or pad ``labels`` to honor the user-provided ``bars_per_side``."""
        if bars_per_side is None:
            return labels
        n = int(bars_per_side)
        if n <= 0:
            return []

        has_center = (K % 2 == 1) and len(labels) > 0
        if include_center and has_center:
            if half_side == "right":
                want = 1 + n
                return (labels[:want] + ["Null Link"] * want)[:want]
            labels = labels[1:] if labels else labels
            return (labels[:n] + ["Null Link"] * n)[:n]

        return (labels[:n] + ["Null Link"] * n)[:n]

    # ----- order sources & links ------------------------------------------- #

    src_names = sorted(
        list(sources.keys()),
        key=lambda x: (re.sub(r"\D", "", str(x)).zfill(8), str(x)),
    )
    entry_by_source = {str(e["source"]): e for e in previous_best_results}

    if link_order:
        link_labels = [f"Link {u}{v}" for (u, v) in link_order]
    else:
        link_labels = []
        seen: set[str] = set()
        for e in previous_best_results:
            for (u, v) in e.get("links", []):
                lab = f"Link {u}{v}"
                if lab not in seen:
                    seen.add(lab)
                    link_labels.append(lab)

    if freqs_by_link:
        for (u1, u2) in sorted(
            freqs_by_link.keys(),
            key=lambda lk: canonical_link_tuple(lk) or (str(lk[0]), str(lk[1])),
        ):
            u1, u2 = canonical_link_tuple((u1, u2)) or (str(u1), str(u2))
            lab = f"Link {u1}{u2}"
            if lab not in link_labels:
                link_labels.append(lab)

    if "Null Link" not in link_labels:
        link_labels.append("Null Link")

    link_to_color = _link_colors(link_labels)

    # ----- geometry & per-source labels ------------------------------------ #

    half_w = base_width / 2.0
    span = base_width if side == "both" else base_width * 0.85
    centers = np.arange(len(src_names)) * (span + gap)

    per_source: list[tuple[str, int, list[str], list[str]]] = []
    for sname in src_names:
        entry = entry_by_source.get(
            str(sname),
            {"source": str(sname), "links": [], "channel_allocation": []},
        )
        K = len(sources[sname]["available_channels"])
        pairs = _counts_for_source(entry, K)

        seq: list[str] = []
        for lbl, c in pairs:
            seq.extend([lbl] * int(c))
        seq = seq[:K]

        if side == "right":
            right_labels, left_labels = seq, []
        elif side == "left":
            right_labels, left_labels = [], seq
        else:
            right_labels, left_labels = _make_half_labels(K, pairs)
            if K % 2 == 1:
                left_labels = [right_labels[0]] + left_labels

        per_source.append((str(sname), K, right_labels, left_labels))

    # Reduce horizontal stretching so sources are not overly spaced out for many
    # sources. Previously used 4.6 * n which produced excessive whitespace.
    fig_w = max(7.5, 1.6 * max(1, len(src_names)))
    fig_h = 2.9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_ylim(0, axes_height)
    ax.set_xlim(centers[0] - span / 2 - 0.05, centers[-1] + span / 2 + 0.05)

    # ----- draw each source ------------------------------------------------ #

    u_cut = _compute_ucut(
        cutoff_frac, profile,
        plateau_frac=plateau_frac, edge_power=edge_power, lobes=lobes,
    )

    for i, (sname, K, right_labels, left_labels) in enumerate(per_source):
        xc = centers[i]
        halves: list[tuple[str, list[str]]] = []
        if side in ("right", "both"):
            halves.append(("right", right_labels))
        if side in ("left", "both"):
            halves.append(("left", left_labels))

        for half_side, labels in halves:
            labels = _force_bar_count(labels, half_side=half_side, K=K)

            cap_path = _sample_half_cap(
                xc, half_w, height, half_side,
                profile=profile, samples=samples,
                plateau_frac=plateau_frac,
                edge_power=edge_power,
                lobes=lobes,
            )
            cap_patch = PathPatch(
                cap_path,
                facecolor="none",
                edgecolor="black",
                lw=float(cap_lw),
                antialiased=True,
                zorder=1,
            )
            ax.add_patch(cap_patch)

            k = len(labels)
            if k == 0:
                continue

            if half_side == "right":
                x_left, x_right = xc, xc + u_cut * half_w
            else:
                x_left, x_right = xc - u_cut * half_w, xc

            edges = np.linspace(x_left, x_right, k + 1)
            eps = half_w * 1e-4  # tiny overlap to suppress hairline gaps

            for j, lbl in enumerate(labels):
                left = edges[j] - (eps if j != 0 else 0.0)
                right = edges[j + 1] + (eps if j != k - 1 else 0.0)
                rect = Rectangle(
                    (left, 0.0),
                    right - left,
                    height,
                    facecolor=link_to_color.get(lbl, (0.6, 0.6, 0.6)),
                    edgecolor="black",
                    linewidth=float(bar_lw),
                    antialiased=True,
                    zorder=2,
                )
                rect.set_clip_path(cap_patch)
                ax.add_patch(rect)

        if side == "right":
            label_x = xc + (u_cut * 0.5 * half_w)
        elif side == "left":
            label_x = xc - (u_cut * 0.5 * half_w)
        else:
            label_x = xc

        ax.text(
            label_x,
            -0.12 * height,
            entity_mathtext(sname),
            ha="center",
            va="top",
            fontsize=mpl.rcParams["axes.labelsize"],
        )

    # ----- legend & framing ------------------------------------------------ #

    used_set: set[str] = set()
    for _, _, rlabels, llabels in per_source:
        used_set.update(rlabels)
        used_set.update(llabels)

    used = [lab for lab in link_labels if lab in used_set and lab != "Null Link"]
    if "Null Link" in used_set:
        used = ["Null Link"] + used

    if used:
        patches: list[Patch] = []
        for label in used:
            if label == "Null Link":
                patches.append(Patch(facecolor=link_to_color[label], edgecolor="black",
                                     label="Unassigned"))
            else:
                m = re.findall(r"(?i)[us](?:er)?\d+", label)
                nice = (
                    f"Link {entity_mathtext(m[0])}{entity_mathtext(m[1])}"
                    if len(m) >= 2 else label
                )
                patches.append(Patch(facecolor=link_to_color[label], edgecolor="black",
                                     label=nice))

        ax.legend(
            handles=patches,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0.0,
            frameon=True,
            fontsize=mpl.rcParams["legend.fontsize"],
        )
        fig.subplots_adjust(left=0.03, right=0.84, bottom=0.16, top=0.96)
    else:
        fig.subplots_adjust(left=0.03, right=0.99, bottom=0.16, top=0.96)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    if title:
        ax.set_title(title, pad=4, fontsize=mpl.rcParams["axes.titlesize"])
    plt.tight_layout(pad=0.15)
    return fig, ax


# --------------------------------------------------------------------------- #
# Source allocation: save wrappers
# --------------------------------------------------------------------------- #

def save_source_allocation(
    previous_best_results,
    sources,
    freqs_by_link=None,
    outdir: str | Path = "outputs",
    filename: str = "source_allocation.svg",
    **kwargs,
):
    """Render with :func:`source_allocation` and save the figure to disk."""
    fig, _ = source_allocation(
        previous_best_results=previous_best_results,
        sources=sources,
        freqs_by_link=freqs_by_link,
        **kwargs,
    )
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_source_allocation(cfg,
    best_result: dict[str, Any] | None,
    sources: dict[str, Any],
    freqs_by_link: dict[tuple[str, str], tuple[list[int], list[int]]] | None = None,
    outdir: str | Path = "outputs",
    filename: str = "source_allocation.svg",
    **kwargs,
):
    """Convert a ``best_result`` to source rows then render and save the cap plot."""
    previous_best_results = _best_result_to_source_rows(best_result)
    if not previous_best_results and not sources:
        return None

    fig, _ = source_allocation(
        previous_best_results=previous_best_results,
        sources=sources,
        freqs_by_link=freqs_by_link,
        link_order=_ordered_link_keys(best_result),
        **kwargs,
    )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Analytical fidelity / rate exploration
# --------------------------------------------------------------------------- #

def fidelity_rate(
    y_1: float = 0.05,
    y_2: float = 0.10,
    fidelity_min: float = 0.80,
    outdir: str | Path = "outputs",
    filename: str = "fidelity_rate.svg",
):
    """
    Plot the analytical fidelity and rate curves vs flux, and mark the
    largest flux at which fidelity meets ``fidelity_min``.
    """
    flux = np.linspace(0, 1, 1000)
    rate = flux ** 2 + (2 * y_1 + 2 * y_2 + 1) * flux + 4 * y_1 * y_2
    fidelity = 0.25 * (1 + 3 * flux / rate)

    fig, ax = make_figure(figsize=(9, 6))
    ax.plot(flux, fidelity, label="Fidelity")
    ax.plot(flux, rate, label="Rate")
    ax.axhline(fidelity_min, linestyle="--", linewidth=1.5,
               label=f"Fidelity Limit = {fidelity_min}", c="k")

    # Find the flux values where fidelity crosses the floor by linear interpolation.
    diff = fidelity - fidelity_min
    cross_idx = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) <= 0)[0]

    x_crosses: list[float] = []
    r_crosses: list[float] = []
    for i in cross_idx:
        x0, x1 = flux[i], flux[i + 1]
        y0, y1p = fidelity[i], fidelity[i + 1]
        if y1p == y0:
            continue
        x_cross = x0 + (fidelity_min - y0) * (x1 - x0) / (y1p - y0)
        x_crosses.append(x_cross)
        r_crosses.append(x_cross ** 2 + (2 * y_1 + 2 * y_2 + 1) * x_cross + 4 * y_1 * y_2)

    if x_crosses:
        best_idx = int(np.argmax(r_crosses))
        x_best = x_crosses[best_idx]
        ax.axvline(x_best, linestyle="--", linewidth=1.5, c="k")
        ax.plot([x_best], [fidelity_min], marker="o", ms=7)
        ax.annotate(
            f"x={x_best:.4f}\nrate={max(r_crosses):.4f}",
            xy=(x_best, fidelity_min),
            xytext=(10, 10),
            textcoords="offset points",
            ha="left",
            va="bottom",
        )

    ax.set_xlabel("Flux over Coincidence Window")
    ax.set_ylabel("Magnitude")
    ax.set_title("Graphical View of Fidelity and Rate Relation")
    ax.set_xlim([float(np.min(flux)), float(np.max(flux))])
    ax.set_ylim([0, 1])
    ax.legend(loc="best")

    return save_figure(fig, outdir, filename)
