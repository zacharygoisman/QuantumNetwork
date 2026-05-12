#plotting/base.py
"""
Plotting defaults
"""

#ZHG
#2026.03.26
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from __future__ import annotations

from pathlib import Path
import math
import re
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


GEM = [
    "#0072BD", "#D95319", "#EDB120", "#7E2F8E",
    "#77AC30", "#4DBEEE", "#A2142F", "#003BFF",
    "#017501", "#FF0000", "#B526FF", "#FF00FF", "#000000",
]


def apply_plot_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })


def make_figure(figsize=(8, 5)):
    apply_plot_style()
    return plt.subplots(figsize=figsize)


def save_figure(fig, outdir: str | Path, filename: str) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


def gem_colors(n: int, start: int = 0) -> list[str]:
    if n <= 0:
        return []
    return [GEM[(start + i) % len(GEM)] for i in range(n)]


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if hasattr(x, "value"):
            v = x.value
            if isinstance(v, (list, tuple, np.ndarray)):
                if len(v) == 0:
                    return default
                return float(v[0])
            return float(v)

        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float)
            if arr.size == 0:
                return default
            if arr.size == 1:
                return float(arr.item())
            return float(arr.sum())

        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    val = safe_float(x, float(default))
    if not np.isfinite(val):
        return default
    return int(round(val))


def option_link_tuple(option: dict[str, Any]) -> tuple[str, str] | None:
    link = option.get("link") or option.get("users")
    if isinstance(link, (tuple, list)) and len(link) == 2:
        return str(link[0]), str(link[1])
    return None


def option_link_label(option: dict[str, Any]) -> str:
    tup = option_link_tuple(option)
    if tup is not None:
        return f"{tup[0]}{tup[1]}"
    link = option.get("link") or option.get("users")
    return str(link)


def entity_mathtext(label: str) -> str:
    def repl(m: re.Match[str]) -> str:
        base = m.group(1).lower()
        idx = int(m.group(2))
        return rf"{base}$_{{{idx}}}$"

    s = str(label)
    s = re.sub(r"(?i)\b(u(?:ser)?|s(?:rc|ource)?)[_\s-]*?(\d+)\b", repl, s)
    s = re.sub(r"(?i)\b([us])(\d+)\b", repl, s)
    return s


def bar_link_label(label: Any) -> str:
    if isinstance(label, (tuple, list)) and len(label) == 2:
        return entity_mathtext(label[0]) + entity_mathtext(label[1])

    if isinstance(label, str):
        for sep in ["-", "–"]:
            if sep in label:
                left, right = label.split(sep, 1)
                return entity_mathtext(left.strip()) + entity_mathtext(right.strip())

    return entity_mathtext(str(label))


def path_edges(path: Iterable[str]) -> list[tuple[str, str]]:
    path = list(path)
    return [tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)]


def path_loss(network, path: Iterable[str]) -> float:
    path = list(path)
    total = 0.0
    for u, v in zip(path, path[1:]):
        total += safe_float(network[u][v].get("loss", 1.0), 1.0)
    return total


def per_link_expr(option: dict[str, Any], allocation_record: dict[str, Any] | None) -> float:
    if allocation_record is None:
        return float("nan")

    k = safe_float(allocation_record.get("k"), float("nan"))
    mu = safe_float(allocation_record.get("mu"), float("nan"))
    y1 = safe_float(option.get("y1"), float("nan"))
    y2 = safe_float(option.get("y2"), float("nan"))

    if not all(np.isfinite(v) for v in (k, mu, y1, y2)):
        return float("nan")

    return mu * mu * k * k + mu * k * (2.0 * (y1 + y2) + 1.0) + 4.0 * y1 * y2


def per_link_log_utility(option: dict[str, Any], allocation_record: dict[str, Any] | None) -> float:
    expr = per_link_expr(option, allocation_record)
    if not np.isfinite(expr) or expr <= 0:
        return float("nan")
    return math.log10(expr)


def link_color_map(link_labels: list[str]) -> dict[str, Any]:
    colors = {lab: c for lab, c in zip(link_labels, gem_colors(len(link_labels), start=1))}
    colors["Null Link"] = GEM[0]
    return colors


def augment_legend_with_frequencies(ax, freqs_by_link: dict | None) -> None:
    if not freqs_by_link:
        return

    handles, labels = ax.get_legend_handles_labels()
    new_labels = []

    for lab in labels:
        s = lab.strip()
        base = s.replace("Link ", "")

        if "-" in base:
            u1, u2 = base.split("-", 1)
        elif "–" in base:
            u1, u2 = base.split("–", 1)
        else:
            new_labels.append(lab)
            continue

        key = (u1.strip(), u2.strip())
        if key in freqs_by_link:
            to_u1, to_u2 = freqs_by_link[key]
            lab = f"Link {u1}-{u2}  (+{list(to_u1)} → {u1},  -{list(to_u2)} → {u2})"
        new_labels.append(lab)

    ax.legend(handles, new_labels, loc="best")