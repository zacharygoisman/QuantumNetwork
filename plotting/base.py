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

# Shared publication font sizes. Use these everywhere so the standalone
# figures and the composite figure have matching typography.
PLOT_FONT_SIZE = 20
PLOT_TITLE_SIZE = PLOT_FONT_SIZE
PLOT_LABEL_SIZE = PLOT_FONT_SIZE
PLOT_TICK_SIZE = PLOT_FONT_SIZE
PLOT_LEGEND_SIZE = PLOT_FONT_SIZE

# Brighter unassigned/surplus-bin gray used by the source-allocation plot.
NULL_LINK_GRAY = "#EDEDED"



def apply_plot_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "axes.titlesize": PLOT_TITLE_SIZE,
        "axes.labelsize": PLOT_LABEL_SIZE,
        "xtick.labelsize": PLOT_TICK_SIZE,
        "ytick.labelsize": PLOT_TICK_SIZE,
        "legend.fontsize": PLOT_LEGEND_SIZE,
        "font.size": PLOT_FONT_SIZE,
        "mathtext.default": "regular",
        "axes.unicode_minus": True,
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


def true_minus(text: Any) -> str:
    """Use the typographic minus sign in manually formatted labels."""
    return str(text).replace("-", "−")


def format_float(value: float, fmt: str = ".2f") -> str:
    """Format a number and replace hyphen-minus with a true minus sign."""
    return true_minus(format(value, fmt))


def _entity_sort_key(label: Any) -> tuple[int, int, str]:
    """Sort sources/users naturally: U1 before U2 before U10."""
    s = str(label)
    m = re.search(r"(?i)\b([us])(?:er|rc|ource)?[_\s-]*?(\d+)\b", s)
    if not m:
        m = re.search(r"(?i)\b([us])(\d+)\b", s)
    if m:
        kind = 0 if m.group(1).lower() == "u" else 1
        return (kind, int(m.group(2)), s)
    nums = re.findall(r"\d+", s)
    return (2, int(nums[0]) if nums else 10**9, s)


def canonical_link_tuple(link: Any) -> tuple[str, str] | None:
    """Return a canonical user-pair tuple sorted by user number."""
    if isinstance(link, (tuple, list)) and len(link) == 2:
        a, b = str(link[0]), str(link[1])
        return tuple(sorted((a, b), key=_entity_sort_key))  # type: ignore[return-value]
    return None


def option_link_tuple(option: dict[str, Any]) -> tuple[str, str] | None:
    link = option.get("link") or option.get("users")
    return canonical_link_tuple(link)


def link_sort_key(link: Any) -> tuple[tuple[int, int, str], tuple[int, int, str]]:
    tup = canonical_link_tuple(link)
    if tup is None:
        return (_entity_sort_key(str(link)), (9, 10**9, str(link)))
    return (_entity_sort_key(tup[0]), _entity_sort_key(tup[1]))


def option_sort_key(option: dict[str, Any]) -> tuple[tuple[int, int, str], tuple[int, int, str]]:
    return link_sort_key(option.get("link") or option.get("users"))


def ordered_link_keys_from_options(options: Iterable[dict[str, Any]]) -> list[tuple[str, str]]:
    """Stable link order: U1-containing links first, then by next-lowest user."""
    seen: dict[tuple[str, str], None] = {}
    for opt in sorted(list(options), key=option_sort_key):
        key = option_link_tuple(opt)
        if key is not None and key not in seen:
            seen[key] = None
    return list(seen.keys())


def sorted_options_by_link(options: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(list(options), key=option_sort_key)

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
    colors["Null Link"] = NULL_LINK_GRAY
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
            lab = true_minus(f"Link {u1}-{u2}  (+{list(to_u1)} → {u1},  -{list(to_u2)} → {u2})")
        new_labels.append(lab)

    ax.legend(handles, new_labels, loc="best")