# plotting/base.py
"""
Shared plotting primitives.

This module groups helpers used by every figure in the project so that the
standalone plots (network, utility bars, source allocation) and the
publication composite render with consistent typography, colors, sort order
and label formatting.

The public surface is intentionally small and is re-exported via ``__all__``
so that :mod:`plotting` consumers have a single import target.
"""

# ZHG
# 2026.03.26
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Twelve-color qualitative palette derived from MATLAB's ``GEM`` scheme,
#: with three additional accent colors appended for plots that need more
#: than the default eight series.
GEM: list[str] = [
    "#0072BD", "#D95319", "#EDB120", "#7E2F8E",
    "#77AC30", "#4DBEEE", "#A2142F", "#003BFF",
    "#017501", "#FF0000", "#B526FF", "#FF00FF", "#000000",
]


# Shared publication font sizes. Use these everywhere so the standalone
# figures and the composite figure have matching typography.
PLOT_FONT_SIZE = 28
PLOT_TITLE_SIZE = PLOT_FONT_SIZE
PLOT_LABEL_SIZE = PLOT_FONT_SIZE
PLOT_TICK_SIZE = PLOT_FONT_SIZE
PLOT_LEGEND_SIZE = PLOT_FONT_SIZE

# Brighter unassigned/surplus-bin gray used by the source-allocation plot.
NULL_LINK_GRAY = "#d5d5d5ff"


# --------------------------------------------------------------------------- #
# Figure / style helpers
# --------------------------------------------------------------------------- #

def apply_plot_style() -> None:
    """Apply the project-wide Matplotlib rcParams used by every standalone plot."""
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


def make_figure(figsize: tuple[float, float] = (8, 5)):
    """Create a styled ``(fig, ax)`` pair with the project rcParams applied."""
    apply_plot_style()
    return plt.subplots(figsize=figsize)


def save_figure(
    fig,
    outdir: str | Path,
    filename: str,
    bbox_inches: str | None = "tight",
) -> Path:
    """Save ``fig`` under ``outdir/filename`` and close it. Returns the full path."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches=bbox_inches, dpi=300)
    plt.close(fig)
    return path


def gem_colors(n: int, start: int = 0) -> list[str]:
    """Return ``n`` colors cycled from :data:`GEM` starting at index ``start``."""
    if n <= 0:
        return []
    return [GEM[(start + i) % len(GEM)] for i in range(n)]


def link_color_map(link_labels: list[str]) -> dict[str, Any]:
    """Map each link label to a color, reserving ``NULL_LINK_GRAY`` for null links."""
    colors: dict[str, Any] = {
        label: color
        for label, color in zip(link_labels, gem_colors(len(link_labels), start=1))
    }
    colors["Null Link"] = NULL_LINK_GRAY
    return colors

def _label_map_from_context(ctx) -> dict:
    """
    Return a node-label map from either a NetworkX graph or a config object.

    Supported:
        network.graph["node_label_map"]
        cfg.node_label_map

    Default:
        {}
    """
    if ctx is None:
        return {}

    graph = getattr(ctx, "graph", None)
    if isinstance(graph, dict) and "node_label_map" in graph:
        return graph.get("node_label_map", {}) or {}

    return getattr(ctx, "node_label_map", {}) or {}


def display_node_label(ctx, node) -> str:
    """
    Return the plot-display label for a node.

    Default behavior:
        S1 -> S1
        U1 -> U1

    Manhattan behavior, when a label map exists:
        S1 -> s_B
        U1 -> u_A
    """
    label_map = _label_map_from_context(ctx)
    return str(label_map.get(node, node))


def _entity_tokens(text: Any) -> list[str]:
    """Extract entity-like tokens such as U1, S2, u_A, or s_B."""
    s = str(text)
    return [m.group(0) for m in _ENTITY_ANY_RE.finditer(s)]


def display_link_label(ctx, link) -> str:
    """
    Return a plot-display label for a two-user link.

    Default:
        ("U1", "U14") -> U1-U14

    Manhattan:
        ("U1", "U14") -> u_A-u_Q
    """
    if isinstance(link, (tuple, list)) and len(link) == 2:
        a, b = link
        return f"{display_node_label(ctx, a)}-{display_node_label(ctx, b)}"

    if isinstance(link, str):
        raw = link.strip()
        if raw.startswith("Link "):
            raw = raw[len("Link "):].strip()

        for sep in ("-", "–"):
            if sep in raw:
                left, right = raw.split(sep, 1)
                return f"{display_node_label(ctx, left.strip())}-{display_node_label(ctx, right.strip())}"

        tokens = _entity_tokens(raw)
        if len(tokens) >= 2:
            return f"{display_node_label(ctx, tokens[0])}-{display_node_label(ctx, tokens[1])}"

    return str(link)


def display_source_label(ctx, source) -> str:
    """
    Return a plot-display label for a source.

    Default:
        S1 -> S1

    Manhattan:
        S1 -> s_B
    """
    return display_node_label(ctx, source)

# --------------------------------------------------------------------------- #
# Numeric coercion
# --------------------------------------------------------------------------- #

def safe_float(x: Any, default: float = float("nan")) -> float:
    """
    Coerce ``x`` to ``float`` while tolerating CVXPY-style wrappers, sequences
    and exotic numeric containers. Falls back to ``default`` on any failure.

    For multi-element sequences the sum is returned, matching the historical
    behavior used by the allocation reports (where channel counts arrive as
    arrays whose total represents the link allocation).
    """
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
    """Round-coerce ``x`` to ``int``; returns ``default`` for non-finite inputs."""
    val = safe_float(x, float(default))
    if not np.isfinite(val):
        return default
    return int(round(val))


# --------------------------------------------------------------------------- #
# Text / label formatting
# --------------------------------------------------------------------------- #

def true_minus(text: Any) -> str:
    """Replace hyphen-minus with the typographic minus sign (U+2212)."""
    return str(text).replace("-", "−")


def format_float(value: float, fmt: str = ".2f") -> str:
    """Format a number with ``fmt`` and apply :func:`true_minus`."""
    return true_minus(format(value, fmt))


# Matches "U1", "U_1", "User 1", "S3", "Src3", "Source 3", etc.
_ENTITY_NUMERIC_RE = re.compile(
    r"(?i)\b(u(?:ser)?|s(?:rc|ource)?)[_\s-]*?(\d+)\b"
)

# Compatibility alias used by the sort helpers below.
_ENTITY_TOKEN_RE = _ENTITY_NUMERIC_RE

# Matches letter-subscript display labels: u_A, s_B, uA, sM, etc.
_ENTITY_LETTER_RE = re.compile(
    r"(?i)\b([us])[_\s-]*?([A-Z])\b"
)

# Matches either a numeric or a letter entity token. Used to parse compact
# labels like "U1U14" or "Link U1U14".
_ENTITY_ANY_RE = re.compile(
    r"(?i)(?:u(?:ser)?|s(?:rc|ource)?)[_\s-]*?\d+|[us][_\s-]*?[A-Z]"
)


def entity_mathtext(label: str) -> str:
    """
    Render entity labels as Matplotlib mathtext.

    Examples:
        U12  -> u$_{12}$
        S3   -> s$_{3}$
        u_A  -> u$_{A}$
        s_B  -> s$_{B}$
    """
    text = str(label)

    def repl_numeric(m: re.Match[str]) -> str:
        base = m.group(1).lower()
        idx = int(m.group(2))
        return rf"{base}$_{{{idx}}}$"

    def repl_letter(m: re.Match[str]) -> str:
        base = m.group(1).lower()
        idx = m.group(2).upper()
        return rf"{base}$_{{{idx}}}$"

    text = _ENTITY_NUMERIC_RE.sub(repl_numeric, text)
    text = _ENTITY_LETTER_RE.sub(repl_letter, text)
    return text


def bar_link_label(label: Any) -> str:
    """Format a link label for a bar tick: a 2-tuple, ``"U1-U2"`` string, etc."""
    if isinstance(label, (tuple, list)) and len(label) == 2:
        return entity_mathtext(label[0]) + entity_mathtext(label[1])

    if isinstance(label, str):
        raw = label.strip()
        if raw.startswith("Link "):
            raw = raw[len("Link "):].strip()

        for sep in ("-", "–"):
            if sep in raw:
                left, right = raw.split(sep, 1)
                return entity_mathtext(left.strip()) + entity_mathtext(right.strip())

        tokens = _entity_tokens(raw)
        if len(tokens) >= 2:
            return entity_mathtext(tokens[0]) + entity_mathtext(tokens[1])

    return entity_mathtext(str(label))


def display_bar_link_label(ctx, link) -> str:
    """
    Format a link label using a display-label map first, then mathtext.

    Default:
        ("U1", "U14") -> u$_{1}$u$_{14}$

    Manhattan:
        ("U1", "U14") -> u$_{A}$u$_{Q}$
    """
    return bar_link_label(display_link_label(ctx, link))


def display_legend_link_label(ctx, link) -> str:
    """
    Format a legend label using a display-label map.

    Default:
        ("U1", "U14") -> Link u$_{1}$u$_{14}$

    Manhattan:
        ("U1", "U14") -> Link u$_{A}$u$_{Q}$
    """
    return f"Link {display_bar_link_label(ctx, link)}"


# --------------------------------------------------------------------------- #
# Sort & ordering helpers
# --------------------------------------------------------------------------- #

def _entity_sort_key(label: Any) -> tuple[int, int, str]:
    """
    Natural sort for entity labels: ``U1 < U2 < U10`` and users before sources.

    Returns a ``(kind, index, original)`` tuple where ``kind`` is 0 for users,
    1 for sources and 2 for anything else.
    """
    s = str(label)
    m = _ENTITY_TOKEN_RE.search(s)
    if m:
        kind = 0 if m.group(1)[0].lower() == "u" else 1
        return (kind, int(m.group(2)), s)
    nums = re.findall(r"\d+", s)
    return (2, int(nums[0]) if nums else 10**9, s)


def canonical_link_tuple(link: Any) -> tuple[str, str] | None:
    """Return a canonical ``(low, high)`` user-pair tuple, or ``None`` if unparseable."""
    if isinstance(link, (tuple, list)) and len(link) == 2:
        a, b = str(link[0]), str(link[1])
        return tuple(sorted((a, b), key=_entity_sort_key))  # type: ignore[return-value]
    return None


def option_link_tuple(option: dict[str, Any]) -> tuple[str, str] | None:
    """Canonical link tuple from an option's ``link`` or ``users`` field."""
    return canonical_link_tuple(option.get("link") or option.get("users"))


def option_link_label(option: dict[str, Any]) -> str:
    """Concatenated user label for an option, e.g. ``"U1U2"``."""
    tup = option_link_tuple(option)
    if tup is not None:
        return f"{tup[0]}{tup[1]}"
    link = option.get("link") or option.get("users")
    return str(link)


def link_sort_key(link: Any) -> tuple[tuple[int, int, str], tuple[int, int, str]]:
    """Sort key for a link: by lower endpoint then upper endpoint."""
    tup = canonical_link_tuple(link)
    if tup is None:
        s = str(link)
        return (_entity_sort_key(s), (9, 10**9, s))
    return (_entity_sort_key(tup[0]), _entity_sort_key(tup[1]))


def option_sort_key(option: dict[str, Any]) -> tuple[tuple[int, int, str], tuple[int, int, str]]:
    """Sort key for an option, dispatching to :func:`link_sort_key`."""
    return link_sort_key(option.get("link") or option.get("users"))


def ordered_link_keys_from_options(options: Iterable[dict[str, Any]]) -> list[tuple[str, str]]:
    """
    Return the unique canonical link tuples of ``options`` in stable order.

    Order preserves U1-containing links first, then ascends by next-lowest user.
    """
    seen: dict[tuple[str, str], None] = {}
    for opt in sorted(list(options), key=option_sort_key):
        key = option_link_tuple(opt)
        if key is not None and key not in seen:
            seen[key] = None
    return list(seen.keys())


def sorted_options_by_link(options: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ``options`` sorted by canonical link order."""
    return sorted(list(options), key=option_sort_key)


# --------------------------------------------------------------------------- #
# Network geometry & utility math
# --------------------------------------------------------------------------- #

def path_edges(path: Iterable[str]) -> list[tuple[str, str]]:
    """Yield the canonical (sorted) edges traversed by ``path``."""
    nodes = list(path)
    return [tuple(sorted((nodes[i], nodes[i + 1]))) for i in range(len(nodes) - 1)]


def path_loss(network, path: Iterable[str]) -> float:
    """Sum the per-edge ``loss`` attribute along ``path``; missing edges contribute 1.0."""
    nodes = list(path)
    return sum(
        safe_float(network[u][v].get("loss", 1.0), 1.0)
        for u, v in zip(nodes, nodes[1:])
    )


def per_link_expr(option: dict[str, Any], allocation_record: dict[str, Any] | None) -> float:
    """
    Compute the per-link rate expression
    ``μ²k² + μk·(2(y1+y2)+1) + 4·y1·y2``.

    Returns NaN when any input is missing or non-finite.
    """
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
    """``log10`` of :func:`per_link_expr`; returns NaN for non-positive expressions."""
    if allocation_record is not None and "link_utility" in allocation_record:
        return safe_float(allocation_record.get("link_utility"), float("nan"))

    expr = per_link_expr(option, allocation_record)
    if not np.isfinite(expr) or expr <= 0:
        return float("nan")

    val = math.log10(expr)

    # Fallback physical correction if available
    if "total_loss" in option and "tau" in option:
        val -= safe_float(option.get("total_loss"), 0.0) / 10.0
        val -= math.log10(safe_float(option.get("tau"), 1.0))

    return val


# --------------------------------------------------------------------------- #
# Legend
# --------------------------------------------------------------------------- #

def augment_legend_with_frequencies(ax, freqs_by_link: dict | None) -> None:
    """
    Append per-direction channel counts to each ``Link Ux-Uy`` entry on ``ax``.

    ``freqs_by_link`` maps a canonical user-pair tuple to ``(to_u1, to_u2)``
    frequency lists. Entries without a matching key are left untouched.
    """
    if not freqs_by_link:
        return

    handles, labels = ax.get_legend_handles_labels()
    new_labels: list[str] = []

    for lab in labels:
        base = lab.strip().replace("Link ", "")

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
            lab = true_minus(
                f"Link {u1}-{u2}  (+{list(to_u1)} → {u1},  -{list(to_u2)} → {u2})"
            )
        new_labels.append(lab)

    ax.legend(handles, new_labels, loc="best")
