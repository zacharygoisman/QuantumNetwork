#network/link_editor.py
"""
Link_Editor: in-place mutation of an already-built network topology.

This module exposes four small functions that operate on a
``networkx.Graph`` returned by ``network.builder.build_network``:

* ``add_link``       -- insert a new undirected edge with a ``loss`` attribute
* ``remove_link``    -- remove an existing undirected edge, keeping every
                        Entangled_Pair in ``cfg.link_pairs`` connected
* ``save_topology``  -- serialize the topology to a JSON snapshot
* ``load_topology``  -- reconstruct a topology from a JSON snapshot

The functions are intentionally minimal: they never touch routing,
allocation, or plotting and they never rebuild the graph from ``cfg``.
Callers are responsible for re-invoking the pipeline (see
``pipeline.runner.run_pipeline_prebuilt``) once they are happy with the
edits.
"""

#ZHG
#2026.07.19
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

import copy
import json
import math
import os
import random
import tempfile
from pathlib import Path

import networkx as nx


# --------------------------------------------------------------------------- #
# Constants for link-pair editing
# --------------------------------------------------------------------------- #

# Default fidelity assigned to a newly added link pair when the caller does
# not specify one. Matches the middle of the usual 0.9-0.95 range used by
# the paper presets.
_DEFAULT_FIDELITY = 0.9


# --------------------------------------------------------------------------- #
# add_link
# --------------------------------------------------------------------------- #

def add_link(topology, u, v, loss=None, *, cfg=None):
    """Insert an undirected edge ``(u, v)`` into ``topology``.

    Parameters
    ----------
    topology : networkx.Graph
        The graph to mutate in place.
    u, v : hashable
        Endpoint node labels. Both must already exist in ``topology`` and
        must be different from one another.
    loss : float, optional
        If given, ``float(loss)`` (which must be finite) is stored as the
        edge's ``loss`` attribute. If omitted, a value is drawn uniformly
        from ``cfg.loss_range``.
    cfg : Config, keyword-only
        Required when ``loss`` is not provided; used to read
        ``cfg.loss_range`` for the random-loss draw.

    Returns
    -------
    tuple
        The pair ``(u, v)``.
    """
    # (1) self-loop check -- reject before touching the graph
    if u == v:
        raise ValueError(f"add_link: cannot add a self-loop on node {u!r}")

    # (2) missing-node check -- name every absent label
    missing = [n for n in (u, v) if not topology.has_node(n)]
    if missing:
        raise ValueError(
            f"add_link: node(s) not in topology: {missing!r}"
        )

    # (3) duplicate-edge check -- has_edge is order-independent for Graph
    if topology.has_edge(u, v):
        raise ValueError(
            f"add_link: edge ({u!r}, {v!r}) already exists"
        )

    # (4) resolve loss value
    if loss is None:
        if cfg is None or not hasattr(cfg, "loss_range"):
            raise ValueError(
                "add_link: cfg with loss_range is required when loss is omitted"
            )
        lo, hi = cfg.loss_range
        loss_val = float(random.uniform(float(lo), float(hi)))
    else:
        try:
            loss_val = float(loss)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"add_link: invalid loss value {loss!r} ({e})"
            ) from None
        if math.isnan(loss_val) or math.isinf(loss_val):
            raise ValueError(
                f"add_link: loss must be finite, got {loss!r}"
            )

    topology.add_edge(u, v, loss=loss_val)
    return (u, v)


# --------------------------------------------------------------------------- #
# remove_link
# --------------------------------------------------------------------------- #

def remove_link(topology, u, v, *, cfg=None):
    """Remove the undirected edge ``(u, v)`` from ``topology``.

    Failure conditions are evaluated in the fixed order:

        (a) missing-node / self-loop check
        (b) edge-existence check
        (c) connectivity preservation for every pair in ``cfg.link_pairs``

    On success the edge is gone from ``topology`` and the pair ``(u, v)``
    is returned. On any failure the topology is left exactly as it was
    before the call.
    """
    # (a) self-loop / missing-node check
    if u == v:
        raise ValueError(f"remove_link: cannot remove a self-loop on node {u!r}")

    missing = [n for n in (u, v) if not topology.has_node(n)]
    if missing:
        raise ValueError(
            f"remove_link: node(s) not in topology: {missing!r}"
        )

    # (b) edge-existence check
    if not topology.has_edge(u, v):
        raise ValueError(
            f"remove_link: no edge between {u!r} and {v!r}"
        )

    # (c) connectivity preservation. Snapshot the edge attributes so that
    # a rollback is byte-identical if a downstream pair loses its path.
    edge_data = dict(topology[u][v])
    topology.remove_edge(u, v)

    link_pairs = getattr(cfg, "link_pairs", None) if cfg is not None else None
    if link_pairs:
        for pair in link_pairs:
            a, b = pair[0], pair[1]
            connected = (
                topology.has_node(a)
                and topology.has_node(b)
                and nx.has_path(topology, a, b)
            )
            if not connected:
                # Restore the edge exactly and raise.
                topology.add_edge(u, v, **edge_data)
                raise ValueError(
                    f"remove_link: removing edge ({u!r}, {v!r}) would leave "
                    f"entangled pair ({a!r}, {b!r}) with no remaining path"
                )

    return (u, v)


# --------------------------------------------------------------------------- #
# Link_pair editing (the colored entanglement demands drawn in the plot)
# --------------------------------------------------------------------------- #

def _canonical(pair):
    """Return an unordered-pair key so ``(a, b)`` and ``(b, a)`` match."""
    return frozenset((pair[0], pair[1]))


def add_link_pair(topology, links, u, v, *, cfg=None, fidelity=None):
    """Add an entangled user pair ``(u, v)`` to the demand set.

    ``links`` is the ``list[tuple[str, str]]`` returned by
    ``network.builder.build_network`` and stored in ``cfg.link_pairs``.
    Both are mutated in place. The pipeline will then route this new
    pair on the next ``run``.

    Failure modes (checked in this order):
        1. self-loop (``u == v``)
        2. missing node in ``topology``
        3. duplicate pair (either order)
        4. no path between ``u`` and ``v`` in the current topology

    On failure ``links``, ``cfg.link_pairs``, ``cfg.fidelity_limit`` and
    ``cfg.num_lnks`` are all left unchanged.
    """
    if u == v:
        raise ValueError(
            f"add_link_pair: cannot pair a node with itself ({u!r})"
        )

    missing = [n for n in (u, v) if not topology.has_node(n)]
    if missing:
        raise ValueError(
            f"add_link_pair: node(s) not in topology: {missing!r}"
        )

    key = _canonical((u, v))
    for existing in links:
        if _canonical(existing) == key:
            raise ValueError(
                f"add_link_pair: pair ({u!r}, {v!r}) already exists"
            )

    if not nx.has_path(topology, u, v):
        raise ValueError(
            f"add_link_pair: no path between {u!r} and {v!r} in the "
            f"current topology"
        )

    # Resolve fidelity threshold before mutating anything.
    if fidelity is None:
        f_val = _DEFAULT_FIDELITY
    else:
        try:
            f_val = float(fidelity)
        except (TypeError, ValueError):
            raise ValueError(
                f"add_link_pair: invalid fidelity value {fidelity!r}"
            ) from None
        if math.isnan(f_val) or math.isinf(f_val):
            raise ValueError(
                f"add_link_pair: fidelity must be finite, got {fidelity!r}"
            )
        if not (0.5 <= f_val < 1.0):
            raise ValueError(
                f"add_link_pair: fidelity must be in [0.5, 1.0), got {f_val}"
            )

    # Commit: append to the runtime list first, then keep cfg in sync.
    links.append((u, v))
    if cfg is not None:
        # cfg.link_pairs may be None on presets that never set it. Create
        # a fresh list in that case so downstream reads work.
        if getattr(cfg, "link_pairs", None) is None:
            cfg.link_pairs = []
        cfg.link_pairs.append((u, v))
        if hasattr(cfg, "fidelity_limit"):
            cfg.fidelity_limit = list(cfg.fidelity_limit) + [f_val]
        if hasattr(cfg, "num_lnks"):
            cfg.num_lnks = len(links)
    return (u, v)


def remove_link_pair(topology, links, u, v, *, cfg=None):
    """Remove an entangled user pair ``(u, v)`` from the demand set.

    Failure modes:
        1. self-loop (``u == v``)
        2. missing node in ``topology``
        3. pair is not currently in ``links``

    On success the matching entry (order-independent) is dropped from
    ``links``, ``cfg.link_pairs`` and the corresponding ``fidelity_limit``
    entry are removed, and ``cfg.num_lnks`` is updated. Returns the pair
    ``(u, v)`` as it was passed in.
    """
    if u == v:
        raise ValueError(
            f"remove_link_pair: cannot remove a self-pair on {u!r}"
        )

    missing = [n for n in (u, v) if not topology.has_node(n)]
    if missing:
        raise ValueError(
            f"remove_link_pair: node(s) not in topology: {missing!r}"
        )

    key = _canonical((u, v))
    idx = next(
        (i for i, existing in enumerate(links) if _canonical(existing) == key),
        None,
    )
    if idx is None:
        raise ValueError(
            f"remove_link_pair: pair ({u!r}, {v!r}) is not in the "
            f"current link set"
        )

    del links[idx]
    if cfg is not None:
        if getattr(cfg, "link_pairs", None):
            cfg_idx = next(
                (i for i, p in enumerate(cfg.link_pairs)
                 if _canonical(p) == key),
                None,
            )
            if cfg_idx is not None:
                del cfg.link_pairs[cfg_idx]
                if hasattr(cfg, "fidelity_limit") and cfg_idx < len(cfg.fidelity_limit):
                    cfg.fidelity_limit = (
                        list(cfg.fidelity_limit[:cfg_idx])
                        + list(cfg.fidelity_limit[cfg_idx + 1:])
                    )
        if hasattr(cfg, "num_lnks"):
            cfg.num_lnks = len(links)
    return (u, v)


# --------------------------------------------------------------------------- #
# save_topology
# --------------------------------------------------------------------------- #

# JSON schema version for Topology_Snapshot files. Bump this whenever the
# on-disk representation changes in an incompatible way.
_SNAPSHOT_VERSION = 1


def _topology_to_snapshot(topology):
    """Build the JSON-safe dict that gets written by ``save_topology``."""
    nodes = []
    for node, data in topology.nodes(data=True):
        nodes.append({
            "id": str(node),
            "node_type": data.get("node_type"),
        })

    edges = []
    for u, v, data in topology.edges(data=True):
        loss = data.get("loss")
        edges.append({
            "u": str(u),
            "v": str(v),
            # ``repr(float)`` keeps full round-trip precision, but json.dump
            # already emits enough digits for a ``float`` to round-trip in
            # CPython, so we can just cast to float here.
            "loss": float(loss) if loss is not None else 0.0,
        })

    return {
        "version": _SNAPSHOT_VERSION,
        "nodes": nodes,
        "edges": edges,
    }


def save_topology(topology, path):
    """Write ``topology`` to ``path`` as a UTF-8 JSON Topology_Snapshot.

    The write is atomic: the new document is fully serialized to a
    temporary file in the same directory and only then renamed over
    ``path``. If any step fails, the temporary file is removed and the
    original file at ``path`` (if any) is left untouched.
    """
    path = Path(path)

    parent = path.parent if str(path.parent) else Path(".")

    snapshot = _topology_to_snapshot(topology)

    try:
        # Same-directory temp file so that ``os.replace`` is atomic on
        # POSIX / NTFS. ``mkstemp`` opens with mode 0600 which is fine.
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".tmp",
            dir=str(parent),
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp_path), str(path))
        except Exception:
            # Clean up the temp file on any failure; swallow secondary
            # errors so we can re-raise the original one below.
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise
    except OSError as e:
        raise OSError(
            f"save_topology: failed to write topology to {str(path)!r}: {e}"
        ) from e


# --------------------------------------------------------------------------- #
# load_topology
# --------------------------------------------------------------------------- #

_REQUIRED_TOP_LEVEL = ("nodes", "edges")
_VALID_NODE_TYPES = ("source", "user")


def _validate_snapshot(snapshot, path):
    """Validate a decoded snapshot dict, in the fixed order documented in
    the requirements. Raises ``ValueError`` on the first defect."""
    # (2) required top-level fields
    for key in _REQUIRED_TOP_LEVEL:
        if key not in snapshot:
            raise ValueError(
                f"load_topology: snapshot at {str(path)!r} is missing "
                f"required field {key!r}"
            )

    nodes = snapshot["nodes"]
    edges = snapshot["edges"]

    if not isinstance(nodes, list):
        raise ValueError(
            f"load_topology: 'nodes' must be a list, got {type(nodes).__name__}"
        )
    if not isinstance(edges, list):
        raise ValueError(
            f"load_topology: 'edges' must be a list, got {type(edges).__name__}"
        )

    # (3) unique node labels
    labels = set()
    for i, node in enumerate(nodes):
        if not isinstance(node, dict) or "id" not in node:
            raise ValueError(
                f"load_topology: node entry #{i} is malformed: {node!r}"
            )
        label = node["id"]
        if label in labels:
            raise ValueError(
                f"load_topology: duplicate node label {label!r}"
            )
        labels.add(label)

    # (4) node_type validation
    for node in nodes:
        nt = node.get("node_type")
        if nt not in _VALID_NODE_TYPES:
            raise ValueError(
                f"load_topology: node {node['id']!r} has invalid node_type "
                f"{nt!r} (must be one of {_VALID_NODE_TYPES})"
            )

    # (5) edge references, self-loop / duplicate checks
    seen_edges = set()
    for i, edge in enumerate(edges):
        if (
            not isinstance(edge, dict)
            or "u" not in edge
            or "v" not in edge
            or "loss" not in edge
        ):
            raise ValueError(
                f"load_topology: edge entry #{i} is malformed: {edge!r}"
            )
        eu, ev = edge["u"], edge["v"]
        if eu not in labels or ev not in labels:
            raise ValueError(
                f"load_topology: edge #{i} references undefined node(s): "
                f"{edge!r}"
            )
        if eu == ev:
            raise ValueError(
                f"load_topology: edge #{i} is a self-loop on {eu!r}"
            )
        key = frozenset((eu, ev))
        if key in seen_edges:
            raise ValueError(
                f"load_topology: duplicate undirected edge "
                f"between {eu!r} and {ev!r}"
            )
        seen_edges.add(key)

    # (6) finite numeric loss
    for i, edge in enumerate(edges):
        loss = edge["loss"]
        # Reject bool explicitly (bool is a subclass of int in Python).
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            raise ValueError(
                f"load_topology: edge #{i} has non-numeric loss {loss!r}"
            )
        loss_f = float(loss)
        if math.isnan(loss_f) or math.isinf(loss_f):
            raise ValueError(
                f"load_topology: edge #{i} has non-finite loss {loss!r}"
            )


def load_topology(path):
    """Reconstruct a ``networkx.Graph`` from a Topology_Snapshot at ``path``.

    Validation is performed in a fixed order and stops at the first
    failure (see ``_validate_snapshot``). The returned graph carries
    ``node_type`` on every node and ``loss`` on every edge.
    """
    path_obj = Path(path)

    if not path_obj.is_file():
        raise FileNotFoundError(
            f"load_topology: no such file: {str(path)}"
        )

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"load_topology: {str(path)!r} is not valid JSON: {e}"
        ) from e

    _validate_snapshot(snapshot, path_obj)

    graph = nx.Graph()
    for node in snapshot["nodes"]:
        graph.add_node(node["id"], node_type=node["node_type"])
    for edge in snapshot["edges"]:
        graph.add_edge(edge["u"], edge["v"], loss=float(edge["loss"]))

    return graph
