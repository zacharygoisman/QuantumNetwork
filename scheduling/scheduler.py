#scheduling/scheduler.py
"""
CP-SAT to check for bandwidth contention and schedule channels.
"""

#ZHG
#2026.03.26
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
from ortools.sat.python import cp_model

# Module-level cache: signatures of equivalent scheduling problems map to results.
# Two combos that share the same paths/sources/channel-pools/k-allocations have
# identical CP-SAT models, so we can reuse the previously found assignment.
_SCHED_CACHE = {}
_SCHED_CACHE_MAX = 4096


def _make_edges(path):
    return tuple(tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1))


def _build_cache_key(links, link_src, k_vals, sources):
    """Stable hashable signature of the scheduling sub-problem."""
    parts = []
    for i, lk in enumerate(links):
        pool = tuple(sorted(sources[link_src[i]]["available_channels"]))
        parts.append((
            link_src[i],
            tuple(lk["path1"]),
            tuple(lk["path2"]),
            int(k_vals[i]),
            pool,
        ))
    return tuple(parts)


def check_interference(combo, alloc_result, sources):
    """
    combo: tuple/list of chosen path options, one per link
    alloc_result: output of allocate_combo(...)
    returns:
        {"success": True, "assignment": {...}} or {"success": False}
    """

    links = list(combo)
    nlinks = len(links)

    paths_u1 = [lk["path1"] for lk in links]
    paths_u2 = [lk["path2"] for lk in links]
    link_src = [lk["source"] for lk in links]

    k_vals = []
    for lk in links:
        rec = alloc_result["allocation"][id(lk)]
        k_vals.append(int(rec["k"]))

    # Per-link edge sets (as sets for fast membership checks)
    edges_u1 = [set(_make_edges(p)) for p in paths_u1]
    edges_u2 = [set(_make_edges(p)) for p in paths_u2]

    # Cache lookup
    cache_key = _build_cache_key(links, link_src, k_vals, sources)
    cached = _SCHED_CACHE.get(cache_key)
    if cached is not None:
        # Rebuild assignment with current link objects but cached channel/swap
        assignment = {}
        for i, lk in enumerate(links):
            entry = cached[i]
            chosen = entry["channels"]
            swapped = entry["swap"]
            if not swapped:
                to_u1 = [int(c) for c in chosen]
                to_u2 = [-int(c) for c in chosen]
            else:
                to_u1 = [-int(c) for c in chosen]
                to_u2 = [int(c) for c in chosen]
            assignment[i] = {
                "link": lk["link"],
                "source": link_src[i],
                "swap": swapped,
                "channels": [int(c) for c in chosen],
                "to_u1": to_u1,
                "to_u2": to_u2,
            }
        return {"success": True, "assignment": assignment}

    # Collect only edges that are actually used by at least one link
    used_edges = set()
    for i in range(nlinks):
        used_edges.update(edges_u1[i])
        used_edges.update(edges_u2[i])

    # For each edge, which links touch it on path1 or path2
    edge_to_links = {e: [] for e in used_edges}
    for i in range(nlinks):
        for e in edges_u1[i] | edges_u2[i]:
            edge_to_links[e].append(i)

    model = cp_model.CpModel()

    swap = [model.NewBoolVar(f"swap_{i}") for i in range(nlinks)]

    # Channel-use vars: only create for channels in this link's source pool.
    use = {}
    src_pool = [tuple(sources[s]["available_channels"]) for s in link_src]
    for i in range(nlinks):
        pool = src_pool[i]
        for c in pool:
            use[i, c] = model.NewBoolVar(f"use_{i}_{c}")
        model.Add(sum(use[i, c] for c in pool) == k_vals[i])

    # Uniqueness within each source
    src_to_idxs = {}
    for i, s in enumerate(link_src):
        src_to_idxs.setdefault(s, []).append(i)

    for s, idxs in src_to_idxs.items():
        pool = sources[s]["available_channels"]
        for c in pool:
            terms = [use[i, c] for i in idxs]
            if len(terms) > 1:
                model.Add(sum(terms) <= 1)

    # Signed-flow vars only on (i, e, c) tuples that can actually be active
    # (i.e. edge e is on path1 or path2 of link i, channel c is in i's pool).
    pos = {}
    neg = {}

    for i in range(nlinks):
        pool = src_pool[i]
        e1 = edges_u1[i]
        e2 = edges_u2[i]
        both = e1 & e2
        only1 = e1 - both
        only2 = e2 - both

        not_swap = swap[i].Not()

        for e in both:
            # Edge appears on both directions of the link: both signed vars
            # mirror the channel-use indicator (channel traverses both ways).
            for c in pool:
                pv = model.NewBoolVar(f"pos_{i}_{e}_{c}")
                nv = model.NewBoolVar(f"neg_{i}_{e}_{c}")
                model.Add(pv == use[i, c])
                model.Add(nv == use[i, c])
                pos[i, e, c] = pv
                neg[i, e, c] = nv

        for e in only1:
            for c in pool:
                pv = model.NewBoolVar(f"pos_{i}_{e}_{c}")
                nv = model.NewBoolVar(f"neg_{i}_{e}_{c}")
                # path1: + when not swapped, - when swapped.
                # pv = use AND not_swap; nv = use AND swap.
                model.AddBoolAnd([use[i, c], not_swap]).OnlyEnforceIf(pv)
                model.AddBoolOr([use[i, c].Not(), swap[i]]).OnlyEnforceIf(pv.Not())
                model.AddBoolAnd([use[i, c], swap[i]]).OnlyEnforceIf(nv)
                model.AddBoolOr([use[i, c].Not(), not_swap]).OnlyEnforceIf(nv.Not())
                pos[i, e, c] = pv
                neg[i, e, c] = nv

        for e in only2:
            for c in pool:
                pv = model.NewBoolVar(f"pos_{i}_{e}_{c}")
                nv = model.NewBoolVar(f"neg_{i}_{e}_{c}")
                # path2: - when not swapped, + when swapped.
                # pv = use AND swap; nv = use AND not_swap.
                model.AddBoolAnd([use[i, c], swap[i]]).OnlyEnforceIf(pv)
                model.AddBoolOr([use[i, c].Not(), not_swap]).OnlyEnforceIf(pv.Not())
                model.AddBoolAnd([use[i, c], not_swap]).OnlyEnforceIf(nv)
                model.AddBoolOr([use[i, c].Not(), swap[i]]).OnlyEnforceIf(nv.Not())
                pos[i, e, c] = pv
                neg[i, e, c] = nv

    #No signed channel collision on the same edge: at most one + and one - 
    # per edge per channel, and only consider (link, channel) pairs that exist.
    for e, link_idxs in edge_to_links.items():
        # Group by channel -> list of link indices that can use channel c on edge e
        per_channel = {}
        for i in link_idxs:
            for c in src_pool[i]:
                if (i, e, c) in pos:
                    per_channel.setdefault(c, []).append(i)
        for c, idxs in per_channel.items():
            if len(idxs) > 1:
                model.Add(sum(pos[i, e, c] for i in idxs) <= 1)
                model.Add(sum(neg[i, e, c] for i in idxs) <= 1)

    solver = cp_model.CpSolver()
    # Lean parameter set: this is a feasibility problem with small models per
    # combo, so heavy linearization/probing usually costs more than it saves.
    solver.parameters.num_search_workers = 1
    solver.parameters.log_search_progress = False
    solver.parameters.cp_model_presolve = True
    solver.parameters.linearization_level = 0
    solver.parameters.cp_model_probing_level = 0
    solver.parameters.max_time_in_seconds = 10.0

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"success": False}

    assignment = {}
    cache_entry = []
    for i, lk in enumerate(links):
        src = link_src[i]
        chosen = [c for c in src_pool[i] if solver.Value(use[i, c]) == 1]
        swapped = bool(solver.Value(swap[i]))

        # Preserve the old sign convention:
        # swap = 0 -> user1 gets +c, user2 gets -c
        # swap = 1 -> user1 gets -c, user2 gets +c
        if not swapped:
            to_u1 = [int(c) for c in chosen]
            to_u2 = [-int(c) for c in chosen]
        else:
            to_u1 = [-int(c) for c in chosen]
            to_u2 = [int(c) for c in chosen]

        assignment[i] = {
            "link": lk["link"],
            "source": src,
            "swap": swapped,
            "channels": [int(c) for c in chosen],
            "to_u1": to_u1,
            "to_u2": to_u2,
        }
        cache_entry.append({"channels": [int(c) for c in chosen], "swap": swapped})

    if len(_SCHED_CACHE) < _SCHED_CACHE_MAX:
        _SCHED_CACHE[cache_key] = cache_entry

    return {"success": True, "assignment": assignment}
