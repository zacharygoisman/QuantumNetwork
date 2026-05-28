#scheduling/scheduler.py
"""
CP-SAT to check for bandwidth contention and schedule channels.
"""

#ZHG
#2026.03.26
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
from ortools.sat.python import cp_model

def _make_edges(path):
    return [tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)]

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

    edges_u1 = [_make_edges(p) for p in paths_u1]
    edges_u2 = [_make_edges(p) for p in paths_u2]
    all_edges = sorted(set(e for el in edges_u1 + edges_u2 for e in el))

    model = cp_model.CpModel()

    swap = [model.NewBoolVar(f"swap_{i}") for i in range(nlinks)]
    use = {}
    pos = {}
    neg = {}

    # create channel-use vars
    for i, src in enumerate(link_src):
        pool = list(sources[src]["available_channels"])
        for c in pool:
            use[i, c] = model.NewBoolVar(f"use_{i}_{c}")
        model.Add(sum(use[i, c] for c in pool) == k_vals[i])

    # uniqueness within a source
    src_to_idxs = {}
    for i, s in enumerate(link_src):
        src_to_idxs.setdefault(s, []).append(i)

    for s, idxs in src_to_idxs.items():
        pool = list(sources[s]["available_channels"])
        for c in pool:
            model.Add(sum(use[i, c] for i in idxs) <= 1)

    # signed flow vars on edges
    for i, src in enumerate(link_src):
        pool = list(sources[src]["available_channels"])
        e1 = set(edges_u1[i])
        e2 = set(edges_u2[i])

        for e in all_edges:
            for c in pool:
                pos[i, e, c] = model.NewBoolVar(f"pos_{i}_{e}_{c}")
                neg[i, e, c] = model.NewBoolVar(f"neg_{i}_{e}_{c}")

                if e not in e1 and e not in e2:
                    model.Add(pos[i, e, c] == 0)
                    model.Add(neg[i, e, c] == 0)
                    continue

                if e in e1 and e in e2:
                    # Edge appears on both path1 and path2 for this link.
                    # In this case the chosen channel traverses the edge in
                    # both directions (one + and one -). Both signed vars
                    # should therefore reflect the use variable (i.e. both
                    # set to use[i,c]) rather than constraining their sum
                    # to equal use (which would forbid both being active).
                    model.Add(pos[i, e, c] == use[i, c])
                    model.Add(neg[i, e, c] == use[i, c])
                elif e in e1:
                    # path1 gets + if not swapped, - if swapped
                    model.Add(pos[i, e, c] <= use[i, c])
                    model.Add(neg[i, e, c] <= use[i, c])

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
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"success": False}

    assignment = {}
    for i, lk in enumerate(links):
        src = link_src[i]
        chosen = [c for c in sources[src]["available_channels"] if solver.Value(use[i, c]) == 1]
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

    return {"success": True, "assignment": assignment}