#routing/paths.py
"""
Functions to compute candidate paths for each user pair (link) based on the network topology and link losses.
"""

#ZHG
#2026.03.24
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
import networkx as nx
from analysis.metrics import compute_ub_max
from routing.pairing import compute_path_loss, compute_y

def build_path_options(network, links, sources, cfg):
    """For each user pair (link), compute candidate path pairs from each source."""
    all_link_options = []
    user_dark_count = _user_dark_count_map(network, cfg)

    for link_idx, (u1, u2) in enumerate(links):
        candidates = []
        f_req = float(max(0.5, cfg.fidelity_limit[link_idx]))
        d_u1 = user_dark_count[str(u1)]
        d_u2 = user_dark_count[str(u2)]

        for s in sources: #Yen will find paths from s to u1 and s to u2 for each source
            paths_u1 = compute_n_paths(network, s, u1, cfg.n_paths_per_leg, cfg)
            paths_u2 = compute_n_paths(network, s, u2, cfg.n_paths_per_leg, cfg)

            #Merge paths for u1 and u2 to create candidate paths.
            for p1 in paths_u1:
                loss1 = compute_path_loss(network, p1)
                for p2 in paths_u2:
                    loss2 = compute_path_loss(network, p2)

                    #Store candidate with its total loss and computed y values for later scoring and pruning, and the allocation.
                    y1 = compute_y(loss1, cfg.tau, d_u1)
                    y2 = compute_y(loss2, cfg.tau, d_u2)

                    path_ub = compute_ub_max(y1, y2, f_min=f_req) #Determine maximum upper bound for this path

                    candidates.append({
                        "link": (u1, u2),
                        "link_idx": link_idx,
                        "source": s,
                        "users": (u1, u2),
                        "path1": p1,
                        "path2": p2,
                        "y1": y1,
                        "y2": y2,
                        "dark_count_1": d_u1,
                        "dark_count_2": d_u2,
                        "fidelity_limit": f_req,
                        "total_loss": loss1 + loss2,
                        "path_ub": path_ub,
                    })
        if not candidates:
            all_link_options.append([])
            continue

        #Determine link upper bound using the maximum path upper bound for each source path candidate.
        link_ub = max(c["path_ub"] for c in candidates)
        for c in candidates:
            c["link_ub"] = link_ub #All candidates have the same link upper bound

        if cfg.upper_bound_sort:
            candidates.sort(key=lambda x: x["path_ub"], reverse=True)
        else:
            candidates.sort(key=lambda x: x["total_loss"]) #Sort by total loss (TODO: Potential for more complex scoring later that also considers path diversity, channel availability, etc.)
        best_k = candidates[:cfg.combo_limit_per_link]

        all_link_options.append(best_k)
    return all_link_options

def _user_dark_count_map(network, cfg):
    """Map each user node to its dark-count rate, ordered by the numeric suffix in the
    node name. Raises ValueError if the count of users doesn't match cfg.dark_count_rate."""
    user_nodes = [
        str(n) for n, data in network.nodes(data=True)
        if data.get("node_type") == "user" or str(n).startswith("U")
    ]
    user_nodes = sorted(user_nodes, key=lambda s: int("".join(ch for ch in s if ch.isdigit()) or 0))

    if len(user_nodes) != len(cfg.dark_count_rate):
        raise ValueError("dark_count_rate must match number of user nodes")

    return {
        user: float(cfg.dark_count_rate[i])
        for i, user in enumerate(user_nodes)
    }

def compute_n_paths(network, source, target, n, cfg):
    """
    Returns up to n lowest-loss paths.
    If diversity is enabled, returns n diverse paths instead.
    """
    try:
        #If diversity is enabled, we will generate more than n paths and then filter them down to n diverse paths.
        n_internal = n
        if cfg.use_diverse_paths:
            n_internal = cfg.diversity_factor * n

        gen = nx.shortest_simple_paths(network, source, target, weight="loss") #Yen's algorithm is implemented in NetworkX to find the lowest loss paths from a user to a source

        #Generate up to n paths simulating n lowest loss paths from Yen's algorithm
        paths = []
        for i, path in enumerate(gen):
            if i >= n_internal:
                break
            paths.append(path)

        #Apply diversity filtering if enabled
        if cfg.use_diverse_paths:
            paths = filter_diverse_paths(paths, n, cfg.diversity_threshold) #Select up to n paths that are sufficiently different based on edge overlap
        else:
            paths = paths[:n] #If not enforcing diversity, just take the top n paths by loss
        return paths

    #If no path exists, return empty list
    except nx.NetworkXNoPath:
        return []

def filter_diverse_paths(paths, k, threshold):
    """
    Select up to k paths that are sufficiently different
    based on edge overlap.
    """
    selected = []

    #Iterate through each path and compare with selected paths
    for p in paths:
        if all(path_overlap(p, q) < threshold for q in selected): #If this path has low overlap with all previously selected paths, then we consider it diverse enough to add to our selection
            selected.append(p)

        if len(selected) >= k: #Stop once we have selected k diverse paths
            break

    return selected

def path_overlap(p1, p2):
    """Computes edge overlap between two paths as a fraction of shared edges over total unique edges."""
    edges1 = set(zip(p1, p1[1:])) #Convert path to set of edges
    edges2 = set(zip(p2, p2[1:]))
    return len(edges1 & edges2) / len(edges1 | edges2) #Overlap is number of shared edges divided by total unique edges across both paths