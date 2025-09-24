import networkx as nx
import numpy as np
import itertools
from ortools.sat.python import cp_model
from collections import defaultdict
import heapq
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class _PathOpt:
    src: str
    path1: tuple
    path2: tuple
    total_loss: float
    user1_loss: float
    user2_loss: float
    y1: float
    y2: float
    ub: float  # per-link UB if this path were chosen

def _per_link_path_ub(F_min, y1, y2):
    """Tight-ish per-path UB for this link choice (reuses your per_link_ub_value)."""
    if F_min < 0.5: F_min = 0.5
    val, *_ = per_link_ub_value(y1, y2, F_min, x_one_channel_cap=None, on_infeasible="none")
    return float(val) if (val is not None and np.isfinite(val)) else float("-inf")

def per_link_ub_value(y1, y2, F_min, x_one_channel_cap, on_infeasible="zero"):
    """
    Per-link best utility for the 'best-possible' line, with small fidelity limits
    treated as 0.5 (useful-entanglement floor).

    Args:
        y1, y2: link parameters
        F_min: requested fidelity floor (will be clamped to >= 0.5)
        x_one_channel_cap: physical cap for x for this link when it's alone on its best source
                           (if None, no physical cap beyond fidelity)
        on_infeasible: "zero" -> return 0 if F constraint infeasible; "none" -> return None

    Returns:
        float (utility = log10(rate) at the per-link bound), or None if infeasible and on_infeasible="none".
    """
    #1) Enforce useful-entanglement floor
    F_eff = max(0.5, float(F_min))

    #2) Build quadratic for F(x) >= F_eff:
    #  t x^2 + (tA - 3) x + tB <= 0, with t = 4*F_eff - 1  (note: with F_eff>=0.5, t >= 1)
    t = 4.0*F_eff - 1.0  #>= 1.0 when F_eff >= 0.5
    A = 2.0*(y1 + y2) + 1.0
    B = 4.0*y1*y2

    #Discriminant
    D = (t*A - 3.0)**2 - 4.0*(t**2)*B
    if D < 0.0:
        #No x satisfies F(x) >= F_eff for this link
        return None, None, None if on_infeasible == "none" else 0.0

    #3) Take the **upper** admissible root (larger root)
    x_fid_max = (3.0 - t*A + math.sqrt(D)) / (2.0*t)
    if x_fid_max < 0.0:
        #Entire feasible interval is negative; no nonnegative x satisfies it
        return None, None, None if on_infeasible == "none" else 0.0

    #4) Apply physical one-channel cap if provided
    x_star = x_fid_max if (x_one_channel_cap is None) else min(x_fid_max, float(x_one_channel_cap))

    #5) Compute utility at the cap
    R = _rate_poly(x_star, y1, y2)
    #(R is always > 0 with these coefficients; safeguard anyway)
    if R <= 0.0:
        return None, None, None if on_infeasible == "none" else 0.0
    return math.log10(R), x_star, R

def _rate_poly(x, y1, y2):
    A = 2.0*(y1 + y2) + 1.0
    B = 4.0*y1*y2
    return x*x + A*x + B

def prepare_path_sets(network, *, top_k_per_link=8, shortlist_by="ub"):
    """
    Build compact, precomputed per-link path options with ub/loss cached.
    Returns a list[path_options], where each path_options is a list[_PathOpt].
    """
    links = network.graph['desired_links']
    out = []
    for L in links:
        F = float(L.get('fidelity_limit', 0.5))
        opts = []
        for p in L.get('paths', []):
            y1 = float(p.get('y1', 0.0))
            y2 = float(p.get('y2', 0.0))
            ub = _per_link_path_ub(F, y1, y2)
            opts.append(_PathOpt(
                src=p['source'],
                path1=tuple(p['path1']),
                path2=tuple(p['path2']),
                total_loss=float(p.get('total_loss', 0.0)),
                user1_loss=float(p.get('user1_loss', 0.0)),
                user2_loss=float(p.get('user2_loss', 0.0)),
                y1=y1, y2=y2, ub=ub
            ))
        if not opts:
            out.append([])
            continue

        # shortlist
        if shortlist_by == "loss":
            opts.sort(key=lambda o: o.total_loss)                # ascending loss
        else:
            opts.sort(key=lambda o: o.ub, reverse=True)          # descending UB
        out.append(opts[:top_k_per_link] if top_k_per_link else opts)
    return out

def combo_stream_best_first(network, path_sets, *, max_combos=None, time_limit_s=None,
                            capacity_fn=None):
    """
    Yield combos lazily in descending sum-UB order.
    Each yielded record mirrors your existing combo dict shape.
    """
    import time as _time
    t0 = _time.time()
    n = len(path_sets)
    if n == 0: 
        return
    # if any link has no options, stop
    if any(len(ps)==0 for ps in path_sets):
        return
    
    def partial_capacity_ok(idxs_prefix):
        # counts for assigned indices
        counts = {}
        for i, pi in enumerate(idxs_prefix):
            p = path_sets[i][pi]
            counts[p.src] = counts.get(p.src, 0) + 1
        # remaining links can only increase counts, never decrease
        if capacity_fn is None:
            return True
        return capacity_fn(counts)

    # Precompute per-link lengths & an UB-only seed (pick best option per link)
    # State tuple: (neg_sum_ub, indices tuple)
    best_idx = tuple(0 for _ in range(n))
    def sum_ub(idxs):
        return sum(path_sets[i][idxs[i]].ub for i in range(n))

    # Max heap via negative key
    heap = [(-sum_ub(best_idx), best_idx)]
    seen = {best_idx}

    produced = 0
    links = network.graph['desired_links']

    while heap:
        if time_limit_s is not None and (_time.time() - t0) >= time_limit_s:
            break
        negub, idxs = heapq.heappop(heap)
        cur_sum_ub = -negub

        # Optional capacity pruning on partials would require implicit BFS levels;
        # Here we check capacity on full assignment (cheap & keeps code simple):
        # Build combo dict
        combo_with_links = []
        total_loss = 0.0
        src_counts = {}
        for i, pi in enumerate(idxs):
            p = path_sets[i][pi]
            total_loss += p.total_loss
            src_counts[p.src] = src_counts.get(p.src, 0) + 1
            combo_with_links.append({
                'link': tuple(links[i]['link']),
                'fidelity_limit': float(links[i]['fidelity_limit']),
                'path': {
                    'source': p.src, 'path1': list(p.path1), 'path2': list(p.path2),
                    'total_loss': p.total_loss, 'user1_loss': p.user1_loss, 'y1': p.y1,
                    'user2_loss': p.user2_loss, 'y2': p.y2
                }
            })

        if capacity_fn is None or capacity_fn(src_counts):
            yield {
                'combo': combo_with_links,
                'total_loss': total_loss,
                'path_id': produced,
                'usage_info': 'untested',
                '_ub_combo': cur_sum_ub
            }
            produced += 1
            if max_combos is not None and produced >= max_combos:
                break

        # expand neighbors by bumping one link’s option index (+1) each time
        for j in range(n):
            nxt = list(idxs)
            if nxt[j] + 1 < len(path_sets[j]):
                nxt[j] += 1
                nxt = tuple(nxt)
                if nxt not in seen and partial_capacity_ok(nxt):
                    seen.add(nxt)
                    heapq.heappush(heap, (-sum_ub(nxt), nxt))
#double_dijkstra finds the lowest loss paths (and losses) for every source and for every link
def double_dijkstra(network, sources, tau = 0, d1 = 0, d2 = 0):
    '''
    Inputs:
        network: network graph object with topology
        sources: list of all sources in the network
        tau, d1, and d2 are constants required for calculating the y1 and y2 values from the loss. They are defaulted to 0 if we do not care about them

    Outputs:
        network: return the modified network object with best path info
    '''

    for link_index in range(len(network.graph['desired_links'])): #Run Double Dijkstra for each link
        best_path_info = [] 
        link_entry = network.graph['desired_links'][link_index]
        u1, u2 = link_entry['link']

        for source in sources: #For each source we want to find the best lowest loss path from each user
            try:
                path_u1 = nx.dijkstra_path(network, source=u1, target=source, weight="loss")
                loss_u1 = nx.dijkstra_path_length(network, source=u1, target=source, weight="loss")
                path_u2 = nx.dijkstra_path(network, source=u2, target=source, weight="loss")
                loss_u2 = nx.dijkstra_path_length(network, source=u2, target=source, weight="loss")
            except nx.NetworkXNoPath:
                continue

            total_loss = loss_u1 + loss_u2 #Total loss of path
            y1_value = tau * d1 * 10 ** (loss_u1 / 10) #Calculate y1 for path. Tau * dark counts / efficiency
            y2_value = tau * d2 * 10 ** (loss_u2 / 10) #Calculate y2 for path. Tau * dark counts / efficiency
            best_path_info.append({'source':source,'path1':path_u1,'path2':path_u2,'total_loss':total_loss, 'user1_loss':loss_u1, 'y1': y1_value, 'user2_loss':loss_u2, 'y2': y2_value}) #Path info packaging

        network.graph['desired_links'][link_index]['paths'] = best_path_info #Store best paths into network

        if not best_path_info: #If no best sources for a link then output as a warning
            print(f"No common sources for link {u1} -- {u2}")

    return network

#Adds y1, y2, and fidelity limit information into network object's link information
def link_info_packaging(network, fidelity_limit, y1 = None, y2 = None):
    '''
    Inputs:
    network is the topology network object
    fidelity_limit is an array of user determined fidelity constraints for each link 
    y1 and y2 are meant to be empty unless we are manually inserting our own y1 and y2 values rather than calculating them from loss

    Outputs:
    network is the topology network object with some additional link information packed into it
    '''
    links = network.graph['desired_links']
    for link_index in range(len(links)):
        if y1 is not None: #If we are manually adding the y1 parameter instead of using the network loss
            network.graph['desired_links'][link_index]['y1'] = y1[link_index]
        if y2 is not None: #If we are manually adding the y2 parameter instead of using the network loss
            network.graph['desired_links'][link_index]['y2'] = y2[link_index]
        network.graph['desired_links'][link_index]['fidelity_limit'] = fidelity_limit[link_index]
    return network

#Creates a list of every possible path from the set of optimal double dijkstra paths. It outputs this list of dictionaries which contain link information for each of the paths
def path_combinations(network):
    """
    Inputs:
    network object containing double dijkstra information
    
    Outputs:
    sorted_results contains a list of every possible path and relavant link information for each in a dictionary
    """
    link_info = network.graph['desired_links'] #Figure out what links we need
    path_lists = [entry['paths'] for entry in link_info] #Extract path lists per link

    combinations = list(itertools.product(*path_lists)) #Generate all combinations

    #Package link info into results
    results = []
    path_id = 0 #Path id for identifying specific paths
    for combo in combinations:
        total_loss = 0
        combo_with_links = []
        for i, path in enumerate(combo):
            total_loss += path.get('total_loss')
            combo_with_links.append({
                'link': link_info[i]['link'],
                'fidelity_limit': link_info[i]['fidelity_limit'],
                'path': path  #full path dict preserved
            })
        results.append({
            'combo': combo_with_links,
            'total_loss': total_loss,
            'path_id': path_id,
            'usage_info': 'untested'
        })
        path_id += 1

    return results


#choose_paths decides which paths to end up using from the total list of paths
def choose_paths(path_combinations): #TODO: Test other means of choosing for source swapping?
    """
    Inputs:
    path_combinations
    
    Outputs:
    path_choice is the chosen path dictionary with all the link information for the set of links
    path_id is the id value associated with the path
    If no path is being used, we return two Nones instead
    """
    untested_paths = [path for path in path_combinations if path.get('usage_info') == 'untested']
    if untested_paths: #Choose the lowest loss case
        path_choice = untested_paths[0]
        path_choice['usage_info'] = 'tested' #Indicate specific path is being used
        path_id = path_choice['path_id']
        return path_choice, path_id
    else: #No remaining paths
        return None, None


#make_edges is a function that finds all the edges from a set of path nodes
def make_edges(path):
    """Turn a node‐path ['A','B','C'] into edges [('A','B'),('B','C')] sorted canonically."""
    return [tuple(sorted((path[i], path[i+1]))) for i in range(len(path)-1)]

#Function to map channel allocation amounts to links in correct order for check_interference
def extract_link_ch_counts(source_dicts, ordered_links):
    """
    Inputs:
    source_dicts contains the list of source dictionaries containing source information like number of channels
    ordered_links is the list of links in the same global order as in the creation functions
    
    Outputs:
    Returns a list of source and channel numbers associated with the specific links we input
    """
    def _to_int_channels(k):
        #Mirrors the safe converter used in plotting
        try:
            if hasattr(k, 'value'):
                v = k.value
                if isinstance(v, (list, tuple, np.ndarray)):
                    if len(v) == 0:
                        return 0
                    return int(round(float(v[0])))
                return int(round(float(v)))
            if isinstance(k, (list, tuple, np.ndarray)):
                arr = np.asarray(k, dtype=float)
                if arr.size == 0:
                    return 0
                if arr.size == 1:
                    return int(round(float(arr.item())))
                return int(round(float(arr.sum())))
            return int(round(float(k)))
        except Exception:
            try:
                return int(round(float(np.asarray(k).item())))
            except Exception:
                return 0

    link_to_count = {}
    for src_info in source_dicts:
        for link, alloc in zip(src_info['links'], src_info['channel_allocation']):
            link_to_count[link] = _to_int_channels(alloc)

    return [link_to_count[link] for link in ordered_links]


#check_interference checks the topology of the network in order to see if frequency channels overlap. If so then it will test other channel orders for the given link paths.
def check_interference(path_choice, results, sources):
    """
    Decide k_i channels per link + an orientation bit (swap),
    s.t. no physical edge ever carries the same (signed) channel twice.
    Returns [(to_u1, to_u2), …]   or None if infeasible.
    """
    def edges(path):
        return [tuple(sorted((path[i], path[i+1])))
                for i in range(len(path)-1)]

    #unpack link data ----------------------------------------------------
    links       = path_choice['combo']
    nlinks      = len(links)
    paths_u1    = [lk['path']['path1'] for lk in links]
    paths_u2    = [lk['path']['path2'] for lk in links]      #← REAL 2nd path
    link_src    = [lk['path']['source'] for lk in links]
    link_names  = [lk['link']            for lk in links]
    k_vals      = extract_link_ch_counts(results, link_names)

    edges_u1    = [edges(p) for p in paths_u1]
    edges_u2    = [edges(p) for p in paths_u2]
    all_edges   = set(e for el in edges_u1+edges_u2 for e in el)
    all_ch      = sorted({c for src in link_src
                            for c in sources[src]['available_channels']})

    #model ---------------------------------------------------------------
    m = cp_model.CpModel()
    swap  = [m.NewBoolVar(f"swap_{i}") for i in range(nlinks)]
    use   = {}              #(i,c)  channel chosen?
    pos   = {}              #(i,e,c) channel c flows + on edge e for link i
    neg   = {}              #(i,e,c) channel c flows - on edge e for link i

    #(1) create all use[i,c] and per-link cardinalities
    for i, src in enumerate(link_src):
        pool = sources[src]['available_channels']
        for c in pool:
            use[i, c] = m.NewBoolVar(f"use_{i}_{c}")
        m.Add(sum(use[i, c] for c in pool) == k_vals[i]) #Set how many channels we can use for each link

    #(2) per-source, per-channel uniqueness
    src_to_idxs = {}
    for i, s in enumerate(link_src):
        src_to_idxs.setdefault(s, []).append(i)
    for s, idxs in src_to_idxs.items():
        pool = sources[s]['available_channels']
        for c in pool:
            #Each channel index at source s can be assigned to at most one link
            m.Add(sum(use[i, c] for i in idxs) <= 1)

    #(3) now create pos/neg vars and reify with swap
    for i, src in enumerate(link_src):
        pool = sources[src]['available_channels']
        for c in pool:
            for e in edges_u1[i] + edges_u2[i]:
                pos[i, e, c] = m.NewBoolVar(f"p_{i}_{e}_{c}")
                neg[i, e, c] = m.NewBoolVar(f"n_{i}_{e}_{c}")

    #reify edge usage -----------------------------------------------
    for i, src in enumerate(link_src):
        pool = sources[src]['available_channels']

        # U1 arm: swap=0 → +c, swap=1 → –c
        for c in pool:
            for e in edges_u1[i]:
                m.Add(pos[i, e, c] == 1).OnlyEnforceIf([use[i, c], swap[i].Not()])
                m.Add(pos[i, e, c] == 0).OnlyEnforceIf(use[i, c].Not())
                m.Add(pos[i, e, c] == 0).OnlyEnforceIf(swap[i])
                m.Add(neg[i, e, c] == 1).OnlyEnforceIf([use[i, c], swap[i]])
                m.Add(neg[i, e, c] == 0).OnlyEnforceIf(use[i, c].Not())
                m.Add(neg[i, e, c] == 0).OnlyEnforceIf(swap[i].Not())

            # U2 arm: opposite sign mapping
            for e in edges_u2[i]:
                m.Add(neg[i, e, c] == 1).OnlyEnforceIf([use[i, c], swap[i].Not()])
                m.Add(neg[i, e, c] == 0).OnlyEnforceIf(use[i, c].Not())
                m.Add(neg[i, e, c] == 0).OnlyEnforceIf(swap[i])
                m.Add(pos[i, e, c] == 1).OnlyEnforceIf([use[i, c], swap[i]])
                m.Add(pos[i, e, c] == 0).OnlyEnforceIf(use[i, c].Not())
                m.Add(pos[i, e, c] == 0).OnlyEnforceIf(swap[i].Not())

    #edge-conflict: ≤1 pos and ≤1 neg per (edge, channel) --------------
    for e in all_edges:
        for c in all_ch:
            pvars = [pos[i, e, c] for i in range(nlinks) if (i, e, c) in pos]
            nvars = [neg[i, e, c] for i in range(nlinks) if (i, e, c) in neg]
            if pvars:  m.Add(sum(pvars) <= 1)
            if nvars:  m.Add(sum(nvars) <= 1)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    if solver.Solve(m) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    #assemble answer -----------------------------------------------------
    assignment = []
    for i, src in enumerate(link_src):
        pool   = sources[src]['available_channels']
        chosen = [c for c in pool if solver.BooleanValue(use[i, c])]
        if solver.BooleanValue(swap[i]):
            assignment.append(([-c for c in chosen], chosen))
        else:
            assignment.append((chosen, [-c for c in chosen]))
    return assignment



def main():
    import create_network
    num_usr, num_src, num_edg, num_lnk= 10, 5, 40, 5
    loss_range=(1,30)
    y1 = [0, 0.0034, 0, 0.0299, 0.0385, 0.0625, 0.0733, 0, 0.1106, 0.125, 0.1489, 0]
    y2 = [0, 0.0006, 0.0357, 0.0051, 0.0066, 0.0107, 0.0126, 0.1818, 0.019, 0.0214, 0.0256, 0.2979]
    fidelity_limit = np.repeat(0.7,len(y1))
    network, node_usr, sources = create_network.create_network(num_usr, num_src, num_edg, loss_range, num_channels_per_source=[None], topology='ring', density = 0.1)
    network = create_network.create_links(network, node_usr, num_lnk, link_pairs = None)
    network = create_network.link_info_packaging(network, y1, y2, fidelity_limit)

    network = double_dijkstra(network, sources)
    combinations = path_combinations(network)
    path_choice = choose_paths(combinations)
    print(path_choice)

if __name__ == "__main__":
    main()