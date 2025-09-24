
#ultimate.py
#Code that calls many functions
#
#-------------------------------------------------------------
import json, csv, math, time, traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import math
import traceback

import create_network
import routing
import allocate_channels
import plot

#==========================
#1) Configuration
#==========================

@dataclass
class Config:
    #Topology
    num_users = 20
    num_sources = 1
    num_edges = 10000
    num_links = 3
    loss_range = (0, 5.0) #(12.0, 32.0)
    topology = "kite"   #'ring' | 'dense' | 'star' | 'kite' | None
    kite_loss_values = {  #Only for kite topology
        ('Charlie',  'Bob'): 0.75, #Charlie to Bob
        ('Erin',    'Bob'): 0.75, #Erin to Bob
        ('Charlie',  'Alice')  : 4, #Charlie to Alice
        ('Erin',    'Alice')  :  0.75, #Erin to Alice
        ('Dave','Alice')  : 1, #Dave to Alice
    }
    density = 0.2               #used when topology='dense'
    num_channels_per_source = None  #None => default in create_network
    #num_channels_per_source = [70, 43, 60, 55, 85, 52, 63, 63, 72, 29, 9, 34, 22]

    #Physics / constraints
    fidelity_limit = np.repeat(0.94, num_links)
    #fidelity_limit = [0.9, 0.88, 0.89, 0.94, 0.86, 0.87, 0.94, 0.91, 0.9, 0.93, 0.89, 0.94, 0.93, 0.85, 0.92, 0.94, 0.91, 0.9, 0.93, 0.89, 0.94, 0.93, 0.85, 0.92]
    tau = 1e-9
    d1 = 100.0
    d2 = 3500.0
    y1 = None #None => default in create_network
    y2 = None #None => default in create_network

    #Pruning / UB control
    enable_fast_prune = True         #prune based on figuring out which cases are impossible based on how many links are traveling through the edges
    skip_below_best_ub = True        #skip combos whose UB can't beat current best
    ub_skip_eps = 1e-9               #small tolerance for float comparisons

    #Salvage (quick retries for great but infeasible combos) #TODO: Get salvage working
    enable_salvage = False
    salvage_k_caps = [2,1]
    salvage_trigger_ratio = 0.92

    #Early stop controls (all optional)
    stop_attempts = 0     #0 disables attempt‑count stop
    stop_successes = 20    #0 disables 'N successes' stop
    loss_multiplier_stop = 1000000  #large => disabled

    # Parallelism
    parallel = True             # turn on/off parallel evaluation
    workers = None              # None => os.cpu_count()
    executor = "process"        # "process" | "thread"

    # Parallel batching & flow control
    parallel_batch_size = 16         # submit at most this many new tasks in a batch
    max_inflight = 64                # cap on tasks concurrently in-flight
    as_completed_yield = 4           # process this many completions before submitting more

    #I/O
    out_dir = Path("outputs")
    make_plots = True #Toggles plot generation
    report_csv = False #Create the csv files

    # Combo-generation controls
    top_k_per_link = 0       # shortlist size per link (None/0 = keep all)
    shortlist_by = "ub"      # "ub" or "loss"
    max_combos = None # hard cap on number of combos pulled
    time_limit_s = None  # wall-clock budget for combo generation

    #Misc
    verbose = False #For printing terminal messages

#==========================
#2) Logging helpers
#==========================

def log(msg, *, cfg: Config):
    if cfg.verbose:
        print(msg)

def _log_exception(where, cfg, **ctx): #Try/except error message helper
    if getattr(cfg, "verbose", False):
        print(f"\n[ERROR] {where}")
        if ctx:
            print("Context:", ", ".join(f"{k}={v}" for k, v in ctx.items()))
        print(traceback.format_exc())

#=======================================
#3) Upper‑bound & analysis helpers
#(kept close to the driver; pure functions)
#=======================================

def _log10_safe(x):
    return float(np.log10(x)) if x > 0 else float('-inf')

def _find_link_record(net, link_tuple):
    for L in net.graph.get('desired_links', []):
        if 'link' in L and tuple(L['link']) == tuple(link_tuple):
            return L
        if 'users' in L and tuple(L['users']) == tuple(link_tuple):
            return L
    return None

def _rate_poly(x, y1, y2):
    A = 2.0*(y1 + y2) + 1.0
    B = 4.0*y1*y2
    return x*x + A*x + B

def per_link_upper_bound(L):
    """
    Take a link-record with possibly many candidate paths, return the
    best (max) per-link UB as a dict: {ub_utility, ub_source, ub_expr, ub_x, ...}
    Now applies the fidelity clamp: F_eff = max(0.5, L['fidelity_limit']).
    """
    F_eff = max(0.5, float(L.get('fidelity_limit', 0.0))) #Chooses fidelity limit with lower bound being 0.5
    best = dict(ub_utility=float('-inf'), ub_source=None, ub_expr=None,
                ub_x=None, ub_path1=None, ub_path2=None)

    for p in L.get('paths', []):   #{'source','path1','path2','y1','y2'}
        y1 = float(p.get('y1', 0.0)); y2 = float(p.get('y2', 0.0))
        #No explicit one-channel cap stored in L → treat as unbounded here
        val, x, expr = per_link_ub_value(y1, y2, F_eff, x_one_channel_cap=None, on_infeasible="none")
        if val is None or not np.isfinite(val):
            continue

        if float(val) > best['ub_utility']:
            best.update(dict(
                ub_utility=float(val),
                ub_source=p.get('source'),
                ub_expr=float(expr),
                ub_x=float(x),
                ub_path1=p.get('path1'),
                ub_path2=p.get('path2'),
            ))
    return best

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


def ideal_upper_bound_at_fidelity(net):
    total = 0.0
    for L in net.graph.get('desired_links', []):
        ub = per_link_upper_bound(L)
        if ub['ub_utility'] is not None and np.isfinite(ub['ub_utility']):
            total += ub['ub_utility']
    return float(total)

def per_link_ub_for_chosen_path(link_obj):
    #Extracts info to find upper bound utilty for link
    L_faux = {
        'fidelity_limit': link_obj.get('fidelity_limit', 0.0),
        'paths': [{
            'source': link_obj['path']['source'],
            'path1':  link_obj['path']['path1'],
            'path2':  link_obj['path']['path2'],
            'y1':     link_obj['path'].get('y1', 0.0),
            'y2':     link_obj['path'].get('y2', 0.0),
        }]
    }
    ub = per_link_upper_bound(L_faux)
    return float(ub.get('ub_utility', float('-inf')))

def combo_upper_bound_at_fidelity(combo):
    #Finds upper bound utility for every link
    return float(sum(per_link_ub_for_chosen_path(lk) for lk in combo['combo']))

def _edges(path_nodes):
    #Turns list of nodes into list of tuple edges
    return [tuple(sorted((path_nodes[i], path_nodes[i+1])))
            for i in range(len(path_nodes)-1)]

def combo_overlap_metrics(combo_dict):
    #Function for determining an assortment of metrics
    links = combo_dict["combo"]
    per_link_edges = []
    edge_loads = Counter()

    for lk in links:
        p1 = lk['path']['path1']; p2 = lk['path']['path2']
        e1 = _edges(p1) if len(p1) > 1 else []
        e2 = _edges(p2) if len(p2) > 1 else []
        es = set(e1 + e2)
        per_link_edges.append(es)
        for e in es:
            edge_loads[e] += 1

    overlap_pairs = 0
    n = len(per_link_edges)
    for i in range(n):
        ei = per_link_edges[i]
        for j in range(i+1, n):
            if ei & per_link_edges[j]:
                overlap_pairs += 1

    path_hops = [len(es) for es in per_link_edges]

    #Return various metrics
    return dict( 
        overlap_pairs=overlap_pairs, #Number of link pairs that share at least one edge
        max_edge_load=max(edge_loads.values()) if edge_loads else 0, #maximum number of links sharing any single edge
        mean_edge_load=(sum(edge_loads.values())/len(edge_loads)) if edge_loads else 0.0, #average link load per used edge
        total_hops=sum(path_hops), #sum of unique edges over all links
        max_hops=max(path_hops) if path_hops else 0, #maximum unique edges in any one link
        #source usage stats
        src_unique=len(Counter(lk['path']['source'] for lk in links)), #Number of distinct sources across links
        src_links_max=max(Counter(lk['path']['source'] for lk in links).values()) if links else 0, #most links coming from any single source
        src_links_mean=(len(links)/len(Counter(lk['path']['source'] for lk in links))) if links else 0.0, #average number of links per source
    )

def _build_source_conflict_graph(combo_dict):
    links = combo_dict['combo']
    edges_per_link = []
    for lk in links:
        p1 = lk['path']['path1']; p2 = lk['path']['path2']
        e1 = _edges(p1) if len(p1) > 1 else []
        e2 = _edges(p2) if len(p2) > 1 else []
        edges_per_link.append(set(e1 + e2))

    src_to_idxs = defaultdict(list)
    for idx, lk in enumerate(links):
        src_to_idxs[lk['path']['source']].append(idx)

    graphs = {}
    for src, idxs in src_to_idxs.items():
        G = {i: set() for i in idxs}
        for i in range(len(idxs)):
            a = idxs[i]; ea = edges_per_link[a]
            for j in range(i+1, len(idxs)):
                b = idxs[j]
                if ea & edges_per_link[b]:
                    G[a].add(b); G[b].add(a)
        graphs[src] = G
    return graphs

def _greedy_clique_lb(G):
    if not G: return 0
    nodes = sorted(G, key=lambda u: len(G[u]), reverse=True)
    best = 1
    for u in nodes:
        clique = {u}
        candidates = set(G[u])
        while candidates:
            v = max(candidates, key=lambda x: len(G[x]))
            if all(v in G[w] for w in clique):
                clique.add(v)
                candidates &= G[v]
            else:
                candidates.remove(v)
        if len(clique) > best:
            best = len(clique)
        if len(G[u]) + 1 <= best:
            break
    return best

def _get_source_capacity(sources, src_id):
    entry = sources.get(src_id, {})
    ch = entry.get('available_channels')
    if isinstance(ch, (list, tuple)):
        return len(ch)
    for key in ('num_channels','K','channels','freqs'):
        v = entry.get(key)
        if isinstance(v, (list, tuple)):
            return len(v)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    return 10**9  #conservative: don't prune by mistake

def _capacity_ok(sources, counts: dict):
    for s, cnt in counts.items():
        cap = sources.get(s, {}).get('available_channels', [])
        cap_n = len(cap) if hasattr(cap, '__len__') else int(cap or 0)
        if cnt > cap_n:   # at least 1 ch per link from a source
            return False
    return True


#==========================
#4) Core pipeline stages
#==========================

def build_network(cfg: Config):
    net, users, sources = create_network.create_network(cfg.num_users, cfg.num_sources, cfg.num_edges, cfg.loss_range, cfg.num_channels_per_source, topology=cfg.topology, density=cfg.density, kite_loss_values=cfg.kite_loss_values)
    net = create_network.create_links(net, users, cfg.num_links)
    net = routing.double_dijkstra(net, sources, cfg.tau, cfg.d1, cfg.d2)
    #attach fidelity limits
    net = routing.link_info_packaging(net, fidelity_limit=cfg.fidelity_limit, y1 = cfg.y1, y2 = cfg.y2)
    return net, sources

def _links_per_source(combo_dict):
    #how many links in this combo use each source?
    return Counter(lk['path']['source'] for lk in combo_dict['combo'])

def enumerate_and_score_combos(net, sources, cfg: Config):
    path_sets = routing.prepare_path_sets(
        net,
        top_k_per_link = getattr(cfg, 'top_k_per_link', 8),
        shortlist_by   = getattr(cfg, 'shortlist_by', 'ub')
    )

    stream = routing.combo_stream_best_first(
        net, path_sets,
        max_combos=getattr(cfg, 'max_combos', None),
        time_limit_s=getattr(cfg, 'time_limit_s', None),
        # keep fast capacity check; it’s cheap
        capacity_fn=lambda counts: _capacity_ok(sources, counts)
    )

    ideal_ub = ideal_upper_bound_at_fidelity(net)
    return stream, ideal_ub


def _eval_worker(args):
    combo, sources, cfg = args
    return evaluate_combo(combo, sources, cfg)

def evaluate_combo(combo, sources, cfg: Config):
    try:
        res, util = allocate_channels.network_allocation(sources, combo, allocation_type='APOPT', initial_conditions=0.001, verbose = cfg.verbose)
        assignment = routing.check_interference(combo, res, sources)

        # #salvage promising infeasible results
        # if assignment is None and cfg.enable_salvage:
        #     ub_est = combo.get('_ub_combo', float('inf'))
        #     is_good = (np.isfinite(ub_est) and util >= cfg.salvage_trigger_ratio * ub_est)
        #     if is_good:
        #         for cap in cfg.salvage_k_caps:
        #             try:
        #                 res2, util2 = allocate_channels.network_allocation(sources, combo, allocation_type='APOPT', initial_conditions=0.001, per_link_k_cap=cap, verbose = cfg.verbose)
        #                 assignment2 = routing.check_interference(combo, res2, sources)
        #                 if assignment2 is not None:
        #                     return dict(
        #                         path_id=combo['path_id'], utility=util2,
        #                         results=res2, assignment=assignment2, routing_ok=True,
        #                         combo=combo, total_loss=combo['total_loss'],
        #                         used_sources=len(res2), salvaged=True, k_cap=cap
        #                     )
        #             except Exception:
        #                 _log_exception("salvage attempt failed", cfg, cap=cap)

        return dict(
            path_id=combo['path_id'],
            utility=util,
            results=res,
            assignment=assignment,
            routing_ok=assignment is not None,
            combo=combo,
            total_loss=combo['total_loss'],
            used_sources=len(res)
        )
    except Exception as e:
        _log_exception("evaluate_combo()", cfg, path_id=combo.get('path_id'))
        return dict(
            path_id=combo.get('path_id', -1),
            error=''.join(traceback.format_exception_only(type(e), e)),
            total_loss=combo.get('total_loss', float('inf')),
            used_sources=0
        )

def select_best(results):
    ok = [r for r in results if r.get('routing_ok')]
    if not ok: return None
    return max(ok, key=lambda r: float(r['utility']))

def postprocess_and_report(cfg: Config, net, sources, best, ideal_ub, all_utilities, dashed_limit = None):
    cfg.out_dir.mkdir(exist_ok=True)
    rows, flux_by_source = summarize_best(net, best['combo'], best['results'], best['assignment'])

    print_link_report(rows) #Output to terminal

    #---- print totals / bounds after all link lines ----
    total_utility = float(best['utility'])
    ideal_ub_val = float(ideal_ub)
    print("\n=== Totals ===")
    print(f"Total Utility (achieved): {total_utility:.6f}")
    gap_ideal = (1.0 - total_utility / ideal_ub_val) * 100.0 if ideal_ub_val > 0 else float('nan') #Percentage difference between the best and ideal
    print(f"Ideal Upper Bound (any paths): {ideal_ub_val:.6f}  (gap: {gap_ideal:.2f}% )")
    #----------------------------------------------------

    if cfg.report_csv:
        #CSV: per-link summary
        with open(cfg.out_dir / 'best_per_link.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

        #JSON: overall best
        with open(cfg.out_dir / 'best_overall.json', 'w', encoding='utf-8') as f:
            json.dump({
                'best_path_id': best['path_id'],
                'utility': float(best['utility']),
                'ideal_ub': float(ideal_ub),
                'used_sources': int(best['used_sources']),
                'salvaged': bool(best.get('salvaged', False)),
                'k_cap': best.get('k_cap')
            }, f, indent=2)

        #Diagnostics CSV
        diagnose_link_utilities(net, rows, best['results'], verbose=cfg.verbose)

    if cfg.make_plots:
        ideal_ub_now = ideal_upper_bound_at_fidelity(net)
        cleaned = [float(u) for u in all_utilities if np.isfinite(u)]
        plot.utility_comparison(cleaned,float(ideal_ub_now), outfile=str(cfg.out_dir / 'utility_comparison.svg'))

        #Stacked source allocation, bars, network plot
        freqs_by_link = {tuple(lk['link']): best['assignment'][i] for i, lk in enumerate(best['combo']['combo'])}
        plot.source_allocation(previous_best_results=best['results'], sources=sources, freqs_by_link=freqs_by_link)
        plot.plot_link_utility_bars(rows, outfile=str(cfg.out_dir / 'link_utility_bars.svg'))
        plot.plot_network_final(net, best['results'], freqs_by_link=freqs_by_link)

def print_link_report(rows):
    import numpy as _np  #local alias to avoid conflicts
    print("\n=== Link Report ===")
    print("Link Name, Utility, Source Name, Total Loss, Number of Channels Given, Channel Frequencies, Channel Flux, Network Path Taken")
    for r in rows:
        link_name = r["link"]
        util = r["link_utility"]
        util_str = f"{util:.6f}" if _np.isfinite(util) else "NA"
        source = r["source"]
        total_loss = r['link_total_loss']
        k = int(r["k_channels"])
        #Keep the sign convention from the solver: user1 list, user2 list
        ch_freqs = f"to_user1={r['freqs_to_user1']}, to_user2={r['freqs_to_user2']}"
        flux = r.get("total_flux_mu_k", None)
        flux_str = (f"{float(flux):.6g}"
                    if (flux is not None and _np.isfinite(float(flux)))
                    else "NA")
        #Show both arms of the path
        path_str = f"{r['path_user1']} | {r['path_user2']}"
        print(f"{link_name}, {util_str}, {source}, {total_loss:.5f}, {k}, {ch_freqs}, {flux_str}, {path_str}")

def summarize_best(net, best_combo, best_res, best_assignment):
    tau = net.graph.get('tau', None)
    src_map = {e['source']: e for e in best_res}

    #how many channels each link actually got (aligned with best_combo['combo'] order)
    link_names = [lk['link'] for lk in best_combo['combo']]
    k_counts   = routing.extract_link_ch_counts(best_res, link_names)

    def _sum_edge_loss(path):
        #robust fallback if we can't find a matching path record
        try:
            return sum(net[u][v].get('loss', 1.0) for u, v in zip(path, path[1:]))
        except Exception:
            return float('nan')

    def _match_path_record(Lrec, src, p1, p2):
        if not Lrec:
            return None
        for pr in Lrec.get('paths', []):
            #exact match on source and the two arms
            if pr.get('source') == src and pr.get('path1') == p1 and pr.get('path2') == p2:
                return pr
        return None

    rows = []
    for i, lk in enumerate(best_combo['combo']):
        link  = lk['link']
        u1, u2 = link
        src   = lk['path']['source']
        p1    = lk['path']['path1']
        p2    = lk['path']['path2']

        entry = src_map[src]
        #index of this link within this source's result entry
        j = entry['links'].index(link) if link in entry['links'] else entry['links'].index(tuple(link))

        pre = entry['prelog_rates'][j]
        per_link_utility = _log10_safe(pre)  #uses your existing helper

        k_i = int(k_counts[i])                      #channels chosen for this link
        to_u1, to_u2 = best_assignment[i]           #channel lists (sign encodes orientation)

        mu = float(entry.get('optimal_mu', np.nan))
        total_flux = float(mu * k_i) if np.isfinite(mu) else np.nan
        x_val = float(tau * total_flux) if (tau is not None and np.isfinite(total_flux)) else None

        #---- Upper bound (correct call) ----
        Lrec = _find_link_record(net, link)
        ub_dict = per_link_upper_bound(Lrec) if Lrec else None
        ub_util = (float(ub_dict.get('ub_utility')) if (ub_dict and ub_dict.get('ub_utility') is not None)
                   else None)

        #---- Total loss for the *chosen* path ----
        total_loss = float('nan')
        pr = _match_path_record(Lrec, src, p1, p2)
        if pr and ('user1_loss' in pr and 'user2_loss' in pr):
            total_loss = float(pr['user1_loss']) + float(pr['user2_loss'])
        else:
            #fallback: sum edge losses along both arms
            total_loss = _sum_edge_loss(p1) + _sum_edge_loss(p2)

        rows.append({
            "link": f"{u1}-{u2}",
            "source": src,
            "path_user1": " -> ".join(p1),
            "path_user2": " -> ".join(p2),
            "k_channels": k_i,
            "freqs_to_user1": list(map(int, to_u1)),
            "freqs_to_user2": list(map(int, to_u2)),
            "mu_per_channel": mu,
            "total_flux_mu_k": total_flux,
            "x_value_tau_mu_k": x_val,
            "link_utility": per_link_utility,
            #if UB exists, keep the larger of UB vs achieved utility for the display
            "ub_utility": (max(ub_util, per_link_utility) if (ub_util is not None) else per_link_utility),
            "ub_source": (ub_dict.get("ub_source") if ub_dict else None),
            "link_total_loss": float(total_loss),
        })

    #per-source flux (still reported)
    flux_by_source = {e['source']: float(e.get('optimal_mu', np.nan)) for e in best_res}
    return rows, flux_by_source


def diagnose_link_utilities(net, rows, best_res, verbose=True):
    vals = np.array([r["link_utility"] for r in rows], dtype=float)
    same = np.allclose(vals, vals[0], rtol=1e-6, atol=1e-9)

    diag = []
    src_map = {e['source']: e for e in best_res}
    for r in rows:
        link_tuple = tuple(r["link"].split("-"))
        L = _find_link_record(net, link_tuple)
        y1 = y2 = None
        if L:
            for p in L['paths']:
                if p['source'] == r['source'] and " -> ".join(p['path1']) == r['path_user1'] and " -> ".join(p['path2']) == r['path_user2']:
                    y1, y2 = p.get('y1', None), p.get('y2', None)
                    break
        mu = r["mu_per_channel"]
        diag.append(dict(link=r["link"], src=r["source"], k=r["k_channels"], mu=mu, y1=y1, y2=y2, util=r["link_utility"], ub=r["ub_utility"]))

    if verbose:
        print("\n=== Link-Utility Diagnostics ===")
        print(f"All equal (≈): {bool(same)}   min={float(np.min(vals)):.6f}  max={float(np.max(vals)):.6f}  std={float(np.std(vals)):.6f}")

    with open("outputs/diagnostics_link_utility.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["link","src","k","mu","y1","y2","util","ub"])
        w.writeheader()
        for d in diag:
            w.writerow(d)
    return dict(all_equal=bool(same), min=float(np.min(vals)), max=float(np.max(vals)), std=float(np.std(vals)))

#==========================
#5) Orchestrator
#==========================

def run_pipeline(cfg: Config = Config()):
    t0 = time.time()
    cfg.out_dir.mkdir(exist_ok=True)
    log(f"Config: {asdict(cfg)}", cfg=cfg)

    #Build network + inputs
    net, sources = build_network(cfg)
    net.graph['tau'] = cfg.tau  #used by summarize

    # Enumerate & score (streaming)
    stream, ideal_ub = enumerate_and_score_combos(net, sources, cfg)

    # rolling min-loss for loss-window gating
    global_min_loss = [float('inf')]
    def in_loss_window(c):
        tl = float(c.get('total_loss', float('inf')))
        if not np.isfinite(tl):
            return False
        # if the multiplier is huge, effectively disable the window
        if cfg.loss_multiplier_stop >= 1e9:
            # still keep track of a min for potential later use
            if tl < global_min_loss[0]:
                global_min_loss[0] = tl
            return True
        # update rolling min
        if tl < global_min_loss[0]:
            global_min_loss[0] = tl
            return True
        return tl <= cfg.loss_multiplier_stop * global_min_loss[0]

    def pull_batch(n):
        """Pull up to n promising combos from the stream, honoring loss window."""
        out = []
        while len(out) < n:
            try:
                c = next(stream)
            except StopIteration:
                break
            if in_loss_window(c):
                out.append(c)
        return out

    dashed_limit = float(ideal_ub)

    best = None
    successes = 0
    attempts = 0
    all_utilities = []   # collect utilities for utility_comparison plot

    #parallel or sequential evaluation
    if cfg.parallel:
        import os
        from collections import deque
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED

        pool_cls = ProcessPoolExecutor if cfg.executor == "process" else ThreadPoolExecutor
        max_workers = cfg.workers or os.cpu_count() or 1

        def ub_promising(next_ub: float, best_util: float | None) -> bool:
            if not cfg.skip_below_best_ub:
                return True
            if best_util is None:
                return True
            if not np.isfinite(next_ub):
                return True
            return next_ub > (best_util + cfg.ub_skip_eps)

        def attempts_ok():  # 0 means disabled
            return (not cfg.stop_attempts) or (attempts < cfg.stop_attempts)

        def successes_ok():
            return (not cfg.stop_successes) or (successes < cfg.stop_successes)

        inflight = {}                  # fut -> combo
        pending  = deque()             # persistent buffer: combos fetched but not yet submitted
        ub_shutdown = [False]          # once True, we stop submitting new tasks (matches sequential UB stop)

        def refill_pending(target_extra: int):
            """
            Pull up to target_extra new items from the stream (after loss-window),
            appending to 'pending'. Never discards anything already in 'pending'.
            """
            added = 0
            while added < target_extra:
                try:
                    c = next(stream)
                except StopIteration:
                    break
                if in_loss_window(c):
                    pending.append(c)
                    added += 1
            return added

        with pool_cls(max_workers=max_workers) as ex:

            def try_submit_up_to(limit: int):
                """
                Submit up to 'limit' new tasks, respecting max_inflight.
                Never discards candidates; extras remain in 'pending'.
                If UB gate (when enabled) says stop, set ub_shutdown=True.
                """
                if ub_shutdown[0]:
                    return 0

                submitted = 0
                best_util_val = float(best['utility']) if best is not None else None

                # Ensure we have enough in pending to try submissions.
                need = max(0, limit - len(pending))
                if need > 0:
                    # small overfetch: try to have a little cushion but don't lose items
                    refill_pending(need)

                # While we have capacity and candidates, submit.
                while (len(inflight) < cfg.max_inflight) and (submitted < limit) and pending:
                    c = pending[0]  # peek; only pop if we will submit
                    next_ub = float(c.get('_ub_combo', float('inf')))

                    if not ub_promising(next_ub, best_util_val):
                        # Match sequential behavior: once UB wall is hit, stop all new submissions.
                        ub_shutdown[0] = True
                        break

                    # Consume and submit this candidate
                    c = pending.popleft()
                    fut = ex.submit(_eval_worker, (c, sources, cfg))
                    inflight[fut] = c
                    submitted += 1

                    # If we still have room to submit but pending is low, try a small top-up.
                    if (submitted < limit) and (len(inflight) < cfg.max_inflight) and (len(pending) == 0):
                        # pull a few more to keep pipe primed
                        refill_pending(max(1, limit - submitted))

                return submitted

            # Seed initial submissions
            try_submit_up_to(min(cfg.parallel_batch_size, cfg.max_inflight))

            while inflight or (attempts_ok() and successes_ok()):
                # If nothing in-flight, try to submit more; if we can't, we’re done.
                if not inflight:
                    if try_submit_up_to(min(cfg.parallel_batch_size, cfg.max_inflight)) == 0:
                        break
                    continue

                done, _ = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    c = inflight.pop(fut)
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = {'error': str(e)}

                    attempts += 1

                    if 'utility' in res and res['utility'] is not None:
                        try:
                            all_utilities.append(float(res['utility']))
                        except Exception:
                            pass

                    if res.get('routing_ok'):
                        successes += 1
                        if best is None or float(res['utility']) > float(best['utility']):
                            best = res

                if attempts_ok() and successes_ok():
                    # Submit a few more based on completions processed
                    try_submit_up_to(max(1, cfg.as_completed_yield))


    else:
        # === Streaming sequential loop (no pre-built combos list) ===

        # rolling min-loss for the loss-window gate
        global_min_loss = [float('inf')]
        def in_loss_window(c):
            tl = float(c.get('total_loss', float('inf')))
            if not np.isfinite(tl):
                return False
            # effectively disable if multiplier is huge
            if cfg.loss_multiplier_stop >= 1e9:
                if tl < global_min_loss[0]:
                    global_min_loss[0] = tl
                return True
            # update rolling min and test gate
            if tl < global_min_loss[0]:
                global_min_loss[0] = tl
                return True
            return tl <= cfg.loss_multiplier_stop * global_min_loss[0]

        # helper: UB gate check on the *next* candidate
        def ub_promising(next_ub, best_util):
            if not cfg.skip_below_best_ub:
                return True
            if best_util is None:
                return True
            if not np.isfinite(next_ub):
                return True
            return next_ub > (best_util + cfg.ub_skip_eps)

        # pull from the stream until exhausted or stop conditions trigger
        while True:
            # get next candidate
            try:
                combo = next(stream)   # <-- stream returned by enumerate_and_score_combos(...)
            except StopIteration:
                break

            # global UB stop (stream is already sorted by _ub_combo descending)
            if best is not None:
                next_ub = float(combo.get('_ub_combo', float('inf')))
                if not ub_promising(next_ub, float(best['utility'])):
                    if cfg.verbose:
                        print(f"[UB stop] next UB={next_ub:.6f} <= best={float(best['utility']):.6f} → stopping search.")
                    break

            # loss-window filter (cheap)
            if not in_loss_window(combo):
                continue

            # evaluate
            attempts += 1
            res = evaluate_combo(combo, sources, cfg)

            if cfg.verbose and 'error' in res:
                print(f"[evaluate_combo returned error] path_id={res.get('path_id')} :: {res['error']}")

            u = res.get('utility')
            if u is not None and np.isfinite(u):
                try:
                    all_utilities.append(float(u))
                except Exception:
                    pass

            if res.get('routing_ok'):
                successes += 1
                if best is None or float(u) > float(best['utility']):
                    best = res

            # early stops (0 means disabled)
            if cfg.stop_attempts and attempts >= cfg.stop_attempts:
                if cfg.verbose:
                    print(f"[stop_attempts] attempts={attempts} reached limit={cfg.stop_attempts}")
                break
            if cfg.stop_successes and successes >= cfg.stop_successes:
                if cfg.verbose:
                    print(f"[stop_successes] successes={successes} reached limit={cfg.stop_successes}")
                break

    if best is None:
        log("No feasible result found.", cfg=cfg) 
        return None

    log(f"Best utility = {best['utility']:.6f} (path_id={best['path_id']})", cfg=cfg)

    #pass sources and all_utilities into reporting
    postprocess_and_report(cfg, net, sources, best, ideal_ub, all_utilities, dashed_limit=dashed_limit)

    log(f"Done in {time.time()-t0:.2f}s.", cfg=cfg)
    if not cfg.verbose:
        print(f"Done in {time.time()-t0:.2f}s.")
    return best

#==========================
#6) CLI entry
#==========================

if __name__ == "__main__":
    run_pipeline(Config())
