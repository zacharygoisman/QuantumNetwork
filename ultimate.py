#ultimate.py
#Code that calls the main functions

import numpy as np, os, traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import time
import math, csv, json
from pathlib import Path


import create_network, routing, allocate_channels, plot #My cool inputs B)

#Worker
def evaluate_combo(combo, sources):
    """Run APOPT + CP-SAT for one path-combo and return a summary dict."""
    try:
        res, util = allocate_channels.network_allocation(
            sources, combo, allocation_type='APOPT', initial_conditions=0.001
        )
        assignment = routing.check_interference(combo, res, sources)  # [(to_u1, to_u2), ...] or None
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
        return dict(
            path_id=combo['path_id'],
            error=''.join(traceback.format_exception_only(type(e), e)),
            total_loss=combo['total_loss'],
            used_sources=0
        )


def _log10_safe(x):
    return float(np.log10(x)) if x > 0 else float('-inf')

def _find_link_record(net, link_tuple):
    """Return the desired_links record for the given (u1,u2) tuple."""
    for L in net.graph.get('desired_links', []):
        # try common shapes
        if 'link' in L and tuple(L['link']) == tuple(link_tuple):
            return L
        if 'users' in L and tuple(L['users']) == tuple(link_tuple):
            return L
        # sometimes the list is just ((u1,u2),paths,...) – extend if you store differently
    return None

def per_link_upper_bound(L):
    """
    Best possible utility for this single link if it could pick any source/path
    and be driven right to the fidelity limit (ignores interference and pools).
    Returns dict with keys: {'ub_utility','ub_source','ub_expr','ub_x','ub_path1','ub_path2'}
    """
    F = L['fidelity_limit']
    denom = 4*F - 1
    if denom <= 0:
        return dict(ub_utility=float('-inf'), ub_source=None, ub_expr=None, ub_x=None, ub_path1=None, ub_path2=None)

    best = dict(ub_utility=float('-inf'), ub_source=None, ub_expr=None, ub_x=None, ub_path1=None, ub_path2=None)
    for p in L['paths']:  # each has: 'source','path1','path2','y1','y2',...
        y1, y2 = p['y1'], p['y2']
        Y = y1 + y2
        alpha = (2*Y + 1) - (3.0/denom)
        disc = alpha*alpha - 16.0*y1*y2
        if disc < 0:
            continue
        r1 = (-alpha + np.sqrt(disc)) / 2.0
        r2 = (-alpha - np.sqrt(disc)) / 2.0
        x  = max(r1, r2)
        if x <= 0:
            continue
        expr = 3.0 * x / denom
        val  = _log10_safe(expr)
        if val > best['ub_utility']:
            best.update(dict(
                ub_utility=float(val),
                ub_source=p['source'],
                ub_expr=float(expr),
                ub_x=float(x),
                ub_path1=p['path1'],
                ub_path2=p['path2'],
            ))
    return best

def summarize_best(net, best_combo, best_res, best_assignment):
    """
    Build per-link rows + per-source flux dict (+ per-link upper bound).
    Also computes actual per-link flux numbers.

    Returns (rows, flux_by_source)
      rows[i] keys:
        link, source, path_user1, path_user2, k_channels,
        freqs_to_user1, freqs_to_user2,
        mu_per_channel, total_flux_mu_k, x_value_tau_mu_k,
        link_utility, ub_utility, ub_source
    """
    # tau is optional; include x if available
    tau = net.graph.get('tau', None)

    # Map src -> per-source entry
    src_map = {e['source']: e for e in best_res}

    link_names = [lk['link'] for lk in best_combo['combo']]
    k_counts   = routing.extract_link_ch_counts(best_res, link_names)  # ints per link

    rows = []
    for i, lk in enumerate(best_combo['combo']):
        link  = lk['link']
        u1, u2 = link
        src   = lk['path']['source']
        p1    = lk['path']['path1']
        p2    = lk['path']['path2']
        y1    = lk['path'].get('y1', None)
        y2    = lk['path'].get('y2', None)

        entry = src_map[src]
        # map to the correct link index inside this source’s arrays
        try:
            j = entry['links'].index(link)
        except ValueError:
            # fall back if tuple shapes differ (e.g., lists) – normalize
            j = entry['links'].index(tuple(link))

        # Stored "prelog" rate (your objective uses log10 of this)
        pre = entry['prelog_rates'][j]
        per_link_utility = _log10_safe(pre)

        # Channel selection & frequencies
        k_i = int(k_counts[i])
        to_u1, to_u2 = best_assignment[i]  # signed freqs

        # Flux details
        mu = float(entry.get('optimal_mu', np.nan))          # per-channel flux (from APOPT)
        total_flux = float(mu * k_i) if np.isfinite(mu) else np.nan
        x_val = float(tau * total_flux) if (tau is not None and np.isfinite(total_flux)) else None

        # Per-link optimistic upper bound (ignore interference, pools)
        Lrec = _find_link_record(net, link)
        ub = per_link_upper_bound(Lrec) if Lrec else dict(ub_utility=None, ub_source=None)

        rows.append({
            "link": f"{u1}-{u2}",
            "source": src,
            "path_user1": " -> ".join(p1),
            "path_user2": " -> ".join(p2),
            "k_channels": k_i,
            "freqs_to_user1": list(map(int, to_u1)),
            "freqs_to_user2": list(map(int, to_u2)),

            # NEW: actual flux numbers
            "mu_per_channel": mu,
            "total_flux_mu_k": total_flux,
            "x_value_tau_mu_k": x_val,

            # utilities: achieved vs single-link ceiling
            "link_utility": per_link_utility,
            "ub_utility": ub.get("ub_utility"),
            "ub_source": ub.get("ub_source"),
        })

    flux_by_source = {e['source']: float(e.get('optimal_mu', np.nan)) for e in best_res}
    return rows, flux_by_source

def ideal_upper_bound_at_fidelity(net):
    """Sum of per-link upper bounds—each link independently at its own best source/path."""
    total = 0.0
    for L in net.graph.get('desired_links', []):
        ub = per_link_upper_bound(L)
        if ub['ub_utility'] is not None and np.isfinite(ub['ub_utility']):
            total += ub['ub_utility']
    return float(total)

def diagnose_link_utilities(net, rows, best_res, verbose=True):
    """
    Quick checks for 'all link utilities look similar'.
    Prints a small table and returns a dict of summary stats.
    """
    vals = np.array([r["link_utility"] for r in rows], dtype=float)
    same = np.allclose(vals, vals[0], rtol=1e-6, atol=1e-9)

    # Gather y1,y2,k,mu for context
    diag = []
    src_map = {e['source']: e for e in best_res}
    for r in rows:
        # recover y1,y2 from net (chosen path)
        link_tuple = tuple(r["link"].split("-"))
        # since your user labels tend to be strings like 'U1', this string split is fine
        L = _find_link_record(net, link_tuple)
        y1 = y2 = None
        if L:
            # find the exact chosen path inside the available paths, if present
            for p in L['paths']:
                if p['source'] == r['source'] and " -> ".join(p['path1']) == r['path_user1'] and " -> ".join(p['path2']) == r['path_user2']:
                    y1, y2 = p.get('y1', None), p.get('y2', None)
                    break

        mu = r["mu_per_channel"]
        diag.append(dict(
            link=r["link"], src=r["source"], k=r["k_channels"], mu=mu, y1=y1, y2=y2,
            util=r["link_utility"], ub=r["ub_utility"]
        ))

    if verbose:
        print("\n=== Link-Utility Diagnostics ===")
        print(f"All equal (≈): {bool(same)}   min={float(np.min(vals)):.6f}  max={float(np.max(vals)):.6f}  std={float(np.std(vals)):.6f}")
        print("If many links share the same fidelity limit and mu,k end up similar, per-link log utility will cluster tightly.")

    # write a small CSV for inspection
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/diagnostics_link_utility.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["link","src","k","mu","y1","y2","util","ub"])
        w.writeheader()
        for d in diag:
            w.writerow(d)

    return dict(all_equal=bool(same), min=float(np.min(vals)), max=float(np.max(vals)), std=float(np.std(vals)))


#Main
def main():
    start_time = time.time() #Optional timer
    num_usr, num_src, num_edg, num_lnk= 20, 3, 100, 6 #Note that number of iterations will be num_src^num_link
    loss_range=(5,25)
    fidelity_limit = np.repeat(0.7,num_lnk)
    tau = 1e-9
    d1 = 100
    d2 = 3500
    sort_type = 'loss' #Sort by average network hops ('hop') or total network loss ('loss)
    link_pairs = None
    num_channels_per_source = [150, 50, 30] #[None] assumes 5 channels per source 
    
    #Stop conditions
    STOP_NUMBER = 0 #Run this many attempts (0 for disabling it)
    if STOP_NUMBER != 0:
        total = num_src**num_lnk
        STOP_PERCENT = STOP_NUMBER/total #Percent between 0 and 1 to determine the percent of attempts we try
        if STOP_PERCENT > 1:
            STOP_PERCENT = 1 #Percent between 0 and 1 to determine the percent of attempts we try
    
    else:
        STOP_PERCENT = 1 #Percent between 0 and 1 to determine the percent of attempts we try (1 to disable)
    LOSS_MULTIPLIER = 1000000 #Percent must be greater than 1. Minimum loss times this is what loss value we stop at (very large number to disable)
    STOP_SUCCESSES = 10 #Stops code after N successes (set to 0 to disable)

    #Network Creation
    net, users, sources = create_network.create_network(
        num_usr, num_src, num_edg, loss_range, num_channels_per_source,
        topology='dense', density=0.4
    )
    
    #Link Creation
    net = create_network.create_links(net, users, num_lnk)

    #Dijkstra Lowest Loss Paths
    net = routing.double_dijkstra(net, sources, tau, d1, d2)

    #Packages the y1, y2, and fidelity limit into each link
    net = routing.link_info_packaging(net, fidelity_limit)

    #Determines the routing combinations we use and in which order they will be run through
    combos      = routing.path_combinations(net, sort_type)
    max_index   = int(len(combos) * STOP_PERCENT) #Calculate the last index to run through
    min_loss    = combos[0]['total_loss'] #Stores lowest loss combination

    #Storing arrays and best values
    all_util, all_util_no_int = [], []
    all_loss, all_srcs        = [], []
    best_util, best_res       = -np.inf, None

    best_combo = None
    best_assignment = None

    #Parallel Search (refillable queue so we can stop after n successes)
    max_workers = max(1, os.cpu_count() - 1)
    eligible = (c for c in combos[:max_index]
                if c['total_loss'] <= LOSS_MULTIPLIER * min_loss)

    successes = 0
    pending = set()

    def submit_up_to(pool, pending_set):
        try:
            while len(pending_set) < max_workers:
                c = next(eligible)
                pending_set.add(pool.submit(evaluate_combo, c, sources))
        except StopIteration:
            pass

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        submit_up_to(pool, pending)

        if not pending:
            print("[main] No eligible combos to evaluate.")
        else:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    res = fut.result()
                    pid = res.get('path_id', '?')

                    if 'error' in res:
                        print(f"[{pid}] worker error → {res['error'].strip()}")
                        all_util.append(-np.inf)
                        all_util_no_int.append(-np.inf)
                        all_loss.append(res['total_loss'])
                        all_srcs.append(res['used_sources'])
                        continue

                    util = res['utility']
                    all_util.append(util)
                    all_loss.append(res['total_loss'])
                    all_srcs.append(res['used_sources'])

                    if res['routing_ok']:
                        successes += 1
                        all_util_no_int.append(util)
                        if util > best_util:
                            best_util       = util
                            best_res        = res['results']
                            best_combo      = res['combo']
                            best_assignment = res['assignment']
                    else:
                        all_util_no_int.append(-np.inf)


                    # Early exit if enough successes found
                    if STOP_SUCCESSES and successes >= STOP_SUCCESSES:
                        for p in pending:
                            p.cancel()
                        pending.clear()
                        break

                if not (STOP_SUCCESSES and successes >= STOP_SUCCESSES):
                    submit_up_to(pool, pending)


    # -------------- plotting & summary ----------------------------------
    plot.utility_interference(all_util, all_util_no_int)
    plot.utility_interference_zoomed(all_util, all_util_no_int)
    plot.utility_vs_loss(all_util, all_util_no_int, all_loss)
    plot.utility_vs_sources(all_util, all_util_no_int, all_srcs)


    print("Done.")
    total_time = time.time() - start_time
    Path("outputs").mkdir(exist_ok=True)

    # Upper bound for whole network (sum of per-link ceilings)
    ideal_ub = ideal_upper_bound_at_fidelity(net)

    if best_res and best_combo and best_assignment:
        rows, flux_by_source = summarize_best(net, best_combo, best_res, best_assignment)

        # Attempts vs utility with dashed ceiling
        plot.utility_comparison([u for u in all_util if np.isfinite(u)], ideal_ub)

        # Per-link max vs actual
        plot.plot_link_utility_bars(rows)

        # Stacked source allocation (uses your existing signature)
        plot.source_allocation(best_res, sources)

        # Add frequencies to the network legend
        freqs_map = {}
        for r in rows:
            u1, u2 = r["link"].split('-')
            freqs_map[(u1, u2)] = (r["freqs_to_user1"], r["freqs_to_user2"])

        plot.plot_network_final(net, best_res, freqs_by_link=freqs_map)

        # Stacked source allocation (uses your existing signature in plot.py)
        plot.source_allocation(best_res, sources)

        # Draw the network with frequency labels in the legend (plot.py needs the optional freqs_by_link arg)
        plot.plot_network_final(net, best_res, freqs_by_link=freqs_map)

        # Per-link max vs actual utility bars
        plot.plot_link_utility_bars(rows)

        # Attempts vs utility with dashed ceiling (sum of per-link ceilings)
        plot.utility_comparison([u for u in all_util if np.isfinite(u)], ideal_ub)


        # Diagnostics for "why are link utilities so similar?"
        diag_stats = diagnose_link_utilities(net, rows, best_res, verbose=True)

        # sanity check: sum of per-link utilities
        sum_per_link = float(np.sum([r["link_utility"] for r in rows]))

        # --- console summary ---
        print("\n=== BEST RESULT SUMMARY ===")
        print(f"Runtime (s):                          {total_time:.3f}")
        print(f"Best network utility (objective):     {best_util:.6f}")
        print(f"Sum of per-link utility (sanity):     {sum_per_link:.6f}")
        print(f"Ideal upper bound (sum of ceilings):  {ideal_ub:.6f}")
        print("Flux by source (optimal_mu):")
        for s, mu in flux_by_source.items():
            print(f"  {s}: {mu:.6e}")

        # --- CSVs ---
        with open("outputs/best_links_table.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "link","source","path_user1","path_user2",
                "k_channels","freqs_to_user1","freqs_to_user2",
                "mu_per_channel","total_flux_mu_k","x_value_tau_mu_k",
                "link_utility","ub_utility","ub_source"
            ])
            w.writeheader()
            for r in rows:
                w.writerow(r)

        with open("outputs/flux_by_source.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source","optimal_mu"])
            for s, mu in flux_by_source.items():
                w.writerow([s, mu])

        with open("outputs/best_summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "runtime_seconds": total_time,
                "best_network_utility": best_util,
                "ideal_upper_bound_utility": ideal_ub,
                "diagnostics": diag_stats,
                "flux_by_source": flux_by_source,
                "links": rows
            }, f, indent=2)

        print("\nWrote:")
        print("  outputs/best_links_table.csv")
        print("  outputs/flux_by_source.csv")
        print("  outputs/diagnostics_link_utility.csv")
        print("  outputs/best_summary.json")
    else:
        print("\n[main] No interference-free allocation found.")
        print(f"Ideal upper bound (sum of ceilings): {ideal_ub:.6f}")




if __name__ == '__main__':
    main()
