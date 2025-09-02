#ultimate.py
#Code that calls the main functions

import numpy as np, os, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import create_network, routing, allocate_channels, plot #My cool inputs B)

#Worker
def evaluate_combo(combo, sources):
    """Run APOPT + CP-SAT for one path-combo and return a summary dict."""
    try:
        res, util = allocate_channels.network_allocation(
            sources, combo, allocation_type='APOPT', initial_conditions=0.001
        )
        ok = routing.check_interference(combo, res, sources)
        return dict(path_id=combo['path_id'],
                    utility=util, results=res,
                    routing_ok=bool(ok),
                    total_loss=combo['total_loss'],
                    used_sources=len(res))
    except Exception as e:
        # propagate the loss value so bookkeeping stays aligned
        return dict(path_id=combo['path_id'],
                    error=''.join(traceback.format_exception_only(type(e), e)),
                    total_loss=combo['total_loss'],
                    used_sources=0)        # placeholder


#Main
def main():
    start_time = time.time() #Optional timer
    num_usr, num_src, num_edg, num_lnk= 10, 6, 100, 5 #Note that number of iterations will be num_src^num_link
    loss_range=(10,20)
    fidelity_limit = np.repeat(0.7,num_lnk)
    tau = 1e-9
    d1 = 100
    d2 = 3500
    sort_type = 'loss' #Sort by average network hops ('hop') or total network loss ('loss)
    link_pairs = None
    num_channels_per_source = [4, 4, 4, 4, 4, 4] #[None] assumes 5 channels per source 
    
    #Stop conditions
    STOP_NUMBER = 100 #Run this many attempts
    if STOP_NUMBER != 0:
        total = num_src**num_lnk
        STOP_PERCENT = STOP_NUMBER/total #Percent between 0 and 1 to determine the percent of attempts we try
        if STOP_PERCENT > 1:
            STOP_PERCENT = 1 #Percent between 0 and 1 to determine the percent of attempts we try
    else:
        STOP_PERCENT = 1 #Percent between 0 and 1 to determine the percent of attempts we try
    LOSS_MULTIPLIER = 100000 #Percent must be greater than 1. Minimum loss times this is what loss value we stop at

    #Network Creation
    net, users, sources = create_network.create_network(
        num_usr, num_src, num_edg, loss_range, num_channels_per_source,
        topology='ring', density=0.5
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

    #Parallel Search
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 1)) as pool:
        futures = {pool.submit(evaluate_combo, c, sources): c['path_id']
                   for c in combos[:max_index]
                   if c['total_loss'] <= LOSS_MULTIPLIER * min_loss}

        for fut in as_completed(futures):
            res  = fut.result()
            pid  = res['path_id']

            # ----------- keep arrays in sync even on errors --------------
            if 'error' in res:
                print(f"[{pid}] worker error → {res['error'].strip()}")
                all_util.append(-np.inf)
                all_util_no_int.append(-np.inf)
                all_loss.append(res['total_loss'])
                all_srcs.append(res['used_sources'])
                continue
            # --------------------------------------------------------------

            util = res['utility']
            all_util.append(util)
            all_loss.append(res['total_loss'])
            all_srcs.append(res['used_sources'])

            if res['routing_ok']:
                all_util_no_int.append(util)
                if util > best_util:
                    best_util, best_res = util, res['results']
            else:
                all_util_no_int.append(-np.inf)

    # -------------- plotting & summary ----------------------------------
    plot.utility_interference(all_util, all_util_no_int)
    plot.utility_interference_zoomed(all_util, all_util_no_int)
    plot.utility_vs_loss(all_util, all_util_no_int, all_loss)
    plot.utility_vs_sources(all_util, all_util_no_int, all_srcs)

    if best_res:
        plot.plot_network_final(net, best_res)
        plot.source_allocation(best_res, sources)
    else:
        print("[main] No interference-free allocation found – skipping "
              "network/source plots.")

    print("Done.")
    total_time = time.time() - start_time
    print(total_time)

    print(best_util, best_res)


if __name__ == '__main__':
    main()
