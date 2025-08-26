# ultimate.py  –  parallel driver, lean and robust  (with “Option A” fix)
import numpy as np, os, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import create_network, routing, allocate_channels, plot
import time

# ---------------- worker -------------------------------------------------
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


# ---------------- main ---------------------------------------------------
def main():
    start_time = time.time()
    # ————————————————————————————————— parameters you’ll tune
    # num_usr, num_src, num_edg, num_lnk = 10, 5, 100, 5
    # loss_range              = (10, 20)
    # fidelity_limit          = np.repeat(0.7, num_lnk)
    # num_channels_per_source = [5] * num_src
    # STOP_PERCENT            = 0.10   # evaluate 10 % of combos
    # LOSS_MULTIPLIER         = 100    # early loss cut-off
    # ——————————————————————————————————————————————————————
    num_usr, num_src, num_edg, num_lnk= 6, 3, 100, 3 #Note that number of iterations will be num_src^num_link
    loss_range=(10,20)
    # y1 = [0, 0.0034, 0, 0.0299, 0.0385, 0.0625, 0.0733, 0, 0.1106, 0.125, 0.1489, 0]
    # y2 = [0, 0.0006, 0.0357, 0.0051, 0.0066, 0.0107, 0.0126, 0.1818, 0.019, 0.0214, 0.0256, 0.2979]
    fidelity_limit = np.repeat(0.7,num_lnk)
    tau = 1e-9
    d1 = 100
    d2 = 3500
    sort_type = 'loss' #Sort by average network hops or total network loss
    link_pairs = None
    num_channels_per_source = [None] #[None] assumes 5 channels per source 
    STOP_PERCENT = 0.01 #Percent between 0 and 1
    STOP_PERCENT = 1#00 * 1/(num_src**num_lnk)
    LOSS_MULTIPLIER = 30 #Percent must be greater than 1


    net, users, sources = create_network.create_network(
        num_usr, num_src, num_edg, loss_range, num_channels_per_source,
        topology='dense', density=0.5
    )
    net = create_network.create_links(net, users, num_lnk)
    net = routing.double_dijkstra(net, sources, tau, d1, d2)
    net = routing.link_info_packaging(net, fidelity_limit)

    combos      = routing.path_combinations(net, 'loss')
    max_index   = int(len(combos) * STOP_PERCENT)
    min_loss    = combos[0]['total_loss']

    # tracking arrays
    all_util, all_util_no_int = [], []
    all_loss, all_srcs        = [], []
    best_util, best_res       = -np.inf, None

    # -------------- parallel search -------------------------------------
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


if __name__ == '__main__':
    main()
