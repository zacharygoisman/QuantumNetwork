#ultimate.py
#Takes all functions to generate and optimize an arbitrary link

#Packages
import numpy as np
import matplotlib.pyplot as plt
import copy

#Custom imports
import create_network
import routing
import allocate_channels
import plot

def main():
    #Inputs
    num_usr, num_src, num_edg, num_lnk= 5, 5, 40, 5 #Note that number of iterations will be num_src^num_link
    loss_range=(10,20)
    # y1 = [0, 0.0034, 0, 0.0299, 0.0385, 0.0625, 0.0733, 0, 0.1106, 0.125, 0.1489, 0]
    # y2 = [0, 0.0006, 0.0357, 0.0051, 0.0066, 0.0107, 0.0126, 0.1818, 0.019, 0.0214, 0.0256, 0.2979]
    fidelity_limit = np.repeat(0.7,num_lnk)
    tau = 1e-9
    d1 = 100
    d2 = 3500
    sort_type = 'loss' #Sort by average network hops or total network loss
    link_pairs = None
    num_channels_per_source = [5, 7, 5, 3, 6] #[None] assumes 5 channels per source 
    STOP_PERCENT = 0.01 #Percent between 0 and 1
    STOP_PERCENT = 20 * 1/(num_src**num_lnk)
    LOSS_MULTIPLIER = 30 #Percent must be greater than 1
    
    #Create Network
    network, node_usr, sources = create_network.create_network(num_usr, num_src, num_edg, loss_range, num_channels_per_source, topology='ring', density = 0.25)
    network = create_network.create_links(network, node_usr, num_lnk, link_pairs)

    #Routing
    network = routing.double_dijkstra(network, sources, tau, d1, d2)
    network = routing.link_info_packaging(network, fidelity_limit)
    

    #Resource Allocation
    combinations = routing.path_combinations(network, sort_type)
    untested_paths = True
    previous_best_utility = -np.inf
    previous_best_path = None
    previous_best_results = None
    index = -1
    n_combos = len(combinations)
    max_index  = int(n_combos * STOP_PERCENT)
    min_loss = combinations[0]['total_loss']

    all_utilities = [] #List of all utilities for plotting. This includes the interference ones
    all_utilities_no_interference = [] #List of all utilities but the ones with interference show -infinity for plotting
    all_network_loss = [] #List of all total network losses for plotting
    total_used_sources = [] #List of total number of used sources in network combo for plotting
    while untested_paths == True: #Run loop until we are out of path combos TODO make this part faster
        index += 1
        print('Completed: '+str(index)+' / '+str(n_combos))

        #Path choice and allocation
        path_choice, path_id = routing.choose_paths(combinations)
        if path_choice is None:
            untested_paths = False
            continue

        #Stopping conditions
        if index >= max_index: #Stop by percent complete
            print(f"Reached {STOP_PERCENT*100:.0f}% of combos ({index}/{n_combos} combos)—stopping.")
            break
        if path_choice['total_loss'] > LOSS_MULTIPLIER * min_loss: #Stop by percent of loss increase
            print(f"Loss {path_choice['total_loss']:.1f} > {LOSS_MULTIPLIER} x min_loss ({min_loss:.1f})—stopping.")
            break

        try:
            results, network_utility = allocate_channels.network_allocation(sources, path_choice, allocation_type = 'APOPT', initial_conditions = 0.001)
        except:
            print('failed')
            combinations[path_id]['usage_info'] = 'failed'
            continue
        all_utilities.append(network_utility)
        all_network_loss.append(combinations[path_id]['total_loss'])
        total_used_sources.append(len(results))
        if network_utility > previous_best_utility:
            channel_routing = routing.check_interference(path_choice, results, sources)
            if channel_routing == False: #Fail if find interference
                combinations[path_id]['usage_info'] = 'interfere'
                continue
            if previous_best_path is not None:
                combinations[previous_best_path]['usage_info'] = 'suboptimal'
            previous_best_utility = network_utility
            all_utilities_no_interference.append(previous_best_utility)
            previous_best_path = path_id
            previous_best_results = results
            previous_best_routing = channel_routing
        else:
            combinations[path_id]['usage_info'] = 'suboptimal'
            all_utilities_no_interference.append(-np.inf)

    #Plot Results

    #Utility Plots
    plot.utility_interference(all_utilities, all_utilities_no_interference)
    plot.utility_interference_zoomed(all_utilities, all_utilities_no_interference)
    plot.utility_vs_loss(all_utilities, all_utilities_no_interference, all_network_loss)
    plot.utility_vs_sources(all_utilities, all_utilities_no_interference, total_used_sources)

    plot.plot_network_final(network, previous_best_results)
    plot.source_allocation(previous_best_results, sources)

    sorted_combo_utils = sorted(zip(combinations, all_utilities), key=lambda cu: cu[1], reverse=True)

    print('test')


if __name__ == "__main__":
    main()