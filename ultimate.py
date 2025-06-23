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
    num_usr, num_src, num_edg, num_lnk= 10, 5, 20, 7
    loss_range=(10,20)
    # y1 = [0, 0.0034, 0, 0.0299, 0.0385, 0.0625, 0.0733, 0, 0.1106, 0.125, 0.1489, 0]
    # y2 = [0, 0.0006, 0.0357, 0.0051, 0.0066, 0.0107, 0.0126, 0.1818, 0.019, 0.0214, 0.0256, 0.2979]
    fidelity_limit = np.repeat(0.7,num_lnk)
    tau = 1e-9
    d1 = 100
    d2 = 3500
    link_pairs = None
    num_channels_per_source = [5, 7, 5, 3, 6] #[None] assumes 5 channels per source 
    
    #Create Network
    network, node_usr, sources = create_network.create_network(num_usr, num_src, num_edg, loss_range, num_channels_per_source, topology='ring', density = 0.1)
    network = create_network.create_links(network, node_usr, num_lnk, link_pairs)

    #Routing
    network = routing.double_dijkstra(network, sources, tau, d1, d2)
    network = routing.link_info_packaging(network, fidelity_limit)
    

    #Resource Allocation
    combinations = routing.path_combinations(network)
    untested_paths = True
    previous_best_utility = -np.inf
    previous_best_path = None
    previous_best_results = None
    all_utilities = [] #List of all utilities for plotting. This includes the interference ones
    all_utilities_no_interference = [] #List of all utilities but the ones with interference show -infinity
    while untested_paths == True: #Run loop until we are out of path combos TODO make this part faster
        path_choice, path_id = routing.choose_paths(combinations)
        if path_choice is None:
            untested_paths = False
            continue
        try:
            results, network_utility = allocate_channels.network_allocation(sources, path_choice, allocation_type = 'APOPT', initial_conditions = 0.001)
        except:
            print('failed')
            combinations[path_id]['usage_info'] = 'failed'
            continue
        all_utilities.append(network_utility)
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
    plot.utility_interference(all_utilities, all_utilities_no_interference)
    plot.plot_network_final(network)
    plot.source_allocation(previous_best_results)

if __name__ == "__main__":
    main()