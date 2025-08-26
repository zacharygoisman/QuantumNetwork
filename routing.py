import networkx as nx
import matplotlib.pyplot as plt
import random
import csv
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
from scipy.optimize import fsolve
import itertools
from ortools.sat.python import cp_model

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
def path_combinations(network, sort_type):
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
                'path': path  # full path dict preserved
            })
        results.append({
            'combo': combo_with_links,
            'total_loss': total_loss,
            'path_id': path_id,
            'usage_info': 'untested'
        })
        path_id += 1

    #Sort
    if sort_type == 'loss':
        sorted_results = sorted(results, key=lambda r: (r['total_loss'], -len({p['path']['source'] for p in r['combo']}))) #Sort combinations by total loss and total number of sources
    elif sort_type == 'hop':
        sorted_results = sorted(results, key=lambda r: (np.mean([len(p['path']['path1'] + p['path']['path2']) - 1 for p in r['combo']]), r['total_loss'], -len({p['path']['source'] for p in r['combo']})))
    return sorted_results


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

    link_to_count = {}
    for src_info in source_dicts:
        for link, alloc in zip(src_info['links'], src_info['channel_allocation']): #Extract the channel allocation values for each correctly mapped link
            link_to_count[link] = int(alloc[0]) #Flatten channel allocation list of floats for each link
    return [link_to_count[link] for link in ordered_links] #List of source and associated channel amounts per link in correct order


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

    # unpack link data ----------------------------------------------------
    links       = path_choice['combo']
    nlinks      = len(links)
    paths_u1    = [lk['path']['path1'] for lk in links]
    paths_u2    = [lk['path']['path2'] for lk in links]      # ← REAL 2nd path
    link_src    = [lk['path']['source'] for lk in links]
    link_names  = [lk['link']            for lk in links]
    k_vals      = extract_link_ch_counts(results, link_names)

    edges_u1    = [edges(p) for p in paths_u1]
    edges_u2    = [edges(p) for p in paths_u2]
    all_edges   = set(e for el in edges_u1+edges_u2 for e in el)
    all_ch      = sorted({c for src in link_src
                            for c in sources[src]['available_channels']})

    # model ---------------------------------------------------------------
    m = cp_model.CpModel()
    swap  = [m.NewBoolVar(f"swap_{i}") for i in range(nlinks)]
    use   = {}              # (i,c)  channel chosen?
    pos   = {}              # (i,e,c) channel c flows + on edge e for link i
    neg   = {}              # (i,e,c) channel c flows - on edge e for link i

    for i, src in enumerate(link_src):
        pool = sources[src]['available_channels']
        for c in pool:
            use[i, c] = m.NewBoolVar(f"use_{i}_{c}")
        m.Add(sum(use[i, c] for c in pool) == k_vals[i])

        for c in pool:
            for e in edges_u1[i] + edges_u2[i]:
                pos[i, e, c] = m.NewBoolVar(f"p_{i}_{e}_{c}")
                neg[i, e, c] = m.NewBoolVar(f"n_{i}_{e}_{c}")

        # reify edge usage -----------------------------------------------
        for c in pool:
            # U1 arm:   swap=0 → +c,  swap=1 → –c
            for e in edges_u1[i]:
                m.Add(pos[i, e, c] == 1).OnlyEnforceIf([use[i, c], swap[i].Not()])
                m.Add(pos[i, e, c] == 0).OnlyEnforceIf(use[i, c].Not())
                m.Add(pos[i, e, c] == 0).OnlyEnforceIf(swap[i])
                m.Add(neg[i, e, c] == 1).OnlyEnforceIf([use[i, c], swap[i]])
                m.Add(neg[i, e, c] == 0).OnlyEnforceIf(use[i, c].Not())
                m.Add(neg[i, e, c] == 0).OnlyEnforceIf(swap[i].Not())

            # U2 arm:   opposite sign mapping
            for e in edges_u2[i]:
                m.Add(neg[i, e, c] == 1).OnlyEnforceIf([use[i, c], swap[i].Not()])
                m.Add(neg[i, e, c] == 0).OnlyEnforceIf(use[i, c].Not())
                m.Add(neg[i, e, c] == 0).OnlyEnforceIf(swap[i])
                m.Add(pos[i, e, c] == 1).OnlyEnforceIf([use[i, c], swap[i]])
                m.Add(pos[i, e, c] == 0).OnlyEnforceIf(use[i, c].Not())
                m.Add(pos[i, e, c] == 0).OnlyEnforceIf(swap[i].Not())

    # edge-conflict: ≤1 pos and ≤1 neg per (edge, channel) --------------
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

    # assemble answer -----------------------------------------------------
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