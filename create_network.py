#create_network.py
#Create an arbitrary network given network parameters
#Main function to give network plot
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

#create_network allows one to make an arbitrary network given user inputs. This also contains the various topology presets.
def create_network(num_usr, num_src, num_edg, loss_range=(1,30), num_channels_per_source=[None], topology=None, density=0.5, kite_loss_values=None, figure_loss_values=None):
    '''
    Inputs:
    num_usr, num_src, num_edg are for setting the number of a particular item in the network
    loss_range is the allowed range of loss values we can send in
    num_channels_per_source establishes the number of channels each source has available. The default amount is 5 each. Can instead specify 
    topology is the topology preset we can set
    density is the edge density percentage between 0 and 1. Only affects dense topologies.

    Outputs:
    network is the networkx topology object
    node_usr are the names of each user node
    source contains the amount of available channels in order of each source
    '''
    random.seed(40)
    network = nx.Graph()

    #Create node labels
    node_src = [f"S{i}" for i in range(num_src)]
    node_usr = [f"U{i}" for i in range(num_usr)]

    #Edge check
    all_nodes = node_src + node_usr
    max_edges = len(all_nodes) * (len(all_nodes) - 1) // 2
    if num_edg > max_edges:
        print(f"Too many edges specified. Setting number of edges to maximum possible: {max_edges}")
        num_edg = max_edges
    min_edges = len(all_nodes) - 1
    if num_edg < min_edges:
        print(f"Too few edges specified. Setting number of edges to minimum possible: {min_edges}")
        num_edg = min_edges

    # --- helper: add edges so graph is connected, then fill up to target_edges ---
    def _make_connected_then_fill(network, all_nodes, target_edges):
        # (1) connected backbone via random spanning tree
        order = all_nodes[:]  # copy
        random.shuffle(order)
        tree_edges = [(order[i-1], order[i]) for i in range(1, len(order))]
        network.add_edges_from(tree_edges)

        # (2) add extra random edges until target_edges
        max_edges = len(all_nodes) * (len(all_nodes) - 1) // 2
        target_edges = max(len(all_nodes) - 1, min(target_edges, max_edges))

        # pool of remaining possible edges
        existing = set(tuple(sorted(e)) for e in network.edges())
        possible_edges = [(u, v) for i, u in enumerate(all_nodes) for v in all_nodes[i+1:]]
        random.shuffle(possible_edges)
        needed = target_edges - (len(all_nodes) - 1)
        for (u, v) in possible_edges:
            if needed <= 0:
                break
            key = (u, v) if u < v else (v, u)
            if key in existing:
                continue
            network.add_edge(u, v)
            existing.add(key)
            needed -= 1

        # (3) safety net: if somehow not connected, stitch components together
        if not nx.is_connected(network):
            comps = [list(c) for c in nx.connected_components(network)]
            base = comps[0]
            for comp in comps[1:]:
                network.add_edge(random.choice(base), random.choice(comp))
                base += comp  # merge for next iteration

    #Topology presets:
    if topology == 'star': #One source in the middle with users surrounding
        node_src = [f"S0"]  #Only one central source
        network.add_nodes_from(node_src, node_type='source')
        network.add_nodes_from(node_usr, node_type='user')
        center_node = node_src[0]
        for u in node_usr: #Add every user onto the center node
            network.add_edge(u, center_node)

    elif topology == 'dense': #Connects nodes based on edge density value
        network.add_nodes_from(node_src, node_type='source')
        network.add_nodes_from(node_usr, node_type='user')

        # number of edges implied by density, but never below the spanning-tree minimum
        target_edges = max(len(all_nodes) - 1, int(round(max_edges * density)))
        _make_connected_then_fill(network, all_nodes, target_edges)

        #Edge selection
        possible_edges = [(u, v) for idx, u in enumerate(all_nodes) for v in all_nodes[idx + 1:]] #Generate all possible edges
        random.shuffle(possible_edges) #Shuffle the edges
        selected_edges = possible_edges[:int(round(max_edges*density))] #Select desired number of edges and add them
        for edge in selected_edges:
            network.add_edge(edge[0], edge[1])

    elif topology == 'ring':
        network.add_nodes_from(node_src, node_type='source')
        network.add_nodes_from(node_usr, node_type='user')

        #Ring generation
        ring_nodes = node_src.copy() #Ensure all sources are part of the ring
        num_ring_nodes = len(ring_nodes) #Form the ring
        if num_ring_nodes > 1:
            for i in range(num_ring_nodes):
                network.add_edge(ring_nodes[i], ring_nodes[(i + 1) % num_ring_nodes])  #Circular connection
        #Add sources to ring
        all_star_nodes = node_usr
        for node in all_star_nodes:
            central_src = random.choice(node_src)  #Randomly select a source from the ring
            network.add_edge(node, central_src)  #Connect to a source in the ring

    elif topology == 'kite': #Kite Topology: 3 users and 2 sources
        node_src = [f"Bob", f"Alice"]
        node_usr = [f"Charlie", f"Erin", f"Dave"]
        network.add_nodes_from(node_src, node_type='source')
        network.add_nodes_from(node_usr, node_type='user')

        # Add the fixed edge pattern
        edges = [
            (node_usr[0], node_src[0]),  # Alice - Alice_S
            (node_usr[1], node_src[0]),  # Bob   - Alice_S
            (node_usr[0], node_src[1]),  # Alice - Bob_S
            (node_usr[1], node_src[1]),  # Bob   - Bob_S
            (node_usr[2], node_src[1]),  # Charlie - Bob_S
        ]
        network.add_edges_from(edges)

        #You can override these by passing kite_loss_values= {...} when calling create_network().
        default_kite_losses = {
            (node_usr[0], node_src[0]): 12.0,  # Alice - Alice_S
            (node_usr[1], node_src[0]): 18.0,  # Bob   - Alice_S
            (node_usr[0], node_src[1]): 15.0,  # Alice - Bob_S
            (node_usr[1], node_src[1]): 10.0,  # Bob   - Bob_S
            (node_usr[2], node_src[1]): 20.0,  # Charlie - Bob_S
        }
        preset = kite_loss_values or default_kite_losses

        # Assign the preset losses
        for (u, v) in edges:
            # allow either direction in the dict
            if (u, v) in preset:
                network[u][v]['loss'] = float(preset[(u, v)])
            elif (v, u) in preset:
                network[u][v]['loss'] = float(preset[(v, u)])
            else:
                # if a specific edge wasn't given, fall back to a reasonable default
                network[u][v]['loss'] = float(np.mean(list(preset.values())))
    elif topology == 'figure':
        node_src = ["S_left", "S_mid", "S_bottom"]
        node_usr = ["U_blue_LT","U_green_LU","U_red_TC","U_green_TR",
                    "U_red_R","U_yellow_MR","U_yellow_ML","U_blue_BL"]

        network.add_nodes_from(node_src, node_type='source')
        color_tag = {
            "U_blue_LT":"blue","U_green_LU":"green","U_red_TC":"red","U_green_TR":"green",
            "U_red_R":"red","U_yellow_MR":"yellow","U_yellow_ML":"yellow","U_blue_BL":"blue"
        }
        for u in node_usr:
            network.add_node(u, node_type='user', color=color_tag.get(u))

        # Wires exactly like your figure
        edges = [
            ("U_blue_LT","S_left"), ("U_blue_LT","U_green_LU"),
            ("U_green_LU","U_yellow_ML"), ("S_left","U_yellow_ML"),
            ("U_green_LU","U_red_TC"), ("U_red_TC","S_mid"),
            ("S_mid","U_yellow_ML"), ("S_mid","U_yellow_MR"),
            ("U_yellow_ML","U_blue_BL"), ("U_blue_BL","S_bottom"),
            ("S_bottom","U_yellow_ML"), ("S_bottom","U_yellow_MR"),
            ("U_yellow_MR","U_red_R"), ("U_green_TR","U_red_R")
        ]
        network.add_edges_from(edges)

        # Fixed coordinates (roughly matching your screenshot)
        pos = {
            "U_blue_LT":  (-3.0,  1.4),
            "S_left":     (-3.0, -0.2),
            "U_green_LU": (-2.0,  0.7),
            "U_yellow_ML":(-2.1, -0.2),
            "U_blue_BL":  (-2.5, -1.3),
            "U_red_TC":   (-0.2,  1.4),
            "S_mid":      ( 0.0,   0.2),
            "S_bottom":   (-0.9,  -1.0),
            "U_yellow_MR":( 1.0,  -0.1),
            "U_red_R":    ( 2.0,   0.3),
            "U_green_TR": ( 1.2,   1.1),
        }
        network.graph['pos'] = pos  # <- only set for the 'figure' preset

        # Optional: preset losses you can override with figure_loss_values
        default_figure_losses = {
            ("U_blue_LT","S_left"):13.0, ("U_blue_LT","U_green_LU"):11.0,
            ("U_green_LU","U_yellow_ML"):9.5, ("S_left","U_yellow_ML"):12.0,
            ("U_green_LU","U_red_TC"):10.5, ("U_red_TC","S_mid"):8.0,
            ("S_mid","U_yellow_ML"):14.0, ("S_mid","U_yellow_MR"):10.0,
            ("U_yellow_ML","U_blue_BL"):11.5, ("U_blue_BL","S_bottom"):7.5,
            ("S_bottom","U_yellow_ML"):9.0, ("S_bottom","U_yellow_MR"):8.5,
            ("U_yellow_MR","U_red_R"):12.5, ("U_green_TR","U_red_R"):9.0
        }
        preset = figure_loss_values or default_figure_losses
        for (u, v) in edges:
            network[u][v]['loss'] = float(preset.get((u, v), preset.get((v, u),
                                        np.mean(list(preset.values())))))

    else: #Arbitrary topology that randomly chooses the edges
        network.add_nodes_from(node_src, node_type='source')
        network.add_nodes_from(node_usr, node_type='user')

        # ensure at least a connected graph with exactly num_edg edges
        _make_connected_then_fill(network, all_nodes, num_edg)

        #Edge selection
        possible_edges = [(u, v) for idx, u in enumerate(all_nodes) for v in all_nodes[idx + 1:]] #Generate all possible edges
        random.shuffle(possible_edges) #Shuffle the edges
        selected_edges = possible_edges[:num_edg] #Select desired number of edges and add them
        for edge in selected_edges:
            network.add_edge(edge[0], edge[1])


    #Loss assigning
    for (u, v) in network.edges(): #Assign loss values to all edges (loss in dB)
        if 'loss' not in network[u][v]:
            network[u][v]['loss'] = random.uniform(loss_range[0], loss_range[1])



    #Initialize available channels for each source
    sources = {}
    if num_channels_per_source is None or (isinstance(num_channels_per_source, list) and num_channels_per_source[0] is None): #If user didn't specify number of channels, default to 5 per source
        num_channels_per_source = np.ones(len(node_src)) * 5
    for s in range(len(node_src)): #Add sources into 
        sources[node_src[s]] = {'available_channels': list(range(1, int(num_channels_per_source[s] + 1)))}

    return network, node_usr, sources


#Function to randomly create links or use user choices, and put them into the network object for future functions
def create_links(network, node_usr, num_lnk = 1, link_pairs = None):
    '''
    Inputs:
    network is the networkx topology object
    node_user are the names of each of the users stored in an array
    num_link is the number of desired links to be randomly created
    link_pairs is an array that indicates which links are wanted to be a pair
    
    Outputs:
    network is the networkx topology object with the new link data inserted into it via dictionaries
    '''
    #Link Generation
    user_pairs = [(u, v) for idx, u in enumerate(node_usr) for v in node_usr[idx + 1:]] #Get every combination of user pairs
    random.shuffle(user_pairs)
    desired_links = []
    if link_pairs: #Remove and add specified links from inputted link_pairs variable
        for link_choice in link_pairs:
            user_pairs.remove(link_choice)
            desired_links.append(link_choice)
            num_lnk -= 1
    if num_lnk > 0: #For remaining links, take from the start of user_pairs
        desired_links += user_pairs[:num_lnk] #Select the first set of links
    
    #Store desired links in the graph
    desired_link_info = []
    for link in desired_links:
        desired_link_info.append({'link': link})
    network.graph['desired_links'] = desired_link_info
    return network

#plot_network plots the graph topology
def plot_network(network):
    # use fixed positions if the graph provides them; otherwise spring_layout
    pos = network.graph.get('pos', nx.spring_layout(network))

    plt.figure(figsize=(12, 10))

    #Draw nodes with colors based on node type.
    node_colors = []
    for node, data in network.nodes(data=True):
        if data['node_type'] == 'source':
            node_colors.append('lightblue')
        elif data['node_type'] == 'user':
            node_colors.append('lightcoral')
        else:
            node_colors.append('gray')
    nx.draw_networkx_nodes(network, pos, node_color=node_colors, node_size=500)

    #Draw all network edges in lightgray.
    nx.draw_networkx_edges(network, pos, edge_color='lightgray', width=1)

    #Draw the links along their computed paths with their unique colors.
    for desired_link in network.graph['desired_links']:
        if 'channels_assigned' in desired_link:
            link_color = desired_link['color']
            paths = desired_link['paths']
            edges_u1 = list(zip(paths['user1'], paths['user1'][1:]))
            edges_u2 = list(zip(paths['user2'], paths['user2'][1:]))
            edges_in_link = edges_u1 + edges_u2
            nx.draw_networkx_edges(
                network, pos,
                edgelist=edges_in_link,
                edge_color=[link_color],
                width=3,
                alpha=0.5
            )

    ##Draw edge labels showing only loss values.
    #edge_labels = {(u, v): f"{data['loss']:.2f}" for u, v, data in network.edges(data=True)}
    #nx.draw_networkx_edge_labels(network, pos, edge_labels=edge_labels, font_size=8)

    ##Draw node labels.
    #nx.draw_networkx_labels(network, pos, font_size=10, font_weight='bold')

    ##Build the legend with link information including channel assignments.
    #legend_elements = []
    #for desired_link in network.graph['desired_links']:
    #    if 'channels_assigned' in desired_link and 'frequencies' in desired_link:
    #        link = desired_link['link']
    #        #Determine the source from one of the paths (user1's path end)
    #        source = desired_link['paths']['user1'][-1]
    #        total_loss = desired_link['total_loss']
    #        link_color = desired_link['color']
    #        freqs = desired_link['frequencies']  #Expects a dict with 'user1' and 'user2'
    #        #Format the label to include the channels in order (user1 then user2)
    #        label = f"Link {link[0]}-{source}-{link[1]}, Channels: {freqs['user1']}/{freqs['user2']}, Loss={total_loss:.2f}"
    #        legend_elements.append(Patch(facecolor=link_color, edgecolor='k', label=label))

    #Add the legend to the plot.
    #plt.legend(handles=legend_elements, loc='best', fontsize='small')
    #plt.title("Network Topology")
    plt.axis('off')
    plt.show()

def main():
    num_usr, num_src, num_edg, num_lnk, top = 7, 1, 20, 5, 'star' #Star
    num_usr, num_src, num_edg, num_lnk, top = 9, 6, 20, 5, 'ring' #Ring
    #num_usr, num_src, num_edg, num_lnk, top = 8, 5, 20, 5, 'dense' #Dense
    num_usr, num_src, num_edg, num_lnk, top = 1, 1, 1, 1, 'kite' #Figure
    loss_range=(1,30)
    network, node_usr, sources = create_network(num_usr, num_src, num_edg, loss_range, num_channels_per_source=[None], topology=top, density = 0.4)
    network = create_links(network, node_usr, num_lnk, link_pairs = None)
    plot_network(network)


if __name__ == "__main__":
    main()