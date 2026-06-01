#network/topology.py
"""
This file contains functions for creating different network topologies based on user parameters.
"""

#ZHG
#2026.03.23
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
import random

#Topology functions:

def _ring_topology(network, nodes, cfg):
    """Creates ring topology by connecting source nodes in a ring and attaching user nodes to random sources."""
    sources = nodes["sources"]
    users = nodes["users"]

    #Connect sources in a ring
    for i in range(len(sources)):
        network.add_edge(sources[i], sources[(i + 1) % len(sources)])

    #Attach users
    for u in users:
        network.add_edge(u, random.choice(sources))

def _dense_topology(network, nodes, cfg):
    """Creates dense topology by first ensuring all nodes are connected and then adding extra edges based on density parameter."""
    all_nodes = nodes["sources"] + nodes["users"]

    _make_connected(network, all_nodes) #Ensure connectivity for all nodes at least once

    #Add extra edges based on density
    possible_edges = [(u, v) for i, u in enumerate(all_nodes) for v in all_nodes[i+1:]] #Generate all possible edges
    random.shuffle(possible_edges) #Shuffle the edges
    target_edges = int(len(possible_edges) * cfg.density)
    for (u, v) in possible_edges[:target_edges]: #Add edges until we reach the target number based on density
        network.add_edge(u, v)

def _star_topology(network, nodes, cfg):
    node_src = "sources" #Only one central source
    center_node = nodes[node_src][0]
    users = nodes["users"]
    for u in users: #Add every user onto the center node
        network.add_edge(u, center_node)

def _custom_topology(network, nodes, cfg):
    if not cfg.custom_edges:
        raise ValueError("custom topology requires cfg.custom_edges")

    valid_nodes = set(nodes["sources"] + nodes["users"])
    for edge in cfg.custom_edges:
        if len(edge) not in (2, 3):
            raise ValueError(f"Invalid custom edge specification: {edge}")

        u, v = edge[0], edge[1]
        if u not in valid_nodes or v not in valid_nodes:
            raise ValueError(f"Custom edge uses undefined node: {edge}")

        network.add_edge(u, v)
        if len(edge) == 3 and edge[2] is not None:
            network[u][v]["loss"] = float(edge[2])

def _random_topology(network, nodes, cfg):
    all_nodes = nodes["sources"] + nodes["users"]
    num_edg = cfg.num_edg

    _make_connected(network, all_nodes) #Ensure connectivity for all nodes at least once

    #Edge selection
    possible_edges = [(u, v) for idx, u in enumerate(all_nodes) for v in all_nodes[idx + 1:]] #Generate all possible edges
    random.shuffle(possible_edges) #Shuffle the edges
    selected_edges = possible_edges[:num_edg] #Select desired number of edges and add them
    for edge in selected_edges:
        network.add_edge(edge[0], edge[1])

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#Topology helper functions:

def _make_connected(network, nodes):
    """Helper function to ensure all nodes are connected by creating a random spanning tree."""
    order = nodes[:]
    random.shuffle(order)

    for i in range(1, len(order)):
        network.add_edge(order[i-1], order[i])