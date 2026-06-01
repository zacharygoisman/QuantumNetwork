#network/builder.py
"""
Build arbitrary network topologies based on user parameters.
"""

#ZHG
#2026.03.20
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from network import topology

def build_network(cfg):
    """Calls main network funcions to create network and links."""
    network, users, sources = create_network(cfg)
    links = create_links(
        users,
        cfg.num_lnks,
        link_pairs=cfg.link_pairs,
        require_disjoint=cfg.require_disjoint_links,
    )
    return network, sources, links

def create_network(cfg):
    """Creates network topology based on parameters in cfg."""
    random.seed(cfg.random_seed)
    network = nx.Graph()

    nodes = _create_nodes(network, cfg)
    _add_edges(network, nodes, cfg)
    _assign_losses(network, cfg)
    sources = _initialize_sources(nodes, cfg)
    return network, nodes["users"], sources

def _create_nodes(network, cfg):
    """Creates source and user nodes and adds them to the network."""

    #Create node labels (use 1-based numbering so labels are S1..SN and U1..UM)
    sources = [f"S{i+1}" for i in range(cfg.num_src)]
    users = [f"U{i+1}" for i in range(cfg.num_usr)]

    #Add nodes to network
    network.add_nodes_from(sources, node_type="source")
    network.add_nodes_from(users, node_type="user")

    return {"sources": sources, "users": users}

def _add_edges(network, nodes, cfg):
    """Calls topology-specific edge creation functions based on cfg.topology."""
    if cfg.topology == "dense":
        topology._dense_topology(network, nodes, cfg)
    elif cfg.topology == "ring":
        topology._ring_topology(network, nodes, cfg)
    elif cfg.topology == "star":
        topology._star_topology(network, nodes, cfg)
    elif cfg.topology == "custom":
        # Use the explicitly provided custom edges in cfg.custom_edges
        topology._custom_topology(network, nodes, cfg)
    # elif cfg.topology == "kite":
    #     topology._kite_topology(network, nodes)
    # elif cfg.topology == "figure":
    #     topology._figure_topology(network, nodes)
    else:
        topology._random_topology(network, nodes, cfg)

def _assign_losses(network, cfg):
    """Assigns loss values to edges in the network, either randomly within a specified range or based on presets for certain topologies."""
    for (u, v) in network.edges(): #Assign loss values to all edges (loss in dB)
        if "loss" not in network[u][v]:
            network[u][v]["loss"] = random.uniform(*cfg.loss_range)

def _initialize_sources(nodes, cfg):
    """Initializes source nodes with available channels based on cfg.num_channels."""
    sources = {}

    if cfg.custom_source_channels:
        for s in nodes["sources"]:
            channels = cfg.custom_source_channels[s]
            sources[s] = {"available_channels": list(channels)}
        return sources

    for i, s in enumerate(nodes["sources"]): #Initialize each source with a list of available channels (1 to num_channels)
        sources[s] = {"available_channels": list(range(1, int(cfg.num_channels[i]) + 1))}
    return sources


#Function to randomly create links or use user choices, and put them into the network object for future functions
def create_links(node_usr, num_lnk=1, link_pairs=None, *, require_disjoint=True):
    user_pairs = [(u, v) for i, u in enumerate(node_usr) for v in node_usr[i+1:]]
    random.shuffle(user_pairs)

    desired_links = []
    used = set()

    #1) Seed with user-provided pairs
    if link_pairs:
        for (u, v) in link_pairs:
            if require_disjoint and (u in used or v in used):
                raise ValueError(f"Overlapping user in {u, v}")
            desired_links.append((u, v))
            used.update([u, v])

    #2) Fill remaining
    remaining = num_lnk - len(desired_links)
    if remaining > 0:
        if require_disjoint:
            for (u, v) in user_pairs:
                if u in used or v in used:
                    continue
                desired_links.append((u, v))
                used.update([u, v])
                if len(desired_links) >= num_lnk:
                    break
        else:
            desired_links += user_pairs[:remaining]

    #3) Warning
    if len(desired_links) < num_lnk and require_disjoint:
        # Not enough disjoint pairs available to satisfy the request.
        print(f"[create_links] Only formed {len(desired_links)} disjoint links; filling remaining with overlapping pairs")
        # Fill remaining allowing overlaps
        for (u, v) in user_pairs:
            if len(desired_links) >= num_lnk:
                break
            if (u, v) in desired_links:
                continue
            desired_links.append((u, v))

    return desired_links