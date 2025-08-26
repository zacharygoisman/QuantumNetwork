#All the plotting code
import matplotlib.pyplot as plt
import numpy as np
import allocate_channels
import networkx as nx
import matplotlib.lines as mlines
import pathlib
pathlib.Path("outputs").mkdir(exist_ok=True)

def plot_network_final(network, previous_best_results):
    if previous_best_results is None:
        print("[plot] No feasible solution to draw – skipping plot_network_final")
        return
    # existing code follows …


def source_allocation(previous_best_results, sources):
    if previous_best_results is None:
        print("[plot] No feasible solution to draw – skipping source_allocation")
        return
    # existing code follows …

#This is a plot that compares the utility between the various reruns of the optimizer
def utility_interference(all_utilities, all_utilities_no_interference):
    """
    Inputs:
    all_utilities is a list of all utilities determined through the optimizer
    all_utilities_no_interference is a list of all utilities with the failed ones being replaced with negative infinity
    """
    x_range = np.arange(1,len(all_utilities)+1) #Optimizer attempt number
    plt.figure(figsize=(10, 6))
    plt.scatter(x_range, all_utilities, s=3)#, label = 'Points with Interference or Less Utility')
    #plt.scatter(x_range, all_utilities_no_interference, s=3, label = 'Best Utility Values')
    #plt.legend(loc = 'best')
    plt.title('Comparing Utilities of Successful and Failed Optimization Attempts')
    plt.xlabel('Successful Rerun Attempts')
    plt.ylabel('Utility')
    #plt.show()
    plt.savefig('outputs/utility_comparison.png')

#This is a plot that compares the utility between the various reruns of the optimizer, but zoomed in
def utility_interference_zoomed(all_utilities, all_utilities_no_interference):
    """
    Inputs:
    all_utilities is a list of all utilities determined through the optimizer
    all_utilities_no_interference is a list of all utilities with the failed ones being replaced with negative infinity
    """
    if not all_utilities_no_interference or np.isneginf(all_utilities_no_interference).all():
        print("[plot] No interference-free solutions – skipping zoomed plot")
        return

    x_range = np.arange(1,len(all_utilities)+1) #Optimizer attempt number
    plt.figure(figsize=(10, 6))
    plt.scatter(x_range, all_utilities, s=3, label = 'Points with Interference or Less Utility')
    plt.scatter(x_range, all_utilities_no_interference, s=3, label = 'Best Utility Values')
    plt.legend(loc = 'best')
    plt.title('Comparing Utilities of Successful and Failed Optimization Attempts')
    plt.xlabel('Successful Rerun Attempts')
    plt.xlim([0,1+np.argmax(all_utilities_no_interference)])
    plt.ylabel('Utility')
    #plt.show()
    plt.savefig('outputs/utility_comparison_zoomed.png')

#This is a plot that compares the utility between the various reruns of the optimizer
def utility_vs_loss(all_utilities, all_utilities_no_interference, loss):
    """
    Inputs:
    all_utilities is a list of all utilities determined through the optimizer
    all_utilities_no_interference is a list of all utilities with the failed ones being replaced with negative infinity
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(loss, all_utilities, s=3, label = 'Points with Interference or Less Utility')
    plt.scatter(loss, all_utilities_no_interference, s=3, label = 'Best Utility Values')
    plt.legend(loc = 'best')
    plt.title('Comparing Utilities of Successful and Failed Optimization Attempts Against Network Loss')
    plt.xlabel('Total Network Loss')
    plt.ylabel('Utility')
    #plt.show()
    plt.savefig('outputs/utility_loss_comparison.png')

#This is a plot that compares the utility with the number of used sources
def utility_vs_sources(all_utilities, all_utilities_no_interference, total_used_sources):
    """
    Inputs:
    all_utilities is a list of all utilities determined through the optimizer
    all_utilities_no_interference is a list of all utilities with the failed ones being replaced with negative infinity
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(total_used_sources, all_utilities, s=3, label = 'Points with Interference or Less Utility')
    plt.scatter(total_used_sources, all_utilities_no_interference, s=3, label = 'Best Utility Values')
    plt.legend(loc = 'best')
    plt.title('Comparing Utilities of Successful and Failed Optimization Attempts Against Total Used Sources')
    plt.xlabel('Total Used Sources')
    plt.ylabel('Utility')
    #plt.show()
    plt.savefig('outputs/utility_source_comparison.png')


#Network plot showing nodes, edges, and link paths TODO: Add link paths
def plot_network_final(network, previous_best_results):
    """
    network: a NetworkX Graph containing your nodes & edges,
             with node attribute 'node_type' in {'source','user',...}
    previous_best_results: list of dicts, each with keys:
        'source'            : e.g. "S0"
        'links'             : list of tuples like [("U3","U4"), ...]
        'channel_allocation': nested list-of-lists, one sublist per link
        'total_channels'    : int
        'utility'           : float
        (etc.)
    """
    # 1. Layout & base figure
    pos = nx.spring_layout(network)
    fig, ax = plt.subplots(figsize=(12, 10))

    # 2. Draw nodes by type
    color_map = {'source': 'lightblue', 'user': 'lightcoral'}
    node_colors = [
        color_map.get(data.get('node_type'), 'gray')
        for n, data in network.nodes(data=True)
    ]
    nx.draw_networkx_nodes(network, pos,
                           node_color=node_colors,
                           node_size=500,
                           ax=ax)

    # 3. Draw all background edges in light gray
    nx.draw_networkx_edges(network, pos,
                           edge_color='lightgray',
                           width=1,
                           ax=ax)

    # 4. Build a unique color for every link‐tuple across all results
    all_links = [link for pr in previous_best_results for link in pr['links']]
    unique_links = sorted(set(all_links))
    cmap = plt.get_cmap('tab20')
    link_colors = {link: cmap(i % 20)
                   for i, link in enumerate(unique_links)}

    # 5. Draw each link’s two half‐paths and collect a legend entry
    legend_handles = []
    legend_labels  = []

    for pr in previous_best_results:
        src = pr['source']
        util = pr['utility']
        ca   = pr.get('channel_allocation', [])

        for j, link in enumerate(pr['links']):
            u1, u2 = link
            color  = link_colors[link]

            # find shortest paths user→source
            try:
                p1 = nx.shortest_path(network, u1, src)
                p2 = nx.shortest_path(network, u2, src)
            except nx.NetworkXNoPath:
                # skip if there is no path in the graph
                continue

            edges = list(zip(p1, p1[1:])) + list(zip(p2, p2[1:]))

            # draw them
            nx.draw_networkx_edges(
                network, pos,
                edgelist=edges,
                edge_color=[color],
                width=3,
                alpha=0.7,
                ax=ax
            )

            # grab this link’s allocation list
            if j < len(ca):
                per_alloc = ca[j]
                # ensure it’s a simple list
                if not isinstance(per_alloc, list):
                    per_alloc = [per_alloc]
            else:
                per_alloc = []

            # legend text: “U3–U4: U=0.045, alloc=[5.0]”
            lbl = f"{u1}–{u2}: U={util:.3f}, alloc={per_alloc}"

            # proxy artist for the legend
            h = mlines.Line2D([], [], color=color,
                              linewidth=3, alpha=0.7)
            legend_handles.append(h)
            legend_labels.append(lbl)

    # 6. Draw edge‐loss labels and node‐labels
    edge_labels = {
        (u, v): f"{d['loss']:.2f}"
        for u, v, d in network.edges(data=True)
    }
    nx.draw_networkx_edge_labels(network, pos,
                                 edge_labels=edge_labels,
                                 font_size=8,
                                 ax=ax)
    nx.draw_networkx_labels(network, pos,
                            font_size=10,
                            font_weight='bold',
                            ax=ax)

    plt.title("Network Topology")
    plt.axis('off')

    # 7. Legend (only if we have any handles)
    if legend_handles:
        ax.legend(legend_handles,
                  legend_labels,
                  loc='upper right',
                  bbox_to_anchor=(1.15, 1))
    else:
        ax.text(1.02, 0.5,
                "No links found in previous_best_results",
                transform=ax.transAxes,
                fontsize=9, va='center')

    plt.tight_layout()
    plt.savefig('outputs/network_plot.png')
    plt.close()

#Bar plot to show channel allocation to each link by source.
def source_allocation(previous_best_results, sources):
    """
    previous_best_results: list of dicts, each with keys
        'source'             (e.g. "S0")
        'links'              (e.g. ["U2", "U3"])
        'channel_allocation' (e.g. [[1,0,1],[0,1,0]])
        'total_channels'     (int)
    sources: dict mapping source → dict with key
        'available_channels': list of ints (so len(...) gives total_channels)
    """
    # map your results by source name
    results_map = {entry['source']: entry for entry in previous_best_results}
    
    # we'll plot in the order of sources.keys()
    source_names = list(sources.keys())
    bar_width    = 0.5

    # collect all links from the **existing** results for coloring
    all_links = {link for entry in previous_best_results for link in entry['links']}
    cmap      = plt.get_cmap('tab20')
    colors    = {link: cmap(i) for i, link in enumerate(sorted(all_links))}
    colors['Unused'] = 'lightgrey'

    fig, ax = plt.subplots(figsize=(10, 6))
    seen_labels = set()

    for i, src in enumerate(source_names):
        # get your existing entry or stub one with zero-used
        if src in results_map:
            entry = results_map[src]
        else:
            entry = {
                'source': src,
                'links': [],
                'channel_allocation': [],
                'total_channels': len(sources[src]['available_channels'])
            }

        bottom = 0
        used   = 0

        # plot each used-link segment
        for j, link in enumerate(entry['links']):
            # defensive: if channel_allocation has fewer rows than links
            count = int(sum(entry['channel_allocation'][j])) if j < len(entry['channel_allocation']) else 0
            if count <= 0:
                continue
            used += count

            # only show legend once per link
            if link not in seen_labels:
                label = str(link)
                seen_labels.add(link)
            else:
                label = "_nolegend_"

            ax.bar(i, count,
                   bottom=bottom,
                   width=bar_width,
                   color=colors[link],
                   label=label)
            bottom += count

        # plot the unused channels
        unused = entry['total_channels'] - used
        if unused > 0 or used == 0:
            if 'Unused' not in seen_labels:
                label = 'Unused'
                seen_labels.add('Unused')
            else:
                label = "_nolegend_"
            ax.bar(i, unused,
                   bottom=bottom,
                   width=bar_width,
                   color=colors['Unused'],
                   label=label)

    # formatting
    ax.set_xticks(range(len(source_names)))
    ax.set_xticklabels(source_names)
    ax.set_ylabel("Number of Channels")
    ax.set_title("Channel Allocation per Source")

    # build a single legend, with "Unused" forced to the end
    handles, labels = ax.get_legend_handles_labels()
    if 'Unused' in labels:
        idx = labels.index('Unused')
        order = [k for k in range(len(labels)) if k != idx] + [idx]
        handles = [handles[k] for k in order]
        labels  = [labels[k]  for k in order]

    ax.legend(handles, labels,
              loc='upper right',
              bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.savefig('outputs/source_allocation.png', dpi=300)
    plt.close()
