#All the plotting code
import matplotlib.pyplot as plt
import numpy as np
import allocate_channels
import networkx as nx
import matplotlib.lines as mlines
import pathlib
pathlib.Path("outputs").mkdir(exist_ok=True)

def _to_int_channels(k):
    """
    Robustly convert channel allocation 'k' into an int.
    Handles: plain numbers, numpy scalars, lists/arrays, GEKKO GKVariables.
    Strategy:
      - GKVariable: use .value[0] if present
      - list/ndarray: if size==1 -> that value; else sum (e.g., 0/1 indicators)
      - fallback: float() + round()
    """
    import numpy as np

    try:
        # GEKKO variable?
        if hasattr(k, 'value'):
            v = k.value
            if isinstance(v, (list, tuple, np.ndarray)):
                if len(v) == 0:
                    return 0
                return int(round(float(v[0])))
            return int(round(float(v)))

        # Numpy array / list
        if isinstance(k, (list, tuple, np.ndarray)):
            arr = np.asarray(k, dtype=float)
            if arr.size == 0:
                return 0
            if arr.size == 1:
                return int(round(float(arr.item())))
            # If it's a vector (e.g., 0/1 flags), sum it
            return int(round(float(arr.sum())))

        # Numpy scalar or plain number
        return int(round(float(k)))
    except Exception:
        # Last-ditch attempt
        try:
            import numpy as np
            return int(round(float(np.asarray(k).item())))
        except Exception:
            return 0


def _augment_legend_with_frequencies(ax, freqs_by_link):
    """
    freqs_by_link: dict{ ('A','B'): (to_u1_list, to_u2_list), ... }
    Updates legend labels to include assigned positive/negative frequencies.
    """
    if not freqs_by_link:
        return
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for lab in labels:
        s = lab.strip()
        base = s.replace('Link ', '')
        if '-' in base:
            u1, u2 = base.split('-', 1)
        elif len(base) == 2:
            u1, u2 = base[0], base[1]
        else:
            new_labels.append(lab)
            continue
        key = (u1, u2)
        if key in freqs_by_link:
            to_u1, to_u2 = freqs_by_link[key]
            lab = f"Link {u1}-{u2}  (+{list(to_u1)} → {u1},  -{list(to_u2)} → {u2})"
        new_labels.append(lab)
    ax.legend(handles, new_labels, loc='best')


    
# --- MATLAB "gem" palette (cycled) + typography defaults ---
GEM = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
       '#77AC30', '#4DBEEE', '#A2142F', '#003BFF',
       '#017501', '#FF0000', '#B526FF', '#FF00FF', '#000000']

def gem_colors(n, start=0):
    return [GEM[(start+i) % len(GEM)] for i in range(n)]

import matplotlib as mpl
mpl.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 14
})

def plot_source_allocation(K_values, allocations_by_link, link_labels):
    """
    Build a stacked bar chart like your reference figure.

    Parameters
    ----------
    K_values : list[int]        # e.g., [5, 10, 20, 40]
    allocations_by_link : list[list[int]]
        Matrix shape (n_links x n_K). Row i is counts for link i across each K.
        Example for 5 links -> [[1,2,4,9], [1,2,4,8], ...]
    link_labels : list[str]     # e.g., ["Null Link","Link AB","Link CD",...]

    Output: outputs/source_allocation.png
    """
    import numpy as np
    import matplotlib.pyplot as plt

    K_values = list(K_values)
    A = np.asarray(allocations_by_link, dtype=float)  # (n_links, n_K)
    n_links, n_k = A.shape
    assert len(K_values) == n_k, "K axis length must match columns of allocations_by_link"

    colors = gem_colors(n_links)
    # If "Null Link" exists, force it to index 0 (like your figure)
    order = list(range(n_links))
    if "Null Link" in link_labels:
        i0 = link_labels.index("Null Link")
        if i0 != 0:
            order.pop(i0); order = [i0] + order
    A = A[order, :]
    link_labels = [link_labels[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 6))
    bottoms = np.zeros(n_k)
    x = np.arange(n_k)

    for i in range(n_links):
        ax.bar(x, A[i], bottom=bottoms,
               color=colors[i], edgecolor='black', linewidth=1.2,
               label=link_labels[i])
        bottoms += A[i]

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in K_values])
    ax.set_xlabel('K')
    ax.set_ylabel('No. of Channels')

    # Place legend nicely
    ax.legend(loc='upper left', frameon=True)

    # Tight layout + save
    plt.tight_layout()
    plt.savefig('outputs/source_allocation.png', dpi=300)
    plt.close()


def plot_link_utility_bars(rows, start_color_index=0, outfile='outputs/link_utility_bars.png'):
    """
    rows: list of dicts from summarize_best(); each row must have:
          'link', 'link_utility', 'ub_utility'
    Bar outline (black) = max per-link utility; fill = actual utility.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if not rows:
        return

    links = [r['link'] for r in rows]
    ub = np.array([float(r['ub_utility']) if r.get('ub_utility') is not None else np.nan for r in rows])
    ac = np.array([float(r['link_utility']) for r in rows])

    x = np.arange(len(links))
    w = 0.72
    colors = gem_colors(len(links), start=start_color_index)

    fig, ax = plt.subplots(figsize=(10, 6))
    # Max bars as hollow boxes
    ax.bar(x, ub, width=w, facecolor='white', edgecolor='black', linewidth=1.5, label='Max per-link utility')
    # Actual bars as filled overlays
    ax.bar(x, ac, width=w*0.82, color=colors, edgecolor='black', linewidth=1.0, label='Actual utility')

    ax.set_xticks(x)
    ax.set_xticklabels(links, rotation=30, ha='right')
    ax.set_xlabel('Link')
    ax.set_ylabel('Utility (log₁₀ units)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()



def utility_comparison(all_utilities, dashed_limit, outfile='outputs/utility_comparison.png'):
    """
    all_utilities : list[float]  (e.g., best utility from each attempt)
    dashed_limit : float         (sum of per-link ceilings = max possible network utility)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if not all_utilities:
        return
    y = np.array(all_utilities, dtype=float)
    x = np.arange(1, len(y)+1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, y, marker='o', linewidth=1.5)
    ax.set_xlabel('Attempts')
    ax.set_ylabel('Utility')

    # Dashed line for absolute network max
    if np.isfinite(dashed_limit):
        ax.axhline(dashed_limit, linestyle='--', linewidth=1.75, label='Max possible (network)')

    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

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
    #plt.title('Comparing Utilities of Successful and Failed Optimization Attempts')
    plt.xlabel('Attempts')
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
    plt.xlabel('Attempts')
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
def plot_network_final(network, previous_best_results, freqs_by_link=None):
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
    if previous_best_results is None:
        print("[plot] No feasible solution to draw – skipping plot_network_final")
        return
    
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

    _augment_legend_with_frequencies(ax, freqs_by_link)
    plt.tight_layout()
    plt.savefig('outputs/network_plot.png', dpi=300)
    plt.close()

#Bar plot to show channel allocation to each link by source.
def source_allocation(previous_best_results, sources):
    """
    Stacked bars per source:
      x-axis: sources
      y-axis: # of channels
      stacks: channels allocated per link (+ 'Unused' remainder)
    Saves: outputs/source_allocation.png
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if not previous_best_results:
        print("[plot] No feasible solution to draw – skipping source_allocation")
        return

    # Collect all link labels across sources
    all_links = set()
    for entry in previous_best_results:
        for lk in entry.get('links', []):
            u1, u2 = lk
            all_links.add(f"Link {u1}{u2}")
    link_labels = sorted(all_links)
    link_labels.append("Unused")  # always last

    # Build matrix: rows = link labels (incl. Unused), cols = sources
    src_names = [e['source'] for e in previous_best_results]
    nS, nL = len(src_names), len(link_labels)
    M = np.zeros((nL, nS), dtype=float)

    for s_idx, entry in enumerate(previous_best_results):
        sname = entry['source']
        total_k = len(sources[sname]['available_channels'])
        alloc = entry.get('channel_allocation', [])
        links = entry.get('links', [])
        used = 0
        for lk, k in zip(links, alloc):
            lbl = f"Link {lk[0]}{lk[1]}"
            try:
                i = link_labels.index(lbl)
                k_ch = _to_int_channels(k)  # <-- use the robust converter
                M[i, s_idx] += k_ch
                used += k_ch
            except ValueError:
                pass
        # Unused remainder (clamped to >= 0)
        rem = max(total_k - used, 0)
        M[link_labels.index("Unused"), s_idx] = rem


    # Plot
    colors = gem_colors(nL)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(nS)
    bottoms = np.zeros(nS)
    for i in range(nL):
        ax.bar(x, M[i], bottom=bottoms, color=colors[i],
               edgecolor='black', linewidth=1.2, label=link_labels[i])
        bottoms += M[i]

    ax.set_xticks(x)
    ax.set_xticklabels(src_names, rotation=15, ha='right')
    ax.set_xlabel('Source')
    ax.set_ylabel('No. of Channels')

    # Put 'Unused' last in legend (already last in link_labels)
    handles, labels = ax.get_legend_handles_labels()
    if 'Unused' in labels:
        idx = labels.index('Unused')
        order = [k for k in range(len(labels)) if k != idx] + [idx]
        handles = [handles[k] for k in order]
        labels  = [labels[k]  for k in order]

    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.savefig('outputs/source_allocation.png', dpi=300)
    plt.close()

