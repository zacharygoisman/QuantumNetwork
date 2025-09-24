#All the plotting code
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import matplotlib.lines as mlines
import pathlib
import numpy as np
pathlib.Path("outputs").mkdir(exist_ok=True)
import matplotlib as mpl

mpl.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 14,
    #ticks inside & on all sides globally
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})


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
        #GEKKO variable?
        if hasattr(k, 'value'):
            v = k.value
            if isinstance(v, (list, tuple, np.ndarray)):
                if len(v) == 0:
                    return 0
                return int(round(float(v[0])))
            return int(round(float(v)))

        #Numpy array / list
        if isinstance(k, (list, tuple, np.ndarray)):
            arr = np.asarray(k, dtype=float)
            if arr.size == 0:
                return 0
            if arr.size == 1:
                return int(round(float(arr.item())))
            #If it's a vector (e.g., 0/1 flags), sum it
            return int(round(float(arr.sum())))

        #Numpy scalar or plain number
        return int(round(float(k)))
    except Exception:
        #Last-ditch attempt
        try:
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


    
#--- MATLAB "gem" palette (cycled) + typography defaults ---
GEM = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
       '#77AC30', '#4DBEEE', '#A2142F', '#003BFF',
       '#017501', '#FF0000', '#B526FF', '#FF00FF', '#000000']

def gem_colors(n, start=0):
    return [GEM[(start+i) % len(GEM)] for i in range(n)]

mpl.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 14
})

def plot_link_utility_bars(rows, start_color_index=0, outfile='outputs/link_utility_bars.svg'):
    """
    rows: list of dicts from summarize_best(); each row must have:
          'link', 'link_utility', 'ub_utility'
    Bar outline (black) = max per-link utility; fill = actual utility.
    """

    if not rows:
        return

    links = [r['link'] for r in rows]
    ub = np.array([float(r['ub_utility']) if r.get('ub_utility') is not None else np.nan for r in rows])
    ac = np.array([float(r['link_utility']) for r in rows])

    x = np.arange(len(links))
    w = 0.72
    colors = gem_colors(len(links), start=start_color_index)

    fig, ax = plt.subplots(figsize=(10, 6))
    #Max bars as hollow boxes
    ax.bar(x, ub, width=w, facecolor='white', edgecolor='black', linewidth=1.5, label='Max per-link utility')

    #Actual bars as filled overlays (all gem blue)
    gem_blue = '#0072BD'
    ax.bar(x, ac, width=w*0.82, color=gem_blue, edgecolor='black', linewidth=1.0, label='Actual utility')

    ax.set_xticks(x)
    ax.set_xticklabels(links, rotation=30, ha='right')
    ax.set_xlabel('Link')
    ax.set_ylabel('Utility (log₁₀ units)')
    ax.legend()

    #ticks: inside, all sides
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, length=6, width=1)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

def utility_comparison(all_utilities, dashed_limit, outfile='outputs/utility_comparison.svg'):

    if not all_utilities:
        print("[plot] all_utilities empty – skipping utility_comparison")
        return

    y = np.array(all_utilities, dtype=float)
    x = np.arange(1, len(y)+1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(x, y, marker='o', linewidth=1.5, s=3, label='Path Combination')

    ax.set_xlabel('Path Combination')
    ax.set_ylabel('Utility')

    if np.isfinite(dashed_limit):
        #tiny epsilon avoids fp “overrun”
        eps = 1e-9 * max(1.0, abs(dashed_limit))
        ax.axhline(dashed_limit + eps, linestyle='--', linewidth=1.75,
                   label='Max possible (network)', color='k')

    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close(fig)

#Network plot showing nodes, edges, and link paths
import matplotlib.patheffects as pe

def plot_network_final(network, previous_best_results, freqs_by_link=None):
    if previous_best_results is None:
        print("[plot] No feasible solution to draw – skipping plot_network_final")
        return

    # keep positions stable if available; otherwise use a seeded spring for reproducibility
    pos = network.graph.get('pos', nx.spring_layout(network, seed=0))
    fig, ax = plt.subplots(figsize=(12, 10))

    # Nodes
    color_map = {'source': 'lightblue', 'user': 'lightcoral'}
    node_colors = [color_map.get(data.get('node_type'), 'gray')
                   for n, data in network.nodes(data=True)]
    nx.draw_networkx_nodes(network, pos, node_color=node_colors, node_size=500, ax=ax)

    # All background edges (light)
    nx.draw_networkx_edges(network, pos, edge_color='lightgray', width=1, ax=ax)

    # Unique link colors (stable)
    all_links = [link for pr in previous_best_results for link in pr['links']]
    unique_links = sorted(set(all_links))
    cmap = plt.get_cmap('tab20')
    link_colors = {link: cmap(i % 20) for i, link in enumerate(unique_links)}

    def path_loss(path):
        return sum(network[u][v].get('loss', 1.0) for u, v in zip(path, path[1:]))

    # Draw selected paths and build legend labels
    for entry in previous_best_results:
        sname   = entry['source']
        links   = entry['links']
        prelogs = entry.get('prelog_rates', [None]*len(links))

        for j, (u1, u2) in enumerate(links):
            # If you have the chosen paths stored, prefer those; otherwise, fall back to dijkstra on 'loss'
            p1 = nx.dijkstra_path(network, u1, sname, weight="loss")
            p2 = nx.dijkstra_path(network, u2, sname, weight="loss")
            e1 = list(zip(p1, p1[1:]))
            e2 = list(zip(p2, p2[1:]))

            # Colored paths overlaid
            nx.draw_networkx_edges(
                network, pos, edgelist=(e1 + e2),
                edge_color=link_colors[(u1, u2)], width=3, alpha=0.6, ax=ax
            )

            total_loss = path_loss(p1) + path_loss(p2)
            link_util = float(np.log10(prelogs[j])) if prelogs[j] is not None else float('nan')

            ch_u1 = []
            if freqs_by_link and (u1, u2) in freqs_by_link:
                ch_u1 = list(freqs_by_link[(u1, u2)][0])

            lbl = f"{u1}–{sname}–{u2} | loss={total_loss:.4f} | U={link_util:.6f} | +{ch_u1}→{u1}"
            ax.plot([], [], color=link_colors[(u1, u2)], label=lbl)

    # === Label loss for ALL edges (used or not) ===
    edge_labels = {}
    if network.is_multigraph():
        # For MultiGraph/MultiDiGraph, include keys so labels attach correctly
        for u, v, k, data in network.edges(keys=True, data=True):
            if u == v:  # skip self-loops
                continue
            edge_labels[(u, v, k)] = f"{float(data.get('loss', 1.0)):.2f}"
    else:
        for u, v, data in network.edges(data=True):
            if u == v:
                continue
            edge_labels[(u, v)] = f"{float(data.get('loss', 1.0)):.2f}"

    texts = nx.draw_networkx_edge_labels(
        network, pos, edge_labels=edge_labels, ax=ax,
        font_size=9, rotate=True, label_pos=0.5
    )
    for t in texts.values():
        t.set_zorder(5)
        t.set_path_effects([pe.withStroke(linewidth=3, foreground='white')])

    nx.draw_networkx_labels(network, pos, font_size=10, font_weight='bold', ax=ax)
    ax.legend(loc='best', fontsize=10)
    ax.set_title("Network Topology")
    plt.tight_layout()
    plt.savefig('outputs/network_plot.svg', dpi=300)
    plt.close()



#Bar plot to show channel allocation to each link by source.
def source_allocation(previous_best_results, sources, freqs_by_link=None):
    """
    Show per-source frequency allocation by channel index (1..K).
    If freqs_by_link is provided (dict{(u1,u2): (to_u1, to_u2)}), we render
    exactly which channel indices are used; otherwise we fall back to counts.

    previous_best_results: best['results'] (list of per-source dicts)
    sources: dict of sources -> {'available_channels': [1..K]}
    freqs_by_link: optional dict mapping each link tuple to (to_u1_list, to_u2_list)
    """
    import numpy as np

    #legend reorder so "Null Link" is first
    def _legend_null_first(ax):
        handles, labels = ax.get_legend_handles_labels()
        if "Null Link" in labels:
            idx = labels.index("Null Link")
            order = [idx] + [i for i in range(len(labels)) if i != idx]
            handles = [handles[i] for i in order]
            labels  = [labels[i]  for i in order]
            ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 1))
        else:
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # If we don't have precise channel assignments yet, keep the old (count) behavior.
    if not freqs_by_link:
        # ---------- legacy count plot ----------
        src_names = [entry['source'] for entry in previous_best_results]
        # collect link labels + Null Link
        link_labels = sorted({f"Link {lk[0]}{lk[1]}"
                              for entry in previous_best_results
                              for lk in entry.get('links', [])})
        link_labels.append("Null Link")

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
                    k_ch = _to_int_channels(k)
                    M[i, s_idx] += k_ch
                    used += k_ch
                except ValueError:
                    pass
            rem = max(total_k - used, 0)
            M[link_labels.index("Null Link"), s_idx] = rem

        # colors: keep "Null Link" as MATLAB blue; others cycle
        colors = gem_colors(nL, start=1)
        if 'Null Link' in link_labels:
            colors[link_labels.index('Null Link')] = GEM[0]

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

        # Legend with "Null Link" first
        _legend_null_first(ax)

        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, length=6, width=1)
        plt.tight_layout()
        plt.savefig('outputs/source_allocation.svg', dpi=300)
        plt.close()
        return

    # ---------- new per-channel index plot ----------
    # Build a mapping from link -> source it’s assigned to (use previous_best_results)
    link_to_source = {}
    for entry in previous_best_results:
        src = entry['source']
        for lk in entry.get('links', []):
            link_to_source[tuple(lk)] = src

    src_names = [entry['source'] for entry in previous_best_results]
    unique_links = sorted({tuple(lk) for entry in previous_best_results for lk in entry.get('links', [])})
    link_labels = [f"Link {u}{v}" for (u, v) in unique_links]
    # Add Null Link category
    link_labels.append("Null Link")

    # Color map: keep "Null Link" as MATLAB blue; others cycle
    colors = {lab: col for lab, col in zip(link_labels, gem_colors(len(link_labels), start=1))}
    colors["Null Link"] = GEM[0]

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(src_names))

    # For legend: show each label only once
    legend_added = set()

    for s_idx, sname in enumerate(src_names):
        K = len(sources[sname]['available_channels'])
        # occupant per level (1..K) for this source
        level_owner = ["Null Link"] * K

        # Fill in occupied levels from channel assignments
        for (u1, u2), (to_u1, to_u2) in freqs_by_link.items():
            if link_to_source.get((u1, u2)) != sname:
                continue
            # whichever list is positive for this link, levels are abs(c)
            used_levels = sorted({abs(c) for c in list(to_u1) + list(to_u2)})
            for cidx in used_levels:
                if 1 <= cidx <= K:
                    level_owner[cidx - 1] = f"Link {u1}{u2}"

        # Draw one bar per channel index (height=1) so vertical position = index
        for level, owner in enumerate(level_owner, start=1):
            label = (owner if owner not in legend_added else "_nolegend_")
            ax.bar(x[s_idx], 1.0, bottom=level - 1,
                   color=colors[owner], edgecolor='black', linewidth=1.2, label=label)
            legend_added.add(owner)

        # If this source is entirely null, annotate so it "shows as such"
        if set(level_owner) == {"Null Link"} and K > 0:
            ax.text(x[s_idx], K + 0.1, "(all null)", ha='center', va='bottom',
                    fontsize=9, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(src_names, rotation=15, ha='right')
    ax.set_xlabel('Source')
    ax.set_ylabel('Channel Frequency')

    # Build legend and reorder to start with "Null Link"
    # (In case no "Null Link" bars were drawn yet, ensure it appears:)
    if "Null Link" not in legend_added:
        ax.plot([], [], color=colors["Null Link"], label="Null Link")
    _legend_null_first(ax)

    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, length=6, width=1)
    plt.tight_layout()
    plt.savefig('outputs/source_allocation.svg', dpi=300)
    plt.close()

def fidelity_rate():
    fig, ax = plt.subplots(figsize=(9, 6))
    y_1 = 0.05
    y_2 = 0.1
    flux = np.linspace(0, 1, 1000)  # finer grid helps intersection accuracy
    rate = flux**2 + (2*y_1 + 2*y_2 + 1)*flux + 4*y_1*y_2
    fidelity = 0.25*(1 + 3*flux/rate)

    fidelity_min = 0.80

    # Plot
    ax.plot(flux, fidelity, label='Fidelity')
    ax.plot(flux, rate, label='Rate')

    ax.axhline(fidelity_min, linestyle='--', linewidth=1.5, label='Fidelity Limit = '+str(fidelity_min), c = 'k')

    diff = fidelity - fidelity_min
    cross_idx = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) <= 0)[0]  # sign change => crossing

    # Linearly interpolate each crossing to get a better estimate
    x_crosses, r_crosses = [], []
    for i in cross_idx:
        x0, x1 = flux[i], flux[i+1]
        y0, y1 = fidelity[i], fidelity[i+1]
        if y1 == y0:
            continue  # avoid division by zero (flat segment exactly at level)
        x_cross = x0 + (fidelity_min - y0) * (x1 - x0) / (y1 - y0)
        x_crosses.append(x_cross)
        r_crosses.append(x_cross**2 + (2*y_1 + 2*y_2 + 1)*x_cross + 4*y_1*y_2)

    if len(x_crosses) > 0:
        x_best = x_crosses[int(np.argmax(r_crosses))]
        # Vertical line (full axis height so it's easy to see)
        ax.axvline(x_best, linestyle='--', linewidth=1.5, c = 'k')
        # Mark the intersection point on the fidelity curve
        ax.plot([x_best], [fidelity_min], marker='o', ms=7)
        ax.annotate(
            f"x={x_best:.4f}\nrate={max(r_crosses):.4f}",
            xy=(x_best, fidelity_min),
            xytext=(10, 10),
            textcoords='offset points',
            ha='left', va='bottom'
        )
    else:
        print("No intersections found for this fidelity_min. Try a different value.")

    ax.set_xlabel('Flux over Coincidence Window')
    ax.set_ylabel('Magnitude')
    ax.set_title('Graphical View of Fidelity and Rate Relation')
    ax.set_xlim([float(np.min(flux)), float(np.max(flux))])
    ax.set_ylim([0, 1])
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()
    plt.close()
    #plt.savefig('outputs/fidelity_rate.svg', dpi=300)



def main():
    fidelity_rate()

if __name__ == "__main__":
    main()