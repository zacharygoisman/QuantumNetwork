#All the plotting code
import matplotlib.pyplot as plt
import numpy as np
import allocate_channels
import networkx as nx

#This is a plot that compares the utility between the various reruns of the optimizer
def utility_interference(all_utilities, all_utilities_no_interference):
    """
    Inputs:
    all_utilities is a list of all utilities determined through the optimizer
    all_utilities_no_interference is a list of all utilities with the failed ones being replaced with negative infinity
    """
    x_range = np.arange(1,len(all_utilities)+1) #Optimizer attempt number
    plt.scatter(x_range, all_utilities, label = 'Points with Interference or Less Utility')
    plt.scatter(x_range, all_utilities_no_interference, label = 'Best Utility Values')
    plt.legend(loc = 'best')
    plt.title('Comparing Utilities of Successful and Failed Optimization Attempts')
    plt.xlabel('Successful Rerun Attempts')
    plt.ylabel('Utility')
    #plt.show()
    plt.savefig('outputs/utility_comparison.png')

#Network plot showing nodes, edges, and link paths TODO: Add link paths
def plot_network_final(network):
    """
    Inputs:
    network is the NetworkX topology object containing all network information such as nodes and edges
    """
    pos = nx.spring_layout(network)
    plt.figure(figsize=(12, 10))

    # Draw nodes with colors based on node type.
    node_colors = []
    for node, data in network.nodes(data=True):
        if data['node_type'] == 'source':
            node_colors.append('lightblue')
        elif data['node_type'] == 'user':
            node_colors.append('lightcoral')
        else:
            node_colors.append('gray')
    nx.draw_networkx_nodes(network, pos, node_color=node_colors, node_size=500)

    # Draw all network edges in lightgray.
    nx.draw_networkx_edges(network, pos, edge_color='lightgray', width=1)

    # Draw the links along their computed paths with their unique colors.
    for desired_link in network.graph['desired_links']:
        if 'channels_assigned' in desired_link:
            link_color = desired_link['color']
            paths = desired_link['paths']
            # Combine both paths (user1->source and user2->source)
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

    # Draw edge labels showing only loss values.
    edge_labels = {(u, v): f"{data['loss']:.2f}" for u, v, data in network.edges(data=True)}
    nx.draw_networkx_edge_labels(network, pos, edge_labels=edge_labels, font_size=8)

    # Draw node labels.
    nx.draw_networkx_labels(network, pos, font_size=10, font_weight='bold')

    # # Build the legend with link information including channel assignments.
    # legend_elements = []
    # for desired_link in network.graph['desired_links']:
    #     if 'channels_assigned' in desired_link and 'frequencies' in desired_link:
    #         link = desired_link['link']
    #         # Determine the source from one of the paths (user1's path end)
    #         source = desired_link['paths']['user1'][-1]
    #         total_loss = desired_link['total_loss']
    #         link_color = desired_link['color']
    #         freqs = desired_link['frequencies']  # Expects a dict with 'user1' and 'user2'
    #         # Format the label to include the channels in order (user1 then user2)
    #         label = f"Link {link[0]}-{source}-{link[1]}, Channels: {freqs['user1']}/{freqs['user2']}, Loss={total_loss:.2f}"
    #         legend_elements.append(Patch(facecolor=link_color, edgecolor='k', label=label))

    # Add the legend to the plot.
    #plt.legend(handles=legend_elements, loc='best', fontsize='small')
    plt.title("Network Topology")
    plt.axis('off')
    #plt.show()
    plt.savefig('outputs/network_plot.png')

#Bar plot to show channel allocation to each link by source. TODO: Include unused source bars and make unused label at bottom
def source_allocation(previous_best_results):
    """
    Inputs:
    previous_best_results contains source and channel allocation results from the interference function
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    source_names = [entry['source'] for entry in previous_best_results]
    bar_width = 0.5
    colors = {}

   # Gather all unique links and assign colors
    all_links = {link for e in previous_best_results for link in e['links']}
    cmap = plt.get_cmap('tab20')
    colors = {link: cmap(i) for i, link in enumerate(sorted(all_links))}
    colors['Unused'] = 'lightgrey'

    seen_labels = set()

    for i, entry in enumerate(previous_best_results):
        bottom = 0
        total = entry['total_channels']

        # plot each link segment
        for j, link in enumerate(entry['links']):
            count = int(sum(entry['channel_allocation'][j])) if j < len(entry['channel_allocation']) else 0
            if count <= 0:
                continue
            # only label the first time we see this link
            if link not in seen_labels:
                label = str(link)
                seen_labels.add(link)
            else:
                label = "_nolegend_"
            ax.bar(i, count, bottom=bottom, width=bar_width,
                   color=colors[link], label=label)
            bottom += count

        # plot unused channels
        unused = total - bottom
        if unused > 0:
            if 'Unused' not in seen_labels:
                label = 'Unused'
                seen_labels.add('Unused')
            else:
                label = "_nolegend_"
            ax.bar(i, unused, bottom=bottom, width=bar_width,
                   color=colors['Unused'], label=label)

    # Formatting
    ax.set_xticks(range(len(source_names)))
    ax.set_xticklabels(source_names)
    ax.set_ylabel("Number of Channels")
    ax.set_title("Channel Allocation per Source")
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    #plt.show()
    plt.savefig('outputs/source_allocation.png')

#Fidelity and rate plot
def fidelity_rate_plot(results, tau, y1_array, y2_array, fidelity_limit):
    l = len(results[0][0]['links'])
    fig, axs = plt.subplots((l + 1) // 2, 2, figsize=(11, 8))
    line_handles = []  # List to store handles for creating a global legend
    line_labels = []  # List to store labels for creating a global legend

    mu_total = 1/tau
    fidelity_limit = 0.7 #TODO: Fix this to be an input
    mu_channel_array = []
    floored_ratio_array = []
    channel_numbers = []
    all_link_used_channels = []
    for result in results:
        mu_channel_array.append(result[0]['mu']/tau)
        floored_ratio_array.append(np.concatenate(result[0]['channel_allocation']))
        channel_numbers.append(result[0]['total_channels'])
        
    for link_number in range(l):
        y1 = y1_array[link_number]
        y2 = y2_array[link_number]
        mu = np.linspace(0, mu_total, 1000)
        markers = ['o', '^', 's', 'd', 'x']
        ax = axs[link_number % ((l + 1) // 2), link_number // ((l + 1) // 2)]
        fidelity, rate = allocate_channels.network_var(tau, mu, y1, y2)
        line1, = ax.plot(tau * mu, fidelity, label='Fidelity', c='tab:blue')
        line2, = ax.plot(tau * mu, rate, label='Rate/Max Rate', c='tab:orange')
        if link_number == 0:  # To prevent duplicates in the legend
            line_handles.extend([line1, line2])
            line_labels.extend(['Fidelity', 'Rate/Max Rate'])
        link_used_channels = []
        link_rates = []
        link_cost_functions = []
        for channel_index in range(len(channel_numbers)):
            fidelity, rate, flux, cost_function = allocate_channels.network_var_points(tau, mu_channel_array[channel_index] * floored_ratio_array[channel_index][link_number], y1, y2, fidelity_limit)
            link_used_channels.append(floored_ratio_array[channel_index][link_number])
            link_rates.append(rate)
            link_cost_functions.append(cost_function)
            scatter = ax.scatter(flux, fidelity, label=str(channel_numbers[channel_index]), marker="$"+str(channel_numbers[channel_index])+"$", c='tab:blue')
            ax.scatter(flux, rate, marker = "$"+str(channel_numbers[channel_index])+"$", c='tab:orange')
            ax.set_xlim([0,0.7])
            ax.set_ylim([0,1])
            if link_number == 0:  # Add only once to legend
                line_handles.append(scatter)
                line_labels.append(str(channel_numbers[channel_index]))
        #ax.axvline(max_flux_array[link_number], color='k', linestyle='--') #Add maximum possible flux line based on the fidelity limit
        ax.axhline(fidelity_limit, color='k', linestyle='--') #Add fidelity limit horizontal line
        all_link_used_channels.append(link_used_channels)
        # all_rates.append(link_rates)
        # all_link_cost_functions.append(link_cost_functions)

    # Add a global legend
    fig.legend(handles=line_handles, labels=line_labels, loc='upper right', bbox_to_anchor=(.99, .95))
    # Add global figure labels
    fig.text(0.5, 0.02, 'x (Dimensionless flux)', ha='center')
    fig.text(0.02, 0.5, 'y (Rate or Fidelity)', va='center', rotation='vertical')
    fig.text(0.5, .95, 'Fidelity and EBR vs Flux Plots', ha='center')
    # Adjust layout, save and close
    plt.tight_layout(rect=[0.03, 0.03, 0.85, 0.95])  # Adjust the tight_layout to accommodate the legend
    plt.savefig('outputs/link_fidelity.png')  # Save the figure to a file
    #plt.show()
    plt.close()

    # #Allocation bars
    # plt.figure()
    # plt.title('Channel Allocation')
    # plt.xlabel('Number of Channels')
    # plt.ylabel('Channels Used')
    # channel_str = []
    # for channel_number in channel_numbers:
    #     channel_str.append(str(channel_number))
    # colors = ['blue','green','red','cyan','magenta','yellow','black','orange','purple','pink','lime','brown','teal']
    # labels = ['Link AB','Link CD','Link EF','Link GH','Link IJ','Link KL','Link MN','Link OP','Link QR','Link ST','Link UV','Link WX','Link YZ']
    # all_link_used_channels = np.array(all_link_used_channels)
    # bottom_val = all_link_used_channels[0] - all_link_used_channels[0]
    # for i in range(len(all_link_used_channels)):
    #     plt.bar(channel_str, np.array(floored_ratio_array)[:,i], bottom = bottom_val, color = colors[i], label = labels[i])
    #     bottom_val+=np.array(floored_ratio_array)[:,i]
    # #plt.bar(channel_str, np.array(k) - np.sum(all_link_used_channels, axis=0), bottom = bottom_val, color = 'tab:blue', label = 'Null Link')
    # plt.legend(loc='best')
    # plt.savefig('outputs/channel_barplot.png')
    # #plt.show()
    # plt.close()

    # Allocation bars
    plt.figure()
    plt.title('Channel Allocation')
    plt.xlabel('Number of Channels')
    plt.ylabel('Channels Used')

    channel_str = [str(c) for c in channel_numbers]

    colors  = ['blue','green','red','cyan','magenta','yellow','black',
            'orange','purple','pink','lime','brown','teal']
    labels  = ['Link AB','Link CD','Link EF','Link GH','Link IJ',
            'Link KL','Link MN','Link OP','Link QR','Link ST',
            'Link UV','Link WX','Link YZ']

    floored_ratio_array = np.asarray(floored_ratio_array)

    # Start the stack at zero for every bar
    bottom_val = np.zeros_like(channel_numbers, dtype=float)

    # --- Plot each real link ----------------------------------------------------
    for i in range(floored_ratio_array.shape[1]):          # loop over links
        plt.bar(channel_str,
                floored_ratio_array[:, i],
                bottom=bottom_val,
                color=colors[i],
                label=labels[i])
        bottom_val += floored_ratio_array[:, i]            # grow the stack

    # --- Add the "Null Link" ----------------------------------------------------
    null_link = np.asarray(channel_numbers) - bottom_val   # residual capacity
    # If any channel is already over-allocated, clip negatives to zero
    null_link = np.clip(null_link, 0, None)

    plt.bar(channel_str,
            null_link,
            bottom=bottom_val,
            color='tab:blue',          # pick any unused colour
            label='Null Link')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('outputs/channel_barplot.png')
    plt.close()

    #Log of rate utility function
    plt.figure()
    plt.title('Log Rate Utility for Ratio Method')
    plt.xlabel('Number of Channels')
    plt.ylabel('Log Rate Utility')
    channel_str = []
    for channel_number in channel_numbers:
        channel_str.append(str(channel_number))
    colors = ['blue','green','red','cyan','magenta','yellow','black','orange','purple','pink','lime','brown','teal']
    labels = ['Link AB','Link CD','Link EF','Link GH','Link IJ','Link KL','Link MN','Link OP','Link QR','Link ST','Link UV','Link WX','Link YZ']
    all_link_used_channels = np.array(all_link_used_channels)
    bottom_val = all_link_used_channels[0] - all_link_used_channels[0]
    mu_channels = mu_channel_array
    rate = allocate_channels.rate_equation(np.array(floored_ratio_array)[:,0]*tau*mu_channels, y1_array[0], y2_array[0]) * 0 #Just to make an array of 0s
    for i in range(len(all_link_used_channels)):
        rate += np.log10( allocate_channels.rate_equation(np.array(floored_ratio_array)[:,i]*tau*mu_channels, y1_array[i], y2_array[i]))
    plt.bar(channel_str, rate) #, bottom = bottom_val, color = colors[i], label = labels[i])
    #bottom_val+=np.log10(rate_equation(np.array(floored_ratio_array)[:,i]*tau*mu_channels, y1_array[i], y2_array[i]))
    #plt.bar(channel_str, np.array(k) - np.sum(all_link_used_channels, axis=0), bottom = bottom_val, color = 'tab:blue', label = 'Null Link')
    plt.legend(loc='best')
    plt.savefig('outputs/rate_utility.png')
    #plt.show()
    plt.close()