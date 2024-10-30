import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import fsolve
import pandas as pd

#All needed inputs
num_users = 20
num_sources = 10
num_connections = 100
user_pairs = [('U1', 'U2'), ('U3', 'U4'), ('U2', 'U5'), ('U20', 'U12'), ('U1', 'U14'), ('U10', 'U15'), ('U3', 'U7'), ('U8', 'U9'), ('U13', 'U16'), ('U11', 'U18'), ('U6', 'U18'), ('U17', 'U20')]
colors = ['red', 'green', 'blue', 'orange', 'purple', 'lime','pink','cyan','yellow','lightblue','maroon','plum']

tau = 1E-9 #Coincidence window in seconds
d1 = 100 
d2 = 3500
fidelity_limits = 0.7 * (1 + (np.random.randn(12)*.000002)) #Changeable fidelity array
available_channels = 1000 #How many channels can we afford to allocate



#Function to create a random network
def create_random_network(num_users, num_sources, num_connections):
    G = nx.Graph()
    
    #Add user nodes
    for i in range(num_users):
        G.add_node(f'U{i+1}', type='user')
    
    #Add source nodes
    for i in range(num_sources):
        G.add_node(f'S{i+1}', type='source')
    
    #Add random connections with random loss values (weights)
    for _ in range(num_connections):
        u = random.choice([f'U{i+1}' for i in range(num_users)] + [f'S{i+1}' for i in range(num_sources)])
        v = random.choice([f'U{i+1}' for i in range(num_users)] + [f'S{i+1}' for i in range(num_sources)])
        
        if u != v and not G.has_edge(u, v):  #Ensure no self-loops or duplicate edges
            loss = random.uniform(7, 30)  #Loss values
            G.add_edge(u, v, weight=loss)
    
    return G

#Function to find the best source that minimizes the total loss between two users
def find_best_source_paths(G, u1, u2):
    min_total_loss = float('inf')
    best_paths = (None, None)
    best_source = None
    for source in [n for n, d in G.nodes(data=True) if d['type'] == 'source']:
        try:
            path1 = nx.shortest_path(G, source=u1, target=source, weight='weight', method='dijkstra')
            loss1 = calculate_path_loss(G, path1)
        except nx.NetworkXNoPath:
            loss1 = float('inf')
            path1 = None
        try:
            path2 = nx.shortest_path(G, source=u2, target=source, weight='weight', method='dijkstra')
            loss2 = calculate_path_loss(G, path2)
        except nx.NetworkXNoPath:
            loss2 = float('inf')
            path2 = None
        total_loss = loss1 + loss2
        if total_loss < min_total_loss:
            min_total_loss = total_loss
            best_paths = (path1, path2)
            best_source = source
    return best_paths, min_total_loss, best_source, loss1, loss2

#Function to find all paths between user pairs via sources
def find_all_paths(G, user_pairs):
    paths = []
    losses = []
    sources = []
    first_losses = []
    second_losses = []
    for u1, u2 in user_pairs:
        best_paths, min_total_loss, best_source, loss1, loss2 = find_best_source_paths(G, u1, u2)
        if min_total_loss == float('inf'):
            print(f"No path found between {u1} and {u2} via any source.")
            paths.append((None, None))
            losses.append(float('inf'))
            sources.append(None)
        else:
            print(f"Best path between {u1} and {u2} via {best_source} with total loss {min_total_loss:.2f}")
            paths.append(best_paths)
            losses.append(min_total_loss)
            sources.append(best_source)
            first_losses.append(loss1)
            second_losses.append(loss2)
    return paths, losses, sources, first_losses, second_losses

#Function to calculate path loss
def calculate_path_loss(G, path):
    if path is None:
        return float('inf')  #Assign infinity if no path exists
    
    loss = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        loss += G[u][v]['weight']
    return loss




#Seed for reproducibility
random.seed(42)

#Generate a random network
network = create_random_network(num_users, num_sources, num_connections)

#Define user pairs to connect

#Find all paths
all_paths, path_losses, sources, first_losses, second_losses = find_all_paths(network, user_pairs)

#Calculate total loss
total_loss = sum(loss for loss in path_losses if loss != float('inf'))

#Display path losses
for idx, ((u1, u2), loss, source) in enumerate(zip(user_pairs, path_losses, sources)):
    if loss == float('inf'):
        print(f"Path {idx+1} between {u1} and {u2}: No Path Found")
    else:
        print(f"Path {idx+1} between {u1} and {u2} via {source}: Total Loss = {loss:.2f}")
print(f"Total Loss across all paths: {total_loss:.2f}")

#Visualize the network and paths
pos = nx.spring_layout(network, seed=42)  #Fixed seed for consistency

#Create a matplotlib figure
plt.figure(figsize=(14, 12))

#Draw all nodes
nx.draw_networkx_nodes(network, pos, node_color='lightblue', node_size=1000)

#Draw all edges
nx.draw_networkx_edges(network, pos, edge_color='gray', alpha=0.6)

#Retrieve and format edge labels (loss values)
edge_labels = nx.get_edge_attributes(network, 'weight')
formatted_edge_labels = {edge: f"{weight:.2f}" for edge, weight in edge_labels.items()}

#Draw edge labels on top
nx.draw_networkx_edge_labels(
    network, pos, 
    edge_labels=formatted_edge_labels, 
    font_color='black', 
    font_size=8,
    label_pos=0.5, 
    bbox=dict(facecolor='white', edgecolor='none', pad=1.0, alpha=0.7)
)

#Draw labels for nodes
nx.draw_networkx_labels(network, pos, font_size=12, font_weight='bold')

#Define colors for paths

#Highlight the paths
for idx, ((path1, path2), source) in enumerate(zip(all_paths, sources)):
    if path1 is None or path2 is None:
        continue  #Skip if no path found
    #Get the edges for path1 and path2
    path_edges1 = [(path1[i], path1[i+1]) for i in range(len(path1)-1)]
    path_edges2 = [(path2[i], path2[i+1]) for i in range(len(path2)-1)]
    #Combine edges
    path_edges = path_edges1 + path_edges2
    nx.draw_networkx_edges(
        network, pos, 
        edgelist=path_edges, 
        width=3, 
        edge_color=colors[idx % len(colors)], 
        alpha=0.5, 
        label=f'Path {idx+1}: {user_pairs[idx][0]}-{user_pairs[idx][1]} via {source} (Loss: {path_losses[idx]:.2f})'
    )

plt.title("Network Paths with Loss Values", fontsize=16)
plt.legend(title="Paths", fontsize=10, title_fontsize=12)
plt.axis('off')  #Hide axis
plt.tight_layout()
#plt.show()
plt.savefig('outputs/combined_random_network.png')





def fidelity_equation(x, y1, y2):
    return 0.25 * (1 + (3*x / (x**2 + x*(2*y1 + 2*y2 + 1) + 4*y1*y2)))

def rate_equation(x, y1, y2):
    return x**2 + x*(2*y1 + 2*y2 + 1) + 4*y1*y2

#Input parameters

#Efficiencies
#eta1 = 0.012 
#eta2 = 0.00021
#Dark count rates

l = len(user_pairs) #Number of links
#channel_numbers = [12, 72, 147, 225,303, 381, 456, 532, 612, 685, 764]#, 100, 500, 1000]
mu_total = 1/tau #total link flux
#Noise parameters
y1_array = tau * d1 / 10**(-np.array(first_losses)/10)
y2_array = tau * d2 / 10**(-np.array(second_losses)/10)

#y1_array = [0, 0.0034, 0, 0.0299, 0.0385, 0.0625, 0.0733, 0, 0.1106, 0.125, 0.1489, 0]
#y2_array = [0, 0.0006, 0.0357, 0.0051, 0.0066, 0.0107, 0.0126, 0.1818, 0.019, 0.0214, 0.0256, 0.2979]

#link_flux = 5*(np.random.rand(6)+0.1) #Create random link flux limits
#print(link_flux)
#minimum_link_flux = min(link_flux)

k = np.arange(l,available_channels) #Number of available channels
#flux = minimum_link_flux
#iterative_channel_allocation = np.ones(len(link_flux)) #This is here so we can add to it after more iterations
#new_flux = flux


#Find the maximum flux
link_flux = []
for link_number in range(l): #Iterates for each set of links
    y1 = y1_array[link_number]
    y2 = y2_array[link_number]
    fidelity_limit = fidelity_limits[link_number]

    def max_flux_equation(x):
        #Fit equation for getting the maximum flux of a channel. I put this function inside the main code in case I need to use the local fidelity_limit variable in the optimizer
        return 0.25 * (1 + 3 * x / (x**2 + (2 * y1 + 2 * y2 + 1) * x + 4 * y1 * y2)) - fidelity_limit #Fidelity limit here to get the highest possible flux x value possible with the fidelity limit
    
    initial_guess = 1.0
    # Solve the equation using fsolve
    max_flux = fsolve(max_flux_equation, initial_guess)[0] #Finds the max flux of the channels
    link_flux.append(max_flux) 
minimum_link_flux = min(link_flux)
flux = minimum_link_flux
new_flux = flux



# Initialize variables to store results
channels_used_list = []
maximum_flux_list = []
allocated_flux_list = []
remainders_list = []
channel_allocations_list = []
rate = []
# Initialize channel allocations with one channel per link
channel_allocations = np.ones(len(link_flux), dtype=int)
# Compute initial maximum flux
maximum_flux = np.min(link_flux / channel_allocations)
# Compute initial allocated flux and remainders
allocated_flux = channel_allocations * maximum_flux
remainders = link_flux - allocated_flux
maximum_rate = 0
for j in range(len(channel_allocations)):
    maximum_rate += np.log10(rate_equation(channel_allocations[j]*maximum_flux, y1_array[j], y2_array[j]))
# Save initial allocations
channels_used_list.append(np.sum(channel_allocations))
maximum_flux_list.append(maximum_flux)
allocated_flux_list.append(allocated_flux.copy())
remainders_list.append(remainders.copy())
channel_allocations_list.append(channel_allocations.copy())
rate.append(maximum_rate)

for channel_number in k:
    min_total_remainder = np.inf
    max_total_rate = -np.inf
    best_allocation = channel_allocations.copy()
    best_maximum_flux = maximum_flux

    # Try adding a channel to each link and choose the best one
    for i in range(len(link_flux)):
        trial_allocations = channel_allocations.copy()
        trial_allocations[i] += 1

        # Compute trial maximum flux
        trial_maximum_flux = np.min(link_flux / trial_allocations)

        # Compute trial allocated flux and remainders
        trial_allocated_flux = trial_allocations * trial_maximum_flux
        trial_remainders = link_flux - trial_allocated_flux
        trial_total_remainder = np.sum(trial_remainders)

        # Rate calculation
        trial_total_rate = 0
        for j in range(len(trial_allocations)):
            trial_total_rate += np.log10(rate_equation(trial_allocations[j]*trial_maximum_flux, y1_array[j], y2_array[j]))

        # Check if this trial is better
        if trial_total_rate > max_total_rate:
            max_total_rate = trial_total_rate
            min_total_remainder = trial_total_remainder
            best_allocation = trial_allocations.copy()
            best_maximum_flux = trial_maximum_flux

    # Update allocations with the best trial
    channel_allocations = best_allocation.copy()
    maximum_flux = best_maximum_flux
    allocated_flux = channel_allocations * maximum_flux
    remainders = link_flux - allocated_flux
    maximum_rate = max_total_rate

    # Save the allocations
    if np.sum(maximum_rate) > np.sum(rate[-1]):
        channels_used_list.append(np.sum(channel_allocations))
        maximum_flux_list.append(maximum_flux)
        allocated_flux_list.append(allocated_flux.copy())
        remainders_list.append(remainders.copy())
        channel_allocations_list.append(channel_allocations.copy())
        rate.append(maximum_rate)
    else:
        channels_used_list.append(np.sum(channel_allocations))
        maximum_flux_list.append(maximum_flux_list[-1])
        allocated_flux_list.append(allocated_flux_list[-1])
        remainders_list.append(remainders_list[-1])
        channel_allocations_list.append(channel_allocations_list[-1])
        rate.append(rate[-1])

# Convert lists to arrays for plotting
channels_used_array = np.array(channels_used_list)
maximum_flux_array = np.array(maximum_flux_list)
allocated_flux_array = np.array(allocated_flux_list)
remainders_array = np.array(remainders_list)
channel_allocations_array = np.array(channel_allocations_list)
rate = np.array(rate)


plt.figure(dpi=300)
plt.plot(channels_used_array, np.sum(remainders_array, axis = 1))
plt.xlabel('Number of Channels')
plt.ylabel('Leftover Flux Remainder')
plt.title('Remainder Per Number of Channels')
#plt.show()
plt.savefig('outputs/combined_flux_remainder.png')

pure_rates = [] #Loop to get pure rates for utility function comparisons
for i in range(len(channel_allocations_array)):
    pure_rate = 0
    for j in range(len(y1_array)):
        pure_rate += rate_equation(channel_allocations_array[i][j]*maximum_flux_array[i], y1_array[j], y2_array[j])
    pure_rates.append(pure_rate)

plt.figure(dpi=300)
plt.plot(channels_used_array, np.sum(remainders_array, axis = 1)/np.sum(link_flux),label='Remainder percentage per channel iteration')
plt.plot(channels_used_array, (1/l)*np.sum(remainders_array/link_flux, axis = 1),label='Average sum of remainder percentages per iteration')
plt.xlabel('Number of Channels')
plt.ylabel('Leftover Flux Remainder')
plt.title('Remainder Per Number of Channels')
plt.yscale('log')
plt.legend(loc='best')
#plt.show()
plt.savefig('outputs/combined_flux_remainder_comp.png')

plt.figure(dpi=300)
plt.plot(channels_used_array, maximum_flux_array)
plt.xlabel('Number of Channels')
plt.ylabel('Used Flux Per Channel')
plt.title('Used Flux Per Number of Channels')
#plt.show()
plt.savefig('outputs/combined_flux.png')


plt.figure(dpi=300)
plt.title('Rate Utility')
plt.xlabel('Number of Channels')
plt.ylabel('Rate Utility')

colors = ['blue','green','red','cyan','magenta','yellow','black','orange','purple','pink','lime','brown','teal']
labels = ['Link AB','Link CD','Link EF','Link GH','Link IJ','Link KL','Link MN','Link OP','Link QR','Link ST','Link UV','Link WX','Link YZ']
plt.plot(channels_used_array, rate, label = 'Sum of Log of Rate Utility')
plt.plot(channels_used_array, np.log10(pure_rates), label = 'Log of Sum of Rates')
plt.plot(channels_used_array, np.array(pure_rates)/l, label = 'Average of Sum of Rates')
plt.legend(loc = 'best')
plt.savefig('outputs/combined_rate_utility.png')
#plt.show()
plt.close()


# Plotting
plt.figure(figsize=(12, 8), dpi=300)

# Plot maximum flux vs. channels available
plt.subplot(3, 1, 1)
plt.plot(channels_used_array, maximum_flux_array)
plt.title('Maximum Flux vs. Channels Available')
plt.xlabel('Number of Channels Available')
plt.ylabel('Maximum Flux')

# Plot remainders vs. channels available
plt.subplot(3, 1, 2)
plt.plot(channels_used_array, np.sum(remainders_array, axis = 1))#, label=f'Link {i+1}')
plt.title('Remainders vs. Channels Available')
plt.xlabel('Number of Channels Available')
plt.ylabel('Remainders')
plt.yscale('log')
# plt.legend()

# Plot channel allocations vs. channels available
plt.subplot(3, 1, 3)
for i in range(len(link_flux)):
    plt.plot(channels_used_array, channel_allocations_array[:, i], label=labels[i], color = colors[i])
plt.title('Channel Allocations vs. Channels Available')
plt.xlabel('Number of Channels Available')
plt.ylabel('Channel Allocations')
plt.legend()

plt.tight_layout()
plt.savefig('outputs/combined_three_plot.png')
#plt.show()
