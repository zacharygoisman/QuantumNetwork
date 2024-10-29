import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def fidelity_equation(x, y1, y2):
    return 0.25 * (1 + (3*x / (x**2 + x*(2*y1 + 2*y2 + 1) + 4*y1*y2)))

def rate_equation(x, y1, y2):
    return x**2 + x*(2*y1 + 2*y2 + 1) + 4*y1*y2

# Input parameters
tau = 1E-9  # Coincidence window in seconds
# Efficiencies
eta1 = 0.012 
eta2 = 0.00021
# Dark count rates
d1 = 100 
d2 = 3500
l = 12  # Number of links
fidelity_limit = 0.7  # Changeable fidelity minimum line value

# Noise parameters
y1_array = [0, 0.0034, 0, 0.0299, 0.0385, 0.0625, 0.0733, 0, 0.1106, 0.125, 0.1489, 0]
y2_array = [0, 0.0006, 0.0357, 0.0051, 0.0066, 0.0107, 0.0126, 0.1818, 0.019, 0.0214, 0.0256, 0.2979]

# Calculate maximum flux per channel for each link
link_flux = []
for link_number in range(l):
    y1 = y1_array[link_number]
    y2 = y2_array[link_number]

    def max_flux_equation(x):
        return 0.25 * (1 + 3 * x / (x**2 + x*(2*y1 + 2*y2 + 1) + 4*y1*y2)) - fidelity_limit
    
    initial_guess = 1.0
    max_flux = fsolve(max_flux_equation, initial_guess)[0]
    link_flux.append(max_flux)

link_flux = np.array(link_flux)

N_max = 100  # Maximum number of channels
k_values = np.arange(N_max, l-1, -1)  # Total number of channels from N_max down to l

channel_allocations_list = []
allocated_flux_list = []
rate_list = []

# Initialize allocations for N_max channels
allocations = np.full(l, N_max // l, dtype=int)
remaining_channels = N_max - allocations.sum()
# Distribute remaining channels
for i in range(remaining_channels):
    allocations[i % l] +=1

per_channel_flux = link_flux.copy()
total_flux = allocations * per_channel_flux
rates = np.array([np.log10(rate_equation(total_flux[i], y1_array[i], y2_array[i])) for i in range(l)])
total_rate = np.sum(rates)

for total_channels in k_values:
    # Adjust allocations to match total_channels
    while allocations.sum() > total_channels:
        # Calculate marginal decreases
        marginal_decreases = []
        for i in range(l):
            if allocations[i] > 1:
                new_alloc = allocations[i] -1
                new_total_flux = new_alloc * per_channel_flux[i]
                new_rate = np.log10(rate_equation(new_total_flux, y1_array[i], y2_array[i]))
                marginal_decrease = rates[i] - new_rate
                marginal_decreases.append((marginal_decrease, i))
        if not marginal_decreases:
            break  # Cannot reduce allocations further
        # Find link with smallest marginal decrease
        marginal_decreases.sort()
        _, i_min = marginal_decreases[0]
        # Update allocations and rates
        allocations[i_min] -= 1
        total_flux[i_min] = allocations[i_min] * per_channel_flux[i_min]
        rates[i_min] = np.log10(rate_equation(total_flux[i_min], y1_array[i_min], y2_array[i_min]))
        total_rate = np.sum(rates)
    # Save results
    channel_allocations_list.append(allocations.copy())
    allocated_flux_list.append(total_flux.copy())
    rate_list.append(total_rate)

# Reverse the lists to match the increasing order of total_channels
channel_allocations_list = channel_allocations_list[::-1]
allocated_flux_list = allocated_flux_list[::-1]
rate_list = rate_list[::-1]
k_values = k_values[::-1]

# Now plotting and saving results

# Convert lists to arrays
channel_allocations_array = np.array(channel_allocations_list)
allocated_flux_array = np.array(allocated_flux_list)
rate_array = np.array(rate_list)

plt.figure(dpi=300)
plt.plot(k_values, rate_array)
plt.xlabel('Number of Channels')
plt.ylabel('Total Rate')
plt.title('Total Rate vs. Number of Channels')
plt.savefig('outputs/total_rate.png')
plt.close()

plt.figure(dpi=300)
plt.plot(k_values, np.sum(allocated_flux_array, axis=1))
plt.xlabel('Number of Channels')
plt.ylabel('Total Allocated Flux')
plt.title('Total Allocated Flux vs. Number of Channels')
plt.savefig('outputs/allocated_flux.png')
plt.close()

plt.figure(figsize=(12, 8), dpi=300)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'pink', 'lime', 'brown']
labels = ['Link AB','Link CD','Link EF','Link GH','Link IJ','Link KL','Link MN','Link OP','Link QR','Link ST','Link UV','Link WX']

for i in range(l):
    plt.plot(k_values, channel_allocations_array[:, i], label=labels[i], color=colors[i])
plt.title('Channel Allocations vs. Number of Channels')
plt.xlabel('Number of Channels')
plt.ylabel('Channel Allocations')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/channel_allocations.png')
plt.close()
