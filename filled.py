import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd

def fidelity_equation(x, y1, y2):
    return 0.25 * (1 + (3*x / (x**2 + x*(2*y1 + 2*y2 + 1) + 4*y1*y2)))

def rate_equation(x, y1, y2):
    return x**2 + x*(2*y1 + 2*y2 + 1) + 4*y1*y2

#Input parameters
tau = 1E-9 #Coincidence window in seconds
#Efficiencies
eta1 = 0.012 
eta2 = 0.00021
#Dark count rates
d1 = 100 
d2 = 3500
l = 12 #Number of links
channel_numbers = [12, 72, 147, 225,303, 381, 456, 532, 612, 685, 764]#, 100, 500, 1000]
mu_total = 1/tau #total link flux
#Noise parameters
y1_array = [0, 0.0034, 0, 0.0299, 0.0385, 0.0625, 0.0733, 0, 0.1106, 0.125, 0.1489, 0]
y2_array = [0, 0.0006, 0.0357, 0.0051, 0.0066, 0.0107, 0.0126, 0.1818, 0.019, 0.0214, 0.0256, 0.2979]
fidelity_limit = 0.7 #Changeable fidelity minimum line value

#link_flux = 5*(np.random.rand(6)+0.1) #Create random link flux limits
# print(link_flux)
# minimum_link_flux = min(link_flux)
k = np.arange(l,1000) #Number of available channels
# flux = minimum_link_flux
# iterative_channel_allocation = np.ones(len(link_flux)) #This is here so we can add to it after more iterations
# new_flux = flux

#Find the maximum flux
link_flux = []
for link_number in range(l): #Iterates for each set of links
    y1 = y1_array[link_number]
    y2 = y2_array[link_number]

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
plt.savefig('outputs/iterate_flux_remainder.png')

plt.figure(dpi=300)
plt.plot(channels_used_array, maximum_flux_array)
plt.xlabel('Number of Channels')
plt.ylabel('Used Flux Per Channel')
plt.title('Used Flux Per Number of Channels')
#plt.show()
plt.savefig('outputs/iterate_flux.png')


plt.figure(dpi=300)
plt.title('Log Rate Utility')
plt.xlabel('Number of Channels')
plt.ylabel('Sum of Log of Rate Utility')

colors = ['blue','green','red','cyan','magenta','yellow','black','orange','purple','pink','lime','brown','teal']
labels = ['Link AB','Link CD','Link EF','Link GH','Link IJ','Link KL','Link MN','Link OP','Link QR','Link ST','Link UV','Link WX','Link YZ']

plt.plot(channels_used_array, rate)
plt.savefig('outputs/iterate_rate_utility.png')
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
plt.savefig('outputs/iterate_three_plot.png')
#plt.show()
