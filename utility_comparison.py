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
k = np.arange(l,200) #Number of available channels
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


#SUM LOG RATE
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


pure_rates = [] #Loop to get pure rates for utility function comparisons
sum_pure_rates = []
for i in range(len(channel_allocations_array)):
    pure_rate = 0
    sum_pure_rate = 0
    for j in range(len(y1_array)):
        pure_rate += np.log10(rate_equation(channel_allocations_array[i][j]*maximum_flux_array[i], y1_array[j], y2_array[j]))
        sum_pure_rate += (rate_equation(channel_allocations_array[i][j]*maximum_flux_array[i], y1_array[j], y2_array[j]))
    pure_rates.append(pure_rate)
    sum_pure_rates.append(sum_pure_rate)




#SUM RATE
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
rate2 = []
# Initialize channel allocations with one channel per link
channel_allocations = np.ones(len(link_flux), dtype=int)
# Compute initial maximum flux
maximum_flux = np.min(link_flux / channel_allocations)
# Compute initial allocated flux and remainders
allocated_flux = channel_allocations * maximum_flux
remainders = link_flux - allocated_flux
maximum_rate = 0
for j in range(len(channel_allocations)):
    maximum_rate += (rate_equation(channel_allocations[j]*maximum_flux, y1_array[j], y2_array[j]))
# Save initial allocations
channels_used_list.append(np.sum(channel_allocations))
maximum_flux_list.append(maximum_flux)
allocated_flux_list.append(allocated_flux.copy())
remainders_list.append(remainders.copy())
channel_allocations_list.append(channel_allocations.copy())
rate2.append(maximum_rate)

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
            trial_total_rate += (rate_equation(trial_allocations[j]*trial_maximum_flux, y1_array[j], y2_array[j]))

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
    if np.sum(maximum_rate) > np.sum(rate2[-1]):
        channels_used_list.append(np.sum(channel_allocations))
        maximum_flux_list.append(maximum_flux)
        allocated_flux_list.append(allocated_flux.copy())
        remainders_list.append(remainders.copy())
        channel_allocations_list.append(channel_allocations.copy())
        rate2.append(maximum_rate)
    else:
        channels_used_list.append(np.sum(channel_allocations))
        maximum_flux_list.append(maximum_flux_list[-1])
        allocated_flux_list.append(allocated_flux_list[-1])
        remainders_list.append(remainders_list[-1])
        channel_allocations_list.append(channel_allocations_list[-1])
        rate2.append(rate2[-1])

# Convert lists to arrays for plotting
channels_used_array = np.array(channels_used_list)
maximum_flux_array2 = np.array(maximum_flux_list)
allocated_flux_array = np.array(allocated_flux_list)
remainders_array2 = np.array(remainders_list)
channel_allocations_array2 = np.array(channel_allocations_list)
rate2 = np.array(rate2)



pure_rates2 = [] #Loop to get pure rates for utility function comparisons
sum_pure_rates2 = []
for i in range(len(channel_allocations_array2)):
    pure_rate = 0
    sum_pure_rate = 0
    for j in range(len(y1_array)):
        pure_rate += np.log10(rate_equation(channel_allocations_array2[i][j]*maximum_flux_array2[i], y1_array[j], y2_array[j]))
        sum_pure_rate += (rate_equation(channel_allocations_array2[i][j]*maximum_flux_array2[i], y1_array[j], y2_array[j]))
    pure_rates2.append(pure_rate)
    sum_pure_rates2.append(sum_pure_rate)


#AVERAGE REMAINDER
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
rate3 = []
# Initialize channel allocations with one channel per link
channel_allocations = np.ones(len(link_flux), dtype=int)
# Compute initial maximum flux
maximum_flux = np.min(link_flux / channel_allocations)
# Compute initial allocated flux and remainders
allocated_flux = channel_allocations * maximum_flux
remainders = link_flux - allocated_flux
maximum_rate = 0
for j in range(len(channel_allocations)):
    maximum_rate += (rate_equation(channel_allocations[j]*maximum_flux, y1_array[j], y2_array[j]))
# Save initial allocations
channels_used_list.append(np.sum(channel_allocations))
maximum_flux_list.append(maximum_flux)
allocated_flux_list.append(allocated_flux.copy())
remainders_list.append(remainders.copy())
channel_allocations_list.append(channel_allocations.copy())
rate3.append(maximum_rate)

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
            trial_total_rate += (rate_equation(trial_allocations[j]*trial_maximum_flux, y1_array[j], y2_array[j]))

        # Check if this trial is better
        if trial_total_remainder < min_total_remainder:
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
    if np.sum(remainders) < np.sum(remainders_list[-1]):
        channels_used_list.append(np.sum(channel_allocations))
        maximum_flux_list.append(maximum_flux)
        allocated_flux_list.append(allocated_flux.copy())
        remainders_list.append(remainders.copy())
        channel_allocations_list.append(channel_allocations.copy())
        rate3.append(maximum_rate)
    else:
        channels_used_list.append(np.sum(channel_allocations))
        maximum_flux_list.append(maximum_flux_list[-1])
        allocated_flux_list.append(allocated_flux_list[-1])
        remainders_list.append(remainders_list[-1])
        channel_allocations_list.append(channel_allocations_list[-1])
        rate3.append(rate3[-1])

# Convert lists to arrays for plotting
channels_used_array = np.array(channels_used_list)
maximum_flux_array3 = np.array(maximum_flux_list)
allocated_flux_array = np.array(allocated_flux_list)
remainders_array3 = np.array(remainders_list)
channel_allocations_array3 = np.array(channel_allocations_list)
rate3 = np.array(rate3)



pure_rates3 = [] #Loop to get pure rates for utility function comparisons
sum_pure_rates3 = []
for i in range(len(channel_allocations_array3)):
    pure_rate = 0
    sum_pure_rate = 0
    for j in range(len(y1_array)):
        pure_rate += np.log10(rate_equation(channel_allocations_array3[i][j]*maximum_flux_array3[i], y1_array[j], y2_array[j]))
        sum_pure_rate += (rate_equation(channel_allocations_array3[i][j]*maximum_flux_array3[i], y1_array[j], y2_array[j]))
    pure_rates3.append(pure_rate)
    sum_pure_rates3.append(sum_pure_rate)



#AVERAGE REMAINDER PERCENTAGE
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
remainder_percentage_list = []
channel_allocations_list = []
rate4 = []
# Initialize channel allocations with one channel per link
channel_allocations = np.ones(len(link_flux), dtype=int)
# Compute initial maximum flux
maximum_flux = np.min(link_flux / channel_allocations)
# Compute initial allocated flux and remainders
allocated_flux = channel_allocations * maximum_flux
remainders = link_flux - allocated_flux
remainder_percentage = (link_flux - allocated_flux)/link_flux
maximum_rate = 0
for j in range(len(channel_allocations)):
    maximum_rate += (rate_equation(channel_allocations[j]*maximum_flux, y1_array[j], y2_array[j]))
# Save initial allocations
channels_used_list.append(np.sum(channel_allocations))
maximum_flux_list.append(maximum_flux)
allocated_flux_list.append(allocated_flux.copy())
remainders_list.append(remainders.copy())
remainder_percentage_list.append(remainder_percentage.copy())
channel_allocations_list.append(channel_allocations.copy())
rate4.append(maximum_rate)

for channel_number in k:
    min_total_remainder = np.inf
    min_total_remainder_percentage = np.inf
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
        trial_total_remainder_percentage = np.sum(trial_remainders/link_flux)

        # Rate calculation
        trial_total_rate = 0
        for j in range(len(trial_allocations)):
            trial_total_rate += (rate_equation(trial_allocations[j]*trial_maximum_flux, y1_array[j], y2_array[j]))

        # Check if this trial is better
        if trial_total_remainder_percentage < min_total_remainder_percentage:
            max_total_rate = trial_total_rate
            min_total_remainder = trial_total_remainder
            min_total_remainder_percentage = trial_total_remainder_percentage
            best_allocation = trial_allocations.copy()
            best_maximum_flux = trial_maximum_flux

    # Update allocations with the best trial
    channel_allocations = best_allocation.copy()
    maximum_flux = best_maximum_flux
    allocated_flux = channel_allocations * maximum_flux
    remainders = link_flux - allocated_flux
    remainder_percentage = (link_flux - allocated_flux)/link_flux
    maximum_rate = max_total_rate

    # Save the allocations
    if np.sum(remainder_percentage) < np.sum(remainder_percentage_list[-1]):
        channels_used_list.append(np.sum(channel_allocations))
        maximum_flux_list.append(maximum_flux)
        allocated_flux_list.append(allocated_flux.copy())
        remainders_list.append(remainders.copy())
        remainder_percentage_list.append(remainder_percentage.copy())
        channel_allocations_list.append(channel_allocations.copy())
        rate4.append(maximum_rate)
    else:
        channels_used_list.append(np.sum(channel_allocations))
        maximum_flux_list.append(maximum_flux_list[-1])
        allocated_flux_list.append(allocated_flux_list[-1])
        remainders_list.append(remainders_list[-1])
        remainder_percentage_list.append(remainder_percentage_list[-1])
        channel_allocations_list.append(channel_allocations_list[-1])
        rate4.append(rate4[-1])

# Convert lists to arrays for plotting
channels_used_array = np.array(channels_used_list)
maximum_flux_array4 = np.array(maximum_flux_list)
allocated_flux_array = np.array(allocated_flux_list)
remainders_array4 = np.array(remainders_list)
channel_allocations_array4 = np.array(channel_allocations_list)
rate4 = np.array(rate4)



pure_rates4 = [] #Loop to get pure rates for utility function comparisons
sum_pure_rates4 = []
for i in range(len(channel_allocations_array4)):
    pure_rate = 0
    sum_pure_rate = 0
    for j in range(len(y1_array)):
        pure_rate += np.log10(rate_equation(channel_allocations_array4[i][j]*maximum_flux_array4[i], y1_array[j], y2_array[j]))
        sum_pure_rate += (rate_equation(channel_allocations_array4[i][j]*maximum_flux_array4[i], y1_array[j], y2_array[j]))
    pure_rates4.append(pure_rate)
    sum_pure_rates4.append(sum_pure_rate)





plt.figure(dpi=300)
plt.plot(channels_used_array, np.sum(remainders_array, axis = 1)/np.sum(link_flux),label='Sum Log Rate (Utility)', alpha = 0.8, color = 'orange', linewidth = 1)
plt.plot(channels_used_array, np.sum(remainders_array2, axis = 1)/np.sum(link_flux),label='Sum Rate', alpha = 0.8, color = 'red', linewidth = 1)
plt.plot(channels_used_array, np.sum(remainders_array3, axis = 1)/np.sum(link_flux),label='Average Remainder', alpha = 0.8, linestyle = 'dashed', color = 'green', linewidth = 2)
plt.plot(channels_used_array, np.sum(remainders_array4, axis = 1)/np.sum(link_flux),label='Average Remainder Percentage', alpha = 0.8, linestyle = 'dotted', color = 'blue', linewidth = 2)
#plt.plot(channels_used_array, (1/l)*np.sum(remainders_array/link_flux, axis = 1),label='Average sum of remainder percentages per iteration')
plt.xlabel('Number of Channels')
plt.ylabel('Leftover Flux Remainder')
plt.title('Average Remainder')
plt.yscale('log')
plt.legend(loc='best')
plt.tick_params(axis='both',which='major',labelsize='large')
plt.tick_params(direction='in',top=True,right=True,length=6)
plt.minorticks_on()
plt.tick_params(which='minor',direction='in',top=True,right=True,length=3)
#plt.show()
plt.savefig('outputs/comparison_remainder1.png')


plt.figure(dpi=300)
plt.plot(channels_used_array, (1/l)*np.sum(remainders_array/link_flux, axis = 1),label='Sum Log Rate (Utility)', alpha = 0.8, color = 'orange', linewidth = 1)
plt.plot(channels_used_array, (1/l)*np.sum(remainders_array2/link_flux, axis = 1),label='Sum Rate', alpha = 0.8, color = 'red', linewidth = 1)
plt.plot(channels_used_array, (1/l)*np.sum(remainders_array3/link_flux, axis = 1),label='Average Remainder', alpha = 0.8, linestyle = 'dashed', color = 'green', linewidth = 2)
plt.plot(channels_used_array, (1/l)*np.sum(remainders_array4/link_flux, axis = 1),label='Average Remainder Percentage', alpha = 0.8, linestyle = 'dotted', color = 'blue', linewidth = 2)
plt.xlabel('Number of Channels')
plt.ylabel('Leftover Flux Remainder')
plt.title('Average Remainder Percentage')
plt.yscale('log')
plt.legend(loc='best')
plt.tick_params(axis='both',which='major',labelsize='large')
plt.tick_params(direction='in',top=True,right=True,length=6)
plt.minorticks_on()
plt.tick_params(which='minor',direction='in',top=True,right=True,length=3)
#plt.show()
plt.savefig('outputs/comparison_remainder2.png')


plt.figure(dpi=300)
plt.plot(channels_used_array, maximum_flux_array, label='Sum Log Rate (Utility)', alpha = 0.8, color = 'orange', linewidth = 1)
plt.plot(channels_used_array, maximum_flux_array2, label='Sum Rate', alpha = 0.8, color = 'red', linewidth = 1)
plt.plot(channels_used_array, maximum_flux_array3, label='Average Remainder', alpha = 0.8, linestyle = 'dashed', color = 'green', linewidth = 2)
plt.plot(channels_used_array, maximum_flux_array4, label='Average Remainder Percentage', alpha = 0.8, linestyle = 'dotted', color = 'blue', linewidth = 2)
plt.xlabel('Number of Channels')
plt.ylabel('Used Flux Per Channel')
plt.title('Used Flux Per Number of Channels')
plt.tick_params(axis='both',which='major',labelsize='large')
plt.tick_params(direction='in',top=True,right=True,length=6)
plt.minorticks_on()
plt.tick_params(which='minor',direction='in',top=True,right=True,length=3)
#plt.show()
plt.legend(loc = 'best')
plt.savefig('outputs/comparison_flux.png')


plt.figure(dpi=300)
plt.title('Sum of Log of Rate Utility')
plt.xlabel('Number of Channels')
plt.ylabel('Rate Utility')
# plt.ylim([-3,1])
plt.plot(channels_used_array, pure_rates, label='Sum Log Rate (Utility)', alpha = 0.8, color = 'orange', linewidth = 1)
plt.plot(channels_used_array, pure_rates2, label='Sum Rate', alpha = 0.8, color = 'red', linewidth = 1)
plt.plot(channels_used_array, pure_rates3, label='Average Remainder', alpha = 0.8, linestyle = 'dashed', color = 'green', linewidth = 2)
plt.plot(channels_used_array, pure_rates4, label='Average Remainder Percentage', alpha = 0.8, linestyle = 'dotted', color = 'blue', linewidth = 2)
#plt.plot(channels_used_array, np.array(pure_rates)/l, label = 'Average of Sum of Rates')
plt.legend(loc = 'best')
plt.tick_params(axis='both',which='major',labelsize='large')
plt.tick_params(direction='in',top=True,right=True,length=6)
plt.minorticks_on()
plt.tick_params(which='minor',direction='in',top=True,right=True,length=3)
plt.savefig('outputs/comparison_rate_utility1.png')
#plt.show()
plt.close()


plt.figure(dpi=300)
plt.title('Sum of Rates')
plt.xlabel('Number of Channels')
plt.ylabel('Rate Utility')
# plt.ylim([-3,1])
plt.plot(channels_used_array, sum_pure_rates, label='Sum Log Rate (Utility)', alpha = 0.8, color = 'orange', linewidth = 1)
plt.plot(channels_used_array, sum_pure_rates2, label='Sum Rate', alpha = 0.8, color = 'red', linewidth = 1)
plt.plot(channels_used_array, sum_pure_rates3, label='Average Remainder', alpha = 0.8, linestyle = 'dashed', color = 'green', linewidth = 2)
plt.plot(channels_used_array, sum_pure_rates4, label='Average Remainder Percentage', alpha = 0.8, linestyle = 'dotted', color = 'blue', linewidth = 2)
#plt.plot(channels_used_array, np.array(pure_rates)/l, label = 'Average of Sum of Rates')
plt.legend(loc = 'best')
plt.tick_params(axis='both',which='major',labelsize='large')
plt.tick_params(direction='in',top=True,right=True,length=6)
plt.minorticks_on()
plt.tick_params(which='minor',direction='in',top=True,right=True,length=3)
plt.savefig('outputs/comparison_rate_utility2.png')
#plt.show()
plt.close()


