#Iterative Ratio Method
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def fidelity_equation(x, y1, y2):
    return 0.25 * (1 + (3*x / (x**2 + x*(2*y1 + 2*y2 + 1) + 4*y1*y2)))

def rate_equation(x, y1, y2):
    return x**2 + x*(2*y1 + 2*y2 + 1) + 4*y1*y2

def network_var(tau, mu, y1, y2):
#Calculates the lines for fidelity and normalized rate
#Calculates the lines
#tau is the coincidence window
#mu is the link flux
#eta is the efficiency of a particular user
#d is the dark count rate of a particular user
    x = tau*mu
    fidelity = fidelity_equation(x, y1, y2)
    rate = rate_equation(x, y1, y2)
    rate_max = rate_equation(1, y1, y2)
    rate_ratio = rate/rate_max
    return fidelity, rate_ratio

def network_var_points(tau, mu, y1, y2, fidelity_min):
#Calculates the points for fidelity and normalized rate, and finds the cost function
#tau is the coincidence window
#mu is the link flux
#eta is the efficiency of a particular user
#d is the dark count rate of a particular user
    x = tau*mu
    def max_flux_equation(x):
        return 0.25 * (1 + 3 * x / (x**2 + (2 * y1 + 2 * y2 + 1) * x + 4 * y1 * y2)) - fidelity_min
    
    fidelity, rate_ratio = network_var(tau, mu, y1, y2)

    # Initial guess for x
    initial_guess = 1.0
    # Solve the equation using fsolve
    max_flux = fsolve(max_flux_equation, initial_guess)
    if fidelity < fidelity_min:
        cost_function = [100]
    else:
        cost_function = np.abs(max_flux - x) #Cost function is difference between 
    return fidelity, rate_ratio, x, cost_function[0]

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
multiplier_cost = []
floored_ratio_array = []
mu_channel_array = []
mu_channels = []

#Initial loop values
for channel_number in channel_numbers: #Loop for each number of channels we have avaliable
    floored_ratio = [0]
    multiplier = 0
    while np.sum(floored_ratio) < channel_number: #While loop runs until we exceed the channel_number
        multiplier += 1
        #Find the maximum flux values for each link (maximum x)
        max_flux_array = []
        for link_number in range(l): #Iterates for each set of links
            y1 = y1_array[link_number]
            y2 = y2_array[link_number]

            def max_flux_equation(x):
                #Fit equation for getting the maximum flux of a channel. I put this function inside the main code in case I need to use the local fidelity_limit variable in the optimizer
                return 0.25 * (1 + 3 * x / (x**2 + (2 * y1 + 2 * y2 + 1) * x + 4 * y1 * y2)) - fidelity_limit #Fidelity limit here to get the highest possible flux x value possible with the fidelity limit
            
            initial_guess = 1.0
            # Solve the equation using fsolve
            max_flux = fsolve(max_flux_equation, initial_guess)[0] #Finds the max flux of the channels
            max_flux_array.append(max_flux) 
        ratio = np.array(max_flux_array)/min(max_flux_array)*multiplier #Takes the ratio of each flux with the smallest channel flux and multiplies this by the ratio
        #This makes the smallest ratio equal to 1
        floored_ratio = np.floor(ratio)
    
        all_link_used_channels = []
        all_rates = []
        all_link_cost_functions = []
        mu_channel = min(max_flux_array)/tau/multiplier #Determines the channel flux based on the smallest channel flux. We also divide by the multiplier to shrink the flux so we fit multiples into the channels
        fidelity_array = []
        for link_number in range(l):
            x = tau*mu_channel*floored_ratio[link_number] #dimensionless flux calculation
            y1 = y1_array[link_number]
            y2 = y2_array[link_number]
            fidelity = fidelity_equation(x, y1, y2) #Calculates the fidelity of the link
            fidelity_array.append(fidelity) #Adds the fidelity into the fidelity array
    mu_channels.append(mu_channel)

    while np.sum(floored_ratio) > channel_number: #Loop to remove channels from ratio with lowest cost increase
        cost_difference = []

        for link_number in range(l):
            y1 = y1_array[link_number]
            y2 = y2_array[link_number]
            x = tau*mu_channel*floored_ratio[link_number] #dimensionless flux calculation
            if floored_ratio[link_number] == 1: #If the number of channels used for this link is 1 then we skip this link
                cost_difference.append(np.inf) #Setting the cost as infinity
                continue
            rate = np.abs(max_flux_array[link_number] - x)
            x = tau*mu_channel*(floored_ratio[link_number] - 1) #dimensionless flux calculation
            new_rate = np.abs(max_flux_array[link_number] - x)
            cost_difference.append(rate - new_rate) #Stores the change in rate

        least_impactful_index = np.argmin(cost_difference) #Find the least impactful link to take a channel from
        floored_ratio[least_impactful_index] = floored_ratio[least_impactful_index] - 1 #Remove a channel from this link
    floored_ratio_array.append(floored_ratio)
    mu_channel_array.append(mu_channel)

#Fidelity and rate plot
fig, axs = plt.subplots((l + 1) // 2, 2, figsize=(11, 8))
line_handles = []  # List to store handles for creating a global legend
line_labels = []  # List to store labels for creating a global legend
for link_number in range(l):
    y1 = y1_array[link_number]
    y2 = y2_array[link_number]
    mu = np.linspace(0, mu_total, 1000)
    markers = ['o', '^', 's', 'd', 'x']
    ax = axs[link_number % ((l + 1) // 2), link_number // ((l + 1) // 2)]
    fidelity, rate = network_var(tau, mu, y1, y2)
    line1, = ax.plot(tau * mu, fidelity, label='Fidelity', c='tab:blue')
    line2, = ax.plot(tau * mu, rate, label='Rate/Max Rate', c='tab:orange')
    if link_number == 0:  # To prevent duplicates in the legend
        line_handles.extend([line1, line2])
        line_labels.extend(['Fidelity', 'Rate/Max Rate'])
    link_used_channels = []
    link_rates = []
    link_cost_functions = []
    for channel_index in range(len(channel_numbers)):
        fidelity, rate, flux, cost_function = network_var_points(tau, mu_channel_array[channel_index] * floored_ratio_array[channel_index][link_number], y1, y2, fidelity_limit)
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
    ax.axvline(max_flux_array[link_number], color='k', linestyle='--') #Add maximum possible flux line based on the fidelity limit
    ax.axhline(fidelity_limit, color='k', linestyle='--') #Add fidelity limit horizontal line
    all_link_used_channels.append(link_used_channels)
    all_rates.append(link_rates)
    all_link_cost_functions.append(link_cost_functions)

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

#Allocation bars
plt.figure()
plt.title('Channel Allocation')
plt.xlabel('Number of Channels')
plt.ylabel('Channels Used')
channel_str = []
for channel_number in channel_numbers:
    channel_str.append(str(channel_number))
colors = ['blue','green','red','cyan','magenta','yellow','black','orange','purple','pink','lime','brown','teal']
labels = ['Link AB','Link CD','Link EF','Link GH','Link IJ','Link KL','Link MN','Link OP','Link QR','Link ST','Link UV','Link WX','Link YZ']
all_link_used_channels = np.array(all_link_used_channels)
bottom_val = all_link_used_channels[0] - all_link_used_channels[0]
for i in range(len(all_link_used_channels)):
    plt.bar(channel_str, np.array(floored_ratio_array)[:,i], bottom = bottom_val, color = colors[i], label = labels[i])
    bottom_val+=np.array(floored_ratio_array)[:,i]
#plt.bar(channel_str, np.array(k) - np.sum(all_link_used_channels, axis=0), bottom = bottom_val, color = 'tab:blue', label = 'Null Link')
plt.legend(loc='best')
plt.savefig('outputs/channel_barplot.png')
#plt.show()
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
rate = rate_equation(np.array(floored_ratio_array)[:,0]*tau*mu_channels, y1_array[0], y2_array[0]) * 0 #Just to make an array of 0s
for i in range(len(all_link_used_channels)):
    rate += np.log10(rate_equation(np.array(floored_ratio_array)[:,i]*tau*mu_channels, y1_array[i], y2_array[i]))
plt.bar(channel_str, rate) #, bottom = bottom_val, color = colors[i], label = labels[i])
#bottom_val+=np.log10(rate_equation(np.array(floored_ratio_array)[:,i]*tau*mu_channels, y1_array[i], y2_array[i]))
#plt.bar(channel_str, np.array(k) - np.sum(all_link_used_channels, axis=0), bottom = bottom_val, color = 'tab:blue', label = 'Null Link')
plt.legend(loc='best')
plt.savefig('outputs/rate_utility.png')
#plt.show()
plt.close()

# #Cost function plot for all multiples. If 100 then fidelity is worse than linit
# plt.figure(dpi=120)
# all_link_cost_functions = np.array(all_link_cost_functions)
# for cost_index in range(len(all_link_cost_functions[0])):
#     multiplier_cost.append(np.sum(all_link_cost_functions[:,cost_index]))
# plt.plot(multipliers, multiplier_cost)
# plt.title('Ratio Method Cost Function Plot')
# plt.xlabel('Multiplier')
# plt.ylabel('Cost')
# plt.tick_params(axis='both',which='major',labelsize='large')
# plt.tick_params(direction='in',top=True,right=True,length=6)
# plt.minorticks_on()
# plt.tick_params(which='minor',direction='in',top=True,right=True,length=3)
# plt.savefig('outputs/link_cost.png')
# #plt.show()
# plt.close()
