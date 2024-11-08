import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import os

def fidelity_equation(x, y1, y2):
    return 0.25 * (1 + (3 * x) / (x**2 + x * (2 * y1 + 2 * y2 + 1) + 4 * y1 * y2))

def rate_equation(x, y1, y2):
    return x**2 + x * (2 * y1 + 2 * y2 + 1) + 4 * y1 * y2

# Create 'outputs' directory if it doesn't exist
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# Input parameters
tau = 1E-9  # Coincidence window in seconds
# Efficiencies
eta1 = 0.012 
eta2 = 0.00021
# Dark count rates
d1 = 100 
d2 = 3500
l = 12  # Number of links
channel_numbers = [12, 72, 147, 225, 303, 381, 456, 532, 612, 685, 764]  # Example channel numbers
mu_total = 1 / tau  # Total link flux
# Noise parameters
y1_array = [0, 0.0034, 0, 0.0299, 0.0385, 0.0625, 0.0733, 0, 0.1106, 0.125, 0.1489, 0]
y2_array = [0, 0.0006, 0.0357, 0.0051, 0.0066, 0.0107, 0.0126, 0.1818, 0.019, 0.0214, 0.0256, 0.2979]
fidelity_limit = 0.7  # Changeable fidelity minimum line value

def compute_max_flux(l, y1_array, y2_array, fidelity_limit):
    link_flux = []
    for link_number in range(l):
        y1 = y1_array[link_number]
        y2 = y2_array[link_number]

        def max_flux_equation(x):
            return fidelity_equation(x, y1, y2) - fidelity_limit

        initial_guess = 1.0
        # Solve the equation using fsolve
        max_flux = fsolve(max_flux_equation, initial_guess)[0]
        link_flux.append(max_flux)
    return np.array(link_flux)

# Compute maximum flux for each link
link_flux = compute_max_flux(l, y1_array, y2_array, fidelity_limit)

# Initialize variables to store results
channels_used_list = []
maximum_flux_list = []
allocated_flux_list = []
remainders_list = []
channel_allocations_list = []
rate = []

# Start with k = 20
initial_k = 100
k_values = list(range(initial_k, 11, -1))  # k from 20 down to 12

# Function to calculate trial_total_rate using a loop
def calculate_trial_total_rate(channel_allocation, flux, y1_array, y2_array):
    trial_total_rate = 0
    for j in range(len(y1_array)):
        rate_val = rate_equation(channel_allocation[j] * flux, y1_array[j], y2_array[j])
        # To avoid log10 of zero or negative, ensure rate_val is positive
        if rate_val > 0:
            trial_total_rate += np.log10(rate_val)
        else:
            # Assign a very low value if rate_val is non-positive
            trial_total_rate += -np.inf
    return trial_total_rate

# Allocate channels for k = 20
# k = initial_k
# flux_total = np.sum(link_flux)
# flux = flux_total / k
# largest_flux = np.min(link_flux)  # Ensure flux does not exceed the smallest max_flux

# if flux > largest_flux:
#     flux = largest_flux

# channel_allocation = np.floor(link_flux / flux).astype(int)
# # Ensure each link has at least one channel
# channel_allocation[channel_allocation == 0] = 1

#Big Bin but only at initial_k
largest_flux = np.min(link_flux)
if initial_k <= np.sum(np.floor(link_flux/largest_flux)): #If we dont have enough channels to completely fill everything
    flux = largest_flux
    channel_allocation = np.ones(len(link_flux))
    filled_percentage = flux*channel_allocation/link_flux #Calculate percent filled of each link
    used_channels = np.sum(channel_allocation)
    while used_channels < initial_k:
        channel_allocation[np.argmin(filled_percentage)] += 1 #Adds one to channel that is least filled
        filled_percentage = channel_allocation/np.floor(link_flux/flux) #Calculate percent filled of each link
        used_channels = np.sum(channel_allocation)
else:
    flux_total = np.sum(link_flux)
    flux = flux_total/initial_k #Simple flux determining method
    if flux > largest_flux: #Makes sure we dont go over the largest flux amount
        flux = largest_flux
    remaining_flux = link_flux - flux #Flux if we remove guaranteed channel
    channel_allocation = np.floor(remaining_flux/flux) #Allocates extra channels
    channel_allocation = channel_allocation + 1 #Readds the guaranteed channel

#For k = 1000 case
# channel_allocation = np.array([95, 94, 85, 85, 82, 73, 69, 43, 55, 49, 37, 10])
# flux = 0.0069926206260949005
# k_values = list(range(777, 11, -1))


channel_allocations_list.append(channel_allocation.copy())
allocated_flux_list.append(flux)
remainder = link_flux - channel_allocation * flux
remainders_list.append(remainder.copy())

# Compute the initial rate using loop
trial_total_rate = calculate_trial_total_rate(channel_allocation, flux, y1_array, y2_array)
rate.append(trial_total_rate)

channels_used_list.append(initial_k)
maximum_flux_list.append(flux)

# Iteratively reduce k from 19 down to 12
current_allocation = channel_allocation.copy()

for k in k_values[1:]:  # Skip the first element (k=20 already allocated)
    # Compute current rates for each link
    current_rates = []
    for j in range(len(y1_array)):
        rate_val = rate_equation(current_allocation[j] * flux, y1_array[j], y2_array[j])
        current_rates.append(rate_val)

    # Compute log rates using loop
    log_rates = []
    for rate_val in current_rates:
        if rate_val > 0:
            log_rates.append(np.log10(rate_val))
        else:
            log_rates.append(-np.inf)  # Assign a very low value if rate_val is non-positive

    # Identify the link with the smallest log(rate)
    # This link contributes the least to the total rate
    link_to_reduce = np.argmax(log_rates)

    # Reduce the channel allocation for the identified link
    if current_allocation[link_to_reduce] > 1:
        current_allocation[link_to_reduce] -= 1
    else:
        pass
        # print(f"Cannot reduce channels for link {link_to_reduce} below 1.")

    # Recompute flux based on new k
    flux = np.min(link_flux / current_allocation)
    if flux > largest_flux:
        flux = largest_flux

    # Update allocations based on new flux
    #channel_allocation = current_allocation * flux

    # To maintain the total number of channels, adjust allocations if necessary
    total_channels = np.sum(current_allocation)
    while total_channels > k:
        # Recompute current rates for each link
        current_rates = []
        for j in range(len(y1_array)):
            rate_val = rate_equation(current_allocation[j] * flux, y1_array[j], y2_array[j])
            current_rates.append(rate_val)

        # Compute log rates using loop
        log_rates = []
        for rate_val in current_rates:
            if rate_val > 0:
                log_rates.append(np.log10(rate_val))
            else:
                log_rates.append(-np.inf)

        # Identify the link with the smallest log(rate)
        link_to_reduce = np.argmin(log_rates)

        if current_allocation[link_to_reduce] > 1:
            current_allocation[link_to_reduce] -= 1
            total_channels -= 1
        else:
            # If cannot reduce further, break the loop
            break

    # Update allocations to current allocation
    current_allocation = current_allocation.copy()

    # Compute remainder
    remainder = link_flux - current_allocation * flux

    # Store results
    channel_allocations_list.append(current_allocation.copy())
    allocated_flux_list.append(flux)
    remainders_list.append(remainder.copy())

    # Compute the total rate using loop
    trial_total_rate = calculate_trial_total_rate(current_allocation, flux, y1_array, y2_array)
    rate.append(trial_total_rate)

    # Store k and flux
    channels_used_list.append(k)
    maximum_flux_list.append(flux)

    print(f"k: {k}")
    print(f"Channel Allocation: {current_allocation}")
    print(f"Flux: {flux}")
    print(f"Total Channels Allocated: {np.sum(current_allocation)}")
    print("-" * 40)

# Convert lists to arrays for plotting and analysis
channels_used_array = np.array(channels_used_list)
maximum_flux_array = np.array(maximum_flux_list)
allocated_flux_array = np.array(allocated_flux_list)
remainders_array = np.array(remainders_list)
channel_allocations_array = np.array(channel_allocations_list)
rate = np.array(rate)

# Generate labels for the links
labels = [f'Link {chr(65 + 2*i)}{chr(66 + 2*i)}' for i in range(l)]  # e.g., 'Link AB', 'Link CD', etc.
colors = plt.cm.viridis(np.linspace(0, 1, l))  # Generate distinct colors for each link

plt.figure(dpi=300)
plt.plot(channels_used_array, np.sum(remainders_array, axis = 1))
plt.xlabel('Number of Channels')
plt.ylabel('Leftover Flux Remainder')
plt.title('Remainder Per Number of Channels')
#plt.show()
plt.savefig('outputs/racoon_flux_remainder.png')

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
plt.savefig('outputs/racoon_flux_remainder_comp.png')

plt.figure(dpi=300)
plt.plot(channels_used_array, maximum_flux_array)
plt.xlabel('Number of Channels')
plt.ylabel('Used Flux Per Channel')
plt.title('Used Flux Per Number of Channels')
#plt.show()
plt.savefig('outputs/racoon_flux.png')


plt.figure(dpi=300)
plt.title('Rate Utility')
plt.xlabel('Number of Channels')
plt.ylabel('Rate Utility')

colors = ['blue','green','red','cyan','magenta','yellow','black','orange','purple','pink','lime','brown','teal']
labels = ['Link AB','Link CD','Link EF','Link GH','Link IJ','Link KL','Link MN','Link OP','Link QR','Link ST','Link UV','Link WX','Link YZ']
plt.plot(channels_used_array, rate, label = 'Sum of Log of Rate Utility')
plt.plot(channels_used_array, np.log10(pure_rates), label = 'Log of Sum of Rates')
#plt.plot(channels_used_array, np.array(pure_rates)/l, label = 'Average of Sum of Rates')
plt.legend(loc = 'best')
plt.savefig('outputs/racoon_rate_utility.png')
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
plt.savefig('outputs/racoon_three_plot.png')
#plt.show()



# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------=++**#%#***+=-----------------------------------------
# ----------------------------------------=+*#%%%#%%%%%%%%%%%%#+--------------------------------------
# --------------------------------------+#%%%%%##%%%%%%%%%%%%%%%%*=-----------------------------------
# -----------------------------------=*#%%%%%#%%%%%%%%%%%%%%%%%%%%%*=---------------------------------
# ---------------------------------=*###########%%%%%%%%%%%%%%%##++++=--------------------------------
# --------------------------------+####%#########%%%%%%%%%%%%%%#**##*++++==---------------------------
# -------------------------------*#%##%%#%%%%####%%%%%%%%%%#####**%%#*++*%%%#+===---------------------
# -----------------------------=*####%#%#%%%%####%%%%%%%%#####%%*+*#@#*#%%#%%%%%%%#*+++++++-----------
# ----------------------------=###%##%%%%%%#%%%##%#%%%%#####%%%#++*#######*#%%###%%*****##+-----------
# ---------------------------=*###%%%%%%%%%%%#%%%#%%%#%#####%@%#++***################%@%#*=-----------
# ---------------------------+######%%%%%%%%%%%%%%%%######%%%%*+*#***++*****#########%%#**=-----------
# --------------------------=####%%%#%%%%%%%%%%%%%%%######%%%*+*#*++====+=+*#******####***------------
# --------------------------+###%#%###%%%%%%%%%%%#########%%#***+====+***+*##++===++*#*+=-------------
# --------------------------*##%#%%%%%%%%%%%#%%%%####%####%%#*+++#@@@%@@@%%%%#%#*+=++*#*=-------------
# -------------------------=##%%##%%%%%##%%%%%%%%#####%##%%%***##%@@@@%%@%%%@@@@@@#*++**--------------
# -------------------------=###%%#####%%#%%%%%%%#*#########%#**#%%%%%@%*+##%%%%@@@@@#+*=--------------
# -------------------------+%##########%#%#%%%%#########%%%##***##*#*+===+##++%@@@@@%#+---------------
# -------------------------*##%%%%####%%###%%%%#*#####%###%##****#%%#===#%%#==+#%%%##*----------------
# ------------------------=*##%%#%####%%%##%%%%##*##%########****##%%%+%%%%@#+*####*=-----------------
# ------------------------+###%%###%###%####%%########%#%####***#%%%%%#%%@@%#%%###*=------------------
# -----------------------+#####%#####%#%%%#%######%%%%%#%##*##*##%%%%#%@@%%%%%#***=-------------------
# ----------------------+**#%###########%%%%###*######%#######*###%%%%@@%%%%##**#+--------------------
# --------------------+#%%########%######%##%##**#%###%##########%%#%@@%%%#%****+---------------------
# -------------------=++*######*##%%#####%%@%%%#*%#%#%%%%%%%###%####%%%%%%##***+=---------------------
# ----------------+*#******####**#######%%%@@@%#*###%%%%%%%%#####*#%%%@%%###**+=----------------------
# -------------=+*#%%%#*****#+---=##**##%%#%%%%##**###%%##%%%###**#%%%@%%##**+=-----------------------
# ---------=+##*++**#%#*#*+=------=*#####%%##%%%##+*####%#%#%#****#%%######**=------------------------
# -------++*%@%*++++****+----------+**####%#%%#%##**+****#*##**++*###*#****++-------------------------
# ----=*##**#%%#+++==--------------=+#######%##%*+***++***#***++-+***##***++--------------------------
# ----=++++++*+==-------------------=*##########=-*##+++**+****+--+*###*+++---------------------------
# -----------------------------------+##%##%##%*--+%#*++++++*+**=--+***+*+----------------------------
# -----------------------------------=%%%%@@@@%+--+%%#*++==+++++=--=++++=-----------------------------
# ------------------------------------*@@%%###*=--=#@%%#*+====+++===+*++=-----------------------------
# -------------------------------------=***+**%###*+=*#%%%#+==++*#%%#*++=-----------------------------
# ------------------------------------------=+*#***+--------++****++%@@%@%%*--------------------------
# -----------------------------------------------------------*##%%#%#=--------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------