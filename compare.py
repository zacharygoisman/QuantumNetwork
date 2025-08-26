#compare.py
#Takes all functions for comparing to Alnas 2022, using the same utility function

#Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from gekko import GEKKO
import math

def compute_fidelity_rate(tau, mu, y1_val, y2_val):
    """
    For a given link (with parameters y1 and y2), compute the maximum possible
    rate from the rate equation
         Rate(x) = [ x^2 + (2*(y1+y2)+1)*x + 4*y1*y2 ] * log2( F(x) )
    where
         F(x) = 0.25*(1 + 3*x / [ x^2 + (2*(y1+y2)+1)*x + 4*y1*y2 ])
    The function samples x in a fixed interval and returns the maximum value.
    """
    x = tau*mu
    A = 2*(y1_val+y2_val) + 1
    B = 4*y1_val*y2_val
    def expr(x):
         return x**2 + A*x + B
    def fidelity(x):
         return 0.25*(1 + 3*x/expr(x))
    def rate(x):
         # Using math.log2 for base-2 logarithm.
         return expr(x) * np.log2(2*fidelity(x))
    return fidelity(x), rate(x), x

def compute_R_max_full(y1_val, y2_val):
    """
    For a given link (with parameters y1 and y2), compute the maximum possible
    rate from the rate equation
         Rate(x) = [ x^2 + (2*(y1+y2)+1)*x + 4*y1*y2 ] * log2( F(x) )
    where
         F(x) = 0.25*(1 + 3*x / [ x^2 + (2*(y1+y2)+1)*x + 4*y1*y2 ])
    The function samples x in a fixed interval and returns the maximum value.
    """
    A = 2*(y1_val+y2_val) + 1
    B = 4*y1_val*y2_val
    def expr(x):
         return x**2 + A*x + B
    def F(x):
         return 0.25*(1 + 3*x/expr(x))
    def rate(x):
         # Using math.log2 for base-2 logarithm.
         return expr(x) * math.log2(2*F(x))
    # Sample x from a small positive value to a reasonable upper bound.
    xs = [i/10000 for i in range(1, 100000)]  # x from 0.0001 to 10
    rates = [rate(x) for x in xs]
    # The maximum possible rate (remember: with F<1, these rates are negative,
    # and the maximum is the one closest to zero)
    max_rate = max(rates)
    return max_rate

def matt_comparison_gecko(K, fidelity_limit, y1, y2, initial):
    N_links = len(y1)
    
    # --------------------------------------------------------
    # 1) Precompute the maximum possible rate for each link.
    # For link i, compute:
    #   R_max = max_{x>=0} { Rate(x) }
    # where Rate(x) = expr(x) * log2(F(x)) and
    #   expr(x) = x^2 + (2*(y1+y2)+1)*x + 4*y1*y2,
    #   F(x) = 0.25*(1 + 3*x/expr(x)).
    # --------------------------------------------------------
    R_max_list = []
    for i in range(N_links):
        R_max = compute_R_max_full(y1[i], y2[i])
        # If by chance R_max is zero, clamp to a small negative number (rates are negative).
        if R_max == 0:
            R_max = -1e-6
        R_max_list.append(R_max)
        #print(R_max)
    
    # --------------------------------------------------------
    # 2) Build the GEKKO model.
    # --------------------------------------------------------
    m = GEKKO(remote=False)
    m.options.IMODE = 3   # steady-state optimization
    m.options.SOLVER = 1  # use APOPT (supports MINLP)
    m.solver_options = [
    'minlp_maximum_iterations 1000',
    'minlp_gap 0.0001',
    'minlp_branch_method 2',
    'nlp_maximum_iterations 2000'
]
    # Decision variable: mu (continuous, positive)
    mu = m.Var(value=0.001, lb=1e-9)
    
    # Decision variables: k_i (integer channels, at least 1, up to K)
    k_vars = [m.Var(value=K//N_links, integer=True, lb=1, ub=K) for _ in range(N_links)]
    
    # Total channels allocated must not exceed K
    m.Equation(sum(k_vars) <= K)
    
    # Fidelity constraints for each link:
    # For each link, we define:
    #   x = mu * k_i,
    #   expr = x^2 + (2*(y1+y2)+1)*x + 4*y1*y2,
    #   F = 0.25*(1 + 3*x/expr),
    # and we enforce F >= fidelity_limit.
    for i in range(N_links):
        x = mu * k_vars[i]
        expr_i = x**2 + (2*(y1[i]+y2[i]) + 1)*x + 4*y1[i]*y2[i]
        F = 0.25*(1 + 3*x/expr_i)
        m.Equation(F >= fidelity_limit[i])
    
    # --------------------------------------------------------
    # 3) Define the objective.
    # For each link, compute the instantaneous rate:
    #   Rate = expr * log2(F)
    # and then the normalized rate is defined as:
    #   Normalized Rate = R_max / Rate.
    # (Because rates are negative, the best (least negative) rate equals R_max,
    # giving a normalized value of 1; worse rates yield a ratio below 1.)
    # We maximize the sum of normalized rates by minimizing its negative.
    # --------------------------------------------------------
    obj = 0
    for i in range(N_links):
        x = mu * k_vars[i]
        expr_i = x**2 + (2*(y1[i]+y2[i]) + 1)*x + 4*y1[i]*y2[i]
        F = 0.25*(1 + 3*x/expr_i)
        rate_expr = expr_i * (m.log(2*F)/math.log(2))
        # Normalized rate: using the precomputed R_max_list[i]
        normalized_rate = rate_expr / R_max_list[i]
        obj += normalized_rate
    m.Obj(-obj)
    
    # --------------------------------------------------------
    # 4) Solve the MINLP.
    # --------------------------------------------------------
    m.solve(disp=False)
    
    # Output the solution:
    print("Optimal mu =", mu.value[0])
    print("Optimal channel allocation (k_i):")
    for i in range(N_links):
        print(f" Link {i+1}: k = {int(round(k_vars[i].value[0]))}")
    
    # For reference, compute the final normalized objective value:
    prelog_rates = []
    objective_value = 0
    for i in range(N_links):
        x_val = mu.value[0] * k_vars[i].value[0]
        expr_val = x_val**2 + (2*(y1[i]+y2[i]) + 1)*x_val + 4*y1[i]*y2[i]
        F_val = 0.25*(1 + 3*x_val/expr_val)
        rate_val = expr_val * (math.log(2*F_val)/math.log(2))
        normalized_rate_val = rate_val / R_max_list[i]
        # print(i,': ',normalized_rate_val)
        objective_value += normalized_rate_val
        prelog_rates.append(normalized_rate_val)
    
    return k_vars, objective_value, mu, prelog_rates

#Fidelity and rate plot
def fidelity_rate_plot(k_list, k_vars_array, objective_value_array, mu_determined, tau, y1_array, y2_array, fidelity_limit, text):
    case_num = text[-1]
    if case_num == '1':
        xlimit = 1.2
    elif case_num == '2':
        xlimit = 0.8
    elif case_num == '3':
        xlimit = 0.2
    elif case_num == '4':
        xlimit = 0.6
    else:
        xlimit = 1
    l = len(y1_array)
    fig, axs = plt.subplots((l + 1) // 2, 2, figsize=(11, 8))
    line_handles = []  # List to store handles for creating a global legend
    line_labels = []  # List to store labels for creating a global legend

    mu_total = 1.2/tau #1.2 just to extend the line further
    mu_channel_array = []
    floored_ratio_array = []
    channel_numbers = []
    all_link_used_channels = []
    for k_index in range(len(k_list)):
        mu_channel_array.append(mu_determined[k_index]/tau)
        floored_ratio_array.append(np.concatenate(k_vars_array[k_index]))
        channel_numbers.append(np.concatenate(k_vars_array[k_index]))
        
    for link_number in range(l):
        y1 = y1_array[link_number]
        y2 = y2_array[link_number]
        mu = np.linspace(1E-6, mu_total, 1000) #1E-6 to prevent divide by 0 errors
        markers = ['o', '^', 's', 'd', 'x']
        ax = axs[link_number % ((l + 1) // 2), link_number // ((l + 1) // 2)]
        fidelity, rate, flux = compute_fidelity_rate(tau, mu, y1, y2)
        line1, = ax.plot(tau * mu, fidelity, label='Fidelity', c='tab:blue')
        line2, = ax.plot(tau * mu, rate, label='Rate/Max Rate', c='tab:orange')
        if link_number == 0:  # To prevent duplicates in the legend
            line_handles.extend([line1, line2])
            line_labels.extend(['Fidelity', 'Rate/Max Rate'])
        link_used_channels = []
        link_rates = []
        #link_cost_functions = []
        for channel_index in range(len(channel_numbers)):
            fidelity, rate, flux = compute_fidelity_rate(tau, mu_channel_array[channel_index] * floored_ratio_array[channel_index][link_number], y1, y2)
            link_used_channels.append(floored_ratio_array[channel_index][link_number])
            link_rates.append(rate)
            #link_cost_functions.append(cost_function)
            scatter = ax.scatter(flux, fidelity, label=str(channel_numbers[channel_index]), marker=markers[channel_index], c='tab:blue')
            ax.scatter(flux, rate, marker = markers[channel_index], c='tab:orange') #marker = "$"+str(channel_numbers[channel_index])+"$", c='tab:orange')
            ax.set_xlim([0,xlimit])
            ax.set_ylim([0,1])
            if link_number == 0:  # Add only once to legend
                line_handles.append(scatter)
                line_labels.append(str(k_list[channel_index]))

        # fidelity_min = fidelity_limit[link_number]
        # def max_flux_equation(x):
        #     return 0.25 * (1 + 3 * x / (x**2 + (2 * y1 + 2 * y2 + 1) * x + 4 * y1 * y2)) - fidelity_min
        # initial_guess = 1
        # max_flux = fsolve(max_flux_equation, initial_guess)
        fidelity_min = fidelity_limit[link_number]

        # Always draw the horizontal fidelity threshold
        ax.axhline(fidelity_min, color='k', linestyle='--')

        # Only try to solve for the x where fidelity == fidelity_min
        # if the threshold actually lies within the sampled fidelity curve:
        def max_flux_equation(x):
            return 0.25 * (1 + 3 * x / (x**2 + (2 * y1 + 2 * y2 + 1) * x + 4 * y1 * y2)) - fidelity_min

        try:
            initial_guess = 1
            max_flux = fsolve(max_flux_equation, initial_guess)

            ax.axvline(max_flux, color='k', linestyle='--') #Add maximum possible flux line based on the fidelity limit
        except Exception:
            # if fsolve still fails, just skip the vertical line
            pass


        
        #ax.axhline(fidelity_limit[link_number], color='k', linestyle='--') #Add fidelity limit horizontal line
        all_link_used_channels.append(link_used_channels)
        # all_rates.append(link_rates)
        # all_link_cost_functions.append(link_cost_functions)

    if l % 2 != 0:
        fig.delaxes(axs[-1, -1])

    # Add a global legend
    fig.legend(handles=line_handles, labels=line_labels, loc='upper right', bbox_to_anchor=(.99, .95))
    # Add global figure labels
    fig.text(0.5, 0.02, 'x (Dimensionless flux)', ha='center')
    fig.text(0.02, 0.5, 'y (Rate or Fidelity)', va='center', rotation='vertical')
    fig.text(0.5, .95, 'Fidelity and EBR vs Flux Plots', ha='center')
    # Adjust layout, save and close
    plt.tight_layout(rect=[0.03, 0.03, 0.85, 0.95])  # Adjust the tight_layout to accommodate the legend
    plt.savefig('outputs/comp_link_fidelity_'+str(case_num)+'.png')  # Save the figure to a file
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
    plt.title('Channel Allocation', fontsize=18)
    plt.xlabel('Number of Channels', fontsize=18)
    plt.ylabel('Channels Used', fontsize=18)

def channel_bar_plot(channel_numbers, k_vars_array, text):
    case_num = text[-1]
    channel_str = [str(c) for c in channel_numbers]

    colors = [
        '#D95F02',   # orange           – Link AB
        '#EFB618',   # golden yellow    – Link CD
        '#7F26B2',   # deep violet      – Link EF
        '#6BA42C',   # olive-ish green  – Link GH
        '#49B5E7',   # sky-blue         – Link IJ
        '#8B1C1A',   # maroon           – Link KL
        '#003BFF',   # vivid blue       – Link MN
        '#017501',   # dark green       – Link OP
        '#FF0000',   # red              – Link QR
        '#B526FF',   # purple           – Link ST
        '#FF00FF',   # magenta          – Link UV
        '#000000'    # black            – Link WX
    ]
    labels  = ['Link AB','Link CD','Link EF','Link GH','Link IJ',
            'Link KL','Link MN','Link OP','Link QR','Link ST',
            'Link UV','Link WX','Link YZ']

    floored_ratio_array = np.asarray(k_vars_array)

    # Start the stack at zero for every bar
    bottom_val = np.zeros_like(channel_numbers, dtype=float)

    channel_sum = []
    for link_number in range(len(floored_ratio_array[:, 0])):
        channel_sum.append(np.sum(floored_ratio_array[link_number]))
    null_link = np.asarray(channel_numbers) - channel_sum   # residual capacity
    # If any channel is already over-allocated, clip negatives to zero
    null_link = np.clip(null_link, 0, None)

    plt.bar(channel_str,
            null_link,
            bottom=bottom_val,
            color='tab:blue',          # pick any unused colour
            label='Null Link')

    bottom_val += null_link

    # --- Plot each real link ----------------------------------------------------
    for i in range(len(k_vars_array[0])):          # loop over links
        plt.bar(channel_str,
                np.concatenate(floored_ratio_array[:,i]),
                bottom=bottom_val,
                color=colors[i],
                label=labels[i])
        bottom_val += np.concatenate(floored_ratio_array[:,i])            # grow the stack



    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('outputs/comp_channel_barplot_'+str(case_num)+'.png')
    plt.close()

def rate_bar_plot(channel_numbers, k_vars_array, objective_value, mu_determined, tau, y1_array, y2_array, fidelity_limit, text):
    case_num = text[-1]
    floored_ratio_array = np.asarray(k_vars_array)

    plt.figure()
    plt.title('Fitness over Total Available Channels', fontsize=18)
    plt.xlabel('Number of Total Available Channels', fontsize=18)
    plt.ylabel('Fitness', fontsize=18)
    channel_str = []
    for channel_number in channel_numbers:
        channel_str.append(str(channel_number))
    # mu_channels = mu_determined
    # rate = rate_equation(np.array(floored_ratio_array)[0,0]*tau*mu_channels, y1_array[0], y2_array[0]) * 0 #Just to make an array of 0s
    # for i in range(len(k_vars_array)):
    #     rate += np.log10(rate_equation(np.concatenate(floored_ratio_array[:,i])*tau*mu_channels, y1_array[i], y2_array[i]))
    plt.bar(channel_str, objective_value) #, bottom = bottom_val, color = colors[i], label = labels[i])
    max_total_fitness = 0.0

    for i, (y1, y2, f_th) in enumerate(zip(y1_array, y2_array, fidelity_limit)):
        # precompute expr‐coeffs
        A = 2*(y1 + y2) + 1
        B = 4*y1*y2
        # the unconstrained R_max for this link
        R_max_i = compute_R_max_full(y1, y2)

        if f_th <= 0:
            # zero‐threshold ⇒ we can achieve R_max ⇒ normalized = 1.0
            normalized_i = 1.0
        else:
            # solve F(x)=f_th
            def fidelity_eq(x):
                return 0.25*(1 + 3*x/(x*x + A*x + B)) - f_th
            # initial guess for x
            x_thresh, = fsolve(fidelity_eq, 1.0)
            # rate at exactly f_th
            expr_val      = x_thresh**2 + A*x_thresh + B
            rate_at_thresh = expr_val * math.log2(2*f_th)
            normalized_i   = rate_at_thresh / R_max_i

        max_total_fitness += normalized_i

    # draw the “best possible” horizontal line
    plt.axhline(max_total_fitness,
                color='k',
                linestyle='--',
                label='Max Total Fitness')

    plt.legend(loc='best', fontsize=12)
    plt.savefig('outputs/comp_rate_utility_'+str(case_num)+'.png')
    #plt.show()
    plt.close()

#Function to trigger all calculations and plots for each case
def run_plots(k_list, fidelity_limit, tau, y1, y2, text, initial = 0.001):
    '''
    Inputs:
    k_list is an array of source channel amounts
    fidelity_limit is an array of fidelity lower bounds for each link
    tau is the coincidence time in seconds
    y1 is the whole y1 array for all links
    y2 is the whole y2 array for all links
    text is a string that differentiates the output plots
    initial is the initial condition mu value that the APOPT solver uses
    '''
    k_vars_array = []
    rate_array = []
    mu_array = []
    objective_array = []
    for k in k_list:
        k_vars, objective_value, mu, prelog_rates = matt_comparison_gecko(k, fidelity_limit, y1, y2, initial)
        k_vars_array.append(k_vars)
        rate_array.append(prelog_rates)
        mu_array.append(mu.value[0])
        objective_array.append(objective_value)
    fidelity_rate_plot(k_list, k_vars_array, rate_array, mu_array, tau, y1, y2, fidelity_limit, text)
    channel_bar_plot(k_list, k_vars_array, text)
    rate_bar_plot(k_list, k_vars_array, objective_array, mu_array, tau, y1, y2, fidelity_limit, text)

def main():
    tau = 1E-9

    #Case 1
    k_list = [5, 10, 20, 40]
    y1 = [0,0.04,0,0.11,0.15]
    y2 = [0,0.007,0.125,0.019,0.025]
    fidelity_limit = np.repeat(0, len(y1))
    text = 'Case 1'
    run_plots(k_list, fidelity_limit, tau, y1, y2, text, initial = 0.001)

    #Case 2
    k_list = [5, 10, 20, 40]
    y1 = [0,0.04,0,0.11,0.15]
    y2 = [0,0.007,0.125,0.019,0.025]
    fidelity_limit = np.repeat(0.7, len(y1))
    text = 'Case 2'
    run_plots(k_list, fidelity_limit, tau, y1, y2, text, initial = 0.001)

    #Case 3
    k_list = [5, 10, 20, 40]
    y1 = [0,0.0034,0.0104,0.0179,0]
    y2 = [0,0.006,0.0018,0.0031,0.0515]
    fidelity_limit = np.repeat(0.9, len(y1))
    text = 'Case 3'
    run_plots(k_list, fidelity_limit, tau, y1, y2, text, initial = 0.001)

    #Case 4
    k_list = [12, 24, 48, 96]
    y1 = [0, 0.0034, 0, 0.0299, 0.0385, 0.0625, 0.0733, 0, 0.1106, 0.125, 0.1489, 0]
    y2 = [0, 0.0006, 0.0357, 0.0051, 0.0066, 0.0107, 0.0126, 0.1818, 0.019, 0.0214, 0.0256, 0.2979]
    fidelity_limit = np.repeat(0.7, len(y1))
    text = 'Case 4'
    run_plots(k_list, fidelity_limit, tau, y1, y2, text, initial = 0.001)


if __name__ == "__main__":
    main()