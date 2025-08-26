#The MATT method (MINLP with APOPT for Transmission Tuning)
from gekko import GEKKO
import numpy as np
import math
from scipy.optimize import fsolve

def fidelity_equation(x, y1, y2):
    return 0.25 * (1 + (3*x / rate_equation(x, y1, y2)))

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

def network_var_points(tau, mu, y1, y2, fidelity_min): #TODO: Idk if I need this
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


def matt(K, fidelity_limit, y1, y2, initial):
    """
    Inputs:

    
    Outputs:

    """
    N_links = len(y1)

    # Create GEKKO model. Setting remote=True uses GEKKO's online solver service.
    m = GEKKO(remote=False)
    m.options.IMODE = 3  # steady-state optimization mode
    m.options.SOLVER = 1  # use APOPT (which supports MINLP)
    m.solver_options = [
        'minlp_maximum_iterations 5000',   # was 1000
        'minlp_gap_tol 0.0001',            # correct keyword
        'minlp_branch_method 2',
        'nlp_maximum_iterations 5000'      # was 2000
    ]


    # Decision variable: mu (continuous, positive)
    mu = m.Var(value=initial, lb=1e-9)

    # Decision variables: k_i (integer, at least 1, and upper bounded by K)
    k_vars = [m.Var(value=K//N_links, integer=True, lb=1, ub=K) for i in range(N_links)]

    # Constraint: Sum of channels must equal K
    m.Equation(sum(k_vars) <= K)

    # Fidelity constraints for each link:
    #  0.25*(1 + (3*mu*k_i)/(mu^2*k_i^2 + mu*k_i*(2*(y1_i+y2_i)+1) + 4*y1_i*y2_i)) >= fidelity_limit
    for i in range(N_links):
        expr = mu**2 * k_vars[i]**2 + mu * k_vars[i]*(2*(y1[i]+y2[i]) + 1) + 4*y1[i]*y2[i]
        m.Equation(0.25*(1 + (3*mu*k_vars[i]) / expr) >= fidelity_limit[i])

    # Objective: maximize the sum over links of log10(expr_i)
    # Since log10(x) = ln(x)/ln(10) and logarithm is monotonic,
    # maximizing the sum of ln(expr) is equivalent.
    # GEKKO minimizes by default so we minimize the negative of the sum.
    obj = 0
    for i in range(N_links):
        expr = mu**2 * k_vars[i]**2 + mu * k_vars[i]*(2*(y1[i]+y2[i]) + 1) + 4*y1[i]*y2[i]
        obj += m.log(expr)
    m.Obj(-obj)  # maximize sum(log(expr))

    #Solve the MINLP
    m.solve(disp=True)

    #Store optimal values for flux and channels
    optimal_mu = mu.value[0]
    optimal_allocation = []
    for i in range(N_links):
        optimal_allocation.append(k_vars[i].value[0])

    # Compute and print the objective value (in base 10)
    prelog_rates = [] #Saving the pre-utility rates for each link
    objective_value = 0
    for i in range(N_links):
        expr_val = mu.value[0]**2 * (k_vars[i].value[0])**2 + mu.value[0] * (k_vars[i].value[0]) * (2*(y1[i]+y2[i]) + 1) + 4*y1[i]*y2[i]
        prelog_rates.append(expr_val)
        objective_value += math.log(expr_val) / math.log(10)
    print("Calculated Objective Value: ", objective_value)
    return k_vars, objective_value, mu, prelog_rates, optimal_mu, optimal_allocation


def network_allocation(sources, path_choice, allocation_type = 'APOPT', initial_conditions = 0.001):
    """
    Inputs:
    sources
    path_choice
    allocation_type
    initial_conditions
    
    Outputs:
    results
    utility_sum
    """
    network_utility = 0
    output = []
    for source in sources:
        k = len(sources[source]['available_channels']) #Total number of available channels in the source (does not matter what the frequency is)

        fidelity_limit = []
        y1 = []
        y2 = []
        links = []
        output_values = {}
        for link in path_choice['combo']: #For each link we are extracting the link specific parameters like fidelity limit, y1, and y2
            if source == link['path']['source']: #If the source is the same then we want to save the link parameters
                fidelity_limit.append(link['fidelity_limit'])
                y1.append(link['path']['y1'])
                y2.append(link['path']['y2'])
                links.append(link['link'])
            else: #If we dont have any links that use the source then we try the next one
                continue

        if links == []: #If there are no links at all that use the source then we try the next source
            continue
        else:
            if allocation_type == 'comparison':
                import compare
                k_vars, objective_value, mu, prelog_rates, optimal_mu, optimal_allocation = compare.matt_comparison_gecko(k, fidelity_limit, y1, y2, initial_conditions)
            else: #Matt's MINLP solver
                k_vars, objective_value, mu, prelog_rates, optimal_mu, optimal_allocation = matt(k, fidelity_limit, y1, y2, initial_conditions)

        #Packing variables
        output_values = {'source':source,'channel_allocation':k_vars,'total_channels':k,'links':links,'utility':objective_value,'mu':mu[0],'prelog_rates':prelog_rates,'optimal_mu':optimal_mu,'optmial_allocation':optimal_allocation}
        output.append(output_values)
        network_utility += objective_value #Total Rate Utility of the network
    

    return output, network_utility

def main():
    import create_network
    import routing
    num_usr, num_src, num_edg, num_lnk= 10, 5, 40, 5
    loss_range=(1,30)
    y1 = [0, 0.0034, 0, 0.0299, 0.0385, 0.0625, 0.0733, 0, 0.1106, 0.125, 0.1489, 0]
    y2 = [0, 0.0006, 0.0357, 0.0051, 0.0066, 0.0107, 0.0126, 0.1818, 0.019, 0.0214, 0.0256, 0.2979]
    fidelity_limit = np.repeat(0.7,len(y1))
    link_pairs = None
    

    network, node_usr, sources = create_network.create_network(num_usr, num_src, num_edg, loss_range, num_channels_per_source=[None], topology='ring', density = 0.1)
    network = create_network.create_links(network, node_usr, num_lnk, link_pairs)
    network = create_network.link_info_packaging(network, y1, y2, fidelity_limit)

    network = routing.double_dijkstra(network, sources)
    combinations = routing.path_combinations(network)
    path_choice = routing.choose_paths(combinations)
    results, network_utility = network_allocation(sources, path_choice, allocation_type = 'APOPT', initial_conditions = 0.001)
    print('')

if __name__ == "__main__":
    main()