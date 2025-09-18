#The MATT method (MINLP with APOPT for Transmission Tuning)
from gekko import GEKKO
import numpy as np
import math
import os
import contextlib

def matt(K, fidelity_limit, y1, y2, initial, per_link_k_cap=None, verbose = True):
    N_links = len(y1)

    #Create GEKKO model
    m = GEKKO(remote=False)
    m.options.IMODE = 3
    m.options.SOLVER = 1
    m.solver_options = [
        'minlp_maximum_iterations 1000',
        'minlp_gap_tol 1e-4',
        'minlp_branch_method 2',
        'nlp_maximum_iterations 2000'
    ]

    #Decision variable: mu (continuous, positive)
    mu = m.Var(value=initial, lb=1e-9)

    if N_links == 1:
        k_vars = [m.Var(value=1, integer=True, lb=1, ub=1)]
        m.Equation(sum(k_vars) == 1)
    else:
        ub_each = K if per_link_k_cap is None else min(K, int(per_link_k_cap))
        k_vars = [m.Var(value=K//N_links, integer=True, lb=1, ub=ub_each) for i in range(N_links)]
        m.Equation(sum(k_vars) <= K)

    #Fidelity constraints per link
    for i in range(N_links):
        expr = mu**2 * k_vars[i]**2 + mu * k_vars[i]*(2*(y1[i]+y2[i]) + 1) + 4*y1[i]*y2[i]
        m.Equation(0.25*(1 + (3*mu*k_vars[i]) / expr) >= fidelity_limit[i])

    #Objective: maximize sum log10(expr_i)
    obj = 0
    for i in range(N_links):
        expr = mu**2 * k_vars[i]**2 + mu * k_vars[i]*(2*(y1[i]+y2[i]) + 1) + 4*y1[i]*y2[i]
        obj += m.log10(expr)
    m.Obj(-obj)

    #Solve MINLP
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            m.solve(disp=False)

    #sanitize k and compute μ analytically
    #Round & clip integers
    ub_each = K if per_link_k_cap is None else min(K, int(per_link_k_cap))
    raw_k = [kv.value[0] for kv in k_vars]
    k_int = [int(round(v)) for v in raw_k]
    k_int = [min(max(1, k), ub_each) for k in k_int]

    #Cheap repair to enforce sum(k) <= K after rounding
    if N_links > 1:
        total = sum(k_int)
        if total > K:
            #Prefer to decrement those rounded up the most
            round_up_amt = [k_int[i] - raw_k[i] for i in range(N_links)]
            while total > K:
                candidates = [(i, round_up_amt[i]) for i in range(N_links) if k_int[i] > 1]
                if not candidates:
                    break
                i_star = max(candidates, key=lambda t: t[1])[0]
                k_int[i_star] -= 1
                total -= 1

    #Closed-form μ* from fidelity equalities (use the UPPER root; then take min across links)
    def mu_upper_for_link(i, k_i):
        F = fidelity_limit[i]
        s = y1[i] + y2[i]
        p = y1[i] * y2[i]
        t = 4.0*F - 1.0  #> 0
        A = t * (k_i**2)
        B = t * k_i * (2.0*s + 1.0) - 3.0 * k_i
        C = 4.0 * t * p
        #Discriminant
        D = B*B - 4.0*A*C
        if D <= 0.0:
            #numerically tight; fall back to current mu value to be safe
            return max(float(mu.value[0]), 1e-9)
        sqrtD = math.sqrt(D)
        #Upper root
        mu_hi = (-B + sqrtD) / (2.0 * A)
        return max(mu_hi, 1e-9)

    mu_star = min(mu_upper_for_link(i, k_int[i]) for i in range(N_links)) if N_links > 0 else float(mu.value[0])

    optimal_mu = mu_star
    optimal_allocation = k_int[:]

    #Compute objective and per-link expr with (μ*, k_int)
    prelog_rates = []
    objective_value = 0.0
    for i in range(N_links):
        ki = optimal_allocation[i]
        expr_val = (optimal_mu**2 * (ki**2) + optimal_mu * ki * (2*(y1[i]+y2[i]) + 1) + 4*y1[i]*y2[i])
        prelog_rates.append(expr_val)
        objective_value += math.log10(expr_val)

    if verbose:
        print("Calculated Objective Value (log10):", objective_value)
    return k_vars, objective_value, mu, prelog_rates, optimal_mu, optimal_allocation

def network_allocation(sources, path_choice, allocation_type='APOPT',initial_conditions=0.001, per_link_k_cap=None, verbose = True):
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
                f_lim = link['fidelity_limit']
                if f_lim < 0.5:
                    f_lim = 0.5
                fidelity_limit.append(f_lim)
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
                k_vars, objective_value, mu, prelog_rates, optimal_mu, optimal_allocation = matt(k, fidelity_limit, y1, y2, initial_conditions, per_link_k_cap=per_link_k_cap, verbose = verbose)

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