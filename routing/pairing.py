#routing/pairing.py
"""
Calculations related to logarithmic path loss and noise parameter y
"""

#ZHG
#2026.03.24
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def compute_path_loss(network, path):
    """Computes the total loss for a given path by summing the losses of its constituent"""
    loss = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        loss += network[u][v].get("loss", 0.0)
    return loss

def compute_y(loss, tau, d):
    """Computes the noise parameter for a path based on its total loss, coincidence window, and dark count rate."""
    return tau * d * 10 ** (loss / 10.0)