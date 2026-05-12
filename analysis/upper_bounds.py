#analysis/upper_bounds.py
"""
Functions to compute upper bounds for path and link combinations.
"""

#ZHG
#2026.03.26
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def combo_path_upper_bound(combo):
    return sum(o["path_ub"] for o in combo)

def combo_link_upper_bound(combo):
    return sum(o["link_ub"] for o in combo)