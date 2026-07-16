#main.py
"""
This is the entry point into my routing and spectrum allocation code. 
The code can create an arbitrary network based on user parameters to 
then have routing and spectrum allocation optimally solved.
"""

#ZHG
#2026.03.20
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from config.presets import contention, manhattan_ilec, paper_dense, simple_contention, star, paper_exhaustive, paper_ring, two_source_three_users_custom, super_dense
from config.custom_topology import my_custom_network
from pipeline.runner import run_pipeline


def main():
    #edit custom_topology.py in config
    #cfg = my_custom_network()  

    # Paper figures. Only the last assignment actually runs; comment out
    # the ones below the one you want to reproduce.
    cfg = simple_contention()
    cfg = paper_ring()
    cfg = manhattan_ilec()

    result = run_pipeline(cfg)
    #print(result)


if __name__ == "__main__":
    main()
