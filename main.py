#main.py
"""
This is the entry point into my routing and spectrum allocation code. 
The code can create an arbitrary network based on user parameters to 
then have routing and spectrum allocation optimally solved.
"""
  
#ZHG
#2026.03.20
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from config.presets import contention, paper_dense, star, paper_exhaustive, paper_ring, two_source_three_users_custom, contention, super_dense
from pipeline.runner import run_pipeline

def main():
    #cfg = paper_exhaustive()
    cfg = paper_ring()
    #cfg = paper_dense()
    #cfg = two_source_three_users_custom()
    #cfg = contention()
    result = run_pipeline(cfg)
    #print(result)

if __name__ == "__main__":
    main()