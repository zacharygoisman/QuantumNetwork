#main.py
"""
This is the entry point into my routing and spectrum allocation code. 
The code can create an arbitrary network based on user parameters to 
then have routing and spectrum allocation optimally solved.
"""
  
#ZHG
#2026.03.20
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from config.presets import contention, paper_dense, star, paper_exhaustive, paper_ring, two_source_three_users_custom, super_dense, custom_topology_preset
from pipeline.runner import run_pipeline

# ── Topology size ────────────────────────────────────────────────────────────
# NUM_USR  : total number of user nodes in the network (U1 … U<NUM_USR>)
# NUM_SRC  : total number of source nodes             (S1 … S<NUM_SRC>)
# NUM_LNKS : number of user-pair links to evaluate
#            must equal the number of entries in link_pairs below
NUM_USR  = 12
NUM_SRC  = 4
NUM_LNKS = 6

def main():
    #cfg = paper_exhaustive()
    #cfg = paper_ring()
    #cfg = paper_dense()
    #cfg = two_source_three_users_custom()
    #cfg = contention()
    #cfg = super_dense()
    #cfg = star()

    cfg = custom_topology_preset(

        # ── Network size ─────────────────────────────────────────────────────
        # These must match the constants defined above AND be consistent with
        # every other parameter below (see individual notes).
        num_usr=NUM_USR,
        num_src=NUM_SRC,
        num_lnks=NUM_LNKS,

        # ── Graph edges ──────────────────────────────────────────────────────
        # List every physical fibre connection in the network.
        # Format: ("NodeA", "NodeB")          — loss assigned randomly from cfg.loss_range
        #         ("NodeA", "NodeB", <dB>)    — explicit loss value in dB
        #
        # Node naming rules:
        #   Sources : S1, S2, … S<NUM_SRC>
        #   Users   : U1, U2, … U<NUM_USR>
        #
        # You can connect:
        #   Source → User   : the normal entanglement distribution edge
        #   Source → Source : a backbone link so sources can relay photons
        #   User   → User   : a direct user-to-user fibre (unusual but valid)
        #
        # Every node referenced here must exist given NUM_USR / NUM_SRC above.
        # Nodes that appear in link_pairs but have no path to any source will
        # produce 0 candidate paths and be skipped during evaluation.
        custom_edges=[
            # S1 cluster — S1 serves users U1-U4 directly
            ("S1", "U1"),
            ("S1", "U2"),
            ("S1", "U3"),
            ("S1", "U4"),

            # S2 cluster — S2 serves U3-U6; U3/U4 are shared with S1
            ("S2", "U3"),
            ("S2", "U4"),
            ("S2", "U5"),
            ("S2", "U6"),

            # S3 cluster — S3 serves U5-U8; U5/U6 are shared with S2
            ("S3", "U5"),
            ("S3", "U6"),
            ("S3", "U7"),
            ("S3", "U8"),

            # S4 cluster — S4 serves U7-U12; U7/U8 are shared with S3
            ("S4", "U7"),
            ("S4", "U8"),
            ("S4", "U9"),
            ("S4", "U10"),
            ("S4", "U11"),
            ("S4", "U12"),

            # Source backbone — allows multi-hop paths across the network
            # e.g. S1 can reach U12 via S1→S2→S3→S4→U12
            ("S1", "S2"),
            ("S2", "S3"),
            ("S3", "S4"),
        ],

        # ── Channels per source ───────────────────────────────────────────────
        # One integer per source, in order S1, S2, … S<NUM_SRC>.
        # Each value is the number of wavelength channels that source can emit.
        # More channels = more simultaneous entanglement pairs available.
        # Length must equal NUM_SRC.
        num_channels=[4, 4, 4, 4],

        # ── Fidelity requirements per link ────────────────────────────────────
        # Minimum acceptable entanglement fidelity for each link in link_pairs,
        # in the same order as link_pairs.  Range: (0.5, 1.0).
        # Higher values are stricter — fewer channel allocations will qualify.
        # Length must equal NUM_LNKS.
        fidelity_limit=[0.9] * NUM_LNKS,

        # ── Dark-count rates per user ─────────────────────────────────────────
        # Detector dark-count rate (counts/second) for each user node,
        # in order U1, U2, … U<NUM_USR>.
        # Lower values mean cleaner detectors and higher achievable fidelity.
        # Length must equal NUM_USR.
        dark_count_rate=[500.0] * NUM_USR,

        # ── Entanglement links (user pairs) ───────────────────────────────────
        # The specific user pairs you want to distribute entanglement between.
        # Format: [("Ua", "Ub"), ("Uc", "Ud"), …]
        #
        # Rules:
        #   • Length must equal NUM_LNKS.
        #   • Both users in each pair must exist (≤ NUM_USR).
        #   • Each pair needs a path through the graph above — if two users
        #     share no common source (directly or via the backbone) the pair
        #     will have 0 candidates and be skipped.
        #   • With require_disjoint_links=True no user may appear in more than
        #     one pair.  Set it to False to allow shared users across links.
        link_pairs=[
            ("U1",  "U12"),
            ("U2",  "U11"),
            ("U3",  "U10"),
            ("U4",  "U9"),
            ("U5",  "U8"),
            ("U6",  "U7"),
        ],

        # ── Disjoint-link constraint ──────────────────────────────────────────
        # True  : each user node may appear in at most one link (default).
        #         Ensures links don't compete for the same detector.
        # False : a user can appear in multiple links simultaneously.
        #         Useful when modelling shared-node or relay scenarios.
        require_disjoint_links=False,
    )

    result = run_pipeline(cfg)
    #print(result)

if __name__ == "__main__":
    main()