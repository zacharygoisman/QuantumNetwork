#config/custom_topology.py
"""
Editable template for defining a custom network topology.

Instructions for editing values are shown below and to run this preset
you must enable the cfg = my_custom_network() in main.
"""

from config.presets import custom_topology_preset


# ── Topology size ────────────────────────────────────────────────────────────
# NUM_USR  : total number of user nodes in the network (U1 … U<NUM_USR>)
# NUM_SRC  : total number of source nodes             (S1 … S<NUM_SRC>)
# NUM_LNKS : number of user-pair links to evaluate
NUM_USR  = 4
NUM_SRC  = 2
NUM_LNKS = 2


def my_custom_network():
    """Custom network topology — edit this function to define your own.

    Returns a Config object ready for run_pipeline().
    """
    return custom_topology_preset(

        # ── Network size ─────────────────────────────────────────────────────
        # Must match the constants above and the lengths of the lists below.
        num_usr=NUM_USR,
        num_src=NUM_SRC,
        num_lnks=NUM_LNKS,

        # ── Graph edges ──────────────────────────────────────────────────────
        # Every physical fibre connection in the network.
        # Format:
        #   ("NodeA", "NodeB")           — loss drawn randomly from cfg.loss_range
        #   ("NodeA", "NodeB", loss_db)  — explicit loss in dB for this edge
        #
        # Node naming:
        #   Sources : S1, S2, …
        #   Users   : U1, U2, …
        #
        # You can connect Source→User (entanglement distribution) and
        # Source→Source (backbone relay).
        custom_edges=[
            # S1 serves U1 and U2
            ("S1", "U1"),
            ("S1", "U2"),

            # S2 serves U3 and U4
            ("S2", "U3"),
            ("S2", "U4"),

            # Backbone link so S1 ↔ S2 can relay photons across the network
            ("S1", "S2"),
        ],

        # ── Channels per source ───────────────────────────────────────────────
        # Number of wavelength channels per source, in order S1, S2, …
        # Length must equal NUM_SRC.
        num_channels=[2, 2],

        # ── Fidelity requirements per link ────────────────────────────────────
        # Minimum acceptable fidelity for each link in link_pairs, same order.
        # Range: (0.5, 1.0). Length must equal NUM_LNKS.
        fidelity_limit=[0.9] * NUM_LNKS,

        # ── Dark-count rates per user ─────────────────────────────────────────
        # Detector dark-count rate (counts/sec) per user, order U1, U2, …
        # Length must equal NUM_USR.
        dark_count_rate=[500.0] * NUM_USR,

        # ── Entanglement links (user pairs) ───────────────────────────────────
        # The user pairs you want to entangle.
        # Length must equal NUM_LNKS, and each pair needs a path through the graph.
        link_pairs=[
            ("U1", "U3"),
            ("U2", "U4"),
        ],

        # ── Disjoint-link constraint ──────────────────────────────────────────
        # True  : each user appears in at most one link (default).
        # False : a user may appear in multiple links (shared-node scenarios).
        require_disjoint_links=True,
    )
