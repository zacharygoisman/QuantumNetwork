#config/presets.py
"""
File to load preset configurations for different network topologies and settings.
"""

#ZHG
#2026.03.20
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
import numpy as np
from .config import Config

def paper_exhaustive():
    #Configuration for Figures 6 and 7 of "Efficient routing and spectrum allocation in arbitrary flex-grid entanglement networks"
    return Config(
        num_usr=8,
        num_src=3,
        num_lnks=4,
        num_channels= [2,2,2],
        fidelity_limit= [0.9407133998740564, 0.9085642634432197, 0.9344355106995621, 0.9070017406330423],#[np.random.uniform(0.90, 0.95) for _ in range(4)],
        dark_count_rate= [809.9685193109326, 237.9591005579212, 737.0439190875217, 691.8037211080392, 493.2091158888405, 813.8575660638049, 997.4666516393447, 866.7240775121699],#[np.random.uniform(100, 1000) for _ in range(8)],
        topology="ring",
        max_combos = 3, #exhaustive search set to None
        n_paths_per_leg = 4,
        random_seed = 1
    )

def paper_ring():
    #Configuration for Figure 8 of "Efficient routing and spectrum allocation in arbitrary flex-grid entanglement networks"
    return Config(
        num_usr=24,
        num_src=7,
        num_lnks=12,
        topology="ring",
        num_channels= [7,7,7,7,7,7,7],
        fidelity_limit= [0.9278049289484264, 0.9410293438393607, 0.9020729445567374, 0.9106374219758467, 0.9129526959829191, 0.9298316405965614, 0.9445800125182414, 0.9163775471428058, 0.9243459587214392, 0.9166085206889515, 0.9199542252091557, 0.9237397971790817],#[np.random.uniform(0.90, 0.95) for _ in range(12)],
        dark_count_rate= [872.1823768091353, 251.50285220867644, 978.6506777896305, 194.90791919710097, 415.7716104280857, 365.67639236380523, 434.69204920178504, 123.45019935702673, 956.7742326930493, 726.8534396284018, 533.5273249593395, 422.17768575525474, 810.1878634697338, 479.63884985152583, 627.9087281987123, 844.5766142692864, 474.2633557012324, 209.01041063885714, 187.93567471840117, 805.9399272561506, 689.3520693331785, 497.3060338139331, 646.3023646405952, 453.96088094617664],#[np.random.uniform(100, 1000) for _ in range(24)],
        max_combos = 1,
        n_paths_per_leg = 1,
        random_seed = 3,
        use_best_first = True,
    )

def paper_dense():
    #Configuration for Figure 9 of "Efficient routing and spectrum allocation in arbitrary flex-grid entanglement networks"
    return Config(
        num_usr=12,
        num_src=3,
        num_lnks=6,
        num_channels= [7, 7, 7],
        topology="dense",
        density= 0.15,
        fidelity_limit= [0.9398199004875115, 0.9335799370727665, 0.9453218374098463, 0.9054871178386069, 0.9029770692798971, 0.9103397555478387],#[np.random.uniform(0.90, 0.95) for _ in range(6)],
        dark_count_rate= [242.48973775874953, 607.416261888995, 798.1705870142911, 264.2816643599733, 687.9398242594576, 260.8033389904631, 952.6118693337143, 365.3253559752819, 451.50292448125356, 323.9246409797049, 406.61248760988224, 691.665750795401],#[np.random.uniform(100, 1000) for _ in range(12)],
        max_combos = 1,
        n_paths_per_leg = 4,
        random_seed = 3,
        use_best_first = True,
    )

def super_dense():
    #Configuration for Figure 9 of "Efficient routing and spectrum allocation in arbitrary flex-grid entanglement networks"
    return Config(
        num_usr=30,
        num_src=6,
        num_lnks=15,
        num_channels= [7, 7, 7, 7, 7, 7],
        topology="dense",
        density= 0.2,
        fidelity_limit= [0.9059764503908941, 0.9172337539992664, 0.9153401718175314, 0.9323339577474602, 0.9124287382839245, 0.9354830512176966, 0.9423647573905697, 0.9439211314831565, 0.923712026059091, 0.9415338717748518, 0.9130574672173623, 0.9116143095309597, 0.9489777035702667, 0.9497239578861959, 0.9422965100473697],#[np.random.uniform(0.90, 0.95) for _ in range(6)],
        dark_count_rate= [905.9613754481101, 242.39445617365465, 501.04962351338776, 487.5065368040025, 358.5120130796217, 860.0012044319201, 220.23413129689305, 127.09204277338496, 475.6415550368487, 508.21898546013495, 872.8080838127084, 983.7689230947295, 970.1831779595382, 170.25280962399853, 797.2473763782796, 149.2424263150669, 500.8220829668604, 216.23330027723895, 129.74661428509023, 222.4574357089607, 455.3484548702772, 618.1015383824848, 647.700836558703, 412.1143157930114, 169.07573478073283, 376.72988184662665, 705.3457271813237, 641.4215496253936, 988.0131785585, 607.3069289606512],#[np.random.uniform(100, 1000) for _ in range(12)],
        max_combos = 1,
        n_paths_per_leg = 4,
        random_seed = 3,
        use_best_first = True,
    )

def star():
    #Generic star topology"
    return Config(
        num_usr=8,
        num_src=1,
        num_lnks=4,
        num_channels= [8],
        fidelity_limit= [0.7,0.7,0.7,0.7],
        dark_count_rate= [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0],
        topology="star",
    )

def custom_topology_preset(
    *,
    num_usr,
    num_src,
    num_lnks,
    custom_edges,
    num_channels,
    fidelity_limit,
    dark_count_rate,
    custom_source_channels=None,
    link_pairs=None,
    require_disjoint_links=True,
    **kwargs,
):
    """
    Convenience preset for explicitly specified topologies.

    custom_edges supports either:
        [("S0", "U0"), ("S0", "U1", 1.2), ...]
    If a loss is omitted, normal random loss assignment still applies via cfg.loss_range.
    """
    # Allow callers to pass explicit per-edge loss values via kwargs.edge_loss_values
    edge_loss_values = kwargs.pop("edge_loss_values", None)
    custom_edges_list = list(custom_edges)
    if edge_loss_values is not None:
        if len(edge_loss_values) != len(custom_edges_list):
            raise ValueError("edge_loss_values must match length of custom_edges")
        new_edges = []
        for e, loss in zip(custom_edges_list, edge_loss_values):
            if len(e) == 3:
                # preserve explicit loss in edge tuple unless caller wants override
                new_edges.append((e[0], e[1], float(loss)))
            else:
                new_edges.append((e[0], e[1], float(loss)))
        custom_edges_list = new_edges

    return Config(
        num_usr=num_usr,
        num_src=num_src,
        num_lnks=num_lnks,
        topology="custom",
        custom_edges=custom_edges_list,
        num_channels=list(num_channels),
        fidelity_limit=list(fidelity_limit),
        dark_count_rate=list(dark_count_rate),
        custom_source_channels=dict(custom_source_channels or {}),
        link_pairs=list(link_pairs) if link_pairs is not None else None,
        require_disjoint_links=require_disjoint_links,
        **kwargs,
    )


def two_source_three_users_custom(edge_losses, num_channels, fidelity_limit, dark_count_rate):
    """Preset: 2 sources, 3 users.

    - S0 connected to U0, U1, U2
    - S1 connected to U0, U1
    - users are not connected to each other

    Use this by setting `cfg = two_source_three_users_custom()` in `main.py`.
    """
    # custom_edges: (node_a, node_b) [, loss]
    base_edges = [
        ("S1", "U1"),
        ("S1", "U2"),
        ("S1", "U3"),
        ("S2", "U1"),
        ("S2", "U2"),
    ]

    edge_losses = [10, 10, 10, 2, 2]

    if edge_losses is not None:
        if len(edge_losses) != len(base_edges):
            raise ValueError("edge_losses must match number of base edges")
        edges = [(u, v, float(loss)) for (u, v), loss in zip(base_edges, edge_losses)]
    else:
        edges = base_edges

    return custom_topology_preset(
        num_usr=3,
        num_src=2,
        num_lnks=3,
        custom_edges=edges,
        num_channels=[3, 3],
        fidelity_limit=[0.9, 0.9, 0.9],
        dark_count_rate=[500.0, 500.0, 500.0],
        random_seed=7,
    )


def contention():
    """Preset: 2 sources, 4 users.

    - S0 connected to U0, U1, U2
    - S1 connected to U0, U1
    - users are not connected to each other

    Use this by setting `cfg = two_source_three_users_custom()` in `main.py`.
    """
    # custom_edges: (node_a, node_b) [, loss]
    base_edges = [
        ("S1", "U1"),
        ("S1", "S2"),
        ("U1", "U3"),
        ("S2", "U1"),
        ("U1", "U4"),
        ("U1", "U2"),
    ]

    edge_losses = [10, 1, 1, 1, 1, 1]

    if edge_losses is not None:
        if len(edge_losses) != len(base_edges):
            raise ValueError("edge_losses must match number of base edges")
        edges = [(u, v, float(loss)) for (u, v), loss in zip(base_edges, edge_losses)]
    else:
        edges = base_edges

    return custom_topology_preset(
        num_usr=4,
        num_src=2,
        num_lnks=2,
        custom_edges=edges,
        num_channels=[1, 1],
        fidelity_limit=[0.9320997273968529, 0.9050930112698425],
        dark_count_rate=[685.3686678691192, 890.5016916171851, 564.1122800179337, 342.4765797067348],
        random_seed=7,
        max_combos=21,
        n_paths_per_leg=4,
    )

def simple_contention():
    """Preset: 2 sources, 4 users.

    - S0 connected to U0, U1, U2
    - S1 connected to U0, U1
    - users are not connected to each other

    Use this by setting `cfg = two_source_three_users_custom()` in `main.py`.
    """
    # custom_edges: (node_a, node_b) [, loss]
    base_edges = [
        ("S1", "U1"),
        ("S1", "S2"),
        ("U1", "U3"),
        ("S2", "U1"),
        ("U1", "U4"),
        ("U1", "U2"),
    ]
    

    edge_losses = [10, 0, 0, 0, 0, 0]

    if edge_losses is not None:
        if len(edge_losses) != len(base_edges):
            raise ValueError("edge_losses must match number of base edges")
        edges = [(u, v, float(loss)) for (u, v), loss in zip(base_edges, edge_losses)]
    else:
        edges = base_edges

    return custom_topology_preset(
        num_usr=4,
        num_src=2,
        num_lnks=2,
        custom_edges=edges,
        num_channels=[1, 1],
        fidelity_limit=[0.5, 0.75],
        dark_count_rate=[0, 0, 0, 0],
        random_seed=1,
        max_combos=21,
        n_paths_per_leg=4,
        tau=1.0,
        link_pairs=[
            ("U1", "U3"),
            ("U2", "U4"),
        ]
    )

def manhattan_ilec():
    """
    Manhattan ILEC topology adapted from Bali et al.'s Table I.

    Original ILEC labels:
        A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q

    Mapping to this code:
        Sources:
            S1 = M
            S2 = N
            S3 = O

        Users:
            U1  = A
            U2  = B
            U3  = C
            U4  = D
            U5  = E
            U6  = F
            U7  = G
            U8  = H
            U9  = I
            U10 = J
            U11 = K
            U12 = L
            U13 = P
            U14 = Q

    Distances are the as-the-crow-flies values in km reported by Bali et al.
    We convert distance to edge loss using an effective metro loss coefficient.
    This is an adaptation to our undirected source/user graph model, not an
    exact reproduction of Bali et al.'s directed internal WSS graph.
    """

    # Effective loss coefficient for metro fiber.
    # Tune this if the rates/fidelities are too easy or too hard.
    fiber_loss_db_per_km = 1

    source_map = {
        "B": "S1",
        "M": "S2",
        "N": "S3",
    }

    user_map = {
        "A": "U1",
        "C": "U2",
        "D": "U3",
        "E": "U4",
        "F": "U5",
        "G": "U6",
        "H": "U7",
        "I": "U8",
        "J": "U9",
        "K": "U10",
        "L": "U11",
        "O": "U12",
        "P": "U13",
        "Q": "U14",
    }

    node_map = {**source_map, **user_map}

    # Upper-triangular distance matrix from Bali et al. Table I.
    # Entries omitted in the table are represented by None and skipped.
    distances_km = {
        ("A", "B"): 0.304,
        ("A", "C"): 1.184,
        ("A", "D"): 2.032,
        ("A", "E"): 3.744,
        ("A", "F"): 5.200,
        ("A", "G"): 4.352,
        ("A", "H"): 5.776,
        ("A", "I"): 6.096,
        ("A", "J"): 5.840,
        ("A", "K"): 7.232,
        ("A", "L"): 7.040,
        ("A", "M"): 8.800,
        ("A", "N"): 9.120,
        ("A", "O"): 10.688,

        ("B", "C"): 0.912,
        ("B", "D"): 1.712,
        ("B", "E"): 3.488,
        ("B", "F"): 5.056,
        ("B", "G"): 4.048,
        ("B", "H"): 5.488,
        ("B", "I"): 5.936,
        ("B", "J"): 5.296,
        ("B", "K"): 6.848,
        ("B", "L"): 6.656,
        ("B", "M"): 8.496,
        ("B", "N"): 8.816,
        ("B", "O"): 10.320,

        ("C", "D"): 2.336,
        ("C", "E"): 2.080,
        ("C", "F"): 3.328,
        ("C", "G"): 2.304,
        ("C", "H"): 3.728,
        ("C", "I"): 4.192,
        ("C", "J"): 3.904,
        ("C", "K"): 5.296,
        ("C", "L"): 5.040,
        ("C", "M"): 6.752,
        ("C", "N"): 7.216,
        ("C", "O"): 9.664,

        ("D", "E"): 2.224,
        ("D", "F"): 3.360,
        ("D", "G"): 2.368,
        ("D", "H"): 3.728,
        ("D", "I"): 2.192,
        ("D", "J"): 4.000,
        ("D", "K"): 5.392,
        ("D", "L"): 4.992,
        ("D", "M"): 6.848,
        ("D", "N"): 7.216,
        ("D", "O"): 8.768,

        ("E", "F"): 1.440,
        ("E", "G"): 1.600,
        ("E", "H"): 2.448,
        ("E", "I"): 2.624,
        ("E", "J"): 1.968,
        ("E", "K"): 3.472,
        ("E", "L"): 3.728,
        ("E", "M"): 5.280,
        ("E", "N"): 5.312,
        ("E", "O"): 6.880,

        ("F", "G"): 1.696,
        ("F", "H"): 1.536,
        ("F", "I"): 1.360,
        ("F", "J"): 0.544,
        ("F", "K"): 2.000,
        ("F", "L"): 2.528,
        ("F", "M"): 4.000,
        ("F", "N"): 3.872,
        ("F", "O"): 5.456,

        ("G", "H"): 1.408,
        ("G", "I"): 1.888,
        ("G", "J"): 2.112,
        ("G", "K"): 3.312,
        ("G", "L"): 2.624,
        ("G", "M"): 4.496,
        ("G", "N"): 5.056,
        ("G", "O"): 6.496,

        ("H", "I"): 0.624,
        ("H", "J"): 1.408,
        ("H", "K"): 2.176,
        ("H", "L"): 1.280,
        ("H", "M"): 3.040,
        ("H", "N"): 3.696,
        ("H", "O"): 5.152,

        ("I", "J"): 1.120,
        ("I", "K"): 1.552,
        ("I", "L"): 1.264,
        ("I", "M"): 2.704,
        ("I", "N"): 3.120,
        ("I", "O"): 4.576,

        ("J", "K"): 1.376,
        ("J", "L"): 2.288,
        ("J", "M"): 3.424,
        ("J", "N"): 3.296,
        ("J", "O"): 4.832,

        ("K", "L"): 2.208,
        ("K", "M"): 2.560,
        ("K", "N"): 1.920,
        ("K", "O"): 3.472,

        ("L", "M"): 1.872,
        ("L", "N"): 3.200,
        ("L", "O"): 4.256,

        ("M", "N"): 4.800,
        ("M", "O"): 6.368,
        ("M", "P"): 2.960,
        ("M", "Q"): 6.096,

        ("N", "O"): 1.536,
        ("N", "Q"): 5.856,

        ("O", "Q"): 4.368,

        ("P", "Q"): 3.040,
    }

    edges = []
    for (a, b), distance_km in distances_km.items():
        u = node_map[a]
        v = node_map[b]
        loss_db = fiber_loss_db_per_km * distance_km
        edges.append((u, v, float(loss_db)))

    return custom_topology_preset(
        num_usr=14,
        num_src=3,
        num_lnks=7,
        custom_edges=edges,
        topology_name="manhattan",

        # Similar scale to your current dense example.
        num_channels=[15, 15, 15],

        # Fixed link pairs so the figure/result are reproducible.
        # These pair geographically separated / topologically separated nodes
        # under the A-Q ordering while avoiding source nodes M, N, O.
        link_pairs=[
            ("U1", "U14"),   # A-Q
            ("U2", "U13"),   # B-P
            ("U3", "U12"),   # C-L
            ("U4", "U11"),   # D-K
            ("U5", "U10"),   # E-J
            ("U6", "U9"),    # F-I
        ],

        fidelity_limit=[0.90428246, 0.91184053, 0.94006372, 0.9291081 , 0.90470643, 0.92165635, 0.92395256],

        dark_count_rate=[177.08425043, 313.12945594, 821.14701869, 623.94583246,
            184.71577802, 489.81424621, 531.14616833, 243.76502317,
            761.11943627, 202.30481793, 452.10537145, 565.06616436,
            487.56521837, 628.11871429],

        random_seed=3,
        max_combos=1,
        n_paths_per_leg=4,
        use_best_first=True,
        require_disjoint_links=True,
    )