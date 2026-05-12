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
        n_paths_per_leg = 4,
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
    """Preset: 2 sources, 3 users.

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

    edge_losses = [10 , 1, 1, 1, 1, 1]

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
        max_combos=20,
        n_paths_per_leg=4,
    )

