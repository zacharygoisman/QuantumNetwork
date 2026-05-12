#config/config.py
"""
Base configuration class for network settings and presents
"""

#ZHG
#2026.03.20
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    #network
    num_usr: int = 3            #Number of users in the network
    num_src: int = 2            #Number of source nodes in the network
    num_lnks: int = 2           #Number of links in the network
    num_edg: int = 10000        #Number of edges in the network (used for random topologies, ignored for structured topologies)
    topology: str = "dense"     #Network topology type (e.g., "dense", "ring", "star")
    density: float = 0.1          #Density of the network (used for random topologies, between 0 and 1)
    loss_range: tuple = (1,10) #Range of link losses (min, max) for random assignment (in dB)
    num_channels: list = field(default_factory=lambda: [5, 5])         #Number of available channels per source (list of ints, length should match num_src)
    #TODO: Completely custom topology
    random_seed: int = 41         #Random seed for reproducibility of random topologies and losses

    # custom network controls
    custom_edges: list[tuple | tuple] = field(default_factory=list)
    custom_source_channels: dict[str, list[int]] = field(default_factory=dict)
    link_pairs: list[tuple[str, str]] | None = None
    require_disjoint_links: bool = True

    #routing
    routing_algorithm: str = "yen"      #Routing algorithm to use (e.g., "yen" for Yen's K-shortest path algorithm)
    n_paths_per_leg: int = 4            #Traditionally seen as K in K-shortest path algorithms. This is the number of paths to consider per leg of the route.
    combo_limit_per_link: int = 10      #Maximum number of path combinations to consider per link (pruning parameter to limit search space)
    use_best_first: bool = True              #Whether to use a best-first search strategy to explore path combinations (instead of brute-force). This can help find good solutions faster, but may miss the optimal solution if the heuristic is not perfect.
    upper_bound_sort: bool = False                           #TODO Make more and update this. Whether to sort path options by their upper bound (instead of total loss) when generating combos. This can help improve fidelity by prioritizing paths with higher potential utility, but may require more careful tuning of the upper bound calculation to ensure good performance.

    #path diversity
    use_diverse_paths: bool = False     #Whether to enforce path diversity (e.g., no shared edges) when generating combos. This can help improve fidelity by reducing correlated losses, but may limit the number of feasible combos.
    diversity_threshold: float = 0.7    #Threshold for path overlap to consider two paths as "diverse". This is a value between 0 and 1, where 0 means completely disjoint paths and 1 means identical paths. For example, if set to 0.7, then two paths that share more than 70% of their edges would be considered non-diverse.
    diversity_factor: int = 3           #how many extra paths to sample

    #physics
    fidelity_limit: list = field(default_factory=lambda: [0.9, 0.95])    #Fidelity limit for each link (list of floats, length should match num_lnks)
    tau: float = 1e-9                    #Coincidence window for photon detection (in seconds)
    dark_count_rate: list = field(default_factory=lambda: [100.0, 3500.0, 1000.0])

    #runtime
    use_upper_bound: bool = True     #Whether to use an upper bound on the utility of path combinations to prune the search space during evaluation
    max_combos: int = 1              #Maximum number of path combinations to evaluate (None for no limit)
    verbose: bool = True               #Whether to print verbose output during execution
    parallel: bool = True
    parallel_workers: int | None = None
    parallel_chunk_size: int = 8

    #output
    output_directory: Path = Path("outputs") #output directory for results and logs


    def copy_with(self, **kwargs):
        """
        Create a copy of the current Config instance with specified fields updated. Example: cfg = paper_dense().copy_with(num_usr=20, verbose=True)
        """
        data = self.__dict__.copy()
        data.update(kwargs)
        return Config(**data)
    
    def __post_init__(self):
        """
        Constraints and validation for configuration parameters. This method is called automatically after the dataclass is initialized.
        """
        if self.num_usr <= 0:
            raise ValueError("num_usr must be positive")
        if self.num_src <= 0:
            raise ValueError("num_src must be positive")
        if self.num_lnks <= 0:
            raise ValueError("num_lnks must be positive")
        if len(self.num_channels) != self.num_src:
            raise ValueError("num_channels list must correspond to num_src")
        if len(self.fidelity_limit) != self.num_lnks:
            raise ValueError("fidelity_limit list must correspond to num_lnks")
        if len(self.dark_count_rate) != self.num_usr:
            raise ValueError("dark_count_rate list must correspond to num_usr")
        if self.parallel_chunk_size <= 0:
            raise ValueError("parallel_chunk_size must be positive")
        if self.custom_source_channels:
            # Expect source keys to use 1-based labels (S1..S<num_src>) to match network/node naming
            expected = {f"S{i+1}" for i in range(self.num_src)}
            missing = expected.difference(self.custom_source_channels.keys())
            if missing:
                raise ValueError(
                    "custom_source_channels must include every source label: "
                    + ", ".join(sorted(missing))
                )
