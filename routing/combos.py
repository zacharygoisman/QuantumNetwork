#routing/combos.py
"""
Functions to generate and evaluate path combinations for routing.
"""

#ZHG
#2026.03.24
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
import itertools
import heapq

def generate_combos(all_link_options, cfg):
    if cfg.use_best_first:
        yield from best_first_search(all_link_options, cfg)
    else:
        yield from brute_force(all_link_options, cfg)


def _get_combo(all_link_options, indices):
    return tuple(all_link_options[i][indices[i]] for i in range(len(all_link_options)))



def _combo_priority_from_indices(all_link_options, indices):
    combo = _get_combo(all_link_options, indices)

    combo_path_ub = sum(opt["path_ub"] for opt in combo)
    n_used_sources = len({opt["source"] for opt in combo})
    combo_total_loss = sum(opt["total_loss"] for opt in combo)

    return (
        combo_total_loss,
        -n_used_sources,
        -combo_path_ub,
        indices,
    )


def best_first_search(all_link_options, cfg):
    L = len(all_link_options)

    start = tuple(0 for _ in range(L))

    heap = [(_combo_priority_from_indices(all_link_options, start), start)]
    visited = {start}
    yielded = 0

    while heap:
        _, indices = heapq.heappop(heap)
        yield _get_combo(all_link_options, indices)
        yielded += 1

        if cfg.max_combos and yielded >= cfg.max_combos:
            break

        for i in range(L):
            new_indices = list(indices)
            new_indices[i] += 1

            if new_indices[i] >= len(all_link_options[i]):
                continue

            new_indices = tuple(new_indices)
            if new_indices in visited:
                continue

            visited.add(new_indices)
            heapq.heappush(
                heap,
                (_combo_priority_from_indices(all_link_options, new_indices), new_indices),
            )

def brute_force(all_link_options, cfg):
    all_indices = list(itertools.product(*(range(len(opts)) for opts in all_link_options)))

    all_indices.sort(
        key=lambda indices: _combo_priority_from_indices(all_link_options, indices)
    )

    count = 0
    for indices in all_indices:
        yield _get_combo(all_link_options, indices)
        count += 1

        if cfg.max_combos and count >= cfg.max_combos:
            break