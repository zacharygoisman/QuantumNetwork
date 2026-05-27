#routing/combos.py
"""
Functions to generate and evaluate path combinations for routing.
"""

#ZHG
#2026.03.24
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
import itertools
import heapq
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

def generate_combos(all_link_options, cfg):
    search_method = getattr(cfg, 'search_method', 'best_first' if cfg.use_best_first else 'brute_force')
    
    if search_method == 'beam_search':
        yield from beam_search(all_link_options, cfg)
    elif search_method == 'parallel_decision_tree':
        yield from parallel_decision_tree(all_link_options, cfg)
    elif cfg.use_best_first or search_method == 'best_first':
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


def beam_search(all_link_options, cfg):
    """
    Beam search: maintains top-k candidates at each level of the decision tree.
    Explores multiple promising paths in parallel while pruning less promising ones.
    """
    L = len(all_link_options)
    beam_width = getattr(cfg, 'beam_width', 5)  # Default beam width
    
    # Start with the first link choice
    beam = []
    for i in range(min(beam_width, len(all_link_options[0]))):
        indices = [i] + [0] * (L - 1)
        priority = _combo_priority_from_indices(all_link_options, indices)
        beam.append((priority, tuple(indices)))
    
    beam.sort()
    yielded = 0
    
    # Expand beam level by level
    for level in range(L):
        if level == L - 1:
            # Last level: yield all candidates in beam
            for priority, indices in beam:
                yield _get_combo(all_link_options, indices)
                yielded += 1
                if cfg.max_combos and yielded >= cfg.max_combos:
                    return
            break
        
        # Expand current beam to next level
        next_beam = []
        for priority, indices in beam:
            indices_list = list(indices)
            next_level = level + 1
            
            # Try all options for the next link
            for next_idx in range(len(all_link_options[next_level])):
                new_indices = indices_list.copy()
                new_indices[next_level] = next_idx
                new_priority = _combo_priority_from_indices(all_link_options, new_indices)
                next_beam.append((new_priority, tuple(new_indices)))
        
        # Keep only top beam_width candidates
        next_beam.sort()
        beam = next_beam[:beam_width]


def _expand_node_worker(args):
    """Worker function for parallel expansion of decision tree nodes."""
    all_link_options, indices, level, L = args
    
    if level >= L:
        return []
    
    expansions = []
    for i in range(level, L):
        new_indices = list(indices)
        new_indices[i] += 1
        
        if new_indices[i] >= len(all_link_options[i]):
            continue
        
        new_indices = tuple(new_indices)
        priority = _combo_priority_from_indices(all_link_options, new_indices)
        expansions.append((priority, new_indices))
    
    return expansions


def parallel_decision_tree(all_link_options, cfg):
    """
    Parallel decision tree search: uses multiple workers to explore the search space.
    Combines best-first search with parallel node expansion.
    """
    L = len(all_link_options)
    max_workers = getattr(cfg, 'max_workers', 4)  # Default number of workers
    use_processes = getattr(cfg, 'use_processes', False)  # Thread vs Process pool
    
    start = tuple(0 for _ in range(L))
    heap = [(_combo_priority_from_indices(all_link_options, start), start)]
    visited = {start}
    yielded = 0
    
    # Choose executor type
    ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    with ExecutorClass(max_workers=max_workers) as executor:
        while heap:
            _, indices = heapq.heappop(heap)
            yield _get_combo(all_link_options, indices)
            yielded += 1
            
            if cfg.max_combos and yielded >= cfg.max_combos:
                break
            
            # Prepare expansion tasks
            expansion_tasks = []
            for i in range(L):
                new_indices = list(indices)
                new_indices[i] += 1
                
                if new_indices[i] >= len(all_link_options[i]):
                    continue
                
                new_indices = tuple(new_indices)
                if new_indices not in visited:
                    expansion_tasks.append(new_indices)
            
            # Process expansions in parallel
            if expansion_tasks:
                # For thread pool, we can directly compute priorities
                for new_indices in expansion_tasks:
                    visited.add(new_indices)
                    priority = _combo_priority_from_indices(all_link_options, new_indices)
                    heapq.heappush(heap, (priority, new_indices))