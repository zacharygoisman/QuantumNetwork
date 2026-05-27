#pipeline/evaluator.py
"""
Evaluates combos by running them through the allocation step and computing utility.
"""

#ZHG
#2026.03.24
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
import time

from analysis.upper_bounds import combo_link_upper_bound, combo_path_upper_bound
from allocation.allocator import allocate_combo
from scheduling.scheduler import check_interference

def _batched(iterable, size):
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            break
        yield batch



def _evaluate_combo_worker(args):
    combo, network, sources, cfg = args
    return evaluate_combo(combo, network, sources, cfg)

def evaluate_stream(combo_stream, network, sources, cfg):
    """Runs through all combos and evaluates them, returning a list of results."""
    if cfg.parallel and not cfg.use_upper_bound:
        return _evaluate_stream_parallel(combo_stream, network, sources, cfg)
    results = []
    best_utility = float("-inf")
    combos_seen = 0
    milestone = 100
    t_start = time.perf_counter()
    
    # Early termination: if we find a solution within X% of theoretical upper bound, stop
    early_term_threshold = 0.95  # 95% of upper bound
    theoretical_ub = None
    
    for combo in combo_stream:
        combo_idx = combos_seen
        combos_seen += 1
        if cfg.use_upper_bound:
            ub = combo_path_upper_bound(combo)
            
            # Track theoretical upper bound from first combo
            if theoretical_ub is None:
                theoretical_ub = combo_link_upper_bound(combo)
            
            if ub < best_utility:
                results.append({
                    "combo_idx": combo_idx,
                    "valid": False,
                    "reason": "ub_pruned",
                    "utility": None,
                    "combo": combo,
                    "combo_path_ub": ub,
                    "combo_link_ub": combo_link_upper_bound(combo),
                    "allocation": {},
                    "assignment": {},
                    "timing": {
                        "ub_s": 0.0,
                        "alloc_s": 0.0,
                        "sched_s": 0.0,
                        "total_s": 0.0,
                    },
                })
                if combos_seen % milestone == 0:
                    print(f"Processed {combos_seen} combinations")
                # record elapsed time even for pruned combos
                results[-1]["elapsed_s"] = time.perf_counter() - t_start
                continue  # skip if cannot reach best utility

        result = evaluate_combo(combo, network, sources, cfg)
        result["combo_idx"] = combo_idx
        # record elapsed wall time from start of evaluation stream until this result returned
        result["elapsed_s"] = time.perf_counter() - t_start
        if result.get("valid"):
            util = result["utility"]
            best_utility = max(best_utility, util)
            
            # Early termination check
            if theoretical_ub is not None and best_utility >= early_term_threshold * theoretical_ub:
                print(f"Early termination: Found solution at {100*best_utility/theoretical_ub:.1f}% of theoretical upper bound")
                results.append(result)
                break

        results.append(result)
        if combos_seen % milestone == 0:
            print(f"Processed {combos_seen} combinations")
            avg_alloc = sum(r.get("timing", {}).get("alloc_s", 0.0) for r in results) / len(results)
            avg_sched = sum(r.get("timing", {}).get("sched_s", 0.0) for r in results) / len(results)
            avg_total = sum(r.get("timing", {}).get("total_s", 0.0) for r in results) / len(results)
            print(
                f"Ran {combos_seen} combos | "
                f"avg alloc {avg_alloc:.3f}s | "
                f"avg sched {avg_sched:.3f}s | "
                f"avg total {avg_total:.3f}s"
            )

        if cfg.max_combos and len(results) >= cfg.max_combos:
            break

    return results

def _evaluate_stream_parallel(combo_stream, network, sources, cfg):
    results = []
    n_seen = 0
    workers = cfg.parallel_workers
    batch_size = max(cfg.parallel_chunk_size, workers or 1)

    t_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for batch in _batched(combo_stream, batch_size):
            remaining = None if cfg.max_combos is None else max(cfg.max_combos - n_seen, 0)
            if remaining == 0:
                break
            if remaining is not None:
                batch = batch[:remaining]
            if not batch:
                break

            tasks = [(combo, network, sources, cfg) for combo in batch]
            batch_results = list(executor.map(_evaluate_combo_worker, tasks))
            # assign combo_idx and elapsed timestamp for each returned result
            batch_start_idx = n_seen
            elapsed_now = time.perf_counter() - t_start
            for i, br in enumerate(batch_results):
                try:
                    br["combo_idx"] = batch_start_idx + i
                except Exception:
                    pass
                try:
                    br["elapsed_s"] = elapsed_now
                except Exception:
                    pass

            results.extend(batch_results)
            n_seen += len(batch_results)

            if cfg.max_combos and n_seen >= cfg.max_combos:
                break

    return results

def evaluate_combo(combo, network, sources, cfg):
    t0 = time.perf_counter()

    combo_path_ub = combo_path_upper_bound(combo) if cfg.use_upper_bound else None
    combo_link_ub = combo_link_upper_bound(combo) if cfg.use_upper_bound else None

    t1 = time.perf_counter()
    alloc_result = allocate_combo(combo, network, sources, cfg)
    t2 = time.perf_counter()

    if not alloc_result["success"]:
        return {
            "valid": False,
            "reason": "allocation_failed",
            "combo": combo,
            "combo_path_ub": combo_path_ub,
            "combo_link_ub": combo_link_ub,
            "allocation": {},
            "assignment": {},
            "timing": {
                "ub_s": t1 - t0,
                "alloc_s": t2 - t1,
                "sched_s": 0.0,
                "total_s": t2 - t0,
            },
        }

    sched_result = check_interference(combo, alloc_result, sources)
    t3 = time.perf_counter()
    if not sched_result["success"]:
        combo_out = []
        for i, opt in enumerate(combo):
            opt_copy = dict(opt)
            opt_copy["allocation"] = dict(alloc_result["allocation"].get(id(opt), {}))
            combo_out.append(opt_copy)
        return {
            "valid": False,
            "reason": "contention_failed",
            "combo": combo,
            "combo_path_ub": combo_path_ub,
            "combo_link_ub": combo_link_ub,
            "allocation": alloc_result["allocation"],
            "assignment": {},
            "timing": {
                "ub_s": t1 - t0,
                "alloc_s": t2 - t1,
                "sched_s": t3 - t2,
                "total_s": t3 - t0,
            },
        }

    combo_out = []
    allocation_out = {}
    assignment_out = {}
    for i, opt in enumerate(combo):
        opt_copy = dict(opt)
        alloc_copy = dict(alloc_result["allocation"].get(id(opt), {}))
        assign_copy = dict(sched_result["assignment"].get(i, {}))
        opt_copy["allocation"] = alloc_copy
        opt_copy["assignment"] = assign_copy
        combo_out.append(opt_copy)
        allocation_out[i] = alloc_copy
        assignment_out[i] = assign_copy

    return {
        "valid": True,
        "reason": "ok",
        "utility": alloc_result["utility"],
        "combo_path_ub": combo_path_ub,
        "combo_link_ub": combo_link_ub,
        "allocation": allocation_out,
        "assignment": assignment_out,
        "combo": tuple(combo_out),
        "timing": {
            "ub_s": t1 - t0,
            "alloc_s": t2 - t1,
            "sched_s": t3 - t2,
            "total_s": t3 - t0,
        },
    }


def select_best(results):
    valid = [r for r in results if r.get("valid")]
    if not valid:
        return None
    return max(valid, key=lambda r: r["utility"])