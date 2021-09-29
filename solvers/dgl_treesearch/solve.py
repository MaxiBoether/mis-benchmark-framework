import dgl
import time
import json
import tempfile
import pickle
import shutil
import gc

import numpy as np
from logzero import logger
from pathlib import Path

from treesearch import tree_search_wrapper
from utils import find_module

import torch.multiprocessing as mp
from torch.multiprocessing import Lock

def _run_single_threaded_treesearch(g, time_budget, self_loop, max_prob_maps, model_prob_maps, cuda_devs, weight_file, reduction, local_search, queue_pruning, noise_as_prob_maps, weighted_queue_pop):
    start_time = time.time() 
    best_solution, best_solution_vertices, best_solution_weight, total_solutions, _, best_solution_time, best_solution_process_time, _ = tree_search_wrapper(0, 1, None, None, weight_file, g, time_budget=time_budget, self_loop=self_loop, max_prob_maps=max_prob_maps, model_prob_maps=model_prob_maps, cuda_devs=cuda_devs, reduction=reduction, local_search=local_search, queue_pruning=queue_pruning, noise_as_prob_maps=noise_as_prob_maps, weighted_queue_pop=weighted_queue_pop)
    end_time = time.time()
    total_time = end_time - start_time

    return best_solution, best_solution_vertices, best_solution_weight, total_solutions, total_time, best_solution_time, best_solution_process_time

def _run_parallel_treesearch(g, threadcount, time_budget, self_loop, max_prob_maps, model_prob_maps, cuda_devs, weight_file, reduction, local_search, queue_pruning, noise_as_prob_maps, weighted_queue_pop):

    ### Build start results for threads single threadedly ###
    start_time = time.time()
    init_best_solution, init_best_solution_vertices, init_best_solution_weight, init_total_solutions, start_results, best_solution_time, best_solution_process_time, opt_found = tree_search_wrapper(0, 1, None, None, weight_file, g, solution_budget=threadcount, self_loop=self_loop, max_prob_maps=max_prob_maps, model_prob_maps=model_prob_maps, cuda_devs=cuda_devs, reduction=reduction, local_search=local_search, queue_pruning=queue_pruning, noise_as_prob_maps=noise_as_prob_maps)
    return_list = []
    # Initialization might already find results for simple graphs, so we need to respect that
    return_list.append({
            "total_solutions": init_total_solutions,
            "mis_vertices": init_best_solution_vertices if init_best_solution is not None else None,
            "mis_weight": init_best_solution_weight if init_best_solution is not None else None,
            "solution": init_best_solution,
            "solution_time": best_solution_time if init_best_solution is not None else None,
            "solution_process_time": best_solution_process_time if init_best_solution is not None else None
        })

    if not opt_found:
        ### Start parallel processing ###
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass # we can only do this once

        lock = Lock()
        pickle_path = Path(tempfile.mkdtemp())
        optimum_found = mp.Value('i', False)
        
        time_used = time.time() - start_time
        time_budget = time_budget - time_used
        processes = [mp.Process(target=tree_search_wrapper, args=(pnum, threadcount, start_results[pnum:pnum+1], lock, weight_file, g, pickle_path, time_budget, None, self_loop, max_prob_maps, model_prob_maps, cuda_devs, reduction, local_search, queue_pruning, noise_as_prob_maps, weighted_queue_pop, optimum_found)) for pnum in range(threadcount)]

        for p in processes:
            p.start()
        
        for p in processes:
            p.join()
        
        for p in processes:
            if p.exitcode != 0:
                logger.error(f"One thread exited with an non-zero exitcode: {p.exitcode}")

        ### Collect Results ###
        end_time = time.time()
        total_time = end_time - start_time
        
        for f_path in pickle_path.rglob("*.pickle"):
            logger.debug(f"Reading file {str(f_path)} into result list.")
            with open(f_path, 'rb') as f:
                result = pickle.load(f)
            return_list.append(result)

        if len(return_list) - 1 != threadcount:
            logger.error(f"Read {len(return_list) - 1} results, but we had {threadcount} threads (+ init). Something is off!")

        logger.debug(f"Deleting temporary directory")
        shutil.rmtree(pickle_path)
        logger.debug("Deleted, closing processes.")

        for p in processes:
            p.close()

        logger.debug("Deleting processes and start results")
        del processes[:]
        del processes
        logger.debug("Deleted.")

    else:
        end_time = time.time()
        total_time = end_time - start_time

        logger.info("Found best solution in initialization, not spawning any processes.")

    ### Find best MWIS over all threads ###

    total_elements = np.sum([result["total_solutions"] for result in return_list])

    best_solution_weight = None
    best_solution_vertices = None
    best_solution = None
    best_solution_time = None
    best_solution_process_time = None

    for result in return_list:
        if "solution" in result.keys() \
            and result["solution"] is not None \
            and (
                (best_solution is None) or
                (best_solution_weight < result["mis_weight"])
            ):
            best_solution = result["solution"]
            best_solution_vertices = result["mis_vertices"]
            best_solution_weight = result["mis_weight"]
            best_solution_time = result["solution_time"]
            best_solution_process_time = result["solution_process_time"]

    if start_results is not None and len(start_results) > 0:
        logger.debug("Deleting start results")
        del start_results[:]
        del start_results

    return best_solution, best_solution_vertices, best_solution_weight, total_elements, total_time, best_solution_time, best_solution_process_time

def solve(args):
    self_loop = args.self_loops
    threadcount = args.num_threads
    max_prob_maps = args.max_prob_maps
    model_prob_maps = args.model_prob_maps
    cuda_devs = args.cuda_devices
    time_budget = args.time_limit
    reduction = args.reduction
    local_search = args.local_search
    queue_pruning = args.queue_pruning
    noise_as_prob_maps = args.noise_as_prob_maps
    weighted_queue_pop = args.weighted_queue_pop

    if not args.pretrained_weights:
        raise ValueError("--pretrained_weights flag is required for solving! Exiting.")

    # initialize reduce lib
    if reduction or local_search:
        from reducelib.reducelib import reducelib
        rdlib = reducelib() # calls cmake if necessary
        del rdlib

    weight_file = args.pretrained_weights
    dgl_graphs = dgl.load_graphs(str(args.input / f"graphs_{'weighted' if args.weighted else 'unweighted'}.dgl"))[0]
    with open(str(args.input / f"graph_names.json"), "r") as f:
        graph_names = json.load(f)

    results = {}
    for idx,g in enumerate(dgl_graphs):
        logger.info(f"Solving graph {idx+1}/{len(dgl_graphs)} ({graph_names[idx]})")

        if self_loop:
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
        else:
            g = dgl.remove_self_loop(g)

        if threadcount > 1:
            best_solution, best_solution_vertices, best_solution_weight, total_solutions, total_time, best_solution_time, best_solution_process_time = _run_parallel_treesearch(g, threadcount, time_budget, self_loop, max_prob_maps, model_prob_maps, cuda_devs, weight_file, reduction, local_search, queue_pruning, noise_as_prob_maps, weighted_queue_pop)
        else:
            best_solution, best_solution_vertices, best_solution_weight, total_solutions, total_time, best_solution_time, best_solution_process_time = _run_single_threaded_treesearch(g, time_budget, self_loop, max_prob_maps, model_prob_maps, cuda_devs, weight_file, reduction, local_search, queue_pruning, noise_as_prob_maps, weighted_queue_pop)

        results[graph_names[idx]] = {
            "total_solutions": int(total_solutions),
            "total_time": total_time,
        }

        logger.info(f"Iterated through {total_solutions} in {total_time} seconds ({float(total_solutions)/float(total_time):.2f} e/s) 🚀🚀")
        if best_solution is not None:
            logger.info(f"Found MWIS: n={best_solution_vertices}, w={best_solution_weight} ✔️✔️")
            results[graph_names[idx]]["mwis_found"] = True
            results[graph_names[idx]]["mwis_vertices"] = int(best_solution_vertices)
            results[graph_names[idx]]["mwis_weight"] = float(best_solution_weight)
            results[graph_names[idx]]["time_to_find_mwis"] = best_solution_time
            results[graph_names[idx]]["process_time_to_find_mwis"] = best_solution_process_time
            results[graph_names[idx]]["mwis"] = np.ravel(best_solution).tolist()
        else:
            logger.info("Did not find any MWIS 😭😭")
            results[graph_names[idx]]["mwis_found"] = False

        with open(args.output / "results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, sort_keys = True, indent=4)

        logger.debug("Calling the garbage collector.")
        del g
        gc.collect()

    logger.info("Done with all graphs, exiting.")

