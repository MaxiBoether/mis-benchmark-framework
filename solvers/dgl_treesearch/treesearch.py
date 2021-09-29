import numpy as np
import time
import torch
import dgl
import gc
import sys
import copy
import pickle

from reducelib.reducelib import reducelib
from utils import _load_model, _locked_log

class _TreeSearch():
    def __init__(self, pid, num_threads, queue, lock, weight_file, pickle_path, g, time_budget, solution_budget, self_loop, max_prob_maps, model_prob_maps, cuda_devs, reduction, local_search, queue_pruning, noise_as_prob_maps, weighted_queue_pop, optimum_found):
        self.queue = queue
        self.num_threads = num_threads
        self.queue_unlabeled_counts = []
        self.neighbor_map = dict()
        self.lock = lock
        self.weight_file = weight_file
        self.pickle_path = pickle_path
        self.g = g
        self.time_budget = time_budget
        self.solution_budget = solution_budget
        self.self_loop = self_loop
        self.max_prob_maps = max_prob_maps
        self.model_prob_maps = model_prob_maps
        self.queue_pruning = queue_pruning
        self.noise_as_prob_maps = noise_as_prob_maps
        self.weighted_queue_pop = weighted_queue_pop
        self.reduction = reduction
        self.local_search = local_search

        self.cuda = bool(cuda_devs) # use cuda if devices are supplied
        self.cuda_dev = cuda_devs[pid % len(cuda_devs)] if self.cuda else None
        self.cuda_devs = cuda_devs

        self.pid = pid
        self.total_solutions = 0
        self.best_solution = None
        self.best_solution_vertices = None
        self.best_solution_weight = None
        self.best_solution_time = None
        self.best_solution_process_time = None

        self.last_status_report = time.monotonic() - 1000
        self.min_unlabeled = 99999999999
        self.max_unlabeled = 0

        self.weighted = torch.any(self.g.ndata["weight"] != 1)
        self.rdlib = reducelib() if reduction or local_search else None 

        if isinstance(optimum_found, bool):
            self.optimum_found = False
        else:
            self.optimum_found = optimum_found # Multiprocessing

        self.update_fn1 = lambda nodes: {'ts_label': torch.ones_like(nodes.data['ts_label'])}
        self.update_fn2 = lambda nodes: {'ts_label': torch.zeros_like(nodes.data['ts_label'])}

        if self.queue_pruning:
            self.max_queue_length = int(min(self.g.num_nodes() * 10, 16384) / num_threads)

        self.labels_given = "label" in self.g.ndata.keys()
        if self.labels_given:
            _lbls = self.g.ndata["label"].detach()
            self.optimal_mwis = torch.sum(self.g.ndata['weight'][_lbls == 1]).item()
            _locked_log(self.lock, f"{self.pid}: Labeled graph given. optimal_mwis={self.optimal_mwis}", "DEBUG")

    def run(self):
        with torch.no_grad():
            self.start_time = time.monotonic()
            self.start_process_time = time.process_time()
            self.load_model()
            self.prepare_queue()
            self.generate_neighborhood_map()
            while time.monotonic() - self.start_time <= self.time_budget:
                should_break = self.search_step()

                if should_break:
                    break

        return self.wrap_up()


    def search_step(self):
        if self.should_break():
            return True

        incomplete_solution = self.pop_incomplete_solution()
        residual = self.create_residual(incomplete_solution)
        self.total_solutions += 1
        
        if self.reduction:
            residual, should_return = self.do_reduction(incomplete_solution, residual)
            if should_return: # reduction might have solved the MIS problem, then we need to stop with the search step here
                return False

        if not self.noise_as_prob_maps and self.cuda:
            residual = residual.to(self.cuda_dev)

        out = self.infer_prob_maps(residual)
        residual = residual.cpu()

        self.explore_solutions(incomplete_solution, residual, out)

        if self.queue_pruning:
            self.prune_queue()

        del residual
        del incomplete_solution

        return False

    def explore_solutions(self, incomplete_solution, residual, prob_maps):
        num_prob_maps = min(prob_maps.shape[1], self.max_prob_maps)
        solutions_to_append = []
        # explore all solutions
        for pmap in range(num_prob_maps):
            # copy incomplete solution, in order to represent final result
            result = copy.deepcopy(incomplete_solution).formats("coo")

            _out = prob_maps[:, pmap].cpu().detach().numpy() # this is the current output we care about

            _sorted = np.flip(np.argsort(_out))
            progress = False

            marked_zero = []
            marked_one = []

            for v in _sorted:
                _v = residual.ndata['id_map'][v][0].item() # _v is the _name_ of the vth node of the residual graph, such that we can find the according node in the original graph
                if _v in marked_zero or _v in marked_one or result.ndata['ts_label'][_v][0].item() > -1:
                    assert progress # if progress is False, something is wrong with the residual...
                    break
                else:
                    progress = True
                    marked_one.append(_v)
                    marked_zero += self.neighbor_map[_v]

            result.apply_nodes(self.update_fn2, v=marked_zero)
            result.apply_nodes(self.update_fn1, v=marked_one)

            tslabels = result.ndata['ts_label'].detach().squeeze(1)
            num_unlabeled = torch.sum(tslabels == -1).item()

            self.min_unlabeled = min(num_unlabeled, self.min_unlabeled)
            self.max_unlabeled = max(num_unlabeled, self.max_unlabeled)

            self.maybe_print_status_report(result)

            if num_unlabeled == 0:
                self.update_best_solution(result)
            else:
                solutions_to_append.append((result, num_unlabeled))

        self.queue.extend(map(lambda x: x[0], solutions_to_append))
        self.queue_unlabeled_counts.extend(map(lambda x: x[1], solutions_to_append))

    def load_model(self):
        self.model = _load_model(self.model_prob_maps, weight_file=self.weight_file, cuda_dev=self.cuda_dev)
        self.model.eval()

    def prepare_queue(self):
        ### Prepare Queue ###
        if self.queue is not None:
            _locked_log(self.lock, f"{self.pid}: Starting with queue of length {len(self.queue)}", "DEBUG")
        else:
            self.queue = []
            # Starting queue is empty, hence we need to push a first graph into it
            if self.g is None:
                raise ValueError("Cannot start with empty queue and no start graph")

            self.g = self.g.cpu().formats("coo")

            # initialize attribute
            self.g.ndata['ts_label'] = torch.tensor(np.full((self.g.num_nodes(), 1), -1), dtype=torch.int8)
            self.g.ndata['id_map'] = torch.tensor(np.arange(self.g.num_nodes()).reshape((self.g.num_nodes(),1)), dtype=torch.int32)

            # we fix self loops before pushing in the queue, so we don't need to do it for every residual
            if self.self_loop:
                self.g = dgl.remove_self_loop(self.g)
                self.g = dgl.add_self_loop(self.g)
            else:
                self.g = dgl.remove_self_loop(self.g)

            self.queue.append(self.g)

        # Precompute unlabeled vertex counts
        for result in self.queue:
            tslabels = result.ndata['ts_label'].detach().squeeze(1)
            self.queue_unlabeled_counts.append(torch.sum(tslabels == -1).item())

    def generate_neighborhood_map(self):
        ### create neighborhood map ###
        _locked_log(self.lock, f"{self.pid}: Generating neighborhood map", "DEBUG")

        for v in self.g.nodes():
            _v = v.item()
            self.neighbor_map[_v] = torch.cat((self.g.predecessors(_v), self.g.successors(_v))).tolist()

        _locked_log(self.lock, f"{self.pid}: Map generated", "DEBUG")

    def should_break(self):
        if isinstance(self.optimum_found, bool):
            opt_found = self.optimum_found
        else:
            opt_found = self.optimum_found.value

        if opt_found:
            _locked_log(self.lock, f"{self.pid}:  Process is exiting, as optimum has been found in some thread", "INFO")
            return True

        if len(self.queue) == 0:
            _locked_log(self.lock, f"{self.pid}: Process is exiting, due to empty queue", "INFO")
            return True

        if self.solution_budget and self.total_solutions > self.solution_budget:
            _locked_log(self.lock, f"{self.pid}: Process is exiting, due to exhausted solution budget", "DEBUG")
            return True

        return False

    def pop_incomplete_solution(self):
        if self.weighted_queue_pop:
            nu = np.array(self.queue_unlabeled_counts)
            unnormalized_pop_p = 1 / nu
            pop_p = unnormalized_pop_p / unnormalized_pop_p.sum() # distribution summing up to 1
        else:
            pop_p = 1 / np.full(len(self.queue), len(self.queue))

        queue_choice = np.random.choice(np.arange(len(self.queue)), p=pop_p)
        incomplete_solution = self.queue.pop(queue_choice)
        self.queue_unlabeled_counts.pop(queue_choice)

        return incomplete_solution

    def create_residual(self, incomplete_solution):
        residual = copy.deepcopy(incomplete_solution).formats("coo")
        to_remove = residual.filter_nodes(lambda nodes: (nodes.data['ts_label'] != -1).squeeze(1))
        residual.remove_nodes(to_remove)

        return residual

    def do_reduction(self, incomplete_solution, residual):
        if self.weighted:
            _, reduction_result = self.rdlib.weighted_reduce_graph(residual)
        else:
            _, reduction_result = self.rdlib.unweighted_reduce_graph(residual)

        marked_zero_residual_ids = np.ravel(np.argwhere(reduction_result == 1)) # KaMIS output is inverted
        marked_one_residual_ids = np.ravel(np.argwhere(reduction_result == 0))

        # if the reduction did stuff
        if marked_zero_residual_ids.shape[0] > 0 or marked_one_residual_ids.shape[0] > 0:
            # map residual vertex ids to original graph vertex ids
            marked_zero = list(map(lambda x: residual.ndata['id_map'][x][0].item(), marked_zero_residual_ids))
            marked_one = list(map(lambda x: residual.ndata['id_map'][x][0].item(), marked_one_residual_ids))

            for v in marked_one:
                marked_zero.extend(self.neighbor_map[v])

            marked_zero = np.array(marked_zero)
            marked_one = np.array(marked_one)

            # Update incomplete solution
            incomplete_solution.apply_nodes(self.update_fn2, v=marked_zero)
            incomplete_solution.apply_nodes(self.update_fn1, v=marked_one)

            # Check if reduction solved MIS
            tslabels = incomplete_solution.ndata['ts_label'].detach().squeeze(1)
            num_unlabeled = torch.sum(tslabels == -1).item()

            if num_unlabeled == 0:
                self.update_best_solution(incomplete_solution)
                return None, True

            # recreate residual graph based on new information
            residual = self.create_residual(incomplete_solution)

        return residual, False

    def update_best_solution(self, labeled_graph):
        if self.local_search and self.rdlib is not None:
            if self.weighted:
                ls_result = self.rdlib.weighted_local_search(labeled_graph)
            else:
                ls_result = self.rdlib.unweighted_local_search(labeled_graph)

            labeled_graph.ndata['ts_label'] = torch.tensor(ls_result, dtype=torch.int8).unsqueeze(1)

        tslabels = labeled_graph.ndata['ts_label'].detach().squeeze(1)
        num_mis = torch.sum(tslabels == 1).item()

        if num_mis > 0:
            mis_vertices = labeled_graph.ndata['id_map'][(tslabels == 1)].to(torch.long)
            mis_weight = torch.sum(labeled_graph.ndata['weight'][mis_vertices].to(torch.float32)).item()

            if (self.best_solution is None) or (self.best_solution_weight < mis_weight):
                _locked_log(self.lock, f"{self.pid}: Found new Maximum {'Weighted' if self.weighted else ''} Independent Set with n = {num_mis} and weight = {mis_weight}", "INFO")
                self.best_solution = mis_vertices.detach().numpy()
                self.best_solution_vertices = num_mis
                self.best_solution_weight = mis_weight
                self.best_solution_time = time.monotonic() - self.start_time
                self.best_solution_process_time = time.process_time() - self.start_process_time
                if self.labels_given and self.best_solution_weight >= self.optimal_mwis:
                    _locked_log(self.lock, f"{self.pid}: Found optimal solution.", "INFO")
                    if isinstance(self.optimum_found, bool):
                        self.optimum_found = True
                    else:
                        with self.lock:
                            self.optimum_found.value = True

        else:
            _locked_log(self.lock, f"{self.pid}: Labeled all vertices in this graph as not belonging to MIS, something is off. Ignoring.", "WARNING")

    def infer_prob_maps(self, residual):
        features = residual.ndata['weight']
        if not self.noise_as_prob_maps:
            # actual model inference
            out = self.model(residual, features)
        else:
            # Replace GNN output probability maps with random noise
            out = torch.rand(features.shape[0], self.max_prob_maps)
        return out

    def maybe_print_status_report(self, result):
        if time.monotonic() - self.last_status_report > 15:
            _locked_log(self.lock, f"{self.pid}: Cuda Device={self.cuda_dev} (out of {self.cuda_devs}). Currently {self.min_unlabeled}/{len(list(result.nodes()))} vertices are unlabeled in the best solution, and {self.max_unlabeled}/{len(list(result.nodes()))} are unlabeled in the worst solution. We have {len(self.queue)} solutions in the queue. The queue is {sys.getsizeof(self.queue)} big. Time spent: {time.monotonic() - self.start_time}. Max Prob Maps = {self.max_prob_maps}. Total solutions done = {self.total_solutions}", "INFO")
            self.last_status_report = time.monotonic()
            gc.collect()

    def prune_queue(self):
        while len(self.queue) > self.max_queue_length:
            g = self.queue.pop(0)
            del g
            self.queue_unlabeled_counts.pop(0)


    def wrap_up(self):
        _locked_log(self.lock,f"{self.pid}: Wrapping up", "INFO")

        if self.pickle_path:
            _locked_log(self.lock,f"{self.pid}: Saving result", "DEBUG")
            with open(self.pickle_path / f"{self.pid}.pickle", 'wb') as f:
                results = { 
                    "total_solutions": self.total_solutions,
                    "mis_vertices": self.best_solution_vertices if self.best_solution is not None else None,
                    "mis_weight": self.best_solution_weight if self.best_solution is not None else None,
                    "solution": self.best_solution,
                    "solution_time": self.best_solution_time,
                    "solution_process_time": self.best_solution_process_time
                }
                pickle.dump(results, f)

            _locked_log(self.lock,f"{self.pid}: Pickle saved! Cleaning up memory.", "DEBUG")
            del self.queue[:]
            del self.queue
            if self.rdlib:
                del self.rdlib
                
            return

        if isinstance(self.optimum_found, bool):
            opt_found = self.optimum_found
        else:
            opt_found = self.optimum_found.value

        _locked_log(self.lock,f"{self.pid}: Cleaning up objects.", "DEBUG")

        _locked_log(self.lock,f"{self.pid}: Returning.", "DEBUG")
        return self.best_solution, self.best_solution_vertices, self.best_solution_weight, self.total_solutions, self.queue, self.best_solution_time, self.best_solution_process_time, opt_found

# warning order to arguments changed check in calls!
def tree_search_wrapper(pid, num_threads, queue, lock, weight_file, g, pickle_path=None, time_budget=600, solution_budget=None, self_loop = False, max_prob_maps=64, model_prob_maps=32, cuda_devs=[0], reduction=False, local_search=False, queue_pruning=False, noise_as_prob_maps=False, weighted_queue_pop=False, optimum_found=False):
    np.random.seed() # reseed as we are multithreaded

    ts = _TreeSearch(pid, num_threads, queue, lock, weight_file, pickle_path, g, time_budget, solution_budget, self_loop, max_prob_maps, model_prob_maps, cuda_devs, reduction, local_search, queue_pruning, noise_as_prob_maps, weighted_queue_pop, optimum_found)
    return ts.run()

def check_for_inconsistency(g, neighbor_map):
    for i in range(g.number_of_nodes()):
        if g.ndata["ts_label"][i][0].item() == 1:
            for neighbor in neighbor_map[i]:
                if i != neighbor and g.ndata["ts_label"][neighbor][0].item() == 1:
                    return True

        if g.ndata["ts_label"][i][0].item() == -1:
            for neighbor in neighbor_map[i]:
                if g.ndata["ts_label"][neighbor][0].item() == 1:
                    return True

    return False
