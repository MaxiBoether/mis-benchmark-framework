import ctypes
import subprocess
import numpy as np

from pathlib import Path
from sys import platform
from logzero import logger
from typing import Tuple

class reducelib:

    def __init__(self, silent_output=True):
        reducelib.__ensure_library_present()
        u_library_path = reducelib.__get_unweighted_library_path()
        self.unweighted_lib = ctypes.CDLL(str(u_library_path))

        w_library_path = reducelib.__get_weighted_library_path()
        self.weighted_lib = ctypes.CDLL(str(w_library_path))

        if silent_output:
            self.unweighted_lib.SilenceOutput()
            self.weighted_lib.SilenceOutput()

    @staticmethod
    def __get_build_path():
        return Path(__file__).parent / "build"

    @staticmethod
    def __get_unweighted_library_path():
        if platform == "darwin":
            library_filename = "libreduceu.dylib"
        else:
            library_filename = "libreduceu.so"
        return reducelib.__get_build_path() / library_filename

    @staticmethod
    def __get_weighted_library_path():
        if platform == "darwin":
            library_filename = "libreducew.dylib"
        else:
            library_filename = "libreducew.so"
        return reducelib.__get_build_path() / library_filename

    @staticmethod
    def __ensure_library_present():
        if not reducelib.__get_unweighted_library_path().exists():
            logger.info('Unweighted not built yet. Building...')
            build_path = reducelib.__get_build_path()
            # Execute `cmake ..` in build folder
            subprocess.run(['cmake', '..'], check=True, cwd=build_path)
            # Execute `make reduce` in build folder
            subprocess.run(['make', '-j8', 'reduceu'], check=True, cwd=build_path)

        if not reducelib.__get_weighted_library_path().exists():
            logger.info('Weighted not built yet. Building...')
            build_path = reducelib.__get_build_path()
            # Execute `cmake ..` in build folder
            subprocess.run(['cmake', '..'], check=True, cwd=build_path)
            # Execute `make reduce` in build folder
            subprocess.run(['make', '-j8', 'reducew'], check=True, cwd=build_path)

    def __CtypeAdj(self, adj):
        adj = adj.tocoo()
        num_edge = adj.nnz
        num_node = adj.shape[0]
        e_list_from = (ctypes.c_int * num_edge)()
        e_list_to = (ctypes.c_int * num_edge)()
        edges = zip(adj.col, adj.row)
        if num_edge:
            a, b = zip(*edges)
            e_list_from[:] = a
            e_list_to[:] = b

        return (num_node, num_edge, ctypes.cast(e_list_from, ctypes.c_void_p), ctypes.cast(e_list_to, ctypes.c_void_p))

    def unweighted_reduce_graph(self, g) -> Tuple[int, np.array]:
        # generate sparse input to pass to C++
        _g = g.remove_self_loop() # create new graph without self loops, as KaMIS expects no self-loops
        n_nodes, n_edges, e_froms, e_tos = self.__CtypeAdj(_g.adj(transpose=False, scipy_fmt="coo")) # we have undirected graphs hence transpose does not matter

        # create output variables
        reduction_result = (ctypes.c_int * (n_nodes))()

        # Call library
        crt_is_size = self.unweighted_lib.UnweightedReduce(n_nodes,
                                    n_edges,
                                    e_froms,
                                    e_tos,
                                    reduction_result)

        # CInt[] to np array
        reduction_result = np.asarray(reduction_result[:])

        return crt_is_size, reduction_result

    def unweighted_local_search(self, g):
        # generate sparse input to pass to C++
        _g = g.remove_self_loop() # create new graph without self loops, as KaMIS expects no self-loops
        n_nodes, n_edges, e_froms, e_tos = self.__CtypeAdj(_g.adj(transpose=False, scipy_fmt="coo")) # we have undirected graphs hence transpose does not matter

        init_mis = (ctypes.c_int * (n_nodes))()
        final_mis = (ctypes.c_int * (n_nodes))()
        current_solution = _g.ndata["ts_label"].detach().squeeze(1).numpy().astype(np.int32)
        init_mis[:] = current_solution
        init_mis =  ctypes.cast(init_mis, ctypes.c_void_p)
        self.unweighted_lib.UnweightedLocalSearch(n_nodes, n_edges, e_froms, e_tos, init_mis, final_mis)
        indset = np.asarray(final_mis[:])

        return indset

    def weighted_reduce_graph(self, g):
        # generate sparse input to pass to C++
        _g = g.remove_self_loop() # create new graph without self loops, as KaMIS expects no self-loops
        n_nodes, n_edges, e_froms, e_tos = self.__CtypeAdj(_g.adj(transpose=False, scipy_fmt="coo")) # we have undirected graphs hence transpose does not matter

        node_weights = (ctypes.c_int * (n_nodes))()
        _weights = _g.ndata["weight"].detach().squeeze(1).numpy().astype(np.int32)
        node_weights[:] = _weights
        node_weights =  ctypes.cast(node_weights, ctypes.c_void_p)

        # create output variables
        reduction_result = (ctypes.c_int * (n_nodes))()

        # Call library
        crt_is_weight = self.weighted_lib.WeightedReduce(n_nodes,
                                    n_edges,
                                    e_froms,
                                    e_tos,
                                    node_weights,
                                    reduction_result)

        # CInt[] to np array
        reduction_result = np.asarray(reduction_result[:])

        return crt_is_weight, reduction_result

    def weighted_local_search(self, g):
        # generate sparse input to pass to C++
        _g = g.remove_self_loop() # create new graph without self loops, as KaMIS expects no self-loops
        n_nodes, n_edges, e_froms, e_tos = self.__CtypeAdj(_g.adj(transpose=False, scipy_fmt="coo")) # we have undirected graphs hence transpose does not matter
        
        node_weights = (ctypes.c_int * (n_nodes))()
        _weights = _g.ndata["weight"].detach().squeeze(1).numpy().astype(np.int32)
        node_weights[:] = _weights
        node_weights =  ctypes.cast(node_weights, ctypes.c_void_p)
        
        init_mis = (ctypes.c_int * (n_nodes))()
        final_mis = (ctypes.c_int * (n_nodes))()
        current_solution = _g.ndata["ts_label"].detach().squeeze(1).numpy().astype(np.int32)
        init_mis[:] = current_solution
        init_mis =  ctypes.cast(init_mis, ctypes.c_void_p)
        self.weighted_lib.WeightedLocalSearch(n_nodes,
                                                n_edges,
                                                e_froms,
                                                e_tos,
                                                node_weights,
                                                init_mis,
                                                final_mis)
        indset = np.asarray(final_mis[:])

        return indset