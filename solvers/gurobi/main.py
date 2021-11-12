import argparse
import logzero
import time
import json
import sys
import io
import re
import os

import pulp as plp
import numpy as np
import networkx as nx
import pandas as pd

from pathlib import Path
from logzero import logger

def solve(G, args, graph_name_stub):
    msg = args.loglevel in ["DEBUG", "INFO"]
    start_time = time.monotonic()
    time_limit = args.time_limit
    weight_dict = nx.get_node_attributes(G, "weight")
    weight_list = [weight_dict[key] for key in sorted(weight_dict.keys())] # not defined whether dictionary returned by networkx is sorted
    wts = np.array(weight_list)

    if not args.quadratic:
        opt_model = plp.LpProblem(name="model")
        adj = nx.adjacency_matrix(G)
        x_vars = {i: plp.LpVariable(cat=plp.LpBinary, name="x_{0}".format(i)) for i in range(wts.size)}
        set_V = set(range(wts.size)) # not sure why we need set here, as the call to range produces a unique list
        constraints = dict()
        ei = 0 # number of constraints counter
        for j in set_V: # we iterate over all nodes (could we be more explicit and use G.nodes here?)
            _, set_N = np.nonzero(adj[j]) # set N are all the vertices that j has a connection to
            for i in set_N:
                constraints[ei] = opt_model.addConstraint(
                    plp.LpConstraint(
                        e=plp.lpSum([x_vars[i], x_vars[j]]),
                        sense=plp.LpConstraintLE,
                        rhs=1,
                        name="constraint_{0}_{1}".format(j,i))) # constraint that we only pick one of two neighboring nodes
                ei = ei + 1

        objective = plp.lpSum(x_vars[i] * wts[i] for i in set_V) # we want to maximize the weight

        opt_model.sense = plp.LpMaximize
        opt_model.setObjective(objective)

        if args.write_mps:
            opt_model.writeMPS(args.output / (graph_name_stub  + ".mps"))
            return

        # redirect stdout by gurobi into internal string buffer
        real_stdout = sys.stdout
        sys.stdout = io.StringIO()

        pulp_gurobi_kwargs = {
            "mip": True,
            "msg": msg,
            "timeLimit": time_limit,
            "Threads": args.num_threads
        }

        if args.prm_file:
            with open(args.prm_file, "r") as a_file:
                for line in a_file:
                    stripped_line = line.strip()
                    splitted = stripped_line.split()
                    pulp_gurobi_kwargs[splitted[0]] = float(splitted[1])
        else:
            pulp_gurobi_kwargs["ImproveStartTime"] = time_limit*0.9

        opt_model.solve(solver=plp.apis.GUROBI(**pulp_gurobi_kwargs))
    
        gurobi_output = sys.stdout.getvalue()
        sys.stdout = real_stdout # set stdout file back to normal
        print(gurobi_output)

        # parse solving time from gurobi output
        time_string_match = re.compile("in (\d+(\.\d*)) seconds").findall(gurobi_output)[0]
        apparent_solve_time = float(time_string_match[0])

        if plp.LpStatus[opt_model.status] == "Optimal":
            opt_df = pd.DataFrame.from_dict(x_vars, orient="index", columns=["variable_object"])
            opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.varValue)
            solu = opt_df[opt_df['solution_value'] > 0].index.to_numpy()
        else:
            # Retrieve best (not optimal) solution
            # GUROBI specific, seems to be a bug in PuLP where varValue is None for non-optimal solutions
            solu = np.nonzero(np.array(opt_model.solverModel.X))[0]
        
        end_time = time.monotonic()

        return solu, wts[solu].sum(), plp.LpStatus[opt_model.status], end_time - start_time, apparent_solve_time

    else:
        import gurobipy as gp
        from gurobipy import GRB

        n = G.number_of_nodes()
        adj = nx.to_numpy_array(G)
        J = np.identity(n)
        A = J - adj

        m = gp.Model("mis")

        x = m.addMVar(shape=n, vtype=GRB.BINARY, name="x")
        m.setObjective(x @ A @ x, GRB.MAXIMIZE)
        m.setParam('TimeLimit', time_limit)
        m.setParam('Threads', 8)

        if args.prm_file:
            with open(args.prm_file, "r") as a_file:
                for line in a_file:
                    stripped_line = line.strip()
                    splitted = stripped_line.split()
                    m.setParam(splitted[0], float(splitted[1]))
        else:
            m.setParam('ImproveStartTime', time_limit*0.9)

        if args.write_mps:
            m.write(args.output / graph_name_stub  + ".mps")
            return

        # redirect stdout by gurobi into internal string buffer
        real_stdout = sys.stdout
        sys.stdout = io.StringIO()

        m.optimize()

        gurobi_output = sys.stdout.getvalue()
        sys.stdout = real_stdout # set stdout file back to normal
        print(gurobi_output)

        # parse solving time from gurobi output
        time_string_match = re.compile("in (\d+(\.\d*)) seconds").findall(gurobi_output)[0]
        apparent_solve_time = float(time_string_match[0])

        solu = np.nonzero(np.array(x.X))[0]

        status = m.status
        if status == GRB.OPTIMAL:
            status = "Optimal"
        else:
            status = str(status)

        end_time = time.monotonic()

        return solu, wts[solu].sum(), status, end_time - start_time, apparent_solve_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gurobi-based MWIS solving.")
    parser.add_argument("input", type=Path, action="store", help="Directory containing input graphs (to be solved/trained on).")
    parser.add_argument("output", type=Path, action="store",  help="Folder in which the output (e.g. json containg statistics and solution will be stored)")

    parser.add_argument("--loglevel", type=str, action="store", default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Verbosity of logging (DEBUG/INFO/WARNING/ERROR)")

    parser.add_argument("--time_limit", type=int, nargs="?", action="store", default=120, help="Time limit in seconds")
    parser.add_argument("--weighted", action="store_true", default=False, help="Whether input is weighted or unweighted.")
    parser.add_argument("--num_threads", type=int, nargs="?", action="store", default=8, help="Maximum number of threads to use.")
    parser.add_argument("--quadratic", action="store_true", default=False, help="Whether a quadratic program should be used instead of linear.")
    parser.add_argument("--write_mps", action="store_true", default=False, help="Instead of solving, write mps output (e.g., for tuning).")
    parser.add_argument("--prm_file", type=Path, nargs="?", action="store", help="Gurobi parameter file (e.g. by grbtune).")
    parser.add_argument("--tuned", action="store_true", default=False, help="dummy parameter for experiments.")

    args = parser.parse_args()

    ### Set logging ###
    if args.loglevel == "DEBUG":
        logzero.loglevel(logzero.DEBUG)
    elif args.loglevel == "INFO":
        logzero.loglevel(logzero.INFO)
    elif args.loglevel == "WARNING":
        logzero.loglevel(logzero.WARNING)
    elif args.loglevel == "ERROR":
        logzero.loglevel(logzero.ERROR)
    else:
        print(f"Unknown loglevel {args.loglevel}, ignoring.")

    logger.debug(f"Gurobi solver got launched with arguments {args}")

    if args.quadratic and args.weighted:
        logger.error("Cannot use --quadratic together with --weighted")
        sys.exit()

    results = {}

    for idx,graph_path in enumerate(args.input.rglob(f"*_{'weighted' if args.weighted else 'unweighted'}.graph")):
        graph_name_stub = os.path.splitext(os.path.basename(graph_path))[0].rsplit('_', 1)[0]

        logger.info(f"Solving graph {graph_path}")
        G = nx.read_gpickle(graph_path)
        if args.write_mps:
            solve(G, args, graph_name_stub)
        else:
            solu, sum, status, total_time, explore_time = solve(G, args, graph_name_stub)

            logger.info(f"Found MWIS: n={len(solu)}, w={sum} ✔️✔️")

            if status != "Optimal":
                logger.info(f"Non-Optimal Gurobi status: {status}")

            results[graph_name_stub] = {
                "total_time": total_time,
                "mwis_vertices": int(len(solu)),
                "mwis_weight": float(sum),
                "mwis": np.ravel(solu).tolist(),
                "gurobi_status": status,
                "gurobi_explore_time": explore_time
            }

            print("")

    if not args.write_mps:
        with open(args.output / "results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, sort_keys = True, indent=4)

    logger.info("Done with all graphs, exiting.")