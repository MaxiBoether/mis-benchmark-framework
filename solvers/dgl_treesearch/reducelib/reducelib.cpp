#include "reducelib.h"
#include "KaMIS/lib/mis/kernel/branch_and_reduce_algorithm.h"
#include "KaMIS/extern/KaHIP/lib/data_structure/graph_access.h"
#include "KaMIS/lib/mis/mis_config.h"
#include "KaMIS/lib/mis/evolutionary/population_mis.h"
#include "KaMIS/app/configuration_mis.h"
#include "KaMIS/lib/mis/ils/ils.h"

#include <algorithm>
#include <vector>
#include <cstdio>

void SilenceOutput()
{
    std::freopen("/dev/null", "a", stdout);
}

int UnweightedReduce(const int num_nodes, const int num_edges, const int* edges_from, const int* edges_to, int* reduction_result)
{
    // build adjacency list
    std::vector<std::vector<int>> adj(num_nodes);
    for (int i=0; i < num_edges; ++i) {
        adj[edges_from[i]].push_back(edges_to[i]);
    }

    // reduce graph using KaMIS

    branch_and_reduce_algorithm full_reducer(adj, adj.size());
    full_reducer.reduce_graph();

	std::vector<int> curr_solution(full_reducer.x); // interpretation: current solution (-1: not determined, 0: not in the vc, 1: in the vc, 2: removed by foldings) => reverse for MIS!
	for (int i = 0; i < num_nodes; ++i)
		reduction_result[i] = curr_solution[i];

	return full_reducer.get_current_is_size();
}

void UnweightedLocalSearch(const int num_nodes, const int num_edges, const int* edges_from, const int* edges_to, int* input, int* output)
{
    // Configurations
    MISConfig mis_config;
    configuration_mis cfg;
    cfg.standard(mis_config); // set our mis_config object to standard configuration

    // Build the adjacency list
    std::vector<std::vector<int>> adj(num_nodes);
    for (int i = 0; i < num_edges; ++i) {
        adj[edges_from[i]].push_back(edges_to[i]);
	}

	// Build the funny sparse adjacency matrix
    std::vector<int> xadj(num_nodes + 1);
    std::vector<int> adjncy(num_edges);
    unsigned int adjncy_counter = 0;
    for (unsigned int i = 0; i < num_nodes; ++i) {
        xadj[i] = adjncy_counter;
        for (int const neighbor : adj[i]) {
            if (neighbor == i) continue;
            if (neighbor == UINT_MAX) continue;
            adjncy[adjncy_counter++] = neighbor;
        }
        std::sort(std::begin(adjncy) + xadj[i], std::begin(adjncy) + adjncy_counter);
    }
    xadj[num_nodes] = adjncy_counter;

	// build graph out of sparse matrix
    graph_access G;
    G.build_from_metis(num_nodes, xadj.data(), adjncy.data());

	// tell graph object about current solution
	forall_nodes(G, node) {
        G.setPartitionIndex(node, input[node]);
    } endfor

	// configure to apply Iterated Local Search at most n times
	mis_config.ils_iterations = std::min(G.number_of_nodes(), mis_config.ils_iterations);

    // Perform ILS
    ils iterate;
    iterate.perform_ils(mis_config, G, mis_config.ils_iterations); // updates parition indices of G at the end to best solution

	// Create individuum for final independent set
    individuum_mis final_mis;
	population_mis island;
	island.init(mis_config, G);
    NodeID *solution = new NodeID[G.number_of_nodes()]; // solution is array of NodeID, but actually it is an array of 1s and 0s... why, KaMIS, why?
    final_mis.solution_size = island.create_solution(G, solution); // takes partition indices of G and creates a solution out of it
    final_mis.solution = solution;

    // island.set_mis_for_individuum(mis_config, G, final_mis); // apply solution (stored in final_mis) to G - redundant we think

    // take information from G and put to output
    forall_nodes(G, node) {
		output[node] = final_mis.solution[node];
    } endfor

	delete[] solution;
    solution = NULL;
}
