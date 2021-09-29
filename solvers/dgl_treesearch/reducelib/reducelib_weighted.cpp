#include "reducelib_weighted.h"
#include "branch_and_reduce_algorithm.h"
#include "ils.h"
#include "KaMIS/wmis/extern/KaHIP/lib/data_structure/graph_access.h"
#include "KaMIS/wmis/lib/mis/mis_config.h"
#include "KaMIS/wmis/app/configuration_mis.h"

#include <algorithm>
#include <vector>
#include <cstdio>

void SilenceOutput()
{
    std::freopen("/dev/null", "a", stdout);
}

int WeightedReduce(const int num_nodes, const int num_edges, const int *edges_from, const int *edges_to,  int *node_weights, int *reduction_result)
{
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
    std::vector<int> adjwgt(num_edges, 1); // required by graph_access but ignored by reduction
    G.build_from_metis_weighted(num_nodes, xadj.data(), adjncy.data(), node_weights, adjwgt.data());

    // reduce graph using KaMIS

    MISConfig mis_config;
    branch_and_reduce_algorithm full_reducer(G, mis_config);
    full_reducer.reduce_graph();

    for (int i = 0; i < num_nodes; ++i) {
        switch(full_reducer.status.node_status[i]) {
            case branch_and_reduce_algorithm::IS_status::not_set:
                reduction_result[i] = -1;
                break;
            case branch_and_reduce_algorithm::IS_status::included:
                reduction_result[i] = 0;
                break;
            case branch_and_reduce_algorithm::IS_status::excluded:
                reduction_result[i] = 1;
                break;
            case branch_and_reduce_algorithm::IS_status::folded:
            default:
                reduction_result[i] = 2;
                break;
        }
    } 

    return full_reducer.get_current_is_weight();
}

void WeightedLocalSearch(const int num_nodes, const int num_edges, const int *edges_from, const int *edges_to,  int *node_weights, int *input, int *output)
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
    std::vector<int> adjwgt(num_edges, 1); // required by graph_access but ignored
    G.build_from_metis_weighted(num_nodes, xadj.data(), adjncy.data(), node_weights, adjwgt.data());

	// tell graph object about current solution
	forall_nodes(G, node) {
        G.setPartitionIndex(node, input[node]);
    } endfor

	// configure to apply Iterated Local Search at most n times
	mis_config.ils_iterations = std::min(G.number_of_nodes(), mis_config.ils_iterations);

    // Perform ILS
    ils iterate(mis_config);
    iterate.perform_ils(G, mis_config.ils_iterations); // updates parition indices of G at the end to best solution

    // take information from G and put to output
    forall_nodes(G, node) {
		output[node] = iterate.best_solution[node];
    } endfor
}
