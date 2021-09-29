#ifndef REDUCELIB_WEIGHTED_H
#define REDUCELIB_WEIGHTED_H

extern "C" int WeightedReduce(const int num_nodes, const int num_edges, const int* edges_from, const int* edges_to, int* node_weights, int* reduction_result);
extern "C" void WeightedLocalSearch(const int num_nodes, const int num_edges, const int* edges_from, const int* edges_to, int* node_weights, int* input, int* output);
extern "C" void SilenceOutput(void);

#endif