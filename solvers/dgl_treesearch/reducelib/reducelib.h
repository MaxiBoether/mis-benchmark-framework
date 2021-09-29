#ifndef REDUCELIB_H
#define REDUCELIB_H

extern "C" int UnweightedReduce(const int num_nodes, const int num_edges, const int* edges_from, const int* edges_to, int* reduction_result);
extern "C" void UnweightedLocalSearch(const int num_nodes, const int num_edges, const int* edges_from, const int* edges_to, int* input, int* output);
extern "C" void SilenceOutput(void);

#endif
