//
// Created by baojian on 8/10/18.
//

#ifndef FAST_PCST_HEAD_TAIL_PROJ_H
#define FAST_PCST_HEAD_TAIL_PROJ_H

#include <math.h>
#include "fast_pcst.h"

typedef struct {
    Array *nodes;
    Array *edges;
    double prize;
    double cost;
} Tree;

typedef struct {
    double val;
    int val_index;
} data_pair;


typedef struct {
    Array *re_nodes;
    Array *re_edges;
    double *prizes;
    double *costs;
    int num_pcst;
    double run_time;
} GraphStat;


GraphStat *make_graph_stat(int p, int m);

bool free_graph_stat(GraphStat *graph_stat);

bool head_proj_exact(
        const EdgePair *edges, const double *costs, const double *prizes,
        int g, double C, double delta, int max_iter, double err_tol, int root,
        PruningMethod pruning, double epsilon, int n, int m, int verbose,
        GraphStat *stat);

bool head_proj_approx(
        const EdgePair *edges, const double *costs, const double *prizes,
        int g, double C, double delta, int max_iter, double err_tol, int root,
        PruningMethod pruning, double epsilon, int n, int m, int verbose,
        GraphStat *stat);

bool tail_proj_exact(
        const EdgePair *edges, const double *costs, const double *prizes,
        int g, double C, double nu, int max_iter, double err_tol, int root,
        PruningMethod pruning, double epsilon, int n, int m, int verbose,
        GraphStat *stat);

bool tail_proj_approx(
        const EdgePair *edges, const double *costs, const double *prizes,
        int g, double C, double nu, int max_iter, double err_tol, int root,
        PruningMethod pruning, double epsilon, int n, int m, int verbose,
        GraphStat *stat);

#endif //FAST_PCST_HEAD_TAIL_PROJ_H
