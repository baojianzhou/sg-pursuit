//
// Created by baojian on 8/11/18.
//

#include "head_tail_proj.h"

typedef struct {
    int p;
    int m;
    EdgePair *edges;
    double *prizes;
    double *costs;
} Data;
char *file_name = "/network/rit/lab/ceashpc/bz383376/data/pcst/MNIST/mnist_test_case_0.txt";

Data *read_mnist_data() {
    int p = 784, m = 1512;
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    double *prizes = malloc(sizeof(double) * p);
    double *costs = malloc(sizeof(double) * m);
    char *line = NULL, tokens[10][20];
    FILE *fp;
    size_t len = 0, num_lines = 0, edge_index = 0;
    if ((fp = fopen(file_name, "r")) == NULL) {
        printf("cannot open: %s!\n", file_name);
        exit(EXIT_FAILURE);
    }
    printf("reading data from: %s\n", file_name);
    while ((getline(&line, &len, fp)) != -1) {
        int tokens_size = 0;
        for (char *token = strtok(line, " ");
             token != NULL; token = strtok(NULL, " ")) {
            strcpy(tokens[tokens_size++], token);
        }
        num_lines++;
        if (strcmp("E", tokens[0]) == 0) {
            int uu = (int) strtol(tokens[1], NULL, 10);
            int vv = (int) strtol(tokens[2], NULL, 10);
            double weight = strtod(tokens[3], NULL);
            edges[edge_index].first = uu;
            edges[edge_index].second = vv;
            costs[edge_index++] = weight;
            continue;
        }
        if (strcmp("N", tokens[0]) == 0) {
            int node = (int) strtol(tokens[1], NULL, 10);
            double prize = strtod(tokens[2], NULL);
            prizes[node] = prize;
            continue;
        }
    }
    fclose(fp);
    Data *graph = malloc(sizeof(Data));
    graph->m = m;
    graph->p = p;
    graph->edges = edges;
    graph->prizes = prizes;
    graph->costs = costs;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            printf("%-6.2f ", prizes[i * 28 + j]);
        }
        printf("\n");
    }
    return graph;
}

void test_head_proj_exact() {
    Data *graph = read_mnist_data();
    PruningMethod pruning = GWPruning;
    int g = 6, sparsity = 100, max_iter = 50, root = -1, n = graph->p;
    int m = graph->m, verbose = 0;
    double C = 2 * (sparsity - 1.) * 100., delta = 1. / 169.;
    double err_tol = 1e-6, epsilon = 1e-6, run_time = 0.0;
    GraphStat *head_stat = make_graph_stat(n, m);
    head_proj_exact(
            graph->edges, graph->costs, graph->prizes, g, C, delta, max_iter,
            err_tol, root, pruning, epsilon, n, m, verbose, head_stat);
    printf("number of head_nodes: %d number of tail_nodes: %d\n",
           head_stat->re_nodes->size, head_stat->re_edges->size);
    printf("number of pcst: %d run_time: %.6f\n",
           head_stat->num_pcst, run_time);
    free_graph_stat(head_stat);
    free(graph->costs), free(graph->prizes), free(graph->edges);
    free(graph);
}

void test_head_proj_approx() {
    Data *graph = read_mnist_data();
    PruningMethod pruning = GWPruning;
    int g = 17, sparsity = 100, max_iter = 50, root = -1, n = graph->p;
    int m = graph->m, verbose = 0;
    double C = 2 * (sparsity - 1.) * graph->costs[0], delta = 1. / 169.;
    double err_tol = 1e-6, epsilon = 1e-6, run_time = 0.0;
    GraphStat *head_stat = make_graph_stat(n, m);
    head_proj_approx(
            graph->edges, graph->costs, graph->prizes, g, C, delta, max_iter,
            err_tol, root, pruning, epsilon, n, m, verbose, head_stat);
    printf("number of head_nodes: %d number of tail_nodes: %d\n",
           head_stat->re_nodes->size, head_stat->re_edges->size);
    printf("number of pcst: %d run_time: %.6f\n",
           head_stat->num_pcst, run_time);
    free_graph_stat(head_stat);
    free(graph->costs), free(graph->prizes), free(graph->edges);
    free(graph);
}

void test_tail_proj_exact() {
    Data *graph = read_mnist_data();
    PruningMethod pruning = GWPruning;
    int g = 1, sparsity = 100, max_iter = 50, root = -1, n = graph->p;
    int m = graph->m, verbose = 0;
    double C = 2 * (sparsity - 1.) * graph->costs[0], nu = 2.5, err_tol = 1e-6;
    double epsilon = 1e-6, run_time = 0.0;
    GraphStat *tail_stat = make_graph_stat(n, m);
    tail_proj_exact(
            graph->edges, graph->costs, graph->prizes, g, C, nu,
            max_iter, err_tol, root, pruning, epsilon, n, m, verbose,
            tail_stat);
    printf("number of head_nodes: %d number of tail_nodes: %d\n",
           tail_stat->re_nodes->size, tail_stat->re_edges->size);
    printf("number of pcst: %d run_time: %.6f\n",
           tail_stat->num_pcst, run_time);
    free_graph_stat(tail_stat);
    free(graph->costs), free(graph->prizes), free(graph->edges);
    free(graph);
}

void test_tail_proj_approx() {
    Data *graph = read_mnist_data();
    PruningMethod pruning = GWPruning;
    int g = 1, sparsity = 100, max_iter = 50, root = -1, n = graph->p;
    int m = graph->m, verbose = 0;
    double C = 2 * (sparsity - 1.) * graph->costs[0], nu = 2.5, err_tol = 1e-6;
    double epsilon = 1e-6, run_time = 0.0;
    GraphStat *tail_stat = make_graph_stat(n, m);
    tail_proj_approx(
            graph->edges, graph->costs, graph->prizes, g, C, nu, max_iter,
            err_tol, root, pruning, epsilon, n, m, verbose, tail_stat);
    printf("number of head_nodes: %d number of tail_nodes: %d\n",
           tail_stat->re_nodes->size, tail_stat->re_edges->size);
    printf("number of pcst: %d run_time: %.6f\n",
           tail_stat->num_pcst, run_time);
    free_graph_stat(tail_stat);
    free(graph->costs), free(graph->prizes), free(graph->edges);
    free(graph);
}

void test_all() {
    test_head_proj_exact();
    test_head_proj_approx();
    test_tail_proj_exact();
    test_tail_proj_approx();
}

int main() {
    test_all();
}