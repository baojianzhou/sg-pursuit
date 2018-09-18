//
// Created by baojian on 8/11/18.
//
#include <Python.h>
#include <numpy/arrayobject.h>
#include "head_tail_proj.h"


static PyObject *proj_head(PyObject *self, PyObject *args) {
    /**
     * DO NOT call this function directly, use the Python Wrapper instead.
     * list of args:
     * args[0]: ndarray dim=(m,2) -- edges of the graph.
     * args[1]: ndarray dim=(m,)  -- weights (positive) of the graph.
     * args[2]: ndarray dim=(n,)  -- the vector needs to be projected.
     * args[3]: integer np.int32  -- number of connected components returned.
     * args[4]: integer np.int32  -- sparsity (positive) parameter.
     * args[5]: double np.float64 -- budget of the graph model.
     * args[6]: double np.float64 -- delta. default is 1. / 169.
     * args[7]: integer np.int32  -- maximal # of iterations in the loop.
     * args[8]: double np.float64 -- error tolerance for minimum nonzero.
     * args[9]: integer np.int32  -- root(default is -1).
     * args[10]: string string    -- pruning ['simple', 'gw', 'strong'].
     * args[11]: double np.float64-- epsilon to control the presion of PCST.
     * args[12]: integer np.int32 -- verbosity level
     * @return: (re_nodes, re_edges, p_x)
     * re_nodes: projected nodes
     * re_edges: projected edges (indices)
     * p_x: projection of x.
     */
    if (self != NULL) { return NULL; }
    PyArrayObject *edges_, *weights_, *x_;
    int g, s, root, max_iter, verbose;
    double budget, delta, epsilon, err_tol;
    char *pruning;
    if (!PyArg_ParseTuple(
            args, "O!O!O!iiddidizdi", &PyArray_Type, &edges_, &PyArray_Type,
            &weights_, &PyArray_Type, &x_, &g, &s, &budget, &delta,
            &max_iter, &err_tol, &root, &pruning, &epsilon, &verbose)) {
        return NULL;
    }
    long n = x_->dimensions[0];  // number of nodes
    long m = edges_->dimensions[0];     // number of edges
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    double *prizes = malloc(sizeof(double) * n);
    double *costs = malloc(sizeof(double) * m);;
    double *x = (double *) PyArray_DATA(x_);
    PyObject *results = PyTuple_New(3);
    PyObject *p_x = PyList_New(n);      // projected x
    for (int i = 0; i < m; i++) {
        edges[i].first = *(int *) PyArray_GETPTR2(edges_, i, 0);
        edges[i].second = *(int *) PyArray_GETPTR2(edges_, i, 1);
        double *wei = (double *) PyArray_GETPTR1(weights_, i);
        costs[i] = *wei + budget / s;
    }
    for (int i = 0; i < n; i++) {
        prizes[i] = (x[i]) * (x[i]);
        PyList_SetItem(p_x, i, PyFloat_FromDouble(0.0));
    }
    double C = 2. * budget;
    GraphStat *head_stat = make_graph_stat((int) n, (int) m);
    head_proj_exact(
            edges, costs, prizes, g, C, delta, max_iter,
            err_tol, root, GWPruning, epsilon, (int) n, (int) m,
            verbose, head_stat);
    PyObject *re_nodes = PyList_New(head_stat->re_nodes->size);
    PyObject *re_edges = PyList_New(head_stat->re_edges->size);
    for (int i = 0; i < head_stat->re_nodes->size; i++) {
        int node_i = head_stat->re_nodes->array[i];
        PyList_SetItem(re_nodes, i, PyInt_FromLong(node_i));
        PyList_SetItem(p_x, node_i, PyFloat_FromDouble(x[node_i]));
    }
    for (int i = 0; i < head_stat->re_edges->size; i++) {
        PyList_SetItem(re_edges, i,
                       PyInt_FromLong(head_stat->re_edges->array[i]));
    }
    PyTuple_SetItem(results, 0, re_nodes);
    PyTuple_SetItem(results, 1, re_edges);
    PyTuple_SetItem(results, 2, p_x);
    free_graph_stat(head_stat);
    free(costs), free(prizes), free(edges);
    return results;
}

static PyObject *proj_tail(PyObject *self, PyObject *args) {
    /**
     * DO NOT call this function directly, use the Python Wrapper instead.
     * list of args:
     * args[0]: ndarray dim=(m,2) -- edges of the graph.
     * args[1]: ndarray dim=(m,)  -- weights (positive) of the graph.
     * args[2]: ndarray dim=(n,)  -- the vector needs to be projected.
     * args[3]: integer np.int32  -- number of connected components returned.
     * args[4]: integer np.int32  -- sparsity (positive) parameter.
     * args[5]: double np.float64 -- budget of the graph model.
     * args[6]: double np.float64 -- nu. default is 2.5
     * args[7]: integer np.int32  -- maximal # of iterations in the loop.
     * args[8]: double np.float32 -- error tolerance for minimum nonzero.
     * args[9]: integer np.int32  -- root(default is -1).
     * args[10]: string string    -- pruning ['simple', 'gw', 'strong'].
     * args[11]: double np.float64-- epsilon to control the presion of PCST.
     * args[12]: integer np.int32 -- verbosity level
     * @return: (re_nodes, re_edges, p_x)
     * re_nodes: projected nodes
     * re_edges: projected edges (indices)
     * p_x: projection of x.
     */
    if (self != NULL) { return NULL; }
    PyArrayObject *edges_, *weights_, *x_;
    int g, s, root, max_iter, verbose;
    double budget, nu, epsilon, err_tol;
    char *pruning;
    //edges, weights, x, g, s, budget, nu, max_iter, err_tol,
    //                     root, pruning, epsilon, verbose
    if (!PyArg_ParseTuple(
            args, "O!O!O!iiddidizdi", &PyArray_Type, &edges_, &PyArray_Type,
            &weights_, &PyArray_Type, &x_, &g, &s, &budget, &nu,
            &max_iter, &err_tol, &root, &pruning, &epsilon, &verbose)) {
        return NULL;
    }
    long n = x_->dimensions[0];  // number of nodes
    long m = edges_->dimensions[0];     // number of edges
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    double *prizes = malloc(sizeof(double) * n);
    double *costs = malloc(sizeof(double) * m);
    double *x = (double *) PyArray_DATA(x_);
    for (int i = 0; i < m; i++) {
        edges[i].first = *(int *) PyArray_GETPTR2(edges_, i, 0);
        edges[i].second = *(int *) PyArray_GETPTR2(edges_, i, 1);
        double *wei = (double *) PyArray_GETPTR1(weights_, i);
        costs[i] = (*wei + budget / s);
    }
    for (int i = 0; i < n; i++) {
        prizes[i] = (x[i]) * (x[i]);
    }
    double C = 2. * budget;
    PyObject *results = PyTuple_New(3);
    PyObject *p_x = PyList_New(n);      // projected x
    for (int i = 0; i < n; i++) {
        prizes[i] = (x[i]) * (x[i]);
        PyList_SetItem(p_x, i, PyFloat_FromDouble(0.0));
    }
    GraphStat *tail_stat = make_graph_stat((int) n, (int) m);
    tail_proj_exact(
            edges, costs, prizes, g, C, nu, max_iter, err_tol, root, GWPruning,
            epsilon, (int) n, (int) m, verbose, tail_stat);
    PyObject *re_nodes = PyList_New(tail_stat->re_nodes->size);
    PyObject *re_edges = PyList_New(tail_stat->re_edges->size);
    for (int i = 0; i < tail_stat->re_nodes->size; i++) {
        int node_i = tail_stat->re_nodes->array[i];
        PyList_SetItem(re_nodes, i, PyInt_FromLong(node_i));
        PyList_SetItem(p_x, node_i, PyFloat_FromDouble(x[node_i]));
    }
    for (int i = 0; i < tail_stat->re_edges->size; i++) {
        PyList_SetItem(re_edges, i,
                       PyInt_FromLong(tail_stat->re_edges->array[i]));
    }
    PyTuple_SetItem(results, 0, re_nodes);
    PyTuple_SetItem(results, 1, re_edges);
    PyTuple_SetItem(results, 2, p_x);
    free_graph_stat(tail_stat), free(costs), free(prizes), free(edges);
    return results;
}

static PyObject *proj_pcst(PyObject *self, PyObject *args) {
    /**
     * DO NOT call this function directly, use the Python Wrapper instead.
     * list of args:
     * args[0]: ndarray dim=(m,2) -- edges of the graph.
     * args[1]: ndarray dim=(n,)  -- prizes of the graph.
     * args[2]: ndarray dim=(m,)  -- costs on nodes.
     * args[3]: integer np.int32  -- root(default is -1).
     * args[4]: integer np.int32  -- number of connected components returned.
     * args[5]: string string     -- pruning none, simple, gw, strong.
     * args[6]: double np.float32 -- epsilon to control the precision.
     * args[7]: integer np.int32  -- verbosity level
     * @return: (re_nodes, re_edges)
     * re_nodes: result nodes
     * re_edges: result edges
     */
    if (self != NULL) { return NULL; }
    PyArrayObject *edges_, *prizes_, *weights_;
    int g, root, verbose;
    char *pruning;
    double epsilon;
    if (!PyArg_ParseTuple(args, "O!O!O!iizdi", &PyArray_Type, &edges_,
                          &PyArray_Type, &prizes_, &PyArray_Type,
                          &weights_, &root, &g, &pruning,
                          &epsilon, &verbose)) { return NULL; }
    long n = prizes_->dimensions[0];    // number of nodes
    long m = edges_->dimensions[0];     // number of edges
    EdgePair *edges = malloc(sizeof(EdgePair) * m);
    double *prizes = (double *) PyArray_DATA(prizes_);
    double *costs = (double *) PyArray_DATA(weights_);
    for (int i = 0; i < m; i++) {
        edges[i].first = *(int *) PyArray_GETPTR2(edges_, i, 0);
        edges[i].second = *(int *) PyArray_GETPTR2(edges_, i, 1);
    }
    GraphStat *stat = make_graph_stat((int) n, (int) m);
    PCST *pcst = make_pcst(edges, prizes, costs, root,
                           g, epsilon, GWPruning, (int) n, (int) m, verbose);
    run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
    PyObject *results = PyTuple_New(2);
    PyObject *re_nodes = PyList_New(stat->re_nodes->size);
    PyObject *re_edges = PyList_New(stat->re_edges->size);
    for (int i = 0; i < stat->re_nodes->size; i++) {
        PyList_SetItem(re_nodes, i, PyInt_FromLong(stat->re_nodes->array[i]));
    }
    for (int i = 0; i < stat->re_edges->size; i++) {
        PyList_SetItem(re_edges, i, PyInt_FromLong(stat->re_edges->array[i]));
    }
    PyTuple_SetItem(results, 0, re_nodes);
    PyTuple_SetItem(results, 1, re_edges);
    free(edges), free(stat);
    return results;
}


/**
 * Here we defined 6 functions.
 *
 * 1. proj_head
 * 2. proj_tail
 * 3. proj_pcst
 * 4. mst: minimal_spanning_tree
 * 5. ghtp_logistic: gradient hard thresholding pursuit for logistic function.
 * 6. graph_ghtp_logistic: graph-constrained ghtp_logistic
 * above 6 functions had been tested on Python2.7.
 *
 * each function is defined in the proj module.
 * 1. function name in your Python program,
 * 2. function name defined in c program,
 * 3. flags, usually is METH_VARARGS
 * 4. some docs.
 */
static PyMethodDef proj_methods[] = {
        {"proj_head", (PyCFunction) proj_head, METH_VARARGS, "Head docs"},
        {"proj_tail", (PyCFunction) proj_tail, METH_VARARGS, "Tail docs"},
        {"proj_pcst", (PyCFunction) proj_pcst, METH_VARARGS, "PCST docs"},
        {NULL, NULL, 0, NULL}};


/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC initproj_module() {
    Py_InitModule3("proj_module", proj_methods, "some docs for head proj.");
    import_array();
}