//
// Created by baojian on 8/11/18.
//
#include <Python.h>
#include <numpy/arrayobject.h>
#include "sparse_algorithms.h"


bool get_data(int n, int p, int m, double *x_tr, double *y_tr, double *w0,
              EdgePair *edges, double *weights,
              PyArrayObject *x_tr_, PyArrayObject *y_tr_,
              PyArrayObject *w0_, PyArrayObject *edges_,
              PyArrayObject *weights_) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            x_tr[i * p + j] = *(double *) PyArray_GETPTR2(x_tr_, i, j);
        }
        y_tr[i] = *(double *) PyArray_GETPTR1(y_tr_, i);
    }
    for (i = 0; i < (p + 1); i++) {
        w0[i] = *(double *) PyArray_GETPTR1(w0_, i);;
    }
    if (edges != NULL) {
        for (i = 0; i < m; i++) {
            edges[i].first = *(int *) PyArray_GETPTR2(edges_, i, 0);
            edges[i].second = *(int *) PyArray_GETPTR2(edges_, i, 1);
            weights[i] = *(double *) PyArray_GETPTR1(weights_, i);
        }
    }
    return true;
}

PyObject *batch_get_result(int p, int max_iter,
                           double total_time, double *wt,
                           double *losses) {
    PyObject *results = PyTuple_New(3);
    PyObject *re_wt = PyList_New(p + 1);
    for (int i = 0; i < p + 1; i++) {
        PyList_SetItem(re_wt, i, PyFloat_FromDouble(wt[i]));
    }
    PyObject *re_losses = PyList_New(max_iter);
    for (int i = 0; i < max_iter; i++) {
        PyList_SetItem(re_losses, i, PyFloat_FromDouble(losses[i]));
    }
    PyObject *re_total_time = PyFloat_FromDouble(total_time);
    PyTuple_SetItem(results, 0, re_wt);
    PyTuple_SetItem(results, 1, re_losses);
    PyTuple_SetItem(results, 2, re_total_time);
    return results;
}


static PyObject *online_sgd_l1_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    int n, p, i, verbose;
    double gamma, lambda;
    // x_tr, y_tr, w0, gamma, lambda_, verbose
    if (!PyArg_ParseTuple(args, "O!O!O!ddi", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &gamma, &lambda, &verbose)) { return NULL; }
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    double *x_tr = malloc(sizeof(double) * (n * p));
    double *y_tr = malloc(sizeof(double) * n);
    double *w0 = malloc(sizeof(double) * (p + 1));
    get_data(n, p, -1, x_tr, y_tr, w0, NULL, NULL,
             x_tr_, y_tr_, w0_, NULL, NULL);
    double *wt = malloc(sizeof(double) * (p + 1));
    double *wt_bar = malloc(sizeof(double) * (p + 1));
    int *nonzeros_wt = malloc(sizeof(int) * n);
    int *nonzeros_wt_bar = malloc(sizeof(int) * n);
    double *pred_prob_wt = malloc(sizeof(double) * n);
    double *pred_label_wt = malloc(sizeof(double) * n);
    double *pred_prob_wt_bar = malloc(sizeof(double) * n);
    double *pred_label_wt_bar = malloc(sizeof(double) * n);
    double *losses = malloc(sizeof(double) * n);
    double total_time = 0.0;
    int *missed_wt = malloc(sizeof(int) * n);
    int *missed_wt_bar = malloc(sizeof(int) * n);
    algo_online_sgd_l1_logit(
            x_tr, y_tr, w0, p, n, gamma, lambda, verbose, wt, wt_bar,
            nonzeros_wt, nonzeros_wt_bar, pred_prob_wt, pred_label_wt,
            pred_prob_wt_bar, pred_label_wt_bar, losses, missed_wt,
            missed_wt_bar, &total_time);
    PyObject *results = PyTuple_New(12);
    PyObject *re_wt = PyList_New(p + 1), *re_wt_bar = PyList_New(p + 1);
    PyObject *re_nonzeros_wt = PyList_New(n);
    PyObject *re_nonzeros_wt_bar = PyList_New(n);
    for (i = 0; i < p + 1; i++) {
        PyList_SetItem(re_wt, i, PyFloat_FromDouble(wt[i]));
        PyList_SetItem(re_wt_bar, i, PyFloat_FromDouble(wt_bar[i]));
    }
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_nonzeros_wt, i,
                       PyInt_FromLong(nonzeros_wt[i]));
        PyList_SetItem(re_nonzeros_wt_bar, i,
                       PyInt_FromLong(nonzeros_wt_bar[i]));
    }
    PyObject *re_pred_prob_wt = PyList_New(n);
    PyObject *re_pred_label_wt = PyList_New(n);
    PyObject *re_pred_prob_wt_bar = PyList_New(n);
    PyObject *re_pred_label_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_pred_prob_wt, i,
                       PyFloat_FromDouble(pred_prob_wt[i]));
        PyList_SetItem(re_pred_label_wt, i,
                       PyFloat_FromDouble(pred_label_wt[i]));
        PyList_SetItem(re_pred_prob_wt_bar, i,
                       PyFloat_FromDouble(pred_prob_wt_bar[i]));
        PyList_SetItem(re_pred_label_wt_bar, i,
                       PyFloat_FromDouble(pred_label_wt_bar[i]));
    }
    PyObject *re_losses = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_losses, i, PyFloat_FromDouble(losses[i]));
    }
    PyObject *re_missed_wt = PyList_New(n);
    PyObject *re_missed_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_missed_wt, i, PyInt_FromLong(missed_wt[i]));
        PyList_SetItem(re_missed_wt_bar, i, PyInt_FromLong(missed_wt_bar[i]));
    }
    PyObject *re_total_time = PyFloat_FromDouble(total_time);
    PyTuple_SetItem(results, 0, re_wt);
    PyTuple_SetItem(results, 1, re_wt_bar);
    PyTuple_SetItem(results, 2, re_nonzeros_wt);
    PyTuple_SetItem(results, 3, re_nonzeros_wt_bar);
    PyTuple_SetItem(results, 4, re_pred_prob_wt);
    PyTuple_SetItem(results, 5, re_pred_label_wt);
    PyTuple_SetItem(results, 6, re_pred_prob_wt_bar);
    PyTuple_SetItem(results, 7, re_pred_label_wt_bar);
    PyTuple_SetItem(results, 8, re_losses);
    PyTuple_SetItem(results, 9, re_missed_wt);
    PyTuple_SetItem(results, 10, re_missed_wt_bar);
    PyTuple_SetItem(results, 11, re_total_time);
    // free all used memory
    free(pred_label_wt), free(pred_prob_wt);
    free(w0), free(wt), free(wt_bar), free(missed_wt), free(missed_wt_bar);
    free(nonzeros_wt), free(nonzeros_wt_bar);
    free(losses), free(x_tr), free(y_tr);
    return results;
}

static PyObject *online_rda_l1_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error !!\n");
        return NULL;
    }
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    int n, p, i, verbose;
    double lambda, gamma, rho;
    //x_tr,y_tr,w0,lambda,gamma,rho,verbose
    if (!PyArg_ParseTuple(args, "O!O!O!dddi", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &lambda, &gamma, &rho, &verbose)) { return NULL; }
    if (lambda <= 0.0 || gamma <= 0.0 || rho <= 0.0) {
        printf("error! negative values.\n");
        exit(0);
    }
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    double *x_tr = malloc(n * p * sizeof(double));
    double *y_tr = malloc(n * sizeof(double));
    double *w0 = malloc((p + 1) * sizeof(double));
    get_data(n, p, -1, x_tr, y_tr, w0, NULL, NULL,
             x_tr_, y_tr_, w0_, NULL, NULL);
    double *wt = malloc(sizeof(double) * (p + 1));
    double *wt_bar = malloc(sizeof(double) * (p + 1));
    int *nonzeros_wt = malloc(sizeof(int) * n);
    int *nonzeros_wt_bar = malloc(sizeof(int) * n);
    double *pred_prob_wt = malloc(sizeof(double) * n);
    double *pred_label_wt = malloc(sizeof(double) * n);
    double *pred_prob_wt_bar = malloc(sizeof(double) * n);
    double *pred_label_wt_bar = malloc(sizeof(double) * n);
    double *losses = malloc(sizeof(double) * n);
    double total_time = 0.0;
    int *missed_wt = malloc(sizeof(int) * n);
    int *missed_wt_bar = malloc(sizeof(int) * n);
    algo_online_rda_l1_logit(
            x_tr, y_tr, w0, p, n, lambda, gamma, rho, verbose, wt, wt_bar,
            nonzeros_wt, nonzeros_wt_bar, pred_prob_wt, pred_label_wt,
            pred_prob_wt_bar, pred_label_wt_bar, losses, missed_wt,
            missed_wt_bar, &total_time);
    PyObject *results = PyTuple_New(12);
    PyObject *re_wt = PyList_New(p + 1), *re_wt_bar = PyList_New(p + 1);
    PyObject *re_nonzeros_wt = PyList_New(n);
    PyObject *re_nonzeros_wt_bar = PyList_New(n);
    for (i = 0; i < p + 1; i++) {
        PyList_SetItem(re_wt, i, PyFloat_FromDouble(wt[i]));
        PyList_SetItem(re_wt_bar, i, PyFloat_FromDouble(wt_bar[i]));
    }
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_nonzeros_wt, i,
                       PyInt_FromLong(nonzeros_wt[i]));
        PyList_SetItem(re_nonzeros_wt_bar, i,
                       PyInt_FromLong(nonzeros_wt_bar[i]));
    }
    PyObject *re_pred_prob_wt = PyList_New(n);
    PyObject *re_pred_label_wt = PyList_New(n);
    PyObject *re_pred_prob_wt_bar = PyList_New(n);
    PyObject *re_pred_label_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_pred_prob_wt, i,
                       PyFloat_FromDouble(pred_prob_wt[i]));
        PyList_SetItem(re_pred_label_wt, i,
                       PyFloat_FromDouble(pred_label_wt[i]));
        PyList_SetItem(re_pred_prob_wt_bar, i,
                       PyFloat_FromDouble(pred_prob_wt_bar[i]));
        PyList_SetItem(re_pred_label_wt_bar, i,
                       PyFloat_FromDouble(pred_label_wt_bar[i]));
    }
    PyObject *re_losses = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_losses, i, PyFloat_FromDouble(losses[i]));
    }
    PyObject *re_missed_wt = PyList_New(n);
    PyObject *re_missed_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_missed_wt, i, PyInt_FromLong(missed_wt[i]));
        PyList_SetItem(re_missed_wt_bar, i, PyInt_FromLong(missed_wt_bar[i]));
    }
    PyObject *re_total_time = PyFloat_FromDouble(total_time);
    PyTuple_SetItem(results, 0, re_wt);
    PyTuple_SetItem(results, 1, re_wt_bar);
    PyTuple_SetItem(results, 2, re_nonzeros_wt);
    PyTuple_SetItem(results, 3, re_nonzeros_wt_bar);
    PyTuple_SetItem(results, 4, re_pred_prob_wt);
    PyTuple_SetItem(results, 5, re_pred_label_wt);
    PyTuple_SetItem(results, 6, re_pred_prob_wt_bar);
    PyTuple_SetItem(results, 7, re_pred_label_wt_bar);
    PyTuple_SetItem(results, 8, re_losses);
    PyTuple_SetItem(results, 9, re_missed_wt);
    PyTuple_SetItem(results, 10, re_missed_wt_bar);
    PyTuple_SetItem(results, 11, re_total_time);
    // free all used memory
    free(pred_label_wt), free(pred_prob_wt);
    free(w0), free(wt), free(wt_bar), free(missed_wt), free(missed_wt_bar);
    free(nonzeros_wt), free(nonzeros_wt_bar);
    free(losses), free(x_tr), free(y_tr);
    return results;
}


static PyObject *online_sgd_l2_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    int n, p, i, verbose;
    double lr, eta;
    if (!PyArg_ParseTuple(args, "O!O!O!ddi", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &lr, &eta, &verbose)) { return NULL; }
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    double *x_tr = (double *) malloc(n * p * sizeof(double));
    double *y_tr = (double *) malloc(n * sizeof(double));
    double *w0 = (double *) malloc((p + 1) * sizeof(double));

    get_data(n, p, -1, x_tr, y_tr, w0, NULL, NULL,
             x_tr_, y_tr_, w0_, NULL, NULL);

    double *wt = malloc(sizeof(double) * (p + 1));
    double *wt_bar = malloc(sizeof(double) * (p + 1));
    int *nonzeros_wt = malloc(sizeof(int) * n);
    int *nonzeros_wt_bar = malloc(sizeof(int) * n);
    double *pred_prob_wt = malloc(sizeof(double) * n);
    double *pred_label_wt = malloc(sizeof(double) * n);
    double *pred_prob_wt_bar = malloc(sizeof(double) * n);
    double *pred_label_wt_bar = malloc(sizeof(double) * n);
    double *losses = malloc(sizeof(double) * n);
    double total_time = 0.0;
    int *missed_wt = malloc(sizeof(int) * n);
    int *missed_wt_bar = malloc(sizeof(int) * n);
    algo_online_sgd_l2_logit(x_tr, y_tr, w0, p, n, lr, eta, verbose, wt,
                             wt_bar,
                             nonzeros_wt, nonzeros_wt_bar, pred_prob_wt,
                             pred_label_wt, pred_prob_wt_bar,
                             pred_label_wt_bar,
                             losses, missed_wt, missed_wt_bar, &total_time);
    PyObject *results = PyTuple_New(12);
    PyObject *re_wt = PyList_New(p + 1), *re_wt_bar = PyList_New(p + 1);
    PyObject *re_nonzeros_wt = PyList_New(n);
    PyObject *re_nonzeros_wt_bar = PyList_New(n);
    for (i = 0; i < p + 1; i++) {
        PyList_SetItem(re_wt, i, PyFloat_FromDouble(wt[i]));
        PyList_SetItem(re_wt_bar, i, PyFloat_FromDouble(wt_bar[i]));
    }
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_nonzeros_wt, i,
                       PyInt_FromLong(nonzeros_wt[i]));
        PyList_SetItem(re_nonzeros_wt_bar, i,
                       PyInt_FromLong(nonzeros_wt_bar[i]));
    }
    PyObject *re_pred_prob_wt = PyList_New(n);
    PyObject *re_pred_label_wt = PyList_New(n);
    PyObject *re_pred_prob_wt_bar = PyList_New(n);
    PyObject *re_pred_label_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_pred_prob_wt, i,
                       PyFloat_FromDouble(pred_prob_wt[i]));
        PyList_SetItem(re_pred_label_wt, i,
                       PyFloat_FromDouble(pred_label_wt[i]));
        PyList_SetItem(re_pred_prob_wt_bar, i,
                       PyFloat_FromDouble(pred_prob_wt_bar[i]));
        PyList_SetItem(re_pred_label_wt_bar, i,
                       PyFloat_FromDouble(pred_label_wt_bar[i]));
    }
    PyObject *re_losses = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_losses, i, PyFloat_FromDouble(losses[i]));
    }
    PyObject *re_missed_wt = PyList_New(n);
    PyObject *re_missed_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_missed_wt, i, PyInt_FromLong(missed_wt[i]));
        PyList_SetItem(re_missed_wt_bar, i, PyInt_FromLong(missed_wt_bar[i]));
    }
    PyObject *re_total_time = PyFloat_FromDouble(total_time);
    PyTuple_SetItem(results, 0, re_wt);
    PyTuple_SetItem(results, 1, re_wt_bar);
    PyTuple_SetItem(results, 2, re_nonzeros_wt);
    PyTuple_SetItem(results, 3, re_nonzeros_wt_bar);
    PyTuple_SetItem(results, 4, re_pred_prob_wt);
    PyTuple_SetItem(results, 5, re_pred_label_wt);
    PyTuple_SetItem(results, 6, re_pred_prob_wt_bar);
    PyTuple_SetItem(results, 7, re_pred_label_wt_bar);
    PyTuple_SetItem(results, 8, re_losses);
    PyTuple_SetItem(results, 9, re_missed_wt);
    PyTuple_SetItem(results, 10, re_missed_wt_bar);
    PyTuple_SetItem(results, 11, re_total_time);
    // free all used memory
    free(pred_label_wt), free(pred_prob_wt);
    free(w0), free(wt), free(wt_bar), free(missed_wt), free(missed_wt_bar);
    free(nonzeros_wt), free(nonzeros_wt_bar);
    free(losses), free(x_tr), free(y_tr);
    return results;
}

//input: x_tr, y_tr, w0, lr, eta, edges, weights, sparsity
static PyObject *online_graph_ghtp_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    int i, n, p, m, sparsity, verbose;
    double lr, eta, *x_tr, *y_tr, *w0, *weights;
    EdgePair *edges;
    PyArrayObject *x_tr_, *y_tr_, *w0_, *edges_, *weights_;
    if (!PyArg_ParseTuple(args, "O!O!O!ddO!O!ii", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &lr, &eta, &PyArray_Type, &edges_, &PyArray_Type,
                          &weights_, &sparsity, &verbose)) { return NULL; }
    n = (int) x_tr_->dimensions[0];         // number of samples
    p = (int) x_tr_->dimensions[1];         // number of features(nodes)
    m = (int) edges_->dimensions[0];        // number of edges
    x_tr = malloc(n * p * sizeof(double));
    y_tr = malloc(n * sizeof(double));
    w0 = malloc((p + 1) * sizeof(double));
    edges = malloc(sizeof(EdgePair) * m);
    weights = malloc(sizeof(double) * m);
    get_data(n, p, m, x_tr, y_tr, w0, edges, weights,
             x_tr_, y_tr_, w0_, edges_, weights_);
    int g = 1;
    int num_pcst = 0;
    int *nonzeros_wt = malloc(sizeof(int) * n);
    int *nonzeros_wt_bar = malloc(sizeof(int) * n);
    int *missed_wt = malloc(sizeof(int) * n);
    int *missed_wt_bar = malloc(sizeof(int) * n);

    double *wt = malloc(sizeof(double) * (p + 1));
    double *wt_bar = malloc(sizeof(double) * (p + 1));
    double *pred_prob_wt = malloc(sizeof(double) * n);
    double *pred_prob_wt_bar = malloc(sizeof(double) * n);
    double *pred_label_wt = malloc(sizeof(double) * n);
    double *pred_label_wt_bar = malloc(sizeof(double) * n);
    double *losses = malloc(sizeof(double) * n);
    double run_time_head = 0.0;
    double run_time_tail = 0.0;
    double total_time = 0.0;
    Array *f_nodes = malloc(sizeof(Array));
    Array *f_edges = malloc(sizeof(Array));
    f_nodes->array = malloc(sizeof(int) * p);
    f_edges->array = malloc(sizeof(int) * p);

    algo_online_graph_ghtp_logit(
            edges, weights, g, sparsity, p, m, x_tr, y_tr, w0, n, lr, eta,
            verbose, wt, wt_bar, nonzeros_wt, nonzeros_wt_bar, pred_prob_wt,
            pred_label_wt, pred_prob_wt_bar, pred_label_wt_bar, f_nodes,
            f_edges, &num_pcst, losses, &run_time_head, &run_time_tail,
            missed_wt, missed_wt_bar, &total_time);
    // to save results
    PyObject *results = PyTuple_New(17);
    PyObject *re_wt = PyList_New(p + 1);
    PyObject *re_wt_bar = PyList_New(p + 1);
    PyObject *re_nonzeros_wt = PyList_New(n);
    PyObject *re_nonzeros_wt_bar = PyList_New(n);
    for (i = 0; i < p + 1; i++) {
        PyList_SetItem(re_wt, i, PyFloat_FromDouble(wt[i]));
        PyList_SetItem(re_wt_bar, i, PyFloat_FromDouble(wt_bar[i]));
    }
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_nonzeros_wt, i, PyInt_FromLong(nonzeros_wt[i]));
        PyList_SetItem(re_nonzeros_wt_bar, i,
                       PyInt_FromLong(nonzeros_wt_bar[i]));
    }
    PyObject *re_pred_prob_wt = PyList_New(n);
    PyObject *re_pred_label_wt = PyList_New(n);
    PyObject *re_pred_prob_wt_bar = PyList_New(n);
    PyObject *re_pred_label_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_pred_prob_wt, i,
                       PyFloat_FromDouble(pred_prob_wt[i]));
        PyList_SetItem(re_pred_label_wt_bar, i,
                       PyFloat_FromDouble(pred_label_wt_bar[i]));
        PyList_SetItem(re_pred_prob_wt_bar, i,
                       PyFloat_FromDouble(pred_prob_wt_bar[i]));
        PyList_SetItem(re_pred_label_wt, i,
                       PyFloat_FromDouble(pred_label_wt[i]));
    }
    PyObject *re_nodes = PyList_New(f_nodes->size);
    PyObject *re_edges = PyList_New(f_edges->size);
    for (i = 0; i < f_nodes->size; i++) {
        PyList_SetItem(re_nodes, i, PyFloat_FromDouble(f_nodes->array[i]));
    }
    for (i = 0; i < f_edges->size; i++) {
        PyList_SetItem(re_edges, i, PyFloat_FromDouble(f_edges->array[i]));
    }
    PyObject *re_num_pcst = PyInt_FromLong(num_pcst);
    PyObject *re_losses = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_losses, i, PyFloat_FromDouble(losses[i]));
    }
    PyObject *re_run_time_head = PyFloat_FromDouble(run_time_head);
    PyObject *re_run_time_tail = PyFloat_FromDouble(run_time_tail);
    PyObject *re_missed_wt = PyList_New(n);
    PyObject *re_missed_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_missed_wt, i, PyInt_FromLong(missed_wt[i]));
        PyList_SetItem(re_missed_wt_bar, i, PyInt_FromLong(missed_wt_bar[i]));
    }
    PyObject *re_total_time = PyFloat_FromDouble(total_time);
    PyTuple_SetItem(results, 0, re_wt);
    PyTuple_SetItem(results, 1, re_wt_bar);
    PyTuple_SetItem(results, 2, re_nonzeros_wt);
    PyTuple_SetItem(results, 3, re_nonzeros_wt_bar);
    PyTuple_SetItem(results, 4, re_pred_prob_wt);
    PyTuple_SetItem(results, 5, re_pred_label_wt);
    PyTuple_SetItem(results, 6, re_pred_prob_wt_bar);
    PyTuple_SetItem(results, 7, re_pred_label_wt_bar);
    PyTuple_SetItem(results, 8, re_nodes);
    PyTuple_SetItem(results, 9, re_edges);
    PyTuple_SetItem(results, 10, re_num_pcst);
    PyTuple_SetItem(results, 11, re_losses);
    PyTuple_SetItem(results, 12, re_run_time_head);
    PyTuple_SetItem(results, 13, re_run_time_tail);
    PyTuple_SetItem(results, 14, re_missed_wt);
    PyTuple_SetItem(results, 15, re_missed_wt_bar);
    PyTuple_SetItem(results, 16, re_total_time);

    //free all used memory
    free(x_tr), free(y_tr);
    free(w0), free(edges), free(weights);
    free(nonzeros_wt), free(nonzeros_wt_bar);
    free(missed_wt), free(missed_wt_bar);
    free(wt), free(wt_bar);
    free(pred_label_wt), free(pred_prob_wt);
    free(pred_label_wt_bar), free(pred_prob_wt_bar);
    free(f_nodes->array), free(f_edges->array);
    free(f_nodes), free(f_edges);
    free(losses);
    return results;
}

//input: x_tr, y_tr, w0, lr, eta, edges, weights, sparsity
static PyObject *online_graph_iht_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    int i, n, p, m, sparsity, verbose;
    double lr, eta, *x_tr, *y_tr, *w0, *weights;
    EdgePair *edges;
    PyArrayObject *x_tr_, *y_tr_, *w0_, *edges_, *weights_;
    if (!PyArg_ParseTuple(args, "O!O!O!ddO!O!ii", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &lr, &eta, &PyArray_Type, &edges_, &PyArray_Type,
                          &weights_, &sparsity, &verbose)) { return NULL; }
    n = (int) (x_tr_->dimensions[0]);       // number of samples
    p = (int) (x_tr_->dimensions[1]);       // number of features(nodes)
    m = (int) edges_->dimensions[0];        // number of edges
    x_tr = malloc(n * p * sizeof(double));
    y_tr = malloc(n * sizeof(double));
    w0 = malloc((p + 1) * sizeof(double));
    edges = malloc(sizeof(EdgePair) * m);
    weights = malloc(sizeof(double) * m);
    get_data(n, p, m, x_tr, y_tr, w0, edges, weights,
             x_tr_, y_tr_, w0_, edges_, weights_);

    int g = 1;
    int num_pcst = 0;
    int *nonzeros_wt = malloc(sizeof(int) * n);
    int *nonzeros_wt_bar = malloc(sizeof(int) * n);
    int *missed_wt = malloc(sizeof(int) * n);
    int *missed_wt_bar = malloc(sizeof(int) * n);

    double *wt = malloc(sizeof(double) * (p + 1));
    double *wt_bar = malloc(sizeof(double) * (p + 1));
    double *pred_prob_wt = malloc(sizeof(double) * n);
    double *pred_prob_wt_bar = malloc(sizeof(double) * n);
    double *pred_label_wt = malloc(sizeof(double) * n);
    double *pred_label_wt_bar = malloc(sizeof(double) * n);
    double *losses = malloc(sizeof(double) * n);
    double run_time_head = 0.0;
    double run_time_tail = 0.0;
    double total_time = 0.0;
    Array *f_nodes = malloc(sizeof(Array));
    Array *f_edges = malloc(sizeof(Array));
    f_nodes->array = malloc(sizeof(int) * p);
    f_edges->array = malloc(sizeof(int) * p);

    algo_online_graph_iht_logit(
            edges, weights, g, sparsity, p, m, x_tr, y_tr, w0, n, lr, eta,
            verbose, wt, wt_bar, nonzeros_wt, nonzeros_wt_bar, pred_prob_wt,
            pred_label_wt, pred_prob_wt_bar, pred_label_wt_bar, f_nodes,
            f_edges, &num_pcst, losses, &run_time_head, &run_time_tail,
            missed_wt, missed_wt_bar, &total_time);    // to save results
    PyObject *results = PyTuple_New(17);
    PyObject *re_wt = PyList_New(p + 1), *re_wt_bar = PyList_New(p + 1);
    PyObject *re_nonzeros_wt = PyList_New(n);
    PyObject *re_nonzeros_wt_bar = PyList_New(n);
    for (i = 0; i < p + 1; i++) {
        PyList_SetItem(re_wt, i, PyFloat_FromDouble(wt[i]));
        PyList_SetItem(re_wt_bar, i, PyFloat_FromDouble(wt_bar[i]));
    }
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_nonzeros_wt, i, PyInt_FromLong(nonzeros_wt[i]));
        PyList_SetItem(re_nonzeros_wt_bar, i,
                       PyInt_FromLong(nonzeros_wt_bar[i]));
    }
    PyObject *re_pred_prob_wt = PyList_New(n);
    PyObject *re_pred_prob_wt_bar = PyList_New(n);
    PyObject *re_pred_label_wt = PyList_New(n);
    PyObject *re_pred_label_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_pred_prob_wt, i,
                       PyFloat_FromDouble(pred_prob_wt[i]));
        PyList_SetItem(re_pred_label_wt, i,
                       PyFloat_FromDouble(pred_label_wt[i]));
        PyList_SetItem(re_pred_prob_wt_bar, i,
                       PyFloat_FromDouble(pred_prob_wt_bar[i]));
        PyList_SetItem(re_pred_label_wt_bar, i,
                       PyFloat_FromDouble(pred_label_wt_bar[i]));
    }
    PyObject *re_nodes = PyList_New(f_nodes->size);
    PyObject *re_edges = PyList_New(f_edges->size);
    for (i = 0; i < f_nodes->size; i++) {
        PyList_SetItem(re_nodes, i,
                       PyFloat_FromDouble(f_nodes->array[i]));
    }
    for (i = 0; i < f_edges->size; i++) {
        PyList_SetItem(re_edges, i,
                       PyFloat_FromDouble(f_edges->array[i]));
    }
    PyObject *re_num_pcst = PyInt_FromLong(num_pcst);
    PyObject *re_losses = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_losses, i, PyFloat_FromDouble(losses[i]));
    }
    PyObject *re_run_time_head = PyFloat_FromDouble(run_time_head);
    PyObject *re_run_time_tail = PyFloat_FromDouble(run_time_tail);
    PyObject *re_missed_wt = PyList_New(n);
    PyObject *re_missed_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_missed_wt, i, PyInt_FromLong(missed_wt[i]));
        PyList_SetItem(re_missed_wt_bar, i, PyInt_FromLong(missed_wt_bar[i]));
    }
    PyObject *re_total_time = PyFloat_FromDouble(total_time);
    PyTuple_SetItem(results, 0, re_wt);
    PyTuple_SetItem(results, 1, re_wt_bar);
    PyTuple_SetItem(results, 2, re_nonzeros_wt);
    PyTuple_SetItem(results, 3, re_nonzeros_wt_bar);
    PyTuple_SetItem(results, 4, re_pred_prob_wt);
    PyTuple_SetItem(results, 5, re_pred_label_wt);
    PyTuple_SetItem(results, 6, re_pred_prob_wt_bar);
    PyTuple_SetItem(results, 7, re_pred_label_wt_bar);
    PyTuple_SetItem(results, 8, re_nodes);
    PyTuple_SetItem(results, 9, re_edges);
    PyTuple_SetItem(results, 10, re_num_pcst);
    PyTuple_SetItem(results, 11, re_losses);
    PyTuple_SetItem(results, 12, re_run_time_head);
    PyTuple_SetItem(results, 13, re_run_time_tail);
    PyTuple_SetItem(results, 14, re_missed_wt);
    PyTuple_SetItem(results, 15, re_missed_wt_bar);
    PyTuple_SetItem(results, 16, re_total_time);
    //free all used memory
    free(x_tr), free(y_tr);
    free(w0), free(edges), free(weights);
    free(nonzeros_wt), free(nonzeros_wt_bar);
    free(missed_wt), free(missed_wt_bar);
    free(wt), free(wt_bar);
    free(pred_label_wt), free(pred_prob_wt);
    free(pred_label_wt_bar), free(pred_prob_wt_bar);
    free(f_nodes->array), free(f_edges->array);
    free(f_nodes), free(f_edges);
    free(losses);
    return results;
}


static PyObject *online_iht_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    int n, p, i, sparsity, verbose;
    double lr, eta;
    if (!PyArg_ParseTuple(args, "O!O!O!ddii", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &lr, &eta, &sparsity, &verbose)) { return NULL; }
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    double *x_tr = (double *) malloc(n * p * sizeof(double));
    double *y_tr = (double *) malloc(n * sizeof(double));
    double *w0 = (double *) malloc((p + 1) * sizeof(double));
    get_data(n, p, -1, x_tr, y_tr, w0, NULL, NULL,
             x_tr_, y_tr_, w0_, NULL, NULL);

    double *wt = malloc(sizeof(double) * (p + 1));
    double *wt_bar = malloc(sizeof(double) * (p + 1));
    int *nonzeros_wt = malloc(sizeof(int) * n);
    int *nonzeros_wt_bar = malloc(sizeof(int) * n);
    double *pred_prob_wt = malloc(sizeof(double) * n);
    double *pred_label_wt = malloc(sizeof(double) * n);
    double *pred_prob_wt_bar = malloc(sizeof(double) * n);
    double *pred_label_wt_bar = malloc(sizeof(double) * n);
    double *losses = malloc(sizeof(double) * n);
    double total_time = 0.0;
    int *missed_wt = malloc(sizeof(int) * n);
    int *missed_wt_bar = malloc(sizeof(int) * n);
    if (sparsity > p) {
        printf("the parameter of sparsity is too large!\n");
        printf("sparsity is: %d while p: %d!\n", sparsity, p);
        exit(0);
    }
    algo_online_iht_logit(x_tr, y_tr, w0, sparsity, p, n, lr, eta, verbose,
                          wt, wt_bar, nonzeros_wt, nonzeros_wt_bar,
                          pred_prob_wt, pred_label_wt, pred_prob_wt_bar,
                          pred_label_wt_bar, losses, missed_wt, missed_wt_bar,
                          &total_time);
    PyObject *results = PyTuple_New(12);
    PyObject *re_wt = PyList_New(p + 1), *re_wt_bar = PyList_New(p + 1);
    PyObject *re_nonzeros_wt = PyList_New(n);
    PyObject *re_nonzeros_wt_bar = PyList_New(n);
    for (i = 0; i < p + 1; i++) {
        PyList_SetItem(re_wt, i, PyFloat_FromDouble(wt[i]));
        PyList_SetItem(re_wt_bar, i, PyFloat_FromDouble(wt_bar[i]));
    }
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_nonzeros_wt, i,
                       PyInt_FromLong(nonzeros_wt[i]));
        PyList_SetItem(re_nonzeros_wt_bar, i,
                       PyInt_FromLong(nonzeros_wt_bar[i]));
    }
    PyObject *re_pred_prob_wt = PyList_New(n);
    PyObject *re_pred_label_wt = PyList_New(n);
    PyObject *re_pred_prob_wt_bar = PyList_New(n);
    PyObject *re_pred_label_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_pred_prob_wt, i,
                       PyFloat_FromDouble(pred_prob_wt[i]));
        PyList_SetItem(re_pred_label_wt, i,
                       PyFloat_FromDouble(pred_label_wt[i]));
        PyList_SetItem(re_pred_prob_wt_bar, i,
                       PyFloat_FromDouble(pred_prob_wt_bar[i]));
        PyList_SetItem(re_pred_label_wt_bar, i,
                       PyFloat_FromDouble(pred_label_wt_bar[i]));
    }
    PyObject *re_losses = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_losses, i, PyFloat_FromDouble(losses[i]));
    }
    PyObject *re_missed_wt = PyList_New(n);
    PyObject *re_missed_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_missed_wt, i, PyInt_FromLong(missed_wt[i]));
        PyList_SetItem(re_missed_wt_bar, i, PyInt_FromLong(missed_wt_bar[i]));
    }
    PyObject *re_total_time = PyFloat_FromDouble(total_time);
    PyTuple_SetItem(results, 0, re_wt);
    PyTuple_SetItem(results, 1, re_wt_bar);
    PyTuple_SetItem(results, 2, re_nonzeros_wt);
    PyTuple_SetItem(results, 3, re_nonzeros_wt_bar);
    PyTuple_SetItem(results, 4, re_pred_prob_wt);
    PyTuple_SetItem(results, 5, re_pred_label_wt);
    PyTuple_SetItem(results, 6, re_pred_prob_wt_bar);
    PyTuple_SetItem(results, 7, re_pred_label_wt_bar);
    PyTuple_SetItem(results, 8, re_losses);
    PyTuple_SetItem(results, 9, re_missed_wt);
    PyTuple_SetItem(results, 10, re_missed_wt_bar);
    PyTuple_SetItem(results, 11, re_total_time);
    // free all used memory
    free(pred_label_wt), free(pred_prob_wt);
    free(pred_label_wt_bar), free(pred_prob_wt_bar);
    free(w0), free(wt), free(wt_bar);
    free(nonzeros_wt), free(nonzeros_wt_bar);
    free(losses), free(x_tr), free(y_tr);
    free(missed_wt), free(missed_wt_bar);
    return results;
}


static PyObject *online_ghtp_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    int n, p, i, sparsity, verbose;
    double lr, eta;
    if (!PyArg_ParseTuple(args, "O!O!O!ddii", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &lr, &eta, &sparsity, &verbose)) { return NULL; }
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    double *x_tr = (double *) malloc(n * p * sizeof(double));
    double *y_tr = (double *) malloc(n * sizeof(double));
    double *w0 = (double *) malloc((p + 1) * sizeof(double));
    get_data(n, p, -1, x_tr, y_tr, w0, NULL, NULL,
             x_tr_, y_tr_, w0_, NULL, NULL);

    double *wt = malloc(sizeof(double) * (p + 1));
    double *wt_bar = malloc(sizeof(double) * (p + 1));
    int *nonzeros_wt = malloc(sizeof(int) * n);
    int *nonzeros_wt_bar = malloc(sizeof(int) * n);
    double *pred_prob_wt = malloc(sizeof(double) * n);
    double *pred_label_wt = malloc(sizeof(double) * n);
    double *pred_prob_wt_bar = malloc(sizeof(double) * n);
    double *pred_label_wt_bar = malloc(sizeof(double) * n);
    double *losses = malloc(sizeof(double) * n);
    double total_time = 0.0;
    int *missed_wt = malloc(sizeof(int) * n);
    int *missed_wt_bar = malloc(sizeof(int) * n);
    if (sparsity > p) {
        printf("the parameter of sparsity is too large!\n");
        printf("sparsity is: %d while p: %d!\n", sparsity, p);
        exit(0);
    }
    algo_online_ghtp_logit(x_tr, y_tr, w0, sparsity, p, n, lr, eta, verbose,
                           wt, wt_bar, nonzeros_wt, nonzeros_wt_bar,
                           pred_prob_wt, pred_label_wt, pred_prob_wt_bar,
                           pred_label_wt_bar, losses, missed_wt, missed_wt_bar,
                           &total_time);
    PyObject *results = PyTuple_New(12);
    PyObject *re_wt = PyList_New(p + 1), *re_wt_bar = PyList_New(p + 1);
    PyObject *re_nonzeros_wt = PyList_New(n);
    PyObject *re_nonzeros_wt_bar = PyList_New(n);
    for (i = 0; i < p + 1; i++) {
        PyList_SetItem(re_wt, i, PyFloat_FromDouble(wt[i]));
        PyList_SetItem(re_wt_bar, i, PyFloat_FromDouble(wt_bar[i]));
    }
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_nonzeros_wt, i, PyInt_FromLong(nonzeros_wt[i]));
        PyList_SetItem(re_nonzeros_wt_bar, i,
                       PyInt_FromLong(nonzeros_wt_bar[i]));
    }
    PyObject *re_pred_prob_wt = PyList_New(n);
    PyObject *re_pred_label_wt = PyList_New(n);
    PyObject *re_pred_prob_wt_bar = PyList_New(n);
    PyObject *re_pred_label_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_pred_prob_wt, i,
                       PyFloat_FromDouble(pred_prob_wt[i]));
        PyList_SetItem(re_pred_label_wt, i,
                       PyFloat_FromDouble(pred_label_wt[i]));
        PyList_SetItem(re_pred_prob_wt_bar, i,
                       PyFloat_FromDouble(pred_prob_wt_bar[i]));
        PyList_SetItem(re_pred_label_wt_bar, i,
                       PyFloat_FromDouble(pred_label_wt_bar[i]));
    }
    PyObject *re_losses = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_losses, i, PyFloat_FromDouble(losses[i]));
    }
    PyObject *re_missed_wt = PyList_New(n);
    PyObject *re_missed_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_missed_wt, i, PyInt_FromLong(missed_wt[i]));
        PyList_SetItem(re_missed_wt_bar, i, PyInt_FromLong(missed_wt_bar[i]));
    }
    PyObject *re_total_time = PyFloat_FromDouble(total_time);
    PyTuple_SetItem(results, 0, re_wt);
    PyTuple_SetItem(results, 1, re_wt_bar);
    PyTuple_SetItem(results, 2, re_nonzeros_wt);
    PyTuple_SetItem(results, 3, re_nonzeros_wt_bar);
    PyTuple_SetItem(results, 4, re_pred_prob_wt);
    PyTuple_SetItem(results, 5, re_pred_label_wt);
    PyTuple_SetItem(results, 6, re_pred_prob_wt_bar);
    PyTuple_SetItem(results, 7, re_pred_label_wt_bar);
    PyTuple_SetItem(results, 8, re_losses);
    PyTuple_SetItem(results, 9, re_missed_wt);
    PyTuple_SetItem(results, 10, re_missed_wt_bar);
    PyTuple_SetItem(results, 11, re_total_time);
    // free all used memory
    free(pred_label_wt), free(pred_prob_wt);
    free(pred_label_wt_bar), free(pred_prob_wt_bar);
    free(w0), free(wt), free(wt_bar);
    free(nonzeros_wt), free(nonzeros_wt_bar);
    free(losses), free(x_tr), free(y_tr);
    free(missed_wt), free(missed_wt_bar);
    return results;
}


static PyObject *online_ghtp_logit_sparse(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    int n, p, i, sparsity, verbose;
    double lr, eta;
    if (!PyArg_ParseTuple(args, "O!O!O!ddii", &PyArray_Type, &x_tr_,
                          &PyArray_Type, &y_tr_, &PyArray_Type, &w0_,
                          &lr, &eta, &sparsity, &verbose)) { return NULL; }
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    double *x_tr = (double *) malloc(n * p * sizeof(double));
    double *y_tr = (double *) malloc(n * sizeof(double));
    double *w0 = (double *) malloc((p + 1) * sizeof(double));

    get_data(n, p, -1, x_tr, y_tr, w0, NULL, NULL,
             x_tr_, y_tr_, w0_, NULL, NULL);


    double *wt = malloc(sizeof(double) * (p + 1));
    double *wt_bar = malloc(sizeof(double) * (p + 1));
    int *nonzeros_wt = malloc(sizeof(int) * n);
    int *nonzeros_wt_bar = malloc(sizeof(int) * n);
    double *pred_prob_wt = malloc(sizeof(double) * n);
    double *pred_label_wt = malloc(sizeof(double) * n);
    double *pred_prob_wt_bar = malloc(sizeof(double) * n);
    double *pred_label_wt_bar = malloc(sizeof(double) * n);
    double *losses = malloc(sizeof(double) * n);
    double total_time = 0.0;
    int *missed_wt = malloc(sizeof(int) * n);
    int *missed_wt_bar = malloc(sizeof(int) * n);
    if (sparsity > p) {
        printf("the parameter of sparsity is too large!\n");
        printf("sparsity is: %d while p: %d!\n", sparsity, p);
        exit(0);
    }
    algo_online_ghtp_logit_sparse(
            x_tr, y_tr, w0, sparsity, p, n, lr, eta, verbose, wt, wt_bar,
            nonzeros_wt, nonzeros_wt_bar, pred_prob_wt, pred_label_wt,
            pred_prob_wt_bar, pred_label_wt_bar, losses, missed_wt,
            missed_wt_bar, &total_time);
    PyObject *results = PyTuple_New(12);
    PyObject *re_wt = PyList_New(p + 1), *re_wt_bar = PyList_New(p + 1);
    PyObject *re_nonzeros_wt = PyList_New(n);
    PyObject *re_nonzeros_wt_bar = PyList_New(n);
    for (i = 0; i < p + 1; i++) {
        PyList_SetItem(re_wt, i, PyFloat_FromDouble(wt[i]));
        PyList_SetItem(re_wt_bar, i, PyFloat_FromDouble(wt_bar[i]));
    }
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_nonzeros_wt, i, PyInt_FromLong(nonzeros_wt[i]));
        PyList_SetItem(re_nonzeros_wt_bar, i,
                       PyInt_FromLong(nonzeros_wt_bar[i]));
    }
    PyObject *re_pred_prob_wt = PyList_New(n);
    PyObject *re_pred_label_wt = PyList_New(n);
    PyObject *re_pred_prob_wt_bar = PyList_New(n);
    PyObject *re_pred_label_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_pred_prob_wt, i,
                       PyFloat_FromDouble(pred_prob_wt[i]));
        PyList_SetItem(re_pred_label_wt, i,
                       PyFloat_FromDouble(pred_label_wt[i]));
        PyList_SetItem(re_pred_prob_wt_bar, i,
                       PyFloat_FromDouble(pred_prob_wt_bar[i]));
        PyList_SetItem(re_pred_label_wt_bar, i,
                       PyFloat_FromDouble(pred_label_wt_bar[i]));
    }
    PyObject *re_losses = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_losses, i, PyFloat_FromDouble(losses[i]));
    }
    PyObject *re_missed_wt = PyList_New(n);
    PyObject *re_missed_wt_bar = PyList_New(n);
    for (i = 0; i < n; i++) {
        PyList_SetItem(re_missed_wt, i, PyInt_FromLong(missed_wt[i]));
        PyList_SetItem(re_missed_wt_bar, i, PyInt_FromLong(missed_wt_bar[i]));
    }
    PyObject *re_total_time = PyFloat_FromDouble(total_time);
    PyTuple_SetItem(results, 0, re_wt);
    PyTuple_SetItem(results, 1, re_wt_bar);
    PyTuple_SetItem(results, 2, re_nonzeros_wt);
    PyTuple_SetItem(results, 3, re_nonzeros_wt_bar);
    PyTuple_SetItem(results, 4, re_pred_prob_wt);
    PyTuple_SetItem(results, 5, re_pred_label_wt);
    PyTuple_SetItem(results, 6, re_pred_prob_wt_bar);
    PyTuple_SetItem(results, 7, re_pred_label_wt_bar);
    PyTuple_SetItem(results, 8, re_losses);
    PyTuple_SetItem(results, 9, re_missed_wt);
    PyTuple_SetItem(results, 10, re_missed_wt_bar);
    PyTuple_SetItem(results, 11, re_total_time);
    // free all used memory
    free(pred_label_wt), free(pred_prob_wt);
    free(pred_label_wt_bar), free(pred_prob_wt_bar);
    free(w0), free(wt), free(wt_bar);
    free(nonzeros_wt), free(nonzeros_wt_bar);
    free(losses), free(x_tr), free(y_tr);
    free(missed_wt), free(missed_wt_bar);
    return results;
}


static PyObject *batch_iht_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    int n, p, sparsity, verbose, max_iter;
    double lr, eta, tol;
    if (!PyArg_ParseTuple(
            args, "O!O!O!dididi", &PyArray_Type, &x_tr_, &PyArray_Type, &y_tr_,
            &PyArray_Type, &w0_, &lr, &sparsity, &tol, &max_iter, &eta,
            &verbose)) { return NULL; }
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    double *x_tr = malloc(sizeof(double) * (n * p));
    double *y_tr = malloc(sizeof(double) * n);
    double *w0 = malloc(sizeof(double) * (p + 1));
    get_data(n, p, -1, x_tr, y_tr, w0, NULL, NULL,
             x_tr_, y_tr_, w0_, NULL, NULL);
    double *wt = malloc(sizeof(double) * (p + 1));
    double *losses = malloc(sizeof(double) * max_iter);
    double total_time = 0.0;
    if (sparsity > p) {
        printf("the parameter of sparsity is too large!\n");
        printf("sparsity is: %d while p: %d!\n", sparsity, p);
        exit(0);
    }
    algo_batch_iht_logit(x_tr, y_tr, w0, sparsity, p, n, lr, eta, max_iter,
                         tol, verbose, wt, losses, &total_time);
    PyObject *results = batch_get_result(p, max_iter, total_time, wt, losses);
    free(losses), free(x_tr), free(y_tr), free(w0), free(wt);
    return results;
}

static PyObject *batch_ghtp_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    int n, p, sparsity, verbose, max_iter;
    double lr, eta, tol;
    if (!PyArg_ParseTuple(
            args, "O!O!O!dididi", &PyArray_Type, &x_tr_, &PyArray_Type, &y_tr_,
            &PyArray_Type, &w0_, &lr, &sparsity, &tol, &max_iter, &eta,
            &verbose)) { return NULL; }
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    double *x_tr = malloc(n * p * sizeof(double));
    double *y_tr = malloc(n * sizeof(double));
    double *w0 = malloc((p + 1) * sizeof(double));
    get_data(n, p, -1, x_tr, y_tr, w0, NULL, NULL,
             x_tr_, y_tr_, w0_, NULL, NULL);

    double *wt = malloc(sizeof(double) * (p + 1));
    double *losses = malloc(sizeof(double) * max_iter);
    double total_time = 0.0;
    if (sparsity > p) {
        printf("the parameter of sparsity is too large!\n");
        printf("sparsity is: %d while p: %d!\n", sparsity, p);
        exit(0);
    }
    algo_batch_ghtp_logit(x_tr, y_tr, w0, sparsity, p, n, lr, eta, max_iter,
                          tol, verbose, wt, losses, &total_time);
    PyObject *results = batch_get_result(p, max_iter, total_time, wt, losses);
    free(losses), free(x_tr), free(y_tr), free(w0), free(wt);
    return results;
}


static PyObject *batch_graph_iht_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    EdgePair *edges;
    int g = 1, num_pcst, n, p, m, sparsity, verbose, max_iter;
    double lr, eta, tol, *weights, *x_tr, *y_tr, *w0, *wt, *losses;
    PyArrayObject *x_tr_, *y_tr_, *w0_, *edges_, *weights_;
    if (!PyArg_ParseTuple(
            args, "O!O!O!dididO!O!i", &PyArray_Type, &x_tr_, &PyArray_Type,
            &y_tr_, &PyArray_Type, &w0_, &lr, &sparsity, &tol, &max_iter, &eta,
            &PyArray_Type, &edges_, &PyArray_Type, &weights_,
            &verbose)) { return NULL; }
    n = (int) (x_tr_->dimensions[0]);       // number of samples
    p = (int) (x_tr_->dimensions[1]);       // number of features
    m = (int) edges_->dimensions[0];        // number of edges
    x_tr = malloc(sizeof(double) * (n * p));
    y_tr = malloc(sizeof(double) * n);
    w0 = malloc(sizeof(double) * (p + 1));
    edges = malloc(sizeof(EdgePair) * m);
    weights = malloc(sizeof(double) * m);
    get_data(n, p, m, x_tr, y_tr, w0, edges, weights,
             x_tr_, y_tr_, w0_, edges_, weights_);
    wt = malloc(sizeof(double) * (p + 1));
    losses = malloc(sizeof(double) * max_iter);
    double run_time_head = 0.0, run_time_tail = 0.0, total_time = 0.0;
    if (sparsity > p) {
        printf("the parameter of sparsity is too large!\n");
        printf("sparsity is: %d while p: %d!\n", sparsity, p);
        exit(0);
    }
    Array *f_nodes = malloc(sizeof(Array));
    Array *f_edges = malloc(sizeof(Array));
    f_nodes->array = malloc(sizeof(int) * p);
    f_edges->array = malloc(sizeof(int) * p);
    algo_batch_graph_iht_logit(
            edges, weights, g, sparsity, p, m, x_tr, y_tr, w0, n, tol,
            max_iter, lr, eta, verbose, wt, f_nodes, f_edges, &num_pcst,
            losses, &run_time_head, &run_time_tail, &total_time);
    PyObject *results = batch_get_result(p, max_iter, total_time, wt, losses);
    free(x_tr), free(y_tr), free(w0), free(edges), free(weights);
    free(wt), free(losses), free(f_nodes->array), free(f_edges->array);
    free(f_nodes), free(f_edges);
    return results;
}


static PyObject *batch_graph_ghtp_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    EdgePair *edges;
    int num_pcst, g = 1, n, p, m, sparsity, verbose, max_iter;
    double lr, eta, tol, *weights, *x_tr, *y_tr, *w0, *wt, *losses;
    PyArrayObject *edges_, *weights_, *x_tr_, *y_tr_, *w0_;
    if (!PyArg_ParseTuple(args, "O!O!O!dididO!O!i",
                          &PyArray_Type, &x_tr_, &PyArray_Type, &y_tr_,
                          &PyArray_Type, &w0_, &lr, &sparsity, &tol, &max_iter,
                          &eta, &PyArray_Type, &edges_, &PyArray_Type,
                          &weights_, &verbose)) { return NULL; }
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    m = (int) edges_->dimensions[0];        // number of edges
    x_tr = malloc(n * p * sizeof(double));
    y_tr = malloc(n * sizeof(double));
    w0 = malloc((p + 1) * sizeof(double));
    edges = malloc(sizeof(EdgePair) * m);
    weights = malloc(sizeof(double) * m);
    get_data(n, p, m, x_tr, y_tr, w0, edges, weights,
             x_tr_, y_tr_, w0_, edges_, weights_);
    wt = malloc(sizeof(double) * (p + 1));
    losses = malloc(sizeof(double) * max_iter);
    double run_time_head = 0.0, run_time_tail = 0.0, total_time = 0.0;
    if (sparsity > p) {
        printf("the parameter of sparsity is too large!\n");
        printf("sparsity is: %d while p: %d!\n", sparsity, p);
        exit(0);
    }
    Array *f_nodes = malloc(sizeof(Array));
    Array *f_edges = malloc(sizeof(Array));
    f_nodes->array = malloc(sizeof(int) * p);
    f_edges->array = malloc(sizeof(int) * p);
    algo_batch_graph_ghtp_logit(
            edges, weights, g, sparsity, p, m, x_tr, y_tr, w0, n, tol,
            max_iter, lr, eta, verbose, wt, f_nodes, f_edges, &num_pcst,
            losses, &run_time_head, &run_time_tail, &total_time);
    PyObject *results = batch_get_result(p, max_iter, total_time, wt, losses);
    free(w0), free(wt), free(edges), free(weights), free(x_tr), free(y_tr);
    free(f_nodes->array), free(f_edges->array);
    free(f_nodes), free(f_edges), free(losses);
    return results;
}


static PyObject *batch_graph_posi_logit(PyObject *self, PyObject *args) {
    if (self != NULL) {
        printf("unknown error for no reason.\n");
        return NULL;
    }
    PyArrayObject *x_tr_, *y_tr_, *w0_;
    int n, p, m, sparsity, verbose, max_iter;
    double lr, eta, tol, *weights;
    EdgePair *edges;
    PyArrayObject *edges_, *weights_;
    if (!PyArg_ParseTuple(args, "O!O!O!dididO!O!i",
                          &PyArray_Type, &x_tr_, &PyArray_Type, &y_tr_,
                          &PyArray_Type, &w0_, &lr, &sparsity, &tol, &max_iter,
                          &eta, &PyArray_Type, &edges_, &PyArray_Type,
                          &weights_, &verbose)) { return NULL; }
    int g = 1;
    n = (int) (x_tr_->dimensions[0]);     // number of samples
    p = (int) (x_tr_->dimensions[1]);     // number of features
    m = (int) edges_->dimensions[0];        // number of edges
    double *x_tr = malloc(n * p * sizeof(double));
    double *y_tr = malloc(n * sizeof(double));
    double *w0 = malloc((p + 1) * sizeof(double));
    edges = malloc(sizeof(EdgePair) * m);
    weights = malloc(sizeof(double) * m);
    get_data(n, p, m, x_tr, y_tr, w0, edges, weights,
             x_tr_, y_tr_, w0_, edges_, weights_);
    int num_pcst;
    double *wt = malloc(sizeof(double) * (p + 1));
    double *losses = malloc(sizeof(double) * max_iter);
    double run_time_head;
    double run_time_tail;
    double total_time = 0.0;
    if (sparsity > p) {
        printf("the parameter of sparsity is too large!\n");
        printf("sparsity is: %d while p: %d!\n", sparsity, p);
        exit(0);
    }
    Array *f_nodes = malloc(sizeof(Array));
    Array *f_edges = malloc(sizeof(Array));
    f_nodes->array = malloc(sizeof(int) * p);
    f_edges->array = malloc(sizeof(int) * p);
    algo_batch_graph_posi_logit(
            edges, weights, g, sparsity, p, m, x_tr, y_tr, w0, n, tol,
            max_iter, lr, eta, verbose, wt, f_nodes, f_edges, &num_pcst,
            losses, &run_time_head, &run_time_tail, &total_time);
    PyObject *results = batch_get_result(p, max_iter, total_time, wt, losses);
    free(w0), free(wt), free(edges), free(weights);
    free(f_nodes->array), free(f_edges->array);
    free(f_nodes), free(f_edges), free(losses), free(x_tr), free(y_tr);
    return results;
}


static PyMethodDef sparse_methods[] = {
        {"online_graph_ghtp_logit",  (PyCFunction) online_graph_ghtp_logit,
                METH_VARARGS, "online_graph_ghtp_logit docs"},
        {"online_graph_iht_logit",   (PyCFunction) online_graph_iht_logit,
                METH_VARARGS, "online_graph_iht_logit docs"},
        {"online_iht_logit",         (PyCFunction) online_iht_logit,
                METH_VARARGS, "online_iht_logit docs"},
        {"online_ghtp_logit",        (PyCFunction) online_ghtp_logit,
                METH_VARARGS, "online_iht_logit docs"},
        {"online_ghtp_logit_sparse", (PyCFunction) online_ghtp_logit_sparse,
                METH_VARARGS, "online_iht_logit_sparse docs"},
        {"online_sgd_l2_logit",      (PyCFunction) online_sgd_l2_logit,
                METH_VARARGS, "online_sgd_l2_logit docs"},
        {"online_sgd_l1_logit",      (PyCFunction) online_sgd_l1_logit,
                METH_VARARGS, "online_sgd_l1_logit docs"},
        {"online_rda_l1_logit",      (PyCFunction) online_rda_l1_logit,
                METH_VARARGS, "online_rda_l1_logit docs"},
        {"batch_iht_logit",          (PyCFunction) batch_iht_logit,
                METH_VARARGS, "batch_iht_logit docs"},
        {"batch_ghtp_logit",         (PyCFunction) batch_ghtp_logit,
                METH_VARARGS, "batch_ghtp_logit docs"},
        {"batch_graph_iht_logit",    (PyCFunction) batch_graph_iht_logit,
                METH_VARARGS, "batch_graph_ghtp_logit docs"},
        {"batch_graph_ghtp_logit",   (PyCFunction) batch_graph_ghtp_logit,
                METH_VARARGS, "batch_graph_ghtp_logit docs"},
        {"batch_graph_posi_logit",   (PyCFunction) batch_graph_posi_logit,
                METH_VARARGS, "batch_graph_posi_logit docs"},
        {NULL, NULL, 0, NULL}};

/** Python version 2 for module initialization */
PyMODINIT_FUNC initsparse_module() {
    Py_InitModule3("sparse_module", sparse_methods,
                   "some docs for sparse learning algorithms.");
    import_array();
}

int main() {
    printf("test of main wrapper!\n");
}