//
// Created by baojian on 8/11/18.
//
#include <cblas.h>
#include "sort.h"
#include "loss.h"
#include "sparse_algorithms.h"

#define sign(x) ((x > 0) -(x < 0))

void min_f_posi(const Array *proj_nodes, const double *x_tr,
                const double *y_tr, int max_iter, double eta, double *wt,
                int n, int p) {
    openblas_set_num_threads(1);
    int i;
    double *loss_grad = (double *) malloc((p + 2) * sizeof(double));
    double *tmp_loss_grad = (double *) malloc((p + 2) * sizeof(double));
    double *wt_tmp = (double *) malloc((p + 1) * sizeof(double));
    /**
     * make sure the start point is a feasible point. here we do a trick:
     * we treat wt as an initial point itself. and of course wt is always a
     * feasible point. A Frank-Wolfe style minimization with
     * backtracking line search. Other algorithms can be considered:
     * Newton's method, trust region, etc.
     */
    double beta, lr, grad_sq;
    for (i = 0; i < max_iter; i++) {
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, n, p);
        beta = 0.8, lr = 1.0;
        grad_sq = 0.5 * cblas_ddot(p + 1, loss_grad + 1, 1, loss_grad + 1, 1);
        while (true) { // using backtracking line search
            cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
            cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
            logistic_loss_grad(wt_tmp, x_tr, y_tr, tmp_loss_grad, eta, n, p);
            if (tmp_loss_grad[0] > (loss_grad[0] - lr * grad_sq)) {
                lr *= beta;
            } else {
                break;
            }
        }
        // projection step
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
        cblas_dscal(p + 1, 0.0, wt_tmp, 1);
        for (int k = 0; k < proj_nodes->size; k++) {
            wt_tmp[proj_nodes->array[k]] = wt[proj_nodes->array[k]];
            // positive constraint.
            if (wt_tmp[proj_nodes->array[k]] < 0.) {
                wt_tmp[proj_nodes->array[k]] = 0.0;
            }
        }
        wt_tmp[p] = wt[p];
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
    }
    free(wt_tmp);
    free(tmp_loss_grad);
    free(loss_grad);
}

void min_f(const Array *proj_nodes, const double *x_tr,
           const double *y_tr, int max_iter, double eta, double *wt,
           int n, int p) {
    openblas_set_num_threads(1);
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *tmp_loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    double beta, lr, grad_sq;
    /**
     * make sure the start point is a feasible point. here we do a trick:
     * we treat wt as an initial point itself. and of course wt is always a
     * feasible point. A Frank-Wolfe style minimization with
     * backtracking line search. Other algorithms can be considered:
     * Newton's method, trust region, etc.
     */
    for (int i = 0; i < max_iter; i++) {
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, n, p);
        beta = 0.8, lr = 1.0;
        grad_sq = 0.5 * cblas_ddot(p + 1, loss_grad + 1, 1, loss_grad + 1, 1);
        while (true) { // using backtracking line search
            cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
            cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
            logistic_loss_grad(wt_tmp, x_tr, y_tr, tmp_loss_grad, eta, n, p);
            if (tmp_loss_grad[0] > (loss_grad[0] - lr * grad_sq)) {
                lr *= beta;
            } else {
                break;
            }
        }
        // projection step
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
        cblas_dscal(p + 1, 0.0, wt_tmp, 1);
        for (int k = 0; k < proj_nodes->size; k++) {
            wt_tmp[proj_nodes->array[k]] = wt[proj_nodes->array[k]];
        }
        wt_tmp[p] = wt[p];
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
    }
    free(wt_tmp);
    free(tmp_loss_grad);
    free(loss_grad);
}

void min_f_sparse(
        const Array *proj_nodes, const double *x_tr,
        const double *y_tr, int max_iter, double eta, double *wt, int n,
        int p) {
    openblas_set_num_threads(1);
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *tmp_loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    double beta, lr, grad_sq;
    /**
     * make sure the start point is a feasible point. here we do a trick:
     * we treat wt as an initial point itself. and of course wt is always a
     * feasible point. A Frank-Wolfe style minimization with
     * backtracking line search. Other algorithms can be considered:
     * Newton's method, trust region, etc.
     */
    for (int i = 0; i < max_iter; i++) {
        logistic_loss_grad_sparse(
                wt, x_tr, y_tr, loss_grad, eta, n, p);
        beta = 0.8, lr = 1.0;
        grad_sq = 0.5 * cblas_ddot(p + 1, loss_grad + 1, 1, loss_grad + 1, 1);
        while (true) { // using backtracking line search
            cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
            cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
            logistic_loss_grad_sparse(
                    wt_tmp, x_tr, y_tr, tmp_loss_grad, eta, n, p);
            if (tmp_loss_grad[0] > (loss_grad[0] - lr * grad_sq)) {
                lr *= beta;
            } else {
                break;
            }
        }
        // projection step
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
        cblas_dscal(p + 1, 0.0, wt_tmp, 1);
        for (int k = 0; k < proj_nodes->size; k++) {
            wt_tmp[proj_nodes->array[k]] = wt[proj_nodes->array[k]];
        }
        wt_tmp[p] = wt[p];
        cblas_dcopy(p + 1, wt_tmp, 1, wt, 1);
    }
    free(wt_tmp);
    free(tmp_loss_grad);
    free(loss_grad);
}


bool algo_online_graph_ghtp_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double lr, double eta, int verbose, double *wt, double *wt_bar,
        int *nonzeros_wt, int *nonzeros_wt_bar, double *pred_prob_wt,
        double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar, Array *re_nodes, Array *re_edges,
        int *num_pcst, double *losses, double *run_time_head,
        double *run_time_tail, int *missed_wt, int *missed_wt_bar,
        double *total_time) {
    openblas_set_num_threads(1);
    int i, root = -1, fmin_max_iter = 20, max_iter = 50;
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1); // w0 --> wt_bar
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    double *tmp_prizes = malloc(sizeof(double) * p);
    double *tmp_costs = malloc(sizeof(double) * m);
    clock_t start_head, start_tail, start_total = clock();
    double C = 2. * (s - 1.), delta = 1. / 169., nu = 2.5;
    double err_tol = 1e-6, budget = (s - 1.), eps = 1e-6, tmp_time;
    int total_missed_wt = 0, total_missed_wt_bar = 0;
    enum PruningMethod pruning = GWPruning;
    for (i = 0; i < num_tr; i++) {
        nonzeros_wt[i] = 0;
        nonzeros_wt_bar[i] = 0;
    }
    for (i = 0; i < m; i++) {
        tmp_costs[i] = costs[i] + budget / (double) s;
    }
    for (int tt = 0; tt < num_tr; tt++) {
        // each time has one sample.
        logistic_predict(x_tr + tt * p, wt, &pred_prob_wt[tt],
                         &pred_label_wt[tt], 0.5, 1, p);
        logistic_predict(x_tr + tt * p, wt_bar, &pred_prob_wt_bar[tt],
                         &pred_label_wt_bar[tt], 0.5, 1, p);
        logistic_loss_grad(wt, x_tr + tt * p, y_tr + tt, loss_grad, eta, 1, p);
        if (pred_label_wt[tt] != y_tr[tt]) {
            total_missed_wt += 1;
        }
        if (pred_label_wt_bar[tt] != y_tr[tt]) {
            total_missed_wt_bar += 1;
        }
        missed_wt[tt] = total_missed_wt;
        missed_wt_bar[tt] = total_missed_wt_bar;
        losses[tt] = loss_grad[0];
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -lr / sqrt(tt + 1.), loss_grad + 1, 1, wt_tmp, 1);
        for (i = 0; i < p; i++) {
            tmp_prizes[i] = wt_tmp[i] * wt_tmp[i];
        }
        start_head = clock();
        head_proj_exact(edges, tmp_costs, tmp_prizes, g, C, delta, max_iter,
                        err_tol, root, pruning, eps, p, m, verbose, re_nodes,
                        re_edges, num_pcst, &tmp_time);
        if (verbose > 0) {
            printf("losses[%d]: %.6f h_nodes:%d",
                   tt, losses[tt], re_nodes->size);
        }
        *run_time_head += ((double) (clock() - start_head)) / CLOCKS_PER_SEC;
        min_f(re_nodes, x_tr, y_tr, fmin_max_iter, eta, wt, tt + 1, p);
        for (i = 0; i < p; i++) {
            tmp_prizes[i] = wt[i] * wt[i];
        }
        start_tail = clock();
        tail_proj_exact(edges, tmp_costs, tmp_prizes, g, C, nu, max_iter,
                        err_tol, root, pruning, eps, p, m, verbose, re_nodes,
                        re_edges, num_pcst, &tmp_time);
        *run_time_tail += ((double) (clock() - start_tail)) / CLOCKS_PER_SEC;
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (i = 0; i < re_nodes->size; i++) {
            wt[re_nodes->array[i]] = wt_tmp[re_nodes->array[i]];
        }
        wt[p] = wt_tmp[p];
        if (verbose > 0) {
            printf("t_nodes:%d p_prob: %.4f total_missed %d/%d\n",
                   re_nodes->size, pred_prob_wt[tt], total_missed_wt, tt + 1);
        }
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), wt, 1, wt_bar, 1);
        for (i = 0; i < p; i++) {
            if (wt[i] != 0.0) {
                nonzeros_wt[tt] += 1;
            }
            if (wt_bar[i] != 0.0) {
                nonzeros_wt_bar[tt] += 1;
            }
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(tmp_costs), free(tmp_prizes), free(loss_grad), free(wt_tmp);
    return true;
}


bool algo_online_graph_iht_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double lr, double eta, int verbose, double *wt, double *wt_bar,
        int *nonzeros_wt, int *nonzeros_wt_bar, double *pred_prob_wt,
        double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar, Array *re_nodes, Array *re_edges,
        int *num_pcst, double *losses, double *run_time_head,
        double *run_time_tail, int *missed_wt, int *missed_wt_bar,
        double *total_time) {
    openblas_set_num_threads(1);
    int i;
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1); // w0 --> wt_bar
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    double *gt_bar = malloc((p + 1) * sizeof(double));
    cblas_dcopy(p + 1, w0, 1, gt_bar, 1);
    double *tmp_prizes = malloc(sizeof(double) * p);
    double *tmp_costs = malloc(sizeof(double) * m);
    clock_t start_head, start_tail, start_total = clock();
    double C = 2. * (s - 1.), delta = 1. / 169., nu = 2.5, err_tol = 1e-6;
    double budget = (s - 1.), eps = 1e-6, tmp_time;
    int root = -1, max_iter = 50;
    int total_missed_wt = 0, total_missed_wt_bar = 0;
    enum PruningMethod pruning = GWPruning;
    for (i = 0; i < num_tr; i++) {
        nonzeros_wt[i] = 0;
        nonzeros_wt_bar[i] = 0;
    }
    for (i = 0; i < m; i++) {
        tmp_costs[i] = costs[i] + budget / (double) s;
    }
    for (int tt = 0; tt < num_tr; tt++) {
        // each time has one sample.
        logistic_predict(x_tr + tt * p, wt, &pred_prob_wt[tt],
                         &pred_label_wt[tt], 0.5, 1, p);
        logistic_predict(x_tr + tt * p, wt_bar, &pred_prob_wt_bar[tt],
                         &pred_label_wt_bar[tt], 0.5, 1, p);
        logistic_loss_grad(wt, x_tr + tt * p, y_tr + tt, loss_grad, eta, 1, p);
        cblas_dscal(p + 1, tt / (tt + 1.), gt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), loss_grad + 1, 1, gt_bar, 1);
        if (pred_label_wt[tt] != y_tr[tt]) {
            total_missed_wt++;
        }
        if (pred_label_wt_bar[tt] != y_tr[tt]) {
            total_missed_wt_bar++;
        }
        missed_wt[tt] = total_missed_wt;
        missed_wt_bar[tt] = total_missed_wt_bar;
        losses[tt] = loss_grad[0];
        for (i = 0; i < p; i++) {
            tmp_prizes[i] = loss_grad[i + 1] * loss_grad[i + 1];
        }
        start_head = clock();
        head_proj_exact(edges, tmp_costs, tmp_prizes, g, C, delta, max_iter,
                        err_tol, root, pruning, eps, p, m, verbose, re_nodes,
                        re_edges, num_pcst, &tmp_time);
        cblas_dcopy(p + 1, loss_grad + 1, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_dscal(p + 1, 0.0, loss_grad + 1, 1);
        for (i = 0; i < re_nodes->size; i++) {
            int cur_node = re_nodes->array[i];
            loss_grad[cur_node + 1] = wt_tmp[cur_node];
        }
        if (verbose > 0) {
            printf("losses[%d]: %.6f h_nodes:%d", tt, losses[tt],
                   re_nodes->size);
        }
        *run_time_head += ((double) (clock() - start_head)) / CLOCKS_PER_SEC;
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -lr / sqrt(tt + 1.), loss_grad + 1, 1, wt_tmp, 1);
        for (i = 0; i < p; i++) {
            tmp_prizes[i] = gt_bar[i] * gt_bar[i];
        }
        start_tail = clock();
        tail_proj_exact(edges, tmp_costs, tmp_prizes, g, C, nu, max_iter,
                        err_tol, root, pruning, eps, p, m, verbose, re_nodes,
                        re_edges, num_pcst, &tmp_time);
        *run_time_tail += ((double) (clock() - start_tail)) / CLOCKS_PER_SEC;
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (i = 0; i < re_nodes->size; i++) {
            wt[re_nodes->array[i]] = wt_tmp[re_nodes->array[i]];
        }
        wt[p] = wt_tmp[p];
        if (verbose > 0) {
            printf("t_nodes:%d p_prob: %.4f total_missed %d/%d\n",
                   re_nodes->size, pred_prob_wt[tt], total_missed_wt, tt + 1);
        }
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), wt, 1, wt_bar, 1);
        for (i = 0; i < p; i++) {
            if (wt[i] != 0.0) {
                nonzeros_wt[tt] += 1;
            }
            if (wt_bar[i] != 0.0) {
                nonzeros_wt_bar[tt] += 1;
            }
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(tmp_costs), free(tmp_prizes), free(loss_grad), free(wt_tmp);
    free(gt_bar);
    return true;
}

bool algo_online_iht_logit(
        const double *x_tr, const double *y_tr, const double *w0, int s, int p,
        int num_tr, double lr, double eta, int verbose, double *wt,
        double *wt_bar, int *nonzeros_wt, int *nonzeros_wt_bar,
        double *pred_prob_wt, double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar, double *losses, int *missed_wt,
        int *missed_wt_bar, double *total_time) {
    openblas_set_num_threads(1);
    int i;
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1); // w0 --> wt_bar
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    double *gt_bar = malloc((p + 1) * sizeof(double));
    cblas_dcopy(p + 1, w0, 1, gt_bar, 1);
    int *sorted_ind = malloc(sizeof(double) * p);
    int total_missed_wt = 0, total_missed_wt_bar = 0;
    clock_t start_total = clock();
    for (i = 0; i < num_tr; i++) {
        nonzeros_wt[i] = 0;
        nonzeros_wt_bar[i] = 0;
    }
    for (int tt = 0; tt < num_tr; tt++) {
        // each time has one sample.
        logistic_predict(x_tr + tt * p, wt, &pred_prob_wt[tt],
                         &pred_label_wt[tt], 0.5, 1, p);
        logistic_predict(x_tr + tt * p, wt_bar, &pred_prob_wt_bar[tt],
                         &pred_label_wt_bar[tt], 0.5, 1, p);
        logistic_loss_grad(wt, x_tr + tt * p, y_tr + tt, loss_grad, eta, 1, p);
        if (pred_label_wt[tt] != y_tr[tt]) {
            total_missed_wt++;
        }
        if (pred_label_wt_bar[tt] != y_tr[tt]) {
            total_missed_wt_bar++;
        }
        missed_wt[tt] = total_missed_wt;
        missed_wt_bar[tt] = total_missed_wt_bar;
        losses[tt] = loss_grad[0];
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp

        cblas_daxpy(p + 1, -lr / sqrt(tt + 1.), loss_grad + 1, 1, wt_tmp, 1);
        cblas_dscal(p + 1, tt / (tt + 1.), gt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), loss_grad + 1, 1, gt_bar, 1);
        arg_magnitude_sort_descend(gt_bar, sorted_ind, p);
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (i = 0; i < s; i++) {
            wt[sorted_ind[i]] = wt_tmp[sorted_ind[i]];
        }
        wt[p] = wt_tmp[p];
        if (verbose > 0) {
            printf("pred_prob:%.4f total_missed:%d/%d\n",
                   pred_prob_wt[tt], total_missed_wt, tt + 1);
        }
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), wt, 1, wt_bar, 1);
        for (i = 0; i < p; i++) {
            if (wt[i] != 0.0) {
                nonzeros_wt[tt] += 1;
            }
            if (wt_bar[i] != 0.0) {
                nonzeros_wt_bar[tt] += 1;
            }
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(loss_grad), free(wt_tmp), free(sorted_ind), free(gt_bar);
    return true;
}


bool algo_online_rda_l1_logit(
        const double *x_tr, const double *y_tr, const double *w0, int p,
        int num_tr, double lambda, double gamma, double rho, int verbose,
        double *wt, double *wt_bar, int *nonzeros_wt, int *nonzeros_wt_bar,
        double *pred_prob_wt, double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar, double *losses, int *missed_wt,
        int *missed_wt_bar, double *total_time) {
    openblas_set_num_threads(1);
    double *loss_grad = malloc(sizeof(double) * (p + 2));
    double *gt_bar = malloc(sizeof(double) * (p + 1));
    double *gt = malloc(sizeof(double) * (p + 1));
    double wei, lambda_t_rda;
    int total_missed_wt = 0, total_missed_wt_bar = 0, i = 0;
    clock_t start_total = clock();
    cblas_dcopy(p + 1, w0, 1, gt, 1); // w0 --> gt_bar
    cblas_dcopy(p + 1, w0, 1, gt_bar, 1); // w0 --> gt_bar
    cblas_dcopy(p + 1, w0, 1, wt, 1); // wt --> wt_bar
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1); // wt --> wt_bar
    for (i = 0; i < num_tr; i++) {
        nonzeros_wt[i] = 0;
        nonzeros_wt_bar[i] = 0;
    }
    for (int tt = 0; tt < num_tr; tt++) {
        // 1. receive a question x_tr_t and predict a value by wt and wt_bar
        logistic_predict(x_tr + tt * p, wt, &pred_prob_wt[tt],
                         &pred_label_wt[tt], 0.5, 1, p);
        logistic_predict(x_tr + tt * p, wt_bar, &pred_prob_wt_bar[tt],
                         &pred_label_wt_bar[tt], 0.5, 1, p);
        // 2. receive the true answer y_tr_t and suffer a loss
        logistic_loss_grad(wt, x_tr + tt * p, y_tr + tt, loss_grad, 0.0, 1, p);
        if (pred_label_wt[tt] != y_tr[tt]) {
            total_missed_wt++;
        }
        if (pred_label_wt_bar[tt] != y_tr[tt]) {
            total_missed_wt_bar++;
            if (verbose > 0) {
                printf("missed!");
            }
        }
        missed_wt[tt] = total_missed_wt;
        missed_wt_bar[tt] = total_missed_wt_bar;
        losses[tt] = loss_grad[0];
        cblas_dcopy(p + 1, loss_grad + 1, 1, gt, 1);
        cblas_dscal(p + 1, 1. / (tt + 1.), gt, 1);
        cblas_daxpy(p + 1, tt / (tt + 1.), gt_bar, 1, gt, 1);
        cblas_dcopy(p + 1, gt, 1, gt_bar, 1);
        // 3. update the model: enhanced l1-rda method.
        wei = -sqrt(tt + 1.) / gamma;
        lambda_t_rda = lambda + (gamma * rho) / sqrt(tt + 1.);
        for (i = 0; i < (p + 1); i++) {
            if (fabs(gt_bar[i]) <= lambda_t_rda) {
                wt[i] = 0.0; //thresholding entries
            } else {
                wt[i] = wei * (gt_bar[i] - lambda_t_rda * sign(gt_bar[i]));
            }
        }
        double norm_wt = 0.0, norm_gt_bar = 0.0;
        for (i = 0; i < (p + 1); i++) {
            norm_wt += wt[i] * wt[i];
            norm_gt_bar += gt_bar[i] * gt_bar[i];
        }
        // 4. online to batch conversion
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1. / (tt + 1.), wt, 1, wt_bar, 1);
        for (i = 0; i < p; i++) {
            if (wt[i] != 0.0) {
                nonzeros_wt[tt] += 1;
            }
            if (wt_bar[i] != 0.0) {
                nonzeros_wt_bar[tt] += 1;
            }
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(loss_grad), free(gt_bar), free(gt);
    return true;
}

bool algo_online_sgd_l1_logit(
        const double *x_tr, const double *y_tr, const double *w0, int p,
        int num_tr, double gamma, double lambda, int verbose, double *wt,
        double *wt_bar, int *nonzeros_wt, int *nonzeros_wt_bar,
        double *pred_prob_wt, double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar, double *losses, int *missed_wt,
        int *missed_wt_bar, double *total_time) {
    openblas_set_num_threads(1);
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1); // w0 --> wt_bar
    double *loss_grad = malloc((p + 2) * sizeof(double));
    int total_missed_wt = 0, total_missed_wt_bar = 0, i = 0;
    clock_t start_total = clock();
    cblas_dcopy(p + 1, wt, 1, wt_bar, 1); // wt --> wt_bar
    for (i = 0; i < num_tr; i++) {
        nonzeros_wt[i] = 0;
        nonzeros_wt_bar[i] = 0;
    }
    double alpha_t = sqrt(2. / num_tr) / gamma;
    for (int tt = 0; tt < num_tr; tt++) {
        //1. receive a question x_t and predict a value.
        logistic_predict(x_tr + tt * p, wt, &pred_prob_wt[tt],
                         &pred_label_wt[tt], 0.5, 1, p);
        logistic_predict(x_tr + tt * p, wt, &pred_prob_wt_bar[tt],
                         &pred_label_wt_bar[tt], 0.5, 1, p);
        //2. receive a true answer y_t and then suffer a loss.
        if (pred_label_wt[tt] != y_tr[tt]) {
            total_missed_wt++;
        }
        if (pred_label_wt_bar[tt] != y_tr[tt]) {
            total_missed_wt_bar++;
        }
        missed_wt[tt] = total_missed_wt;
        missed_wt_bar[tt] = total_missed_wt_bar;
        logistic_loss_grad(wt, x_tr + tt * p, y_tr + tt, loss_grad, 0.0, 1, p);
        losses[tt] = loss_grad[0];
        //3. update the model:
        for (i = 0; i < p + 1; i++) {
            wt[i] += -alpha_t * (loss_grad[i + 1] + lambda * sign(wt[i]));
        }
        if (verbose > 0) {
            printf("pred_prob: %.4f total_missed %d/%d\n",
                   pred_prob_wt[tt], total_missed_wt_bar, tt + 1);
        }
        //4. online to batch conversion.
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), wt, 1, wt_bar, 1);
        for (i = 0; i < p; i++) {
            if (wt[i] != 0.0) {
                nonzeros_wt[tt] += 1;
            }
            if (wt_bar[i] != 0.0) {
                nonzeros_wt_bar[tt] += 1;
            }
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(loss_grad);
    return true;
}

bool algo_online_sgd_l2_logit(
        const double *x_tr, const double *y_tr, const double *w0, int p,
        int num_tr, double lr, double eta, int verbose, double *wt,
        double *wt_bar, int *nonzeros_wt, int *nonzeros_wt_bar,
        double *pred_prob_wt, double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar, double *losses, int *missed_wt,
        int *missed_wt_bar, double *total_time) {
    openblas_set_num_threads(1);
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1); // w0 --> wt_bar
    double *loss_grad = malloc((p + 2) * sizeof(double));
    int total_missed_wt = 0, total_missed_wt_bar = 0, i = 0;
    clock_t start_total = clock();
    if (verbose > 0) {
        printf("learning rate: %lf num_tr: %d p: %d\n", lr, num_tr, p);
    }
    cblas_dcopy(p + 1, wt, 1, wt_bar, 1); // wt --> wt_bar
    for (i = 0; i < num_tr; i++) {
        nonzeros_wt[i] = 0;
        nonzeros_wt_bar[i] = 0;
    }
    for (int tt = 0; tt < num_tr; tt++) {
        // each time has one sample.
        logistic_predict(x_tr + tt * p, wt, &pred_prob_wt[tt],
                         &pred_label_wt[tt], 0.5, 1, p);
        logistic_predict(x_tr + tt * p, wt, &pred_prob_wt_bar[tt],
                         &pred_label_wt_bar[tt], 0.5, 1, p);
        logistic_loss_grad(wt, x_tr + tt * p, y_tr + tt, loss_grad, eta, 1, p);
        if (pred_label_wt[tt] != y_tr[tt]) {
            total_missed_wt++;
        }
        if (pred_label_wt_bar[tt] != y_tr[tt]) {
            total_missed_wt_bar++;
        }
        missed_wt[tt] = total_missed_wt;
        missed_wt_bar[tt] = total_missed_wt_bar;
        losses[tt] = loss_grad[0];
        cblas_daxpy(p + 1, -lr / sqrt(tt + 1.), loss_grad + 1, 1, wt, 1);
        if (verbose > 0) {
            printf("pred_prob: %.4f total_missed %d/%d\n",
                   pred_prob_wt[tt], total_missed_wt_bar, tt + 1);
        }
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), wt, 1, wt_bar, 1);
        for (i = 0; i < p; i++) {
            if (wt[i] != 0.0) {
                nonzeros_wt[tt] += 1;
            }
            if (wt_bar[i] != 0.0) {
                nonzeros_wt_bar[tt] += 1;
            }
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(loss_grad);
    return true;
}

bool algo_online_ghtp_logit(
        const double *x_tr, const double *y_tr, const double *w0, int s, int p,
        int num_tr, double lr, double eta, int verbose, double *wt,
        double *wt_bar, int *nonzeros_wt, int *nonzeros_wt_bar,
        double *pred_prob_wt, double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar, double *losses, int *missed_wt,
        int *missed_wt_bar, double *total_time) {
    openblas_set_num_threads(1);
    int i;
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1); // w0 --> wt_bar
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    int *sorted_ind = malloc(sizeof(double) * p);
    int total_missed_wt = 0, total_missed_wt_bar = 0, max_iter = 20;
    Array *re_nodes = malloc(sizeof(Array));
    re_nodes->array = malloc(sizeof(int) * p);
    clock_t start_total = clock();
    if (verbose > 0) {
        printf("learning rate: %lf, num_tr: %d ,p: %d\n", lr, num_tr, p);
    }
    for (i = 0; i < num_tr; i++) {
        nonzeros_wt[i] = 0;
        nonzeros_wt_bar[i] = 0;
    }
    for (int tt = 0; tt < num_tr; tt++) {
        // each time has one sample.
        logistic_predict(x_tr + tt * p, wt, &pred_prob_wt[tt],
                         &pred_label_wt[tt], 0.5, 1, p);
        logistic_predict(x_tr + tt * p, wt_bar, &pred_prob_wt_bar[tt],
                         &pred_label_wt_bar[tt], 0.5, 1, p);
        logistic_loss_grad(wt, x_tr + tt * p, y_tr + tt, loss_grad, eta, 1, p);
        if (pred_label_wt[tt] != y_tr[tt]) {
            total_missed_wt++;
        }
        if (pred_label_wt_bar[tt] != y_tr[tt]) {
            total_missed_wt_bar++;
        }
        missed_wt[tt] = total_missed_wt;
        missed_wt_bar[tt] = total_missed_wt_bar;
        losses[tt] = loss_grad[0];
        if (verbose > 0) {
            printf("losses[%d]:%.6f sparsity:%d ", tt, losses[tt], s);
        }
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -lr / sqrt(tt + 1.), loss_grad + 1, 1, wt_tmp, 1);
        arg_magnitude_sort_descend(wt_tmp, sorted_ind, p);
        re_nodes->size = 0;
        for (i = 0; i < s; i++) {
            re_nodes->array[re_nodes->size++] = sorted_ind[i];
        }
        min_f(re_nodes, x_tr, y_tr, max_iter, eta, wt, tt + 1, p);
        if (verbose > 0) {
            printf("pred_prob:%.4f total_missed:%d/%d\n",
                   pred_prob_wt[tt], total_missed_wt, tt + 1);
        }
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), wt, 1, wt_bar, 1);
        for (i = 0; i < p; i++) {
            if (wt[i] != 0.0) {
                nonzeros_wt[tt] += 1;
            }
            if (wt_bar[i] != 0.0) {
                nonzeros_wt_bar[tt] += 1;
            }
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(loss_grad), free(wt_tmp), free(sorted_ind);
    return true;
}


bool algo_online_ghtp_logit_sparse(
        const double *x_tr, const double *y_tr, const double *w0, int s, int p,
        int num_tr, double lr, double eta, int verbose, double *wt,
        double *wt_bar, int *nonzeros_wt, int *nonzeros_wt_bar,
        double *pred_prob_wt, double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar, double *losses, int *missed_wt,
        int *missed_wt_bar, double *total_time) {
    openblas_set_num_threads(1);
    int i;
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    cblas_dcopy(p + 1, w0, 1, wt_bar, 1); // w0 --> wt_bar
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    int *sorted_ind = malloc(sizeof(double) * p);
    int total_missed_wt = 0, total_missed_wt_bar = 0, max_iter = 50;
    Array *re_nodes = malloc(sizeof(Array));
    re_nodes->array = malloc(sizeof(int) * p);
    clock_t start_total = clock();
    if (verbose > 0) {
        printf("learning rate: %lf, num_tr: %d ,p: %d\n", lr, num_tr, p);
    }
    for (i = 0; i < num_tr; i++) {
        nonzeros_wt[i] = 0;
        nonzeros_wt_bar[i] = 0;
    }
    for (int tt = 0; tt < num_tr; tt++) {
        // each time has one sample.
        logistic_predict(x_tr + tt * p, wt, &pred_prob_wt[tt],
                         &pred_label_wt[tt], 0.5, 1, p);
        logistic_predict(x_tr + tt * p, wt_bar, &pred_prob_wt_bar[tt],
                         &pred_label_wt_bar[tt], 0.5, 1, p);
        logistic_loss_grad_sparse(
                wt, x_tr + tt * p, y_tr + tt, loss_grad, eta, 1, p);
        if (pred_label_wt[tt] != y_tr[tt]) {
            total_missed_wt++;
        }
        if (pred_label_wt_bar[tt] != y_tr[tt]) {
            total_missed_wt_bar++;
        }
        missed_wt[tt] = total_missed_wt;
        missed_wt_bar[tt] = total_missed_wt_bar;
        losses[tt] = loss_grad[0];
        if (verbose > 0) {
            printf("losses[%d]:%.6f sparsity:%d ", tt, losses[tt], s);
        }
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -lr / sqrt(tt + 1.), loss_grad + 1, 1, wt_tmp, 1);
        arg_magnitude_sort_descend(wt_tmp, sorted_ind, p);
        re_nodes->size = 0;
        for (i = 0; i < s; i++) {
            re_nodes->array[re_nodes->size++] = sorted_ind[i];
        }
        min_f_sparse(re_nodes, x_tr, y_tr, max_iter, eta, wt, tt + 1, p);
        if (verbose > 0) {
            printf("pred_prob:%.4f total_missed:%d/%d\n",
                   pred_prob_wt[tt], total_missed_wt, tt + 1);
        }
        cblas_dscal(p + 1, tt / (tt + 1.), wt_bar, 1);
        cblas_daxpy(p + 1, 1 / (tt + 1.), wt, 1, wt_bar, 1);
        for (i = 0; i < p; i++) {
            if (wt[i] != 0.0) {
                nonzeros_wt[tt] += 1;
            }
            if (wt_bar[i] != 0.0) {
                nonzeros_wt_bar[tt] += 1;
            }
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(loss_grad), free(wt_tmp), free(sorted_ind);
    return true;
}

bool algo_batch_iht_logit(
        const double *x_tr, const double *y_tr, const double *w0, int s,
        int p, int num_tr, double lr, double eta, int max_iter, double tol,
        int verbose, double *wt, double *losses, double *total_time) {
    openblas_set_num_threads(1);
    int i;
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    int *sorted_ind = malloc(sizeof(double) * p);
    Array *re_nodes = malloc(sizeof(Array));
    re_nodes->array = malloc(sizeof(int) * p);
    clock_t start_total = clock();
    if (verbose > 0) {
        printf("learning rate: %lf, num_tr: %d ,p: %d\n", lr, num_tr, p);
    }
    for (int tt = 0; tt < max_iter; tt++) {
        // each time has one sample.
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, num_tr, p);
        losses[tt] = loss_grad[0];
        if (verbose > 0) {
            printf("losses[%d]:%.6f sparsity:%d ", tt, losses[tt], s);
        }
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
        arg_magnitude_sort_descend(wt_tmp, sorted_ind, p);
        for (i = 0; i < s; i++) {
            wt[sorted_ind[i]] = wt_tmp[sorted_ind[i]];
        }
        wt[p] = wt_tmp[p];
        if (tt >= 1 && (fabs(losses[tt] - losses[tt - 1]) < tol)) {
            break; // stop earlier when it almost stops decreasing the loss
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(loss_grad), free(wt_tmp), free(sorted_ind);
    free(re_nodes->array), free(re_nodes);
    return true;
}

bool algo_batch_ghtp_logit(
        const double *x_tr, const double *y_tr, const double *w0, int s,
        int p, int num_tr, double lr, double eta, int max_iter, double tol,
        int verbose, double *wt, double *losses, double *total_time) {
    openblas_set_num_threads(1);
    int i;
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    int *sorted_ind = malloc(sizeof(double) * p);
    Array *re_nodes = malloc(sizeof(Array));
    re_nodes->array = malloc(sizeof(int) * p);
    clock_t start_total = clock();
    if (verbose > 0) {
        printf("learning rate: %lf, num_tr: %d ,p: %d\n", lr, num_tr, p);
    }
    for (int tt = 0; tt < max_iter; tt++) {
        // each time has one sample.
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, num_tr, p);
        losses[tt] = loss_grad[0];
        if (verbose > 0) {
            printf("losses[%d]:%.6f sparsity:%d ", tt, losses[tt], s);
        }
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
        arg_magnitude_sort_descend(wt_tmp, sorted_ind, p);
        re_nodes->size = 0;
        for (i = 0; i < s; i++) {
            re_nodes->array[re_nodes->size++] = sorted_ind[i];
        }
        min_f(re_nodes, x_tr, y_tr, max_iter, eta, wt, num_tr, p);
        if (tt >= 1 && (fabs(losses[tt] - losses[tt - 1]) < tol)) {
            break; // stop earlier when it almost stops decreasing the loss
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(loss_grad), free(wt_tmp), free(sorted_ind);
    free(re_nodes->array), free(re_nodes);
    return true;
}

bool algo_batch_graph_iht_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double tol, int max_iter, double lr, double eta, int verbose,
        double *wt, Array *re_nodes, Array *re_edges, int *num_pcst,
        double *losses, double *run_time_head, double *run_time_tail,
        double *total_time) {
    openblas_set_num_threads(1);
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    double *loss_grad = malloc(sizeof(double) * (p + 2));
    double *wt_tmp = malloc(sizeof(double) * (p + 1));
    double *tmp_prizes = malloc(sizeof(double) * p);
    double *tmp_costs = malloc(sizeof(double) * m);
    clock_t start_head, start_tail, start_total = clock();
    double C = 2. * (s - 1.), delta = 1. / 169., nu = 2.5, err_tol = 1e-6;
    double budget = (s - 1.), eps = 1e-6, tmp_time;
    int i, root = -1;
    enum PruningMethod pruning = GWPruning;
    for (i = 0; i < m; i++) { tmp_costs[i] = costs[i] + budget / (double) s; }
    for (int tt = 0; tt < max_iter; tt++) {
        // each time has one sample.
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, num_tr, p);
        losses[tt] = loss_grad[0];
        for (i = 0; i < p; i++) {
            tmp_prizes[i] = loss_grad[i + 1] * loss_grad[i + 1];
        }
        start_head = clock();
        head_proj_exact(edges, tmp_costs, tmp_prizes, g, C, delta, max_iter,
                        err_tol, root, pruning, eps, p, m, verbose, re_nodes,
                        re_edges, num_pcst, &tmp_time);
        cblas_dcopy(p + 1, loss_grad + 1, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_dscal(p + 1, 0.0, loss_grad + 1, 1);
        for (i = 0; i < re_nodes->size; i++) {
            int cur_node = re_nodes->array[i];
            loss_grad[cur_node + 1] = wt_tmp[cur_node];
        }
        *run_time_head += ((double) (clock() - start_head)) / CLOCKS_PER_SEC;
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
        for (i = 0; i < p; i++) {
            tmp_prizes[i] = wt_tmp[i] * wt_tmp[i];
        }
        start_tail = clock();
        tail_proj_exact(edges, tmp_costs, tmp_prizes, g, C, nu, max_iter,
                        err_tol, root, pruning, eps, p, m, verbose, re_nodes,
                        re_edges, num_pcst, &tmp_time);
        *run_time_tail += ((double) (clock() - start_tail)) / CLOCKS_PER_SEC;
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (i = 0; i < re_nodes->size; i++) {
            int cur_node = re_nodes->array[i];
            wt[cur_node] = wt_tmp[cur_node];
        }
        wt[p] = wt_tmp[p];
        if ((tt >= 1 && (fabs(losses[tt] - losses[tt - 1]) < tol)) ||
            (tt >= 1 && (losses[tt] >= losses[tt - 1]))) {
            break; // stop earlier when it almost stops decreasing the loss
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(tmp_costs), free(tmp_prizes), free(loss_grad), free(wt_tmp);
    return true;
}

bool algo_batch_graph_ghtp_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double tol, int max_iter, double lr, double eta, int verbose,
        double *wt, Array *re_nodes, Array *re_edges, int *num_pcst,
        double *losses, double *run_time_head, double *run_time_tail,
        double *total_time) {
    openblas_set_num_threads(1);
    int i, root = -1, fmin_max_iter = 20;
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    double *tmp_prizes = malloc(sizeof(double) * p);
    double *tmp_costs = malloc(sizeof(double) * m);
    clock_t start_head, start_tail, start_total = clock();
    double C = 2. * (s - 1.), delta = 1. / 169., nu = 2.5;
    double err_tol = 1e-6, budget = (s - 1.), eps = 1e-6, tmp_time;
    enum PruningMethod pruning = GWPruning;
    for (i = 0; i < m; i++) {
        tmp_costs[i] = costs[i] + budget / (double) s;
    }
    for (int tt = 0; tt < max_iter; tt++) {
        // each time has one sample.
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, num_tr, p);
        losses[tt] = loss_grad[0];
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
        for (i = 0; i < p; i++) {
            tmp_prizes[i] = wt_tmp[i] * wt_tmp[i];
        }
        start_head = clock();
        head_proj_exact(edges, tmp_costs, tmp_prizes, g, C, delta, max_iter,
                        err_tol, root, pruning, eps, p, m, verbose, re_nodes,
                        re_edges, num_pcst, &tmp_time);
        *run_time_head += ((double) (clock() - start_head)) / CLOCKS_PER_SEC;
        min_f(re_nodes, x_tr, y_tr, fmin_max_iter, eta, wt, num_tr, p);
        for (i = 0; i < p; i++) {
            tmp_prizes[i] = wt[i] * wt[i];
        }
        start_tail = clock();
        tail_proj_exact(edges, tmp_costs, tmp_prizes, g, C, nu, max_iter,
                        err_tol, root, pruning, eps, p, m, verbose, re_nodes,
                        re_edges, num_pcst, &tmp_time);
        *run_time_tail += ((double) (clock() - start_tail)) / CLOCKS_PER_SEC;
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (i = 0; i < re_nodes->size; i++) {
            wt[re_nodes->array[i]] = wt_tmp[re_nodes->array[i]];
        }
        wt[p] = wt_tmp[p];
        if ((tt >= 1 && (fabs(losses[tt] - losses[tt - 1]) < tol)) ||
            (tt >= 1 && (losses[tt] >= losses[tt - 1]))) {
            break; // stop earlier when it almost stops decreasing the loss
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(tmp_costs), free(tmp_prizes), free(loss_grad), free(wt_tmp);
    return true;
}


bool algo_batch_graph_posi_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double tol, int max_iter, double lr, double eta, int verbose,
        double *wt, Array *re_nodes, Array *re_edges, int *num_pcst,
        double *losses, double *run_time_head, double *run_time_tail,
        double *total_time) {
    openblas_set_num_threads(1);
    int i, root = -1, fmin_max_iter = 20;
    cblas_dcopy(p + 1, w0, 1, wt, 1); // w0 --> wt
    double *loss_grad = malloc((p + 2) * sizeof(double));
    double *wt_tmp = malloc((p + 1) * sizeof(double));
    double *tmp_prizes = malloc(sizeof(double) * p);
    double *tmp_costs = malloc(sizeof(double) * m);
    clock_t start_head, start_tail, start_total = clock();
    double C = 2. * (s - 1.), delta = 1. / 169., nu = 2.5;
    double err_tol = 1e-6, budget = (s - 1.), eps = 1e-6, tmp_time;
    enum PruningMethod pruning = GWPruning;
    for (i = 0; i < m; i++) {
        tmp_costs[i] = costs[i] + budget / (double) s;
    }
    for (int tt = 0; tt < max_iter; tt++) {
        // each time has one sample.
        logistic_loss_grad(wt, x_tr, y_tr, loss_grad, eta, num_tr, p);
        losses[tt] = loss_grad[0];
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1); // wt --> wt_tmp
        cblas_daxpy(p + 1, -lr, loss_grad + 1, 1, wt_tmp, 1);
        for (i = 0; i < p; i++) {
            tmp_prizes[i] = wt_tmp[i] * wt_tmp[i];
        }
        start_head = clock();
        head_proj_exact(edges, tmp_costs, tmp_prizes, g, C, delta, max_iter,
                        err_tol, root, pruning, eps, p, m, verbose, re_nodes,
                        re_edges, num_pcst, &tmp_time);
        if (verbose > 0) {
            printf("losses[%d]: %.6f h_nodes:%d",
                   tt, losses[tt], re_nodes->size);
        }
        *run_time_head += ((double) (clock() - start_head)) / CLOCKS_PER_SEC;
        min_f_posi(re_nodes, x_tr, y_tr, fmin_max_iter, eta, wt, num_tr, p);
        for (i = 0; i < p; i++) {
            tmp_prizes[i] = wt[i] * wt[i];
        }
        start_tail = clock();
        tail_proj_exact(edges, tmp_costs, tmp_prizes, g, C, nu, max_iter,
                        err_tol, root, pruning, eps, p, m, verbose, re_nodes,
                        re_edges, num_pcst, &tmp_time);
        *run_time_tail += ((double) (clock() - start_tail)) / CLOCKS_PER_SEC;
        cblas_dcopy(p + 1, wt, 1, wt_tmp, 1);
        cblas_dscal(p + 1, 0.0, wt, 1);
        for (i = 0; i < re_nodes->size; i++) {
            wt[re_nodes->array[i]] = wt_tmp[re_nodes->array[i]];
        }
        wt[p] = wt_tmp[p];
        if (tt >= 1 && (fabs(losses[tt] - losses[tt - 1]) < tol)) {
            break; // stop earlier when it almost stops decreasing the loss
        }
    }
    *total_time = (double) (clock() - start_total) / CLOCKS_PER_SEC;
    free(tmp_costs), free(tmp_prizes), free(loss_grad), free(wt_tmp);
    return true;
}