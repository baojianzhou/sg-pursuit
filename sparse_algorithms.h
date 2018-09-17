//
// Created by baojian on 8/11/18.
//

#ifndef FAST_PCST_SPARSE_ALGORITHMS_H
#define FAST_PCST_SPARSE_ALGORITHMS_H

#include "head_tail_proj.h"

bool algo_online_graph_iht_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double lr, double eta, int verbose, double *wt, double *wt_bar,
        int *nonzeros_wt, int *nonzeros_wt_bar, double *pred_prob_wt,
        double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar,
        Array *re_nodes, Array *re_edges, int *num_pcst, double *losses,
        double *run_time_head, double *run_time_tail, int *missed_wt,
        int *missed_wt_bar, double *total_time);

bool algo_online_iht_logit(
        const double *x_tr, const double *y_tr, const double *w0, int s, int p,
        int num_tr, double lr, double eta, int verbose, double *wt,
        double *wt_bar, int *nonzeros_wt, int *nonzeros_wt_bar,
        double *pred_prob_wt, double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar, double *losses, int *missed_wt,
        int *missed_wt_bar, double *total_time);


bool algo_online_sgd_l1_logit(
        const double *x_tr, const double *y_tr, const double *w0, int p,
        int num_tr, double lr, double eta, int verbose, double *wt,
        double *wt_bar, int *nonzeros_wt, int *nonzeros_wt_bar,
        double *pred_prob_wt, double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar, double *losses, int *missed_wt,
        int *missed_wt_bar, double *total_time);

bool algo_online_rda_l1_logit(
        const double *x_tr, const double *y_tr, const double *w0, int p,
        int num_tr, double lambda, double gamma, double rho, int verbose,
        double *wt, double *wt_bar, int *nonzeros_wt, int *nonzeros_wt_bar,
        double *pred_prob_wt, double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar, double *losses, int *missed_wt,
        int *missed_wt_bar, double *total_time);

bool algo_online_sgd_l2_logit(
        const double *x_tr, const double *y_tr, const double *w0, int p,
        int num_tr, double lr, double eta, int verbose, double *wt,
        double *wt_bar, int *nonzeros_wt, int *nonzeros_wt_bar,
        double *pred_prob_wt, double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar, double *losses, int *missed_wt,
        int *missed_wt_bar, double *total_time);

bool algo_online_graph_ghtp_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double lr, double eta, int verbose, double *wt, double *wt_bar,
        int *nonzeros_wt, int *nonzeros_wt_bar, double *pred_prob_wt,
        double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar,
        Array *re_nodes, Array *re_edges, int *num_pcst, double *losses,
        double *run_time_head, double *run_time_tail, int *missed_wt,
        int *missed_wt_bar, double *total_time);

bool algo_online_ghtp_logit(
        const double *x_tr, const double *y_tr, const double *w0, int s, int p,
        int num_tr, double lr, double eta, int verbose, double *wt,
        double *wt_bar, int *nonzeros_wt, int *nonzeros_wt_bar,
        double *pred_prob_wt, double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar, double *losses, int *missed_wt,
        int *missed_wt_bar, double *total_time);

bool algo_online_ghtp_logit_sparse(
        const double *x_tr, const double *y_tr, const double *w0, int s, int p,
        int num_tr, double lr, double eta, int verbose, double *wt,
        double *wt_bar, int *nonzeros_wt, int *nonzeros_wt_bar,
        double *pred_prob_wt, double *pred_label_wt, double *pred_prob_wt_bar,
        double *pred_label_wt_bar, double *losses, int *missed_wt,
        int *missed_wt_bar, double *total_time);

bool algo_batch_ghtp_logit(
        const double *x_tr, const double *y_tr, const double *w0, int s, int p,
        int num_tr, double lr, double eta, int max_iter, double tol,
        int verbose, double *wt, double *losses, double *total_time);

bool algo_batch_iht_logit(
        const double *x_tr, const double *y_tr, const double *w0, int s,
        int p, int num_tr, double lr, double eta, int max_iter, double tol,
        int verbose, double *wt, double *losses, double *total_time);

bool algo_batch_graph_ghtp_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double tol, int max_iter, double lr, double eta, int verbose,
        double *wt, Array *re_nodes, Array *re_edges, int *num_pcst,
        double *losses, double *run_time_head, double *run_time_tail,
        double *total_time);

bool algo_batch_graph_iht_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double tol, int max_iter, double lr, double eta, int verbose,
        double *wt, Array *re_nodes, Array *re_edges, int *num_pcst,
        double *losses, double *run_time_head, double *run_time_tail,
        double *total_time);

bool algo_batch_graph_posi_logit(
        const EdgePair *edges, const double *costs, int g, int s, int p, int m,
        const double *x_tr, const double *y_tr, const double *w0, int num_tr,
        double tol, int max_iter, double lr, double eta, int verbose,
        double *wt, Array *re_nodes, Array *re_edges, int *num_pcst,
        double *losses, double *run_time_head, double *run_time_tail,
        double *total_time);

#endif //FAST_PCST_SPARSE_ALGORITHMS_H
