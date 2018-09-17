//
// Created by baojian on 8/16/18.
//

#include <cblas.h>
#include <stdio.h>
#include <stdbool.h>
#include "loss.h"

bool test_matrix_vector() {
    int n = 3, p = 4;
    double x_tr[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    double w[] = {1., 1., 1., 1.};
    double yz[] = {1., 2., 3.};
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                n, p, 1., x_tr, p, w, 1, 1., yz, 1);
    printf("matrix * vector: %lf %lf %lf\n", yz[0], yz[1], yz[2]);
    return true;
}

bool test_log_sum_exp() {
    double x[] = {1., 0., -1.};
    int x_len = 3;
    printf("log_sum_exp of x: %lf\n", log_sum_exp(x, x_len));
    return true;
}

bool test_logistic() {
    double x[] = {1., 0., -1.}, out[3];
    int x_len = 3;
    logistic(x, out, x_len);
    printf("log_sum_exp of x: %lf %lf %lf\n", out[0], out[1], out[2]);
    return true;
}

int main() {
    test_matrix_vector();
    test_log_sum_exp();
    test_logistic();
}