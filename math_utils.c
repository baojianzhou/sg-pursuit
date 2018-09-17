//
// Created by baojian on 8/24/18.
//
#include <math.h>
#include "math_utils.h"

double norm_l2(double *x, int x_len) {
    double result = 0.0;
    for (int i = 0; i < x_len; i++) {
        result += x[i] * x[i];
    }
    return sqrt(result);
}