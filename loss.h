//
// Created by baojian on 8/16/18.
//

#ifndef FAST_PCST_LOSS_H
#define FAST_PCST_LOSS_H

#include <math.h>

/**
 * Quote from Scipy:
 * The expit function, also known as the logistic function,
 * is defined as expit(x) = 1/(1+exp(-x)).
 * It is the inverse of the logit function.
 * expit is also known as logistic. Please see logistic
 * @param x
 * @param out
 * @param x_len
 */
void expit(const double *x, double *out, int x_len);


/**
 * Compute the log of the logistic function, ``log(1 / (1 + e ** -x))
 * @param x input vector
 * @param out log of the logistic value.
 * @param x_len size of x
 */
void log_logistic(const double *x, double *out, int x_len);

/**
 * logistic is also known as expit. Please see expit.
 * @param x
 * @param out
 * @param x_len
 */
void logistic(const double *x, double *out, int x_len);

double log_sum_exp(const double *x, int x_len);

/**
 * Computes the logistic loss and gradient.
 * Parameters
 * ----------
 * @param w: (n_features + 1,) Coefficient vector.
 * @param x: (n_samples, n_features)  Training data. (CblasRowMajor)
 * @param y: (n_samples,)           Array of labels.
 * @param alpha:   float Regularization parameter. equal to 1 / C.
 * @param weight: (n_samples,) optional
 *          Array of weights that are assigned to individual samples.
 *          If not provided, then each sample is given unit weight.
 * @param n_samples: number of samples
 * @param n_features: number of features
 * @return (loss, grad) (1,(n_features,)) or (1,(n_features + 1,)) loss, grad
 */
void logistic_loss_grad(const double *w,
                        const double *x_tr,
                        const double *y_tr,
                        double *loss_grad,
                        double eta,
                        int n_samples,
                        int n_features);

void logistic_loss_grad_sparse(const double *w,
                               const double *x_tr,
                               const double *y_tr,
                               double *loss_grad,
                               double eta,
                               int n_samples,
                               int n_features);

void logistic_predict(const double *x_te,
                      const double *wt,
                      double *pred_prob,
                      double *pred_label,
                      double threshold,
                      int n,
                      int p);

#endif //FAST_PCST_LOSS_H
