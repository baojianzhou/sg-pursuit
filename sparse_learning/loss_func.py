# -*- coding: utf-8 -*-
__all__ = ["loss_logistic_intercept_dot"]
import numpy as np
from scipy.special import expit


def _log_logistic_sigmoid(n_samples, n_features, X, out):
    for i in range(n_samples):
        for j in range(n_features):
            if X[i, j] > 0:
                out[i, j] = -np.log(1 + np.exp(-X[i, j]))
            else:
                out[i, j] = X[i, j] - np.log(1 + np.exp(X[i, j]))
    return out


def log_logistic(X, out=None):
    """Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.
    This implementation is numerically stable because it splits positive and
    negative values::
        -log(1 + exp(-x_i))     if x_i > 0
        x_i - log(1 + exp(x_i)) if x_i <= 0
    For the ordinary logistic function, use ``scipy.special.expit``.
    Parameters
    ----------
    X : array-like, shape (M, N) or (M, )
        Argument to the logistic function
    out : array-like, shape: (M, N) or (M, ), optional:
        Preallocated output array.
    Returns
    -------
    out : array, shape (M, N) or (M, )
        Log of the logistic function evaluated at every point in x
    Notes
    -----
    See the blog post describing this implementation:
    http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
    """
    is_1d = X.ndim == 1
    X = np.atleast_2d(X)
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape
    if out is None:
        out = np.empty_like(X)
    _log_logistic_sigmoid(n_samples, n_features, X, out)

    if is_1d:
        return np.squeeze(out)
    return out


def loss_logistic_intercept_dot(w, X, y):
    """Computes y * np.dot(X, w).
    It takes into consideration if the intercept should be fit or not.
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    Returns
    -------
    w : ndarray, shape (n_features,)
        Coefficient vector without the intercept weight (w[-1]) if the
        intercept should be fit. Unchanged otherwise.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data. Unchanged.
    yz : float
        y * np.dot(X, w).
    """
    c = 0.
    if w.size == X.shape[1] + 1:
        c = w[-1]
        w = w[:-1]
    z = np.dot(X, w) + c
    yz = y * z
    return w, c, yz


def loss_logistic_loss_and_grad(w, X, y, alpha, sample_weight=None):
    """Computes the logistic loss and gradient.
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.
    Returns
    -------
    out : float
        Logistic loss.
    grad : ndarray, shape (n_features,) or (n_features + 1,)
        Logistic gradient.
    """
    n_samples, n_features = X.shape
    grad = np.empty_like(w)
    c = 0.
    if w.size == X.shape[1] + 1:
        c = w[-1]
        w = w[:-1]
    yz = y * (np.dot(X, w) + c)
    if sample_weight is None:
        sample_weight = np.ones(n_samples)
    # Logistic loss is the negative of the log of the logistic function.
    out = -np.sum(sample_weight * log_logistic(yz)) + .5 * alpha * np.dot(w, w)
    z0 = sample_weight * (expit(yz) - 1) * y
    grad[:n_features] = np.dot(X.T, z0) + alpha * w
    # Case where we fit the intercept.
    if grad.shape[0] > n_features:
        grad[-1] = z0.sum()
    return out, grad


def test():
    w = np.asarray([1, 2, 1, 1.], dtype=np.float64)
    x = np.asarray([[1, 2, 1], [-3, 4, -1], [-3, 4, -1]], dtype=np.float64)
    y = np.asarray([1, -1, 1], dtype=np.float64)
    w, c, yz = loss_logistic_intercept_dot(w, x, y)
    print('w: ', w)
    print('c: ', c)
    print('yz: ', yz)
    import time

    start_time = time.time()
    out = log_logistic(X=(np.random.normal(0.0, 1.0, 100000000).reshape(10000, 10000)))
    print(out)
    print('run time: ', (time.time() - start_time))


from sklearn.linear_model.logistic import _logistic_loss_and_grad
from sklearn.linear_model.logistic import _logistic_loss
from sklearn.linear_model.logistic import _logistic_grad_hess
import time

start_time = time.time()
out, grad = _logistic_loss_and_grad(w=np.random.normal(0.0, 1.0, 10000),
                                    X=np.random.normal(0.0, 1.0, 10000 * 10000).reshape(10000, 10000),
                                    y=np.asarray([1, -1] * 5000),
                                    alpha=0.5, sample_weight=None)

print('run time: ', time.time() - start_time)
print(out)
print(grad)
print('-' * 50)
out, grad = _logistic_loss_and_grad(w=np.asarray([0., 0., 0., 1.]),
                                    X=np.asarray([[1., 2., 3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.]]),
                                    y=np.asarray([1, -1, 1]),
                                    alpha=0.5, sample_weight=None)
print(out)
print(grad)

print('-' * 50)
out, grad = _logistic_loss_and_grad(w=np.asarray([0., 0., 0., 1., -1., 0.5]),
                                    X=np.asarray([[1., 2., 3., 4., -2], [1., 2., 3., 4., -.3]]),
                                    y=np.asarray([1, -1]),
                                    alpha=0.5, sample_weight=None)
print(out)
print(grad)

print('-' * 50)
out, grad = _logistic_loss_and_grad(w=np.asarray([0., 0., 0., 1., -1.]),
                                    X=np.asarray([[1., 2., 3., 4., -2], [1., 2., 3., 4., -.3]]),
                                    y=np.asarray([1, -1]),
                                    alpha=0.5, sample_weight=None)
print(out)
print(grad)

print('-' * 50)
out, grad = _logistic_loss_and_grad(w=np.asarray([0., 0., 0., 0., -0.]),
                                    X=np.asarray([[1., 2., 3., 4., -2], [1., 2., 3., 4., -.3]]),
                                    y=np.asarray([1, -1]),
                                    alpha=0.5, sample_weight=None)
print(out)
print(grad)

print('-' * 50)
out = _logistic_loss(w=np.asarray([0., 0., 0., 0., -0.]),
                     X=np.asarray([[1., 2., 3., 4., -2], [1., 2., 3., 4., -.3]]),
                     y=np.asarray([1, -1]),
                     alpha=0.5, sample_weight=None)
print(out)
