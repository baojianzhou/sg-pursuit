# -*- coding: utf-8 -*-
__all__ = ["ghtp_logistic_py", "graph_ghtp_logistic_py"]
import numpy as np

try:
    import proj_module
    try:
        from proj_module import ghtp_logistic
        from proj_module import graph_ghtp_logistic
    except ImportError:
        print('cannot find these two functions: ghtp_logistic, graph_ghtp_logistic')
        exit(0)
except ImportError:
    print('cannot find the package proj_module')


def ghtp_logistic_py(x_tr, y_tr, w0, lr, sparsity, tol, max_iter, eta):
    """
    :param x_tr: (n,p) training data
    :param y_tr: (n,) testing data
    :param w0: initial point, default is zero
    :param lr: learning rate
    :param sparsity: sparsity k
    :param tol: tolerance of algorithm for stop condition.
    :param max_iter: maximal number of iterations for GHTP algorithm.
    :param eta: regularization parameter eta
    :return: [losses, wt, intercept]
    """
    return ghtp_logistic(x_tr, y_tr, w0, lr, sparsity, tol, max_iter, eta)


def graph_ghtp_logistic_py(x_tr, y_tr, w0, lr, sparsity, tol, maximal_iter, eta,
                           edges, weights, g, budget, delta, mu):
    """
    :param x_tr: (n,p) training data
    :param y_tr: (n,) testing data
    :param w0: initial point, default is zero
    :param lr: learning rate
    :param sparsity: sparsity k
    :param tol: tolerance of algorithm for stop condition.
    :param maximal_iter: maximal number of iterations for GHTP algorithm.
    :param eta: regularization parameter eta
    :param edges: ndarray(m,2), where m is the number of edges in the input graph.
    :param weights: ndarray(m,), where m is the number of edges in the input graph.
    :param g: int, the number of connected components.
    :param budget: budget constraint in weighted graph model.
    :param delta: for tail projection.
    :param mu: for head projection.
    :return: [losses, wt, intercept]
    """
    pass
