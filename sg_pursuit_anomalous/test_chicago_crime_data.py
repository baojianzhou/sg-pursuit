# -*- coding: utf-8 -*-
# !/usr/bin/python
import cPickle
import bz2
import numpy as np
from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj


class FuncEMS(object):
    def __init__(self, data_matrix, lambda_):
        self.data_matrix = data_matrix
        n, p = data_matrix.shape
        self.n = n
        self.p = p
        self.lambda_ = lambda_

    def get_fun_val(self, x, y):
        sum_x, sum_y = np.sum(x), np.sum(y)
        w_y = np.dot(self.data_matrix, y)
        xt_w_y = np.dot(x, w_y)
        reg = .5 * (self.lambda_ * (sum_y ** 2.))
        func_val = - xt_w_y / np.sqrt(sum_x) + reg
        return func_val

    def get_initial_point(self, k, s):
        x0, y0 = np.zeros(self.n), np.zeros(self.p)
        x, y = [], []
        res = []
        scores = np.zeros(self.p)
        for j in range(self.p):
            wt_j = self.data_matrix.T[j]
            # descending order
            sorted_indices = np.argsort(wt_j)[-k:]
            x, y = np.zeros(self.n), np.zeros(self.p)
            x[sorted_indices] = 1.
            y[j] = 1.
            func_val = self.get_fun_val(x, y)
            scores[j] = -func_val
            res.append(sorted_indices)
        func_val = -1.
        indices = np.argsort(scores)[-s:]
        for j in range(s):
            y1 = np.zeros(self.p)
            y1[[indices[_] for _ in range(j + 1)]] = 1.
            x1 = np.zeros(self.n)
            x1[res[indices[0]]] = 1.
            func_val1 = self.get_fun_val(x1, y1)
            if func_val == -1. or func_val > func_val1:
                y = [indices[_] for _ in range(j + 1)]
                x = res[indices[0]]
        for j in range(s):
            x1, y1 = np.zeros(self.n), np.zeros(self.p)
            x1[res[indices[j]]], y1[indices[j]] = 1., 1.
            func_val1 = self.get_fun_val(x1, y1)
            if func_val1 == -1 or func_val > func_val1:
                y = [_ for _ in indices[j]]
                x = [_ for _ in res[indices[j]]]
        x0[x], y0[y] = 1., 1.
        return x0, y0

    def get_argmin_f_xy(self, x0, y0, omega_x, omega_y):
        lr, max_iter = 0.01, 1000
        xt, yt = np.copy(x0), np.copy(y0)
        indicator_x, indicator_y = np.zeros_like(x0), np.zeros_like(y0)
        indicator_x[omega_x] = 1.
        indicator_y[omega_y] = 1.
        for i in range(max_iter):
            loss, grad_x, grad_y = self.get_loss_grad(xt, yt)
            x_pre, y_pre = np.copy(xt), np.copy(yt)
            xt = FuncEMS.update_minimizer(grad_x, indicator_x, xt, 5, lr)
            yt = FuncEMS.update_minimizer(grad_y, indicator_y, yt, 5, lr)
            diff_norm_x = np.sqrt((xt - x_pre) ** 2.)
            diff_norm_y = np.sqrt((yt - y_pre) ** 2.)
            if diff_norm_x <= 1e-6 and diff_norm_y <= 1e-6:
                break
        return xt, yt

    def get_loss_grad(self, x, y):
        sum_x, sum_y = np.sum(x), np.sum(y)
        w_y = np.dot(self.data_matrix, y)
        xt_w_y = np.dot(x, w_y)
        reg = .5 * (self.lambda_ * (sum_y ** 2.))
        func_val = - xt_w_y / np.sqrt(sum_x) + reg
        if sum_x == 0.0 or sum_y == 0.0:
            print('GradientX:Input x vector values are all Zeros !!!')
            exit(0)
        term1 = (1. / np.sqrt(sum_x)) * w_y
        term2 = .5 * xt_w_y / (np.sqrt(sum_x) * sum_x)
        grad_x = -term1 + term2
        x_w = np.dot(self.data_matrix.T, x)
        grad_y = 1. / np.sqrt(sum_x) * x_w + self.lambda_ * y
        if np.isnan(grad_x).any() or np.isnan(grad_y).any():
            print('something is wrong. gradient x or y')
        return func_val, grad_x, grad_y

    @staticmethod
    def update_minimizer(grad_x, indicator_x, x, bound, step_size):
        normalized_x = np.zeros_like(grad_x)
        for j in range(len(x)):
            normalized_x[j] = (x[j] - step_size * grad_x[j]) * indicator_x[j]
        sorted_indices = np.argsort(normalized_x)
        num_non_posi = 0
        for j in range(len(x)):
            if normalized_x[j] <= 0.0:
                num_non_posi += 1
                normalized_x[j] = 0.
            elif normalized_x[j] > 1.:
                normalized_x[j] = 1.
        if num_non_posi == len(x):
            print('siga-1 is too large and all values'
                  ' in the gradient are non-positive.')
            for i in range(bound):
                normalized_x[sorted_indices[i]] = 1.
        return normalized_x


def data_convert():
    root_path = '/network/rit/lab/ceashpc/' \
                'bz383376/data/icdm17/CrimesOfChicago/'
    for test_case in range(52):
        for event_type in ['BATTERY', 'BURGLARY']:
            data = {'test_case': test_case, 'data_matrix': [], 'edges': [],
                    'costs': [], 'event_type': event_type,
                    'true_sub_graph': [], 'true_sub_feature': []}
            print(event_type, test_case)
            with open(root_path + 'graph/processed_%s_test_case_%d.txt'
                      % (event_type, test_case)) as f:
                n, m, p = 0, 0, 0
                for ind, each_line in enumerate(f.readlines()):
                    each_line = each_line.lstrip().rstrip()
                    if ind == 0:
                        data['n'] = int(each_line.split(' ')[0])
                        data['p'] = int(each_line.split(' ')[1])
                        n, p = data['n'], data['p']
                    elif 1 <= ind <= n:
                        arr = [float(_) for _ in each_line.split(' ')]
                        data['data_matrix'].append(arr)
                    elif ind == n + 1:
                        data['m'] = int(each_line)
                        m = int(each_line)
                    elif n + 2 <= ind <= m + n + 1:
                        edge = [each_line.split(' ')[0],
                                each_line.split(' ')[1]]
                        data['edges'].append(edge)
                        data['costs'].append(
                            float(each_line.split(' ')[2]))
                    elif ind == m + n + 2:
                        sub_graph = [int(_) for _ in each_line.split(' ')]
                        data['true_sub_graph'] = sub_graph
                    elif ind == m + n + 3:
                        sub_feature = [int(_) for _ in each_line.split(' ')]
                        data['true_sub_feature'] = sub_feature
            data['data_matrix'] = np.asarray(data['data_matrix'],
                                             dtype=np.float64)
            data['edges'] = np.asarray(data['edges'], dtype=int)
            data['costs'] = np.asarray(data['costs'],
                                       dtype=np.float64)
            file_name = root_path + 'chicago_%s_case_%d.pkl' % \
                        (event_type, test_case)
            bz2_f = bz2.BZ2File(file_name, 'wb')
            cPickle.dump(data, bz2_f)


def normalize_gradient(x, grad):
    normalize_grad = np.zeros_like(grad)
    for i in range(len(grad)):
        if grad[i] < 0.0 and x[i] == 1.:
            normalize_grad[i] = 0.0
        elif grad[i] > 0.0 and x[i] == 0.0:
            normalize_grad[i] = 0.0
        else:
            normalize_grad[i] = grad[i]
    return normalize_grad


def identify_direction(grad, s):
    if np.sum(grad) == 0.0:
        return set()
    gamma_y = set()
    grad_sq = grad * grad
    sorted_indices = np.argsort(grad_sq)[-s:]
    for i in range(s):
        if i < len(sorted_indices) and grad_sq[sorted_indices[i]] > 0.0:
            gamma_y.add(sorted_indices[i])
    return gamma_y


def sg_pursuit_algo(para):
    k, s, data_matrix, edges, costs, max_iter, lambda_ = para
    g, t = 1, 5
    budget = k - g + 0.0
    func = FuncEMS(data_matrix=data_matrix, lambda_=lambda_)
    xt, yt = func.get_initial_point(k, s)
    while True:
        func_val, grad_x, grad_y = func.get_loss_grad(xt, yt)
        grad_x = normalize_gradient(xt, grad_x)
        grad_y = normalize_gradient(yt, grad_x)
        re_head = head_proj(edges, costs, grad_x, g, k, budget=budget,
                            delta=1. / 169., err_tol=1e-6, max_iter=50,
                            root=-1, pruning='strong', epsilon=1e-6, verbose=0)
        re_nodes, re_edges, p_x = re_head
        gamma_x, gamma_y = set(re_nodes), identify_direction(grad_y, 2 * s)
        supp_x = set([ind for ind, _ in enumerate(xt) if _ != 0.0])
        supp_y = set([ind for ind, _ in enumerate(yt) if _ != 0.0])
        omega_x, omega_y = gamma_x.union(supp_x), gamma_y.union(supp_y)
        bx, by = func.get_argmin_f_xy(xt, yt, omega_x, omega_y)
        re_tail = tail_proj(edges, costs, bx, 1, k,
                            root=-1, max_iter=50, budget=budget, nu=2.5)
        re_nodes, re_edges, p_x = re_tail
        psi_x, psi_y = re_nodes, identify_direction(by, s)
        x_pre, y_pre = xt, yt
        xt, yt = np.zeros_like(xt), np.zeros_like(yt)
        xt[psi_x], yt[psi_y] = bx[psi_x], by[psi_y]
        gap_x, gap_y = np.linalg.norm(xt - x_pre), np.linalg.norm(yt - y_pre)
        if (gap_x < 1e-3 and gap_y < 1e-3) or max_iter > t:
            break
    return xt, yt


def main():
    root_path = '/network/rit/lab/ceashpc/' \
                'bz383376/data/icdm17/CrimesOfChicago/'
    for test_case in range(52):
        for event_type in ['BATTERY', 'BURGLARY']:
            file_name = root_path + 'chicago_%s_case_%d.pkl' % \
                        (event_type, test_case)
            chicago_data = cPickle.load(bz2.BZ2File(file_name))
            lambda_, max_iter = 10., 5
            import networkx as nx
            G = nx.Graph()
            for edge in chicago_data['edges']:
                G.add_edge(edge[0], edge[1])
            print(len(nx.connected_components(G)))
            true_x = np.zeros(chicago_data['n'])
            true_y = np.zeros(chicago_data['p'])
            true_x[chicago_data['true_sub_graph']] = 1.
            true_y[chicago_data['true_sub_feature']] = 1.
            func = FuncEMS(chicago_data['data_matrix'], lambda_)
            true_val = func.get_fun_val(true_x, true_y)
            print('true value: %.4f' % true_val)
            para = (len(chicago_data['true_sub_graph']) / 2,
                    len(chicago_data['true_sub_feature']),
                    chicago_data['data_matrix'],
                    chicago_data['edges'],
                    chicago_data['costs'], max_iter, lambda_)
            sg_pursuit_algo(para)
            pass


if __name__ == '__main__':
    main()
