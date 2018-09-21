# -*- coding: utf-8 -*-
import os
import bz2
import sys
import time
import cPickle
import numpy as np
import multiprocessing
from itertools import product
from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

root_path = '../input/data_chicago/'


def node_pre_rec_fm(true_nodes, pred_nodes):
    """ Return the precision, recall and f-measure.
    :param true_nodes:
    :param pred_nodes:
    :return: precision, recall and f-measure """
    true_nodes, pred_nodes = set(true_nodes), set(pred_nodes)
    pre, rec, fm = 0.0, 0.0, 0.0
    if len(pred_nodes) != 0:
        pre = len(true_nodes & pred_nodes) / float(len(pred_nodes))
    if len(true_nodes) != 0:
        rec = len(true_nodes & pred_nodes) / float(len(true_nodes))
    if (pre + rec) > 0.:
        fm = (2. * pre * rec) / (pre + rec)
    return [pre, rec, fm]


# elevated mean scan statistic
class FuncEMS(object):
    def __init__(self, data_matrix=None, lambda_=10.):
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
        result_indices, result_scores = [], np.zeros(self.p)
        for j in range(self.p):
            # descending order
            sorted_indices = np.argsort(self.data_matrix.T[j])[-k:]
            x, y = np.zeros(self.n), np.zeros(self.p)
            x[sorted_indices], y[j] = 1., 1.
            func_val = self.get_fun_val(x, y)
            result_scores[j] = -func_val
            result_indices.append(sorted_indices)
        func_val = -1.
        indices = np.argsort(result_scores)[::-1]
        for j in range(s):
            x1, y1 = np.zeros(self.n), np.zeros(self.p)
            y1[[indices[_] for _ in range(j + 1)]] = 1.
            x1[result_indices[indices[0]]] = 1.
            func_val1 = self.get_fun_val(x1, y1)
            if func_val == -1. or func_val > func_val1:
                y = [indices[_] for _ in range(j + 1)]
                x = result_indices[indices[0]]
                func_val = func_val1
        for j in range(s):
            x1, y1 = np.zeros(self.n), np.zeros(self.p)
            x1[result_indices[indices[j]]], y1[indices[j]] = 1., 1.
            func_val1 = self.get_fun_val(x1, y1)
            if func_val1 == -1 or func_val > func_val1:
                y = [indices[j]]
                x = result_indices[indices[j]]
                func_val = func_val1
        x0[x], y0[y] = 1., 1.
        return x0, y0

    def get_argmin_f_xy(self, x0, y0, omega_x, omega_y):
        lr, max_iter = 0.01, 1000
        xt, yt = np.copy(x0), np.copy(y0)
        indicator_x, indicator_y = np.zeros_like(x0), np.zeros_like(y0)
        indicator_x[list(omega_x)] = 1.
        indicator_y[list(omega_y)] = 1.
        for i in range(max_iter):
            loss, grad_x, grad_y = self.get_loss_grad(xt, yt)
            x_pre, y_pre = np.copy(xt), np.copy(yt)
            xt = FuncEMS.update_minimizer(grad_x, indicator_x, xt, 5, lr)
            yt = FuncEMS.update_minimizer(grad_y, indicator_y, yt, 5, lr)
            diff_norm_x = np.linalg.norm(xt - x_pre)
            diff_norm_y = np.linalg.norm(yt - y_pre)
            if diff_norm_x <= 1e-6 and diff_norm_y <= 1e-6:
                break
        return xt, yt

    def get_loss_grad(self, x, y):
        sum_x, sum_y = np.sum(x), np.sum(y)
        if sum_x == 0.0 or sum_y == 0.0:
            print('gradient_x: input x vector values are all zeros !!!')
            exit(0)
        w_y = np.dot(self.data_matrix, y)
        xt_w_y = np.dot(x, w_y)
        reg = .5 * (self.lambda_ * (sum_y ** 2.))
        func_val = - xt_w_y / np.sqrt(sum_x) + reg
        term2 = .5 * xt_w_y / (np.sqrt(sum_x) * sum_x)
        grad_x = -(1. / np.sqrt(sum_x)) * w_y + term2
        x_w = np.dot(self.data_matrix.T, x)
        grad_y = -1. / np.sqrt(sum_x) * x_w + self.lambda_ * y
        if np.isnan(grad_x).any() or np.isnan(grad_y).any():
            print('something is wrong. gradient x or y')
        return func_val, grad_x, grad_y

    @staticmethod
    def update_minimizer(grad_x, indicator_x, x, bound, step_size):
        normalized_x = (x - step_size * grad_x) * indicator_x
        sorted_indices = np.argsort(normalized_x)
        num_non_posi = len(np.where(normalized_x <= 0.0))
        normalized_x[normalized_x <= 0.0] = 0.
        normalized_x[normalized_x > 1.] = 1.
        if num_non_posi == len(x):
            print('siga-1 is too large and all values'
                  ' in the gradient are non-positive.')
            for i in range(bound):
                normalized_x[sorted_indices[i]] = 1.
        return normalized_x


def data_convert():
    for test_case, event_type in product(range(52), ['BATTERY', 'BURGLARY']):
        data = {'n': 0,
                'p': 0,
                'test_case': test_case,
                'data_matrix': [],
                'edges': [],
                'costs': [],
                'event_type': event_type,
                'true_sub_graph': [],
                'true_sub_feature': [],
                'true_x': None,
                'true_y': None}
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
                    data['costs'].append(float(each_line.split(' ')[2]))
                elif ind == m + n + 2:
                    sub_graph = [int(_) for _ in each_line.split(' ')]
                    data['true_sub_graph'] = sub_graph
                    data['true_x'] = np.zeros(n)
                    data['true_x'][sub_graph] = 1.
                elif ind == m + n + 3:
                    sub_feature = [int(_) for _ in each_line.split(' ')]
                    data['true_sub_feature'] = sub_feature
                    data['true_y'] = np.zeros(p)
                    data['true_y'][sub_feature] = 1.
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
    k, s, data_matrix, edges, costs, max_iter, true_nodes, true_features = para
    func = FuncEMS(data_matrix=data_matrix)
    xt, yt = func.get_initial_point(k, s)
    print(np.nonzero(xt)[0])
    print(np.nonzero(yt)[0])
    print(np.linalg.norm(xt), np.linalg.norm(yt))
    print(node_pre_rec_fm(true_nodes=true_nodes, pred_nodes=np.nonzero(xt)[0]))
    print(node_pre_rec_fm(true_nodes=true_features, pred_nodes=np.nonzero(yt)[0]))
    run_time_head_tail = 0.0
    for tt in range(max_iter):
        iter_time = time.time()
        print(tt)
        func_val, grad_x, grad_y = func.get_loss_grad(xt, yt)
        grad_x = normalize_gradient(xt, grad_x)
        grad_y = normalize_gradient(yt, grad_y)
        start_time = time.time()
        re_head = head_proj(
            edges=edges, weights=costs, x=grad_x, g=1, s=k, budget=k - 1.,
            delta=1. / 169., max_iter=50, err_tol=1e-6, root=-1,
            pruning='strong', epsilon=1e-6, verbose=0)
        re_nodes, re_edges, p_x = re_head
        run_time_head_tail += time.time() - start_time
        gamma_x, gamma_y = set(re_nodes), identify_direction(grad_y, 2 * s)
        supp_x = set([ind for ind, _ in enumerate(xt) if _ != 0.0])
        supp_y = set([ind for ind, _ in enumerate(yt) if _ != 0.0])
        omega_x, omega_y = gamma_x.union(supp_x), gamma_y.union(supp_y)
        bx, by = func.get_argmin_f_xy(xt, yt, omega_x, omega_y)
        start_time = time.time()
        re_tail = tail_proj(
            edges=edges, weights=costs, x=bx, g=1, s=k, budget=k - 1., nu=2.5,
            max_iter=50, err_tol=1e-6, root=-1, pruning='strong', verbose=0,
            epsilon=1e-6)
        re_nodes, re_edges, p_x = re_tail
        run_time_head_tail += time.time() - start_time
        psi_x, psi_y = re_nodes, identify_direction(by, s)
        x_pre, y_pre = xt, yt
        xt, yt = np.zeros_like(xt), np.zeros_like(yt)
        xt[list(psi_x)], yt[list(psi_y)] = bx[list(psi_x)], by[list(psi_y)]
        gap_x, gap_y = np.linalg.norm(xt - x_pre), np.linalg.norm(yt - y_pre)
        if gap_x < 1e-3 and gap_y < 1e-3:
            break
        print('iteration time: %.4f, head_tail time: %.4f' %
              (time.time() - iter_time, run_time_head_tail))
    return xt, yt


def run_single_process(para):
    event_type, test_case = para
    file_name = 'chicago_%s_case_%d.pkl' % (event_type, test_case)
    chicago_data = cPickle.load(bz2.BZ2File(root_path + file_name))
    true_x = np.zeros(chicago_data['n'])
    true_y = np.zeros(chicago_data['p'])
    true_x[chicago_data['true_sub_graph']] = 1.
    true_y[chicago_data['true_sub_feature']] = 1.
    func = FuncEMS(chicago_data['data_matrix'])
    true_val = func.get_fun_val(true_x, true_y)
    print('true value: %.4f' % true_val)
    para = (len(chicago_data['true_sub_graph']) / 2,
            len(chicago_data['true_sub_feature']),
            chicago_data['data_matrix'],
            chicago_data['edges'],
            chicago_data['costs'], 5,
            chicago_data['true_sub_graph'],
            chicago_data['true_sub_feature'])
    xt, yt = sg_pursuit_algo(para)
    n_pre_rec_fm = node_pre_rec_fm(
        true_nodes=chicago_data['true_sub_graph'],
        pred_nodes=np.nonzero(xt)[0])
    f_pre_rec_fm = node_pre_rec_fm(
        true_nodes=chicago_data['true_sub_feature'],
        pred_nodes=np.nonzero(yt)[0])
    print(n_pre_rec_fm)
    print(f_pre_rec_fm)
    return n_pre_rec_fm, f_pre_rec_fm


def main():
    num_cpu = int(sys.argv[1])
    input_paras = [_ for _ in product(['BATTERY', 'BURGLARY'], range(52))]
    pool = multiprocessing.Pool(processes=num_cpu)
    results_pool = pool.map(run_single_process, input_paras)
    pool.close()
    pool.join()
    cPickle.dump(results_pool, open('../output/output_chicago.pkl', 'wb'))
    for result in results_pool:
        pre, rec, fm = result[0]
        print('node_pre_rec_fm: %.4f %.4f %.4f' % (pre, rec, fm))
        pre, rec, fm = result[1]
        print('feature_pre_rec_fm: %.4f %.4f %.4f' % (pre, rec, fm))


if __name__ == '__main__':
    main()
