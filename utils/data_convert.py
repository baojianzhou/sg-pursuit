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
