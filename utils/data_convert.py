# -*- coding: utf-8 -*-
import os
import bz2
import sys
import time
import cPickle
import numpy as np
from itertools import product

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

root_path = '../input/data_chicago/'


def data_convert_chicago_crime():
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
        data['data_matrix'] = np.asarray(data['data_matrix'], dtype=np.float64)
        data['edges'] = np.asarray(data['edges'], dtype=int)
        data['costs'] = np.asarray(data['costs'], dtype=np.float64)
        file_name = root_path + 'chicago_%s_case_%d.pkl' % \
                    (event_type, test_case)
        bz2_f = bz2.BZ2File(file_name, 'wb')
        cPickle.dump(data, bz2_f)


def data_convert_yelp():
    root_path = '/home/baojian/Yelp/'
    for test_case, event_type in product([21], ['2014_2015']):
        data = {'test_case': test_case,
                'n': 0,
                'p': 0,
                'k': 0,
                's': 0,
                'm': 0,
                'data_matrix': [],
                'nodes_dict': dict(),
                'words_dict': dict(),
                'edges': [],
                'costs': [],
                'event_type': event_type,
                'true_sub_graph': [],  # nodes
                'true_sub_feature': [],  # words
                'true_x': None,
                'true_y': None}
        print(event_type, test_case)
        with open(root_path + 'graph/processed_%s_test_case_%d.txt'
                  % (event_type, test_case)) as f:
            n, p, k, s, m = 0, 0, 0, 0, 0
            for ind, each_line in enumerate(f.readlines()):
                each_line = each_line.lstrip().rstrip()
                if ind == 0:
                    data['n'] = int(each_line.split(' ')[0])
                    data['p'] = int(each_line.split(' ')[1])
                    data['k'] = int(each_line.split(' ')[2])
                    data['s'] = int(each_line.split(' ')[3])
                    data['m'] = int(each_line.split(' ')[4])
                    n, p, m = data['n'], data['p'], data['m']
                elif 1 <= ind <= n:
                    arr = [float(_) for _ in each_line.split(' ')]
                    data['data_matrix'].append(arr)
                elif n + 1 <= ind <= 2 * n:
                    key_ = each_line.split(' ')[0]
                    data['nodes_dict'][key_] = int(each_line.split(' ')[1])
                elif 2 * n + 1 <= ind <= 2 * n + p:
                    key_ = each_line.split(' ')[0]
                    data['words_dict'][key_] = int(each_line.split(' ')[1])
                elif 2 * n + p + 1 == ind:
                    sub_graph = [int(_) for _ in each_line.split(' ')]
                    data['true_sub_graph'] = sub_graph
                    data['true_x'] = np.zeros(n)
                    data['true_x'][sub_graph] = 1.
                elif 2 * n + p + 2 == ind:
                    sub_feature = [int(_) for _ in each_line.split(' ')]
                    data['true_sub_feature'] = sub_feature
                    data['true_y'] = np.zeros(p)
                    data['true_y'][sub_feature] = 1.
                elif 2 * n + p + 3 <= ind <= 2 * n + p + 3 + m:
                    edge = [each_line.split(' ')[0],
                            each_line.split(' ')[1]]
                    data['edges'].append(edge)
                    data['costs'].append(1.0)  # use default value.
        data['data_matrix'] = np.asarray(data['data_matrix'], dtype=np.float64)
        data['edges'] = np.asarray(data['edges'], dtype=int)
        data['costs'] = np.asarray(data['costs'], dtype=np.float64)
        file_name = root_path + 'chicago_%s_case_%d.pkl' % \
                    (event_type, test_case)
        bz2_f = bz2.BZ2File(file_name, 'wb')
        cPickle.dump(data, bz2_f)


data_convert_yelp()
