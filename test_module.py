#!/usr/bin/python
import time
import numpy as np
from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj
from sparse_learning.fast_pcst import fast_pcst
from sparse_learning.graph_utils import simu_graph
from sparse_learning.graph_utils import minimal_spanning_tree
from sparse_learning.proj_algo import HeadTailWrapper
from sparse_learning.ghtp_algo import ghtp_logistic_py
from sparse_learning.ghtp_algo import graph_ghtp_logistic_py


def test_proj_algo():
    print('-' * 100)
    edges, weights = simu_graph(25)  # get grid graph
    sub_graph = [6, 7, 8, 9]
    x = np.random.normal(0.0, 0.1, 25)
    x[sub_graph] = 5.
    n, m = len(weights), edges.shape[1]
    re = head_proj(edges=edges, weights=weights, x=x, g=1, s=4, budget=3.,
                   delta=1. / 169., err_tol=1e-6, max_iter=30, root=-1,
                   pruning='strong', epsilon=1e-6, verbose=0)
    re_nodes, re_edges, p_x = re
    print('test1 result head nodes: ', re_nodes)
    print('test1 result head edges: ', re_edges)
    re = head_proj(edges=edges, weights=weights, x=np.zeros(n), g=1, s=4,
                   budget=3., delta=1. / 169., err_tol=1e-6, max_iter=30,
                   root=-1, pruning='strong', epsilon=1e-6, verbose=0)
    re_nodes, re_edges, p_x = re
    print('test2 result head nodes: ', re_nodes)
    print('test2 result head edges: ', re_edges)
    re = tail_proj(edges=edges, weights=weights, x=x, g=1, s=4, root=-1,
                   max_iter=20, budget=3., nu=2.5)
    re_nodes, re_edges, p_x = re
    print('test3 result tail nodes: ', re_nodes)
    print('test3 result tail edges: ', re_nodes)
    re = tail_proj(edges=edges, weights=weights, x=np.zeros(n), g=1, s=4,
                   root=-1, max_iter=20, budget=3., nu=2.5)
    re_nodes, re_edges, p_x = re
    print('test4 result tail nodes: ', re_nodes)
    print('test4 result tail edges: ', re_nodes)
    wrapper = HeadTailWrapper(edges=edges, weights=weights)
    re = wrapper.run_head(x=x, g=1, s=4, budget=3., delta=1. / 169.)
    re_nodes, re_edges, p_x = re
    print('test5 result head nodes: ', re_nodes)
    print('test5 result head edges: ', re_nodes)
    re = wrapper.run_tail(x=x, g=1, s=4, budget=3, nu=2.5)
    re_nodes, re_edges, p_x = re
    print('test6 result tail nodes: ', re_nodes)
    print('test6 result tail edges: ', re_nodes)


def test_fast_pcst():
    print('-' * 100)
    edges, weights = simu_graph(25)  # get grid graph
    n, m = len(weights), edges.shape[1]
    x = np.random.normal(0.0, 0.1, 25)
    sub_graph = [6, 7, 8, 9]
    x[sub_graph] = 5.
    # edges, prizes, weights, root, g, pruning, epsilon, verbose
    re = fast_pcst(edges=edges, prizes=x ** 2., weights=weights, root=-1, g=1,
                   pruning='gw', epsilon=1e-6, verbose=0)
    re_nodes, re_edges = re
    print('test7 result pcst nodes: ', re_nodes)
    print('test7 result pcst edges: ', re_nodes)
    re = fast_pcst(edges=edges, prizes=np.zeros(n), weights=weights, root=-1,
                   g=1, pruning='gw', epsilon=1e-6, verbose=0)
    re_nodes, re_edges = re
    print('test8 result pcst nodes: ', re_nodes)
    print('test8 result pcst edges: ', re_nodes)
    re = fast_pcst(edges=edges, prizes=x ** 2., weights=weights, root=-1, g=1,
                   pruning='strong', epsilon=1e-6, verbose=0)
    re_nodes, re_edges = re
    print('test9 result pcst nodes: ', re_nodes)
    print('test9 result pcst edges: ', re_nodes)


def test_mst():
    print('-' * 100)
    edges, weights = simu_graph(25, rand=True)  # get grid graph
    start_time = time.time()
    selected_indices = minimal_spanning_tree(edges=edges, weights=weights, num_nodes=25)
    print('run time:', (time.time() - start_time))
    for index in selected_indices:
        print(index, weights[index])
    selected_edges = {(i, j): None for (i, j) in edges[selected_indices]}
    import networkx as nx
    from pylab import rcParams
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import subplots_adjust
    subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    rcParams['figure.figsize'] = 14, 14
    G = nx.Graph()
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    pos, edge_posi = dict(), dict()
    length, width, index = 5, 5, 0
    for i in range(length):
        for j in range(width):
            G.add_node(index)
            pos[index] = (j, length - i)
            if (j, length - i) in selected_edges or (length - i, j) in selected_edges:
                edge_posi[index] = (j, length - i)
            index += 1
    nx.draw_networkx_nodes(G, pos, node_size=100, nodelist=range(length * width), node_color='gray')
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=2, edge_color='r')
    plt.axis('off')
    plt.show()


def test_mst_performance():
    edges, weights = simu_graph(1000000, rand=True)  # get grid graph
    x = np.random.normal(0.0, 0.1, 1000000)
    sub_graph = range(10000, 11000)
    x[sub_graph] = 50.
    start_time = time.time()
    re = fast_pcst(edges=edges, prizes=x ** 2., weights=weights, root=-1, g=1,
                   pruning='strong', epsilon=1e-6, verbose=0)
    re_nodes1, re_edges1 = re
    print('run time of original pcst: ', (time.time() - start_time))
    start_time = time.time()
    selected_indices = minimal_spanning_tree(edges=edges, weights=weights, num_nodes=1000000)
    print('run time of mst:', (time.time() - start_time))
    start_time = time.time()
    edges, weights = edges[selected_indices], weights[selected_indices]
    re = fast_pcst(edges=edges, prizes=x ** 2., weights=weights, root=-1, g=1,
                   pruning='strong', epsilon=1e-6, verbose=0)
    print('run time of original pcst: ', (time.time() - start_time))
    re_nodes2, re_edges2 = re
    print(len(re_nodes1))
    print(len(re_nodes2))
    print(len(set(re_nodes1).intersection(re_nodes2)))


def test_graph_ghtp():
    x_tr = np.asarray([[1., 2., 3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.]], dtype=np.float64)
    y_tr = np.asarray([1., 1., -1.], dtype=np.float64)
    w0 = np.asarray([0., 0., 0., 0., 0.])
    lr = 0.1
    sparsity = 2
    tol = 1e-6
    max_iter = 50
    eta = 1e-3
    ghtp_logistic_py(x_tr=x_tr, y_tr=y_tr, w0=w0, lr=lr, sparsity=sparsity, tol=tol, max_iter=max_iter, eta=eta)


def main():
    test_graph_ghtp()


if __name__ == '__main__':
    test_proj_algo()
