# -*- coding: utf-8 -*-
import random
import numpy as np

__all__ = ['simu_graph', 'draw_graph', 'node_pre_rec_fm',
           'minimal_spanning_tree', 'random_walk']


def simu_graph(num_nodes, rand=False, graph_type='grid'):
    """
    To generate a grid graph. Each node has 4-neighbors.
    :param num_nodes: number of nodes in the graph.
    :param rand: if rand True, then generate random weights in (0., 1.)
    :param graph_type: ['grid', 'chain']
    :return: edges and corresponding to unite weights.
    """
    edges, weights = [], []
    if graph_type == 'grid':
        length = int(np.sqrt(num_nodes))
        width, index = length, 0
        for i in range(length):
            for j in range(width):
                if (index % length) != (length - 1):
                    edges.append((index, index + 1))
                    if index + length < int(width * length):
                        edges.append((index, index + length))
                else:
                    if index + length < int(width * length):
                        edges.append((index, index + length))
                index += 1
        edges = np.asarray(edges, dtype=int)
    elif graph_type == 'chain':
        for i in range(num_nodes - 1):
            edges.append((i, i + 1))
    else:
        edges = []

    # generate weights of the graph
    if rand:
        weights = []
        while len(weights) < len(edges):
            rand_x = np.random.random()
            if rand_x > 0.:
                weights.append(rand_x)
        weights = np.asarray(weights, dtype=np.float64)
    else:
        weights = np.ones(len(edges), dtype=np.float64)
    return edges, weights


def random_walk(k, edges, restart=-1.0, start_node=None):
    """
    random walk method to generate a subgraph.
    :param k: the number of nodes in true subgraph
    :param edges: the list of edges.
    :param start_node:
    :param restart: with a fix probability to restart.
    :return: a true subgraph. a list of edges.
    """
    adj = dict()
    for edge in edges:
        if edge[0] not in adj:
            adj[edge[0]] = set()
        adj[edge[0]].add(edge[1])
        if edge[1] not in adj:
            adj[edge[1]] = set()
        adj[edge[1]].add(edge[0])
    subgraph_nodes = set()
    subgraph_edges = set()
    if start_node is None:  # random select an initial node.
        start_node = list(adj.keys())[random.randrange(0, len(adj))]
    next_node = start_node
    subgraph_nodes.add(start_node)
    if k == 1:
        return subgraph_nodes, subgraph_edges
    while True:
        next_neighbor = list(adj[next_node])
        rand_nei = next_neighbor[random.randrange(0, len(next_neighbor))]
        subgraph_nodes.add(rand_nei)
        edge_1, edge_2 = (next_node, rand_nei), (rand_nei, next_node)
        if edge_1 not in subgraph_edges and edge_2 not in subgraph_edges:
            subgraph_edges.add(edge_1)
            subgraph_edges.add(edge_2)
        next_node = rand_nei  # go to next node
        if len(subgraph_nodes) >= k:  # get a connected k-subgraph
            break
        if random.random() < restart:
            next_node = start_node
    return subgraph_nodes, subgraph_edges


def draw_graph(sub_graph, edges, length, width):
    """
    To draw a grid graph.
    :param sub_graph:
    :param edges:
    :param length:
    :param width:
    :return:
    """
    import networkx as nx
    from pylab import rcParams
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import subplots_adjust
    subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    rcParams['figure.figsize'] = 14, 14

    G = nx.Graph()
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    pos = dict()
    index = 0
    for i in range(length):
        for j in range(width):
            G.add_node(index)
            pos[index] = (j, length - i)
            index += 1
    nx.draw_networkx_nodes(G, pos, node_size=100,
                           nodelist=range(33 * 33), node_color='gray')
    nx.draw_networkx_nodes(G, pos, node_size=100,
                           nodelist=sub_graph, node_color='b')
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)
    plt.axis('off')
    plt.show()


def node_pre_rec_fm(true_feature, pred_feature):
    """
    Return the precision, recall and f-measure.
    :param true_feature:
    :param pred_feature:
    :return: pre, rec and fm
    """
    true_feature, pred_feature = set(true_feature), set(pred_feature)
    pre, rec, fm = 0.0, 0.0, 0.0
    if len(pred_feature) != 0:
        pre = len(true_feature & pred_feature) / float(len(pred_feature))
    if len(true_feature) != 0:
        rec = len(true_feature & pred_feature) / float(len(true_feature))
    if (pre + rec) > 0.:
        fm = (2. * pre * rec) / (pre + rec)
    return pre, rec, fm


def minimal_spanning_tree(edges, weights, num_nodes):
    """
    Find the minimal spanning tree of a graph.
    :param edges: ndarray dim=(m,2) -- edges of the graph.
    :param weights: ndarray dim=(m,)  -- weights of the graph.
    :param num_nodes: int, number of nodes in the graph.
    :return: (the edge indices of the spanning tree)
    """
    try:
        from proj_module import mst
    except ImportError:
        print('cannot find this functions: proj_pcst')
        exit(0)
    select_indices = mst(edges, weights, num_nodes)
    return select_indices
