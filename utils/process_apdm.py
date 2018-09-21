from collections import defaultdict
import numpy as np
import cPickle
import bz2
def read_APDM_data(path):
    data = []
    nodes = []
    adjList = defaultdict(list)
    trueNodes = []
    trueFea = []
    lines = open(path).readlines()
    n = -1
    for idx, line in enumerate(lines):
        if line.strip().startswith('truth_features'):
            trueFea = map(int, str(line.strip().split('=')[1]).split())
            n = idx + 5
            break

    for idx in range(n, len(lines)):
        # print lines[idx]
        line = lines[idx]
        if line.find('END') >= 0:
            n = idx + 4
            break
        else:
            items = line.split(' ')
            nodes.append(int(items[0]))
            data.append(map(float, items[1:]))

    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            n = idx + 4
            break
        else:
            vertices = line.split(' ')
            edge = [0] * 2
            edge[0] = int(vertices[0])
            edge[1] = int(vertices[1])

            if edge[1] not in adjList[edge[0]]:
                adjList[edge[0]].append(edge[1])
            if edge[0] not in adjList[edge[1]]:
                adjList[edge[1]].append(edge[0])

    true_subgraph = []
    for idx in range(n, len(lines)):
        line = lines[idx]
        if line.find('END') >= 0:
            break
        else:
            items = line.split(' ')
            true_subgraph.append(int(items[0]))
            true_subgraph.append(int(items[1]))
    true_subgraph = sorted(list(set(true_subgraph)))
    return nodes, dict(adjList), data, true_subgraph, trueFea

def varying_attributes():
    root="/home/apdm05/workspace/data/sg-pursuit/DenseSubgraph_APDM/VaryingNumOfAttributes/"
    #VaryingNumOfAttributes_APDM_Dense_subgraph_in_0.35_out_0.1_numClusters_10_TrueSGSize_30_FeasNum_20_trueFeasNum_5_sigmas1_0.0316_case_0.txt
    for num_feat in [20,40,80,100][:]:
        datas={}
        for case in range(50)[:]:
            filename="VaryingNumOfAttributes_APDM_Dense_subgraph_in_0.35_out_0.1_numClusters_10_TrueSGSize_30_FeasNum_%d_trueFeasNum_10_sigmas1_0.0316_case_%d.txt"%(num_feat,case)
            print filename
            data = {'test_case': case, 'data_matrix': [], 'edges': [],
                    'costs': [], 'event_type': "VaryingAtt_in_0.35_out_0.1_numClusters_10_TrueSGSize_30_trueFeasNum_10_sigmas1_0.0316",
                    'true_sub_graph': [], 'true_sub_feature': []}
            lines = open(root+filename).readlines()
            n = -1
            for idx, line in enumerate(lines):
                if line.strip().startswith('numNodes'):
                    data['n']=int(str(line.strip().split('=')[1]))
                    continue
                if line.strip().startswith('numOfFeatures'):
                    data['p']=int(str(line.strip().split('=')[1]))
                    continue
                if line.strip().startswith('truth_features'):
                    data['true_sub_feature'] = map(int, str(line.strip().split('=')[1]).split())
                    n = idx + 5
                    break

            for idx in range(n, len(lines)):
                # print lines[idx]
                line = lines[idx]
                if line.find('END') >= 0:
                    n = idx + 4
                    break
                else:
                    items = line.split(' ')
                    data['data_matrix'].append(map(float, items[1:]))

            for idx in range(n, len(lines)):
                line = lines[idx]
                if line.find('END') >= 0:
                    n = idx + 4
                    break
                else:
                    vertices = line.split(' ')
                    data['edges'].append([int(vertices[0]),int(vertices[1])])
                    data['costs'].append(float(vertices[2]))

            true_subgraph = []
            for idx in range(n, len(lines)):
                line = lines[idx]
                if line.find('END') >= 0:
                    break
                else:
                    items = line.split(' ')
                    data['true_sub_graph'].append(int(items[0]))
                    data['true_sub_graph'].append(int(items[1]))

            data['true_sub_graph'] = sorted(list(set(data['true_sub_graph'])))
            data['data_matrix'] = np.asarray(data['data_matrix'],
                                             dtype=np.float64)
            data['edges'] = np.asarray(data['edges'], dtype=int)
            data['costs'] = np.asarray(data['costs'],
                                       dtype=np.float64)
            datas[case]=data
        file_name = "/home/apdm05/workspace/data/sg-pursuit/DenseSubgraphPkls/VaryingAttribute_numAtt_%d.pkl"%(num_feat)
        bz2_f = bz2.BZ2File(file_name, 'wb')
        cPickle.dump(datas, bz2_f)

def varying_NumClusters():
    root="/home/apdm05/workspace/data/sg-pursuit/DenseSubgraph_APDM/VaryingNumOfClusters/"
    #VaryingNumOfAttributes_APDM_Dense_subgraph_in_0.35_out_0.1_numClusters_10_TrueSGSize_30_FeasNum_20_trueFeasNum_5_sigmas1_0.0316_case_0.txt
    for num_cluster in [10,12,14,15,20,25][:]:
        datas={}
        for case in range(50)[:]:
            filename="VaryingNumOfClusters_APDM_Dense_subgraph_in_0.35_out_0.1_numClusters_%d_TrueSGSize_30_FeasNum_100_trueFeasNum_10_sigmas1_0.0316_case_%d.txt"%(num_cluster,case)
            print filename
            data = {'test_case': case, 'data_matrix': [], 'edges': [],
                    'costs': [], 'event_type': "VaryingNumOfClusters_in_0.35_out_0.1_TrueSGSize_30_FeasNum_100_trueFeasNum_10_sigmas1_0.0316",
                    'true_sub_graph': [], 'true_sub_feature': []}
            lines = open(root+filename).readlines()
            n = -1
            for idx, line in enumerate(lines):
                if line.strip().startswith('numNodes'):
                    data['n']=int(str(line.strip().split('=')[1]))
                    continue
                if line.strip().startswith('numOfFeatures'):
                    data['p']=int(str(line.strip().split('=')[1]))
                    continue
                if line.strip().startswith('truth_features'):
                    data['true_sub_feature'] = map(int, str(line.strip().split('=')[1]).split())
                    n = idx + 5
                    break

            for idx in range(n, len(lines)):
                # print lines[idx]
                line = lines[idx]
                if line.find('END') >= 0:
                    n = idx + 4
                    break
                else:
                    items = line.split(' ')
                    data['data_matrix'].append(map(float, items[1:]))

            for idx in range(n, len(lines)):
                line = lines[idx]
                if line.find('END') >= 0:
                    n = idx + 4
                    break
                else:
                    vertices = line.split(' ')
                    data['edges'].append([int(vertices[0]),int(vertices[1])])
                    data['costs'].append(float(vertices[2]))

            true_subgraph = []
            for idx in range(n, len(lines)):
                line = lines[idx]
                if line.find('END') >= 0:
                    break
                else:
                    items = line.split(' ')
                    data['true_sub_graph'].append(int(items[0]))
                    data['true_sub_graph'].append(int(items[1]))

            data['true_sub_graph'] = sorted(list(set(data['true_sub_graph'])))
            data['data_matrix'] = np.asarray(data['data_matrix'],
                                             dtype=np.float64)
            data['edges'] = np.asarray(data['edges'], dtype=int)
            data['costs'] = np.asarray(data['costs'],
                                       dtype=np.float64)
            datas[case]=data
        file_name = "/home/apdm05/workspace/data/sg-pursuit/DenseSubgraphPkls/VaryingNumClusters_numCluster_%d.pkl"%(num_cluster)
        bz2_f = bz2.BZ2File(file_name, 'wb')
        cPickle.dump(datas, bz2_f)

def varying_ClusterSize():
    root="/home/apdm05/workspace/data/sg-pursuit/DenseSubgraph_APDM/VaryingClusterSizes/"
    #VaryingNumOfAttributes_APDM_Dense_subgraph_in_0.35_out_0.1_numClusters_10_TrueSGSize_30_FeasNum_20_trueFeasNum_5_sigmas1_0.0316_case_0.txt
    for cluster_sizes in [(30,100),(30,150),(30,200),(30,300),(30,400)][:]:
        datas={}
        divv=0.0
        for case in range(50)[:]:
            filename="VaryingClusterSizes_APDM_Dense_subgraph_in_0.35_out_0.1_numClusters_10_clusterSizeLower_%d_clusterSizeUpper_%d_FeasNum_100_trueFeasNum_10_sigmas1_0.0316_case_%d.txt"%(cluster_sizes[0],cluster_sizes[1],case)
            print filename
            data = {'test_case': case, 'data_matrix': [], 'edges': [],
                    'costs': [], 'event_type': "VaryingClusterSizes_in_0.35_out_0.1_numClusters_10_FeasNum_100_trueFeasNum_10_sigmas1_0.0316.txt",
                    'true_sub_graph': [], 'true_sub_feature': []}
            try:
                lines = open(root+filename).readlines()
            except:
                print("File not exists",filename)
                continue
            n = -1
            for idx, line in enumerate(lines):
                if line.strip().startswith('numNodes'):
                    data['n']=int(str(line.strip().split('=')[1]))
                    continue
                if line.strip().startswith('numOfFeatures'):
                    data['p']=int(str(line.strip().split('=')[1]))
                    continue
                if line.strip().startswith('truth_features'):
                    data['true_sub_feature'] = map(int, str(line.strip().split('=')[1]).split())
                    n = idx + 5
                    break

            for idx in range(n, len(lines)):
                # print lines[idx]
                line = lines[idx]
                if line.find('END') >= 0:
                    n = idx + 4
                    break
                else:
                    items = line.split(' ')
                    data['data_matrix'].append(map(float, items[1:]))

            for idx in range(n, len(lines)):
                line = lines[idx]
                if line.find('END') >= 0:
                    n = idx + 4
                    break
                else:
                    vertices = line.split(' ')
                    data['edges'].append([int(vertices[0]),int(vertices[1])])
                    data['costs'].append(float(vertices[2]))

            true_subgraph = []
            for idx in range(n, len(lines)):
                line = lines[idx]
                if line.find('END') >= 0:
                    break
                else:
                    items = line.split(' ')
                    data['true_sub_graph'].append(int(items[0]))
                    data['true_sub_graph'].append(int(items[1]))

            data['true_sub_graph'] = sorted(list(set(data['true_sub_graph'])))
            data['data_matrix'] = np.asarray(data['data_matrix'],
                                             dtype=np.float64)
            data['edges'] = np.asarray(data['edges'], dtype=int)
            data['costs'] = np.asarray(data['costs'],
                                       dtype=np.float64)

            file_name = "/home/apdm05/workspace/data/sg-pursuit/DenseSubgraphPkls/VaryingClusterSizes_numCluster_%d_%d_case-%d.pkl"%(cluster_sizes[0],cluster_sizes[1],case)
            bz2_f = bz2.BZ2File(file_name, 'wb')
            cPickle.dump(data, bz2_f)


# varying_attributes()
# varying_NumClusters()
varying_ClusterSize()
root="/home/apdm05/workspace/data/sg-pursuit/DenseSubgraph_APDM/VaryingNumOfAttributes/"
nodes, adjList, data, true_subgraph, trueFea=read_APDM_data(root+"VaryingNumOfAttributes_APDM_Dense_subgraph_in_0.35_out_0.1_numClusters_10_TrueSGSize_30_FeasNum_20_trueFeasNum_10_sigmas1_0.0316_case_0.txt")

print nodes
print data
print true_subgraph, trueFea