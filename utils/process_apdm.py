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

read_APDM_data()