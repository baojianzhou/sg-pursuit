import bz2
import cPickle,datetime
import pandas as pd
import numpy as np

mappings = {} # contains mapped userIds as the code requires them to be continuous
def main(path_pickle,path_csv):
    '''
    root_path = 'input/data_chicago/'
    file_name = 'chicago_BATTERY_case_0.pkl'
    chicago_data = cPickle.load(bz2.BZ2File(root_path + file_name))
    print chicago_data.keys()
    '''

    #'''
    edges, data_matrix, cost, spammers = get_edges()

    spammer_dict = {
        'costs': cost,
        'data_matrix': data_matrix,
        'edges':edges,
        'event_type':'SPAM',
        'm':len(edges),
        'n':len(data_matrix), # cross check
        #'p':56,
        'p':64,
        'true_sub_feature':range(0,63 + 1),
        'true_sub_graph':spammers,
        'test_case':{}
    }
    print 'creating pkl.. ',datetime.time()
    output = open('spam.pkl', 'wb')
    cPickle.dump(spammer_dict, output)
    output.close()

    #'''

def get_edges():
    edges = []
    unique_users = []
    data_matrix = []
    cost = []
    mappings = {}
    # get all spammers
    print 'getting spammers'
    spammers = []
    users_list = pd.read_csv('usersdata.csv', delimiter='\t', header=None)
    user_ids = []
    index = 0
    for i in users_list.index[:20000]:
        mappings[users_list.at[i, 0]] = index
        user_ids.append(users_list.at[i, 0])
        # print 'spammers ', datetime.time()
        if users_list.at[i, 4] == 1:
            #print users_list.at[i, 0]
            spammers.append(index)
            index += 1

    #edge_chunk = pd.read_csv('relations.csv',delimiter='\t', header=None)
    edge_chunk = pd.read_csv('xaa',delimiter='\t', header=None)

    print 'getting edges'
    for i in edge_chunk.index:
        # print 'edges ',datetime.time()
        #edges.append([edge_chunk.at[i, 2], edge_chunk.at[i, 3]])
        if edge_chunk.at[i, 2] in mappings and edge_chunk.at[i, 3] in mappings:
            edges.append([mappings[edge_chunk.at[i, 2]], mappings[edge_chunk.at[i, 3]]])

    #bigrams = pd.read_csv('sequence_bigram_features.csv')
    bigrams = pd.read_csv('bigram_1.csv')
    bigrams = bigrams[bigrams['userId'].isin(user_ids)] #.isin(users_list[0])


    #data_matrix = bigrams[bigrams['userId'].isin(unique_users)]

    print 'getting bigrams'
    index = 0
    for i in bigrams.index:
        index += 1
        temp = bigrams.loc[[i]].drop(columns = ['userId'])
        data_matrix.append(temp.values.tolist()[0])
        #print temp.values.tolist()[0]
        #print 'bigram -- no user id should be there ',bigrams.loc[[i]][1:]
        cost.append(1.0)

    edges = np.array(edges)
    data_matrix = np.array(data_matrix)
    cost = np.array(cost)
    print len(edges), len(data_matrix), len(data_matrix[0])
    return edges,data_matrix,cost,spammers



if __name__ == '__main__':
    main('input/data_chicago/chicago_BATTERY_case_0.pkl','./')