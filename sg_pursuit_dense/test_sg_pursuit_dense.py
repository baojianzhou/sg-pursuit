import cPickle,bz2
from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj
import numpy as np
import time,datetime
import copy
import multiprocessing
# from sg_pursuit_dense import *
# from utils.base_function import *

"""-----------------------------------------------------------
Basic Fucntions
--------------------------------------------------------------"""

def getSupp(x):
    return set(np.nonzero(x)[0])


""""
grad_i { = 0       1) if grad_i<0 and x_i=1
     *             2)or if grad_i>0 and x_i =0
     * = grad_i     ,otherwise

"""


def normalized_Gradient(x, gradient):
    if len(gradient) == 0 or len(x) == 0:
        print("Error: gradient/x is None...  (normalized_Gradient)")

    normalizedGradient = np.zeros_like(x)
    for i in range(len(gradient)):
        if gradient[i] < 0.0 and x[i] == 1.0:
            normalizedGradient[i] = 0.0
        elif gradient[i] > 0.0 and x[i] == 0.0:
            normalizedGradient[i] = 0.0
        else:
            normalizedGradient[i] = gradient[i]

    return normalizedGradient


"""
identify the direction, which means a subset of indices will be returned.
The goal is to maximize ||gradient_R||^2 s.t. |R| < s
"""


def identifyDirection(gradient, s):

    if s > len(gradient): s=0.5*s

    if len(gradient) == 0 or s <= 0 or s > len(gradient):
        print("Error: gradient is None...  (argMax_nabla_y_Fxy)")
    # if the vector is zero, just return an empty set
    # if np.sum(gradient): return np.array([])
    squareGradient = gradient * gradient
    indexes = squareGradient.argsort()[::-1]
    gammaY = set()
    for i in range(s):
        if i < len(indexes) and squareGradient[indexes[i]] > 0.0:
            gammaY.add(indexes[i])
    # print(type(gammaY))
    return gammaY


"""
Return the projected vector
param vector: the vector to be projected.
param sset:    the projection set.
return the projected vector.
"""


def projectionOnVector(vector, sset):
    # if vector == None or sset == None: return None
    projectedVector = np.zeros_like(vector)
    for i in range(len(vector)):
        if i in sset: projectedVector[i] = vector[i]

    return projectedVector


def getIndicateVector(S,ssize):
    if ssize<=0: return None
    x=np.zeros(ssize)
    for i in S: x[i]=1.0
    return x

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

"""
rank nodes
"""
def ranknodes(nodes,k,A):
    degrees=np.zeros(len(nodes))
    rank=np.zeros(np.min([len(nodes),k]))
    for i in range(len(nodes)):
        for j in nodes:
            if A[nodes[i]][j]>0.0:
                degrees[i]+=1.0

    indexes = degrees.argsort()[::-1]
    for i in range(len(rank)):
        rank[i]=nodes[indexes[i]]

    return rank



"""

"""
def get_fun(X,Y,W,A,n,p,lambda0=0): #(x,y,W,A,lambda0)
    x=np.zeros(n)
    y=np.zeros(p)
    # print len(y),X,Y
    for i in X:
        x[int(i)]=1.0
    for i in Y:
        y[int(i)]=1.0
    fval=PCA_getFuncValue(x,y,W,A,lambda0)#x,y,W,A,lambda0
    return fval#x,y,W,A,lambda0


def sorted_indexes(x):
    return np.array(x).argsort()[::-1]

"""
Input : A: adjacency matrix
        k: the total sparsity of x
        s: the maximum number of |y|<s

"""
def calcualte_initial_val(W,A,n,p,k,s):


    res=[]
    scores=[]
    trials=[2,3,4,5,6]

    x0=np.zeros(n)
    y0=np.zeros(p)

    for i in range(len(trials)):
        res.append([])
        scores.append([0.0 for i in range(p)])

    for j in range(p):
        for i in range(n):
            nns=list(np.nonzero(A[i])[0])
            # print nns
            dists=[0.0 for _ in range(len(nns))]
            for kk in range(len(nns)):
                dists[kk]=np.abs(W[nns[kk]][j]-W[i][j])
                # print nns[kk],i,j,W[nns[kk]][j],W[i][j]
                # time.sleep(1000)
            # print sorted(dists)

            if len(nns)>1.0:
                x0[i]=-np.percentile(dists,50)
            else:
                x0[i] = -np.mean(dists)

        indexes = x0.argsort()[::-1]

        ii=0
        for r in trials:
            S=[]
            for i in range(k*r):
                S.append(indexes[i])
            if len(S)>0:
                rank=ranknodes(S,k,A)
                # print len(S),S
                # print "rank",rank
                # time.sleep(1000)
                fval=get_fun(rank,[j],W,A,n,p,0) #X,Y,W,A,n,p,lambda0=0

                scores[ii][j]=-fval
                res[ii].append(rank)
            else:
                scores[ii][j]=-1000
                res[ii].append([])
            ii+=1

    fval=-1
    Y=[]
    X=[]
    for ii in range(len(trials)):
        indexes = sorted_indexes(scores[ii])
        Y1=[]
        for i in range(s):
            if scores[ii][indexes[i]]>-1000:
                Y1.append(indexes[i])
        # print ii,indexes[0],len(res),len(res[ii])
        X1=res[ii][indexes[0]]
        fval1=get_fun(X1,Y1,W,A,n,p,0)
        if fval==-1 or fval>fval1:
            Y=[]
            for i in Y1:
                Y.append(i)
            X=[]
            for i in X1:
                X.append(i)
            fval=fval1

        for j in range(s):
            X1=res[ii][indexes[j]]
            Y1=[]
            Y1.append(indexes[j])
            fval1=get_fun(X1,Y1,W,A,n,p,0)

            if fval == -1 or fval > fval1:
                Y = []
                for i in Y1:
                    Y.append(i)
                X = []
                for i in X1:
                    X.append(i)
                fval = fval1

        for j in range(2*s):
            if indexes[j] not in Y:
                YY=[]
                for i in Y:
                    YY.append(i)
                YY.append(indexes[j])
                fval1=get_fun(X,YY,W,A,n,p,0)
                if fval == -1 or fval > fval1:
                    Y = []
                    for i in YY:
                        Y.append(i)
                    fval=fval1

    for i in X: x0[int(i)]=1.0
    for i in Y: y0[int(i)]=1.0

    if np.sum(x0) == 0:
        print("NOTICE: no good initialization of x can be idetnfied!!!!!!!!!!!!!!!!!!!!!!!!!")
        x0[0] = 1.0
        x0[1] = 1.0
        x0[2] = 1.0

    if np.sum(y0) == 0:
        print("NOTICE: no good initialization of y can be idetnfied!!!!!!!!!!!!!!!!!!!!!!!!!")
        y0[0] = 1
    return x0,y0



"""---------------------------------------------------------------------------
PCA Score Function

----------------------------------------------------------------------------------"""

sigmas1=0.1
sigmas2=1.0
"""
PCA score : \sigma_{1..n} (w_iy - xWy/1^Tx)^2 - lambda*xAx/1^Tx
    input: x,y,W numpy arrays
    output: float value
"""


def PCA_getFuncValue(x,y,W,A,lambda0):
    funcValue=0.0
    # if len(x)!=len(W) or len(y)!=len(W[0]):
    #     print("Error:Invalid parameter....(PCA_getFuncValue)")
    #     return None
    # elif np.sum(x)==0:
    #     print("Error:X vector all zeros....(PCA_getFuncValue)")
    #     return None
    # elif  np.sum(y)==0:
    #     print("Error:Y vector all zeros....(PCA_getFuncValue)")
    #     return None

    Ix=np.ones(len(x))
    xT1=np.sum(x)
    xTW = W.T.dot(x)
    xW_1Tx=xTW*(1.0/xT1)
    term1=(1.0/sigmas1)*np.multiply(W-np.outer(Ix,xW_1Tx),W-np.outer(Ix,xW_1Tx)).dot(y)
    term2=(1.0/sigmas2)*np.multiply(W,W).dot(y)
    diff=term1-term2

    funcValue=diff.dot(x)

    if lambda0>0:
        ATx=np.dot(A,x)
        ATx_1Tx=(1.0/xT1)*ATx
        funcValue=funcValue - lambda0*np.dot(x,ATx_1Tx)

    return funcValue


"""
PCA gradient of x
    input: x,y,W, numpy arrays
    output: gradient vector,is a numpy array
"""
def PCA_gradientX(x,y,W,A,lambda0,adjust=0.0):
    if len(x)!=len(W) or len(y)!=len(W[0]):
        print("Error:Invalid parameter....(PCA_getFuncValue)")
        return None
    elif np.sum(x)==0:
        print("Error:X vector all zeros....(PCA_getFuncValue)")
        return None
    elif  np.sum(y)==0:
        print("Error:Y vector all zeros....(PCA_getFuncValue)")
        return None


    non_zero_count=0.0 #[0,1]
    for i in range(len(x)):
        if x[i]>0.0 and x[i]<1.0: non_zero_count+=1.0
    if non_zero_count>0.0:
        xTW = np.dot(W.T,x)
    else:
        xTW = median_WTx(x,W)


    Ix = np.ones(len(x))
    xT1 = np.sum(x)

    xW_1Tx = xTW * (1.0 / xT1)
    term1 = (1.0 / (sigmas1+adjust)) * np.multiply(W - np.outer(Ix, xW_1Tx), W - np.outer(Ix, xW_1Tx)).dot(y)
    term2 = (1.0 / sigmas2) * np.multiply(W, W).dot(y)
    ATx = np.dot(A,x)
    ATx_1Tx = 1.0 / xT1 * ATx
    dense_term=lambda0*(ATx_1Tx-(1.0/(xT1*xT1)*x.dot(ATx)))*Ix
    gradient=term1-term2-dense_term

    return gradient



"""
PCA gradient of y
    input: x,y,W, numpy arrays
    output: gradient vector,is a numpy array
    \delta_x f(x,y)=2* \sigma_{1..n} (w_i - W^Ty/1^Tx)(w_i - W^Ty/1^Tx)^T.y
"""


def PCA_gradientY(x,y,W,A,lambda0=0.0,adjust=0.0):
    if len(x) != len(W) or len(y) != len(W[0]):
        print("Error:Invalid parameter....(PCA_getFuncValue)")
        return None
    elif np.sum(x) == 0:
        print("Error:X vector all zeros....(PCA_getFuncValue)")
        return None
    elif np.sum(y)==0:
        print("Error:Y vector all zeros....(PCA_getFuncValue)")
        return None


    non_zero_count=0.0
    for i in range(len(x)):
        if x[i]>0.0 and x[i]<1.0: non_zero_count+=1.0
    if non_zero_count>0.0:
        xTW = W.T.dot(x)
    else:
        xTW = median_WTx(x,W)


    Ix = np.ones(len(x))
    xT1 = np.sum(x)

    xW_1Tx = xTW * (1.0 / xT1)
    # print(len(xW_1Tx))
    # print len(W.transpose()),len(W.transpose()[0])
    # print len(np.outer(xW_1Tx,Ix)),len(np.outer(xW_1Tx,Ix)[0])
    term1 = (1.0 / (sigmas1+adjust)) * (np.multiply(W.transpose() - np.outer(xW_1Tx,Ix), W.transpose() - np.outer(xW_1Tx,Ix))).dot(x)
    term2 = (1.0 / sigmas2) * np.multiply(W.transpose(), W.transpose()).dot(x)
    # print len(term1),len(term2)
    gradient=term1-term2

    return gradient

def PCA_multiGradientDecent4PCAScore(x0,y0,OmegaX,OmegaY,W,A,lambda0,maxIter=1000,stepSize=0.01):
    # print("Start argmin f(x,y) ...")
    indicatorX = getIndicateVector(OmegaX,len(x0))
    indicatorY = getIndicateVector(OmegaY, len(y0))
    x=copy.deepcopy(x0)
    y=copy.deepcopy(y0)
    for i in xrange(maxIter):
        # t=0.0
        # while True:
        #     gradientX=PCA_gradientX(x,y,W,A,lambda0,t*0.01)
        #     t+=1.0
        #     if np.min(gradientX)>0.0: break
        #
        # t = 0.0
        # while True:
        #     gradientY = PCA_gradientY(x, y, W, A, lambda0, t * 0.01)
        #     t += 1.0
        #     if np.min(gradientY) > 0.0: break
        gradientX = PCA_gradientX(x, y, W, A, lambda0)
        gradientY = PCA_gradientY(x, y, W, A, lambda0)

        xOld = copy.deepcopy(x)
        yOld = copy.deepcopy(y)
        x = updatedMinimizerX(gradientX,indicatorX,x,stepSize,5)
        y = updatedMinimizerY(gradientY, indicatorY,y,stepSize,5)

        diffNormX = np.linalg.norm(x-xOld)
        diffNormY = np.linalg.norm(y-yOld)
        # if i % 100 == 0: print("processes: {}".format(i))

        if diffNormX<=1e-6  and diffNormY<=1e-6:break




    return (x,y)


def updatedMinimizerX(gradientX,indicatorX,x,stepSize,bound=5):
    normalizedX= (x - stepSize*gradientX)*indicatorX
    indexes=normalizedX.argsort()[::-1]
    cnt=0
    for j in xrange(len(x)):
        if normalizedX[j]<=0.0:
            cnt+=1.0
            normalizedX[j]=0.0
        elif normalizedX[j]>=1.0:
            normalizedX[j]=1.0
    if cnt==len(x):
        # print("!!!!!Warning: Sigmas1 is too large and all values in the gradient vector are nonpositive!!!")
        for i in xrange(bound):
            normalizedX[indexes[i]]=1.0

    return normalizedX


def updatedMinimizerY(gradientY,indicatorY,y,stepSize,bound=5):
    normalizedY= (y - stepSize*gradientY)*indicatorY
    indexes=normalizedY.argsort()[::-1]
    cnt=0
    for j in xrange(len(y)):
        if normalizedY[j]<=0.0:
            cnt+=1.0
            normalizedY[j]=0.0
        elif normalizedY[j]>=1.0:
            normalizedY[j]=1.0
    if cnt==len(y):
        # print("!!!!!Warning: Sigmas1 is too small and all values in the gradient vector are nonpositive!!!")
        for i in xrange(bound):
            normalizedY[indexes[i]]=1.0

    return normalizedY

def median_WTx(x,W):
    medians=np.zeros_like(W[0])

    S=list(getSupp(x))
    for i in range(len(W[0])):
        vals=np.zeros(len(S))
        for k,j in enumerate(S):
            vals[k]=x[j]*W[j][i]
        medians[i]=np.percentile(vals,50)*np.sum(x)
    # print S
    # print medians
    # time.sleep(1000)
    return medians


"""------------------------------------------------------------------
SG-Pusuit dense subgraph detecction main algorithm

----------------------------------------------------------------------"""

"""
Main function
SG-Pursuit Algorithm
Input: 


output:

"""


def sg_pursuit_dense(edges, edgeCost, k, s, W,A,lambda0, maxIter, g):
    start_time = time.time()
    num_nodes = len(W)
    num_feats = len(W[0])
    print("num_nodes:%d num_feat:%d"%(num_nodes,num_feats))
    # initialize values x0,y0
    # xi = np.zeros(num_nodes)
    # yi = np.zeros(num_feats)
    # for i in random.choice(range(num_nodes), k): xi[i] = random.random()
    # for i in random.choice(range(num_feats), s): xi[i] = random.random()
    (xi,yi)=calcualte_initial_val(W,A,num_nodes,num_feats,20,3) #W,A,n,p,k,s
    # print sum(xi),len(getSupp(xi))
    # print sum(yi),len(getSupp(yi))
    # time.sleep(100)
    old_func_value=-1
    func_value=-1
    for numOfIter in range(maxIter):
        # print("SG-Pursuit: Iteration:------{}------".format(numOfIter))

        # calcualte normalized gradient
        t0=time.time()
        gradientFx = PCA_gradientX(xi, yi,W,A,lambda0)
        gradientFy = PCA_gradientY(xi, yi,W,A,lambda0)
        gradientFx = normalized_Gradient(xi, gradientFx)
        gradientFy = normalized_Gradient(yi, gradientFy)
        """Algorithm 1: line 6 """
        (result_nodes, result_edges, p_x) = head_proj(edges=edges, weights=edgeCost, x=gradientFx, g=g, s=k, budget=k-1.,
                                                      delta=1. / 169., err_tol=1e-6, max_iter=50, root=-1,
                                                      pruning='strong', epsilon=1e-6, verbose=0)
        gammaX = set(result_nodes)

        """line 7 """
        gammaY = identifyDirection(gradientFy, 2 * s)
        """line 8 """
        omegaX = gammaX.union(getSupp(xi))
        """line 9"""
        omegaY = gammaY.union(getSupp(yi))
        """line 10"""
        (bx, by) = PCA_multiGradientDecent4PCAScore(xi, yi, omegaX, omegaY, W,A,lambda0, maxIter=1000, stepSize=0.01)
        """line 11"""
        (result_nodes, result_edges, p_x) = tail_proj(edges=edges, weights=edgeCost, x=bx, g=1, s=k, budget=k - 1., nu=2.5,
            max_iter=50, err_tol=1e-6, root=-1, pruning='strong', verbose=0,
            epsilon=1e-6)
        psiX = set(result_nodes)
        """line 12"""
        psiY = identifyDirection(by, s)

        xOld = xi
        yOld = yi

        """line 13"""
        xi = projectionOnVector(bx, psiX)
        """line 14"""
        yi = projectionOnVector(by, psiY)

        func_value = PCA_getFuncValue(xi, yi, W,A,lambda0)

        if old_func_value==-1:
            old_func_value=func_value
        else:
            if np.abs(old_func_value-func_value)<0.001:
                break
            else:
                old_func_value=func_value

    running_time = time.time() - start_time

    return xi, yi,running_time,func_value


"""************************************************************************************************
Experiment codes

*************************************************************************************************"""


"""
Multi-procesing 
"""
class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means we should exit
                # print '%s: Exiting' % proc_name
                break
            # print '%s: %s' % (proc_name, next_task)
            answer = next_task()
            self.result_queue.put(answer)
        return

class task_sg_pursuit(object):
    def __init__(self, edges, edgeCost, k, s, W,A,lambda0,true_subgraph,true_feats,case, maxIter, g):
        self.edges = edges
        self.edgeCost=edgeCost
        self.k=k
        self.s = s
        self.W = W
        self.A = A
        self.lambda0 = lambda0
        self.maxIter = maxIter
        self.g = g
        self.case=case
        self.true_subgraph=true_subgraph
        self.true_feats=true_feats

    def __call__(self):
        # this is the place to do your work
        # time.sleep(0.1) # pretend to take some time to do our work
        #admm(omega, b, X_b, y_t, Y, X, edge_up_nns,edge_down_nns, omega_0, R, dict_paths, psl, approx, report_stat)
        xi, yi, running_time,func_value = sg_pursuit_dense(self.edges, self.edgeCost, self.k, self.s, self.W,self.A,self.lambda0, self.maxIter, self.g)
        return xi, yi, running_time,self.true_subgraph,self.true_feats,self.case,func_value

    def __str__(self):
        return '%s' % (self.p0)



def adj_matrix(edges,n):
    A=np.zeros((n,n))
    for (e1,e2) in edges:
        A[e1][e2]=1.0
        A[e2][e1] = 1.0

    return A

def test_varying_num_attr():
    data_folder="../input/dense_supgraph/simu_fig4/"
    out_file="result-VaryingAtt.txt"
    with open("../output/"+out_file, "a+") as op:
        op.write("\n\n###  "+str(datetime.datetime.now())+"  ###\n")
    for num_feat in [20,40,80,100][:]:
        node_prf=[[],[],[]]
        feat_prf=[[],[],[]]
        running_times=[]
        filename="VaryingAttribute_numAtt_%d.pkl"%(num_feat)
        datas = cPickle.load(bz2.BZ2File(data_folder + filename))
        num_jobs = 0
        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()
        # Start consumers

        num_consumers = 50  # number of cores
        print num_feat, 'Creating %d consumers' % num_consumers
        consumers = [Consumer(tasks, results)
                     for i in range(num_consumers)]
        for w in consumers:
            w.start()

        for case,data in datas.items()[:]:
            k=len(data["true_sub_graph"])/2
            s=len(data["true_sub_feature"])
            lambda0=5.0
            A=adj_matrix(data["edges"],data["n"])
            # print(data["true_sub_graph"])
            # print(data["true_sub_feature"])
            #edges, edgeCost, k, s, W,A,lambda0, maxIter=5, g=1.0, B=3.
            # xi,yi,running_time=sg_pursuit_dense(data["edges"], data["costs"], k, s,data["data_matrix"] ,A, lambda0, maxIter=5, g=1)
            tasks.put(task_sg_pursuit(data["edges"], data["costs"], k, s,data["data_matrix"] ,A, lambda0,data['true_sub_graph'],data['true_sub_feature'],case, maxIter=5, g=1))
            num_jobs +=1.0
            data={}

        for i in range(num_consumers):
            tasks.put(None)
        while num_jobs:
            xi,yi,running_time,true_subgraph,true_feats,case,func_value = results.get()
            n_pre_rec_fm = node_pre_rec_fm(
                true_nodes=true_subgraph,
                pred_nodes=np.nonzero(xi)[0])
            f_pre_rec_fm = node_pre_rec_fm(
                true_nodes=true_feats,
                pred_nodes=np.nonzero(yi)[0])
            print("Case:{}, Node:{} ,Feat:{} ,Func_value:{} ,Running:{} ".format(case,n_pre_rec_fm,f_pre_rec_fm,func_value,running_time))
            node_prf[0].append(n_pre_rec_fm[0])
            node_prf[1].append(n_pre_rec_fm[1])
            node_prf[2].append(n_pre_rec_fm[2])

            feat_prf[0].append(f_pre_rec_fm[0])
            feat_prf[1].append(f_pre_rec_fm[1])
            feat_prf[2].append(f_pre_rec_fm[2])

            running_times.append(running_time)
            num_jobs-=1.0




        print(">>VaryingAtt: %d %f %f %f\n"%(num_feat,np.mean(node_prf[2]),np.mean(feat_prf[2]),round(np.mean(running_times),2)))
        with open("../output/"+out_file,"a+") as op:
            op.write("VaryingAtt: %d %f %f %f\n"%(num_feat,np.mean(node_prf[2]),np.mean(feat_prf[2]),round(np.mean(running_times),2)))

def test_varying_num_cluster():
    data_folder="../input/dense_supgraph/simu_fig4/"
    out_file="result-VaryingNumCluster.txt"
    with open("../output/"+out_file, "a+") as op:
        op.write("\n\n###  "+str(datetime.datetime.now())+"  ###\n")
    for num_cluster in [10,12,14,15,20,25][:]:
        node_prf=[[],[],[]]
        feat_prf=[[],[],[]]
        running_times=[]
        filename="VaryingNumClusters_numCluster_%d.pkl"%(num_cluster)
        datas = cPickle.load(bz2.BZ2File(data_folder + filename))
        num_jobs = 0
        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()
        # Start consumers
        num_consumers = 50  # number of cores.
        print num_cluster, 'Creating %d consumers' % num_consumers
        consumers = [Consumer(tasks, results)
                     for i in range(num_consumers)]
        for w in consumers:
            w.start()
        for case, data in datas.items()[:]:
            k = len(data["true_sub_graph"]) / 2
            s = len(data["true_sub_feature"])
            lambda0 = 5.0
            A = adj_matrix(data["edges"], data["n"])

            # edges, edgeCost, k, s, W,A,lambda0, maxIter=5, g=1.0, B=3.
            # xi,yi,running_time=sg_pursuit_dense(data["edges"], data["costs"], k, s,data["data_matrix"] ,A, lambda0, maxIter=5, g=1)
            tasks.put(
                task_sg_pursuit(data["edges"], data["costs"], k, s,data["data_matrix"] ,A, lambda0,data['true_sub_graph'],data['true_sub_feature'],case, maxIter=5, g=1))
            num_jobs += 1.0

        for i in range(num_consumers):
            tasks.put(None)
        while num_jobs:
            xi, yi, running_time, true_subgraph, true_feats, case, func_value = results.get()
            n_pre_rec_fm = node_pre_rec_fm(
                true_nodes=true_subgraph,
                pred_nodes=np.nonzero(xi)[0])
            f_pre_rec_fm = node_pre_rec_fm(
                true_nodes=true_feats,
                pred_nodes=np.nonzero(yi)[0])
            print("Case:{}, Node:{} ,Feat:{} ,Func_value:{} ,Running:{} ".format(case, n_pre_rec_fm,
                                                                                 f_pre_rec_fm, func_value,
                                                                                 running_time))
            node_prf[0].append(n_pre_rec_fm[0])
            node_prf[1].append(n_pre_rec_fm[1])
            node_prf[2].append(n_pre_rec_fm[2])

            feat_prf[0].append(f_pre_rec_fm[0])
            feat_prf[1].append(f_pre_rec_fm[1])
            feat_prf[2].append(f_pre_rec_fm[2])

            running_times.append(running_time)
            num_jobs -= 1.0

        print("VaryingAtt: num_attirbute:%d avg_node_fm:%f   avg_feat_fm%f avg_runtime:%f"%(num_cluster,np.mean(node_prf[2]),np.mean(feat_prf[2]),round(np.mean(running_times),2)))
        with open("../output/"+out_file,"a+") as op:
            op.write("VaryingNumCluster %f %f %f %f\n"%(num_cluster,np.mean(node_prf[2]),np.mean(feat_prf[2]),round(np.mean(running_times),2)))


def test_varying_cluster_size():
    data_folder="../input/dense_supgraph/simu_fig4/"
    out_file="result-VaryingClusterSize.txt"
    with open("../output/"+out_file, "a+") as op:
        op.write("\n\n###  "+str(datetime.datetime.now())+"  ###\n")
    for cluster_sizes in [(30,100),(30,150),(30,200),(30,300),(30,400)][:]:
        node_prf=[[],[],[]]
        feat_prf=[[],[],[]]
        running_times=[]
        num_jobs = 0
        tasks = multiprocessing.Queue()
        results = multiprocessing.Queue()
        # Start consumers
        num_consumers = 50  # number of cores.
        print iter, 'Creating %d consumers' % num_consumers
        consumers = [Consumer(tasks, results)
                     for i in range(num_consumers)]
        for w in consumers:
            w.start()
        for case in range(50)[:]:
            filename="VaryingClusterSizes_numCluster_%d_%d_case-%d.pkl"%(cluster_sizes[0],cluster_sizes[1],case)
            try:
                data = cPickle.load(bz2.BZ2File(data_folder + filename))
            except:
                print("File not exist..")
                continue
            k = len(data["true_sub_graph"]) / 2
            s = len(data["true_sub_feature"])
            lambda0 = 5.0
            A = adj_matrix(data["edges"], data["n"])



            # edges, edgeCost, k, s, W,A,lambda0, maxIter=5, g=1.0, B=3.
            # xi,yi,running_time=sg_pursuit_dense(data["edges"], data["costs"], k, s,data["data_matrix"] ,A, lambda0, maxIter=5, g=1)
            tasks.put(
                task_sg_pursuit(data["edges"], data["costs"], k, s,data["data_matrix"] ,A, lambda0,data['true_sub_graph'],data['true_sub_feature'],case, maxIter=5, g=1))
            num_jobs += 1.0

        for i in range(num_consumers):
            tasks.put(None)
        while num_jobs:
            xi, yi, running_time, true_subgraph, true_feats, case, func_value = results.get()
            n_pre_rec_fm = node_pre_rec_fm(
                true_nodes=true_subgraph,
                pred_nodes=np.nonzero(xi)[0])
            f_pre_rec_fm = node_pre_rec_fm(
                true_nodes=true_feats,
                pred_nodes=np.nonzero(yi)[0])
            print("Case:{}, Node:{} ,Feat:{} ,Func_value:{} ,Running:{} ".format(case, n_pre_rec_fm,
                                                                                 f_pre_rec_fm, func_value,
                                                                                 running_time))
            node_prf[0].append(n_pre_rec_fm[0])
            node_prf[1].append(n_pre_rec_fm[1])
            node_prf[2].append(n_pre_rec_fm[2])

            feat_prf[0].append(f_pre_rec_fm[0])
            feat_prf[1].append(f_pre_rec_fm[1])
            feat_prf[2].append(f_pre_rec_fm[2])

            running_times.append(running_time)
            num_jobs -= 1.0

        print("VaryingClusterSize: cluster size:%d - %d avg_node_fm:%f   avg_feat_fm%f avg_runtime:%f"%(cluster_sizes[0],cluster_sizes[0],np.mean(node_prf[2]),np.mean(feat_prf),np.mean(running_times)))
        with open("../output/"+out_file,"a+") as op:
            op.write("VaryingClusterSize %d_%d %f %f %f\n"%(cluster_sizes[0],cluster_sizes[0],np.mean(node_prf[2]),np.mean(feat_prf),round(np.mean(running_times),2)))






def main():
    test_varying_num_attr()
    # test_varying_num_cluster()
    # test_varying_cluster_size()




if __name__ == '__main__':
    main()