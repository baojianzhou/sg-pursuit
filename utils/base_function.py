import numpy as np
from functions.PCA import *
def getSupp(x):
    return set(np.nonzero(x)[0])


""""
grad_i { = 0       1) if grad_i<0 and x_i=1
     *             2)or if grad_i>0 and x_i =0
     * = grad_i     ,otherwise

"""


def normalized_Gradient(x, gradient):
    if gradient == None or len(gradient) == 0 or x == None or len(x) == 0:
        print("Error: gradient/x is None...  (normalized_Gradient)")

    normalizedGradient = np.zeros_like(x)
    for i in range(len(gradient)):
        if gradient[i] < 0.0 and x[i] == 1.0:
            normalizedGradient[i] = 0.0
        elif gradient[i] > 0.0 and x[i] == 0.0:
            normalizedGradient[i] = 0.0
        else:
            normalizedGradient[i] = gradient[i]


"""
identify the direction, which means a subset of indices will be returned.
The goal is to maximize ||gradient_R||^2 s.t. |R| < s
"""


def identifyDirection(gradient, s):
    if gradient == None or len(gradient) == 0 or s <= 0 or s > len(gradient):
        print("Error: gradient is None...  (argMax_nabla_y_Fxy)")
    # if the vector is zero, just return an empty set
    if np.sum(gradient): return np.array([])
    squareGradient = gradient * gradient
    indexes = squareGradient.argsort()[::-1]
    gammaY = set()
    for i in range(s):
        if i < len(indexes) and squareGradient[indexes[i]] > 0.0:
            gammaY.add(indexes[i])

    return gammaY


"""
Return the projected vector
param vector: the vector to be projected.
param sset:    the projection set.
return the projected vector.
"""


def projectionOnVector(vector, sset):
    if vector == None or set == None: return None
    projectedVector = np.zeros_like(vector)
    for i in range(len(vector)):
        if i in sset: projectedVector[i] = vector[i]

    return projectedVector


def getIndicateVector(S,ssize):
    if ssize<=0: return None
    x=np.zeros(ssize)
    for i in S: x[i]=1.0
    return x


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
def get_fun(X,Y,W,n,p,lambda0=0):
    x=np.zeros(n)
    y=np.zeros(p)

    for i in X:
        x[i]=1.0
    for i in Y:
        y[i]=1.0

    return PCA_getFuncValue(x,y,W,None,lambda0)#A,W,lambda0

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
            nns=set(np.nonzero(A[i])[0])
            dists=[0.0 for i in range(len(nns))]
            for kk in range(len(nns)):
                dists[kk]=np.abs(W[kk][j]-W[i][j])

            if len(nns)>1.0:
                x0[i]=np.percentile(dists,50)*-1.0
            else:
                x0[i] = np.mean(dists)*-1.0

        indexes = x0.argsort()[::-1]

        ii=0
        for r in trials:
            S=[]
            for i in range(k*r):
                S.append(indexes[i])
            if len(S)>0:
                rank=ranknodes(S,k,A)
                fval=get_fun(rank,[j],0)
                scores[ii][j]=fval*-1
                res[ii].append(rank)
            else:
                scores[ii][j]=-1000
                res[ii].append([])
            ii+=1.0

    fval=-1
    Y=[]
    X=[]
    for ii in range(len(trials)):
        indexes = scores[ii].argsort()[::-1]
        Y1=[]
        for i in range(s):
            if scores[ii][indexes[i]]>-1000:
                Y1.append(indexes[i])

        X1=res[ii][indexes[0]]
        fval1=get_fun(X1,Y1,0)
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
            fval1=get_fun(X1,Y1,0.0)

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
                YY.add(indexes[j])
                fval1=get_fun(X,YY,0.0)
                if fval == -1 or fval > fval1:
                    Y = []
                    for i in YY:
                        Y.append(i)
                    fval=fval1

    for i in X: x0[i]=1.0
    for i in Y: y0[i]=1.0

    if np.sum(x0) == 0:
        print("NOTICE: no good initialization of x can be idetnfied!!!!!!!!!!!!!!!!!!!!!!!!!")
        x0[0] = 1
        x0[1] = 1
        x0[2] = 1

    if np.sum(y0) == 0:
        print("NOTICE: no good initialization of y can be idetnfied!!!!!!!!!!!!!!!!!!!!!!!!!")
        y0[0] = 1
    return x0,y0






