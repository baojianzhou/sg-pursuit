from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj
import time
import random
from fucntions.EMS import *



def getSupp(x):
    return set(np.nonzero(x)[0])
""""
grad_i { = 0       1) if grad_i<0 and x_i=1
     *             2)or if grad_i>0 and x_i =0
     * = grad_i     ,otherwise

"""
def normalized_Gradient(x,gradient):
    if gradient==None or len(gradient)==0 or x==None or len(x)==0:
        print("Error: gradient/x is None...  (normalized_Gradient)")

    normalizedGradient=np.zeros_like(x)
    for i in range(len(gradient)):
        if gradient[i]<0.0 and x[i]==1.0:
            normalizedGradient[i]=0.0
        elif gradient[i]>0.0 and x[i]==0.0:
            normalizedGradient[i] = 0.0
        else:
            normalizedGradient[i]=gradient[i]

"""
identify the direction, which means a subset of indices will be returned.
The goal is to maximize ||gradient_R||^2 s.t. |R| < s
"""
def identifyDirection(gradient,s):
    if gradient==None or len(gradient)==0 or s<=0 or s>len(gradient):
        print("Error: gradient is None...  (argMax_nabla_y_Fxy)")
    #if the vector is zero, just return an empty set
    if np.sum(gradient): return np.array([])
    squareGradient=gradient*gradient
    indexes = squareGradient.argsort()[::-1]
    gammaY=set()
    for i in range(s):
        if i<len(indexes) and squareGradient[indexes[i]]>0.0:
            gammaY.add(indexes[i])

    return gammaY

"""
Return the projected vector
param vector: the vector to be projected.
param sset:    the projection set.
return the projected vector.
"""

def projectionOnVector(vector,sset):
    if vector==None or set==None: return None
    projectedVector=np.zeros_like(vector)
    for i in range(len(vector)):
        if i in sset: projectedVector[i]=vector[i]

    return projectedVector

"""
Main function
SG-Pursuit Algorithm
Input: 
    
    
output:

"""
def SG_Pursuit(edges,edgeCost,k,s,W,maxIter=10,g=1.0,B=3.):
    start_time=time.time()
    num_nodes=len(W)
    num_feats=len(W[0])

    #initialize values x0,y0
    xi=np.zeros(num_nodes)
    yi=np.zeros(num_feats)
    for i in random.choice(range(num_nodes),k): xi[i]=random.random()
    for i in random.choice(range(num_feats), s): xi[i] = random.random()

    numOfIter=0.0
    while(True):
        print("SG-Pursuit: Iteration:------{}------".format(numOfIter))
        #calcualte normalized gradient
        gradientFx=EMS_gradientX(xi,yi,W)
        gradientFy=EMS_gradientY(xi,yi,W)
        gradientFx = normalized_Gradient(xi,gradientFx)
        gradientFy = normalized_Gradient(yi,gradientFy)
        """Algorithm 1: line 6 """
        (result_nodes, result_edges, p_x) = head_proj(edges=edges, weights=edgeCost, x=gradientFx, g=g, s=k, budget=B,
                       delta=1. / 169., err_tol=1e-6, max_iter=30, root=-1,
                       pruning='strong', epsilon=1e-6, verbose=0)
        gammaX=set(result_nodes)

        """line 7 """
        gammaY=identifyDirection(gradientFy,2*s)
        """line 8 """
        omegaX=gammaX.union(getSupp(xi))
        """line 9"""
        omegaY=gammaY.union(getSupp(yi))
        """line 10"""
        (bx,by)=EMS_multiGradientDecent4EMSScore(xi,yi,omegaX,omegaY,W,maxIter=1000,stepSize=0.01)
        """line 11"""
        (result_nodes, result_edges, p_x)=tail_proj(edges=edges, weights=edgeCost, x=bx, g=g, s=k, root=-1,
                  max_iter=20, budget=3., nu=2.5)
        psiX=set(result_nodes)
        """line 12"""
        psiY=identifyDirection(by,s)

        xOld=xi
        yOld=yi

        """line 13"""
        xi = projectionOnVector(bx,psiX)
        """line 14"""
        yi = projectionOnVector(by,psiY)

        funcValue=EMS_getFuncValue(xi,yi,W)

        gapX = np.sqrt(np.sum((xi - xOld) ** 2))
        gapY = np.sqrt(np.sum((yi - yOld) ** 2))
        numOfIter+=1

        if (gapX<1e-3 and gapY<1e-3) or numOfIter>maxIter:
            break
    running_time=time.time()-start_time

    return xi,yi,funcValue,running_time