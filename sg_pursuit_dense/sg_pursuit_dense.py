from sparse_learning.proj_algo import head_proj
from sparse_learning.proj_algo import tail_proj
import time
import random
from functions.PCA import *
from utils.base_function import *



"""
Main function
SG-Pursuit Algorithm
Input: 


output:

"""


def sg_pursuit_dense(edges, edgeCost, k, s, W,A,lambda0, maxIter=5, g=1.0, B=3.):
    start_time = time.time()
    num_nodes = len(W)
    num_feats = len(W[0])
    print("num_nodes:%d num_feat:%d"%(num_nodes,num_feats))
    # initialize values x0,y0
    xi = np.zeros(num_nodes)
    yi = np.zeros(num_feats)
    # for i in random.choice(range(num_nodes), k): xi[i] = random.random()
    # for i in random.choice(range(num_feats), s): xi[i] = random.random()
    (xi,yi)=calcualte_initial_val(W,A,num_nodes,num_feats,k,s)
    old_func_value=-1;
    for numOfIter in range(maxIter):
        print("SG-Pursuit: Iteration:------{}------".format(numOfIter))

        # calcualte normalized gradient
        gradientFx = PCA_gradientX(xi, yi,W,A,lambda0)
        gradientFy = PCA_gradientY(xi, yi,W,A,lambda0)
        gradientFx = normalized_Gradient(xi, gradientFx)
        gradientFy = normalized_Gradient(yi, gradientFy)
        """Algorithm 1: line 6 """
        (result_nodes, result_edges, p_x) = head_proj(edges=edges, weights=edgeCost, x=gradientFx, g=g, s=k, budget=B,
                                                      delta=1. / 169., err_tol=1e-6, max_iter=30, root=-1,
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
        (result_nodes, result_edges, p_x) = tail_proj(edges=edges, weights=edgeCost, x=bx, g=g, s=k, root=-1,
                                                      max_iter=20, budget=3., nu=2.5)
        psiX = set(result_nodes)
        """line 12"""
        psiY = identifyDirection(by, s)

        xOld = xi
        yOld = yi

        """line 13"""
        xi = projectionOnVector(bx, psiX)
        """line 14"""
        yi = projectionOnVector(by, psiY)

        func_value = PCA_getFuncValue(xi, yi, W)

        if old_func_value==-1:
            old_func_value=func_value
        else:
            if np.abs(old_func_value-func_value)<0.001:
                break
            else:
                old_func_value=func_value

    running_time = time.time() - start_time

    return xi, yi