import numpy as np


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