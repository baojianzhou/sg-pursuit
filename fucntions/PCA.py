import numpy as np
import copy
from utils.base_function import *
"""
PCA score : \sigma_{1..n} (w_iy - xWy/1^Tx)^2 - lambda*xAx/1^Tx
    input: x,y,W numpy arrays
    output: float value
"""
def PCA_getFuncValue(x,y,A,W,lambda0,sigma1,sigma2):
    funcValue=0.0
    if len(x)!=len(W) or len(y)!=len(W[0]):
        print("Error:Invalid parameter....(PCA_getFuncValue)")
        return None
    elif np.sum(x)==0:
        print("Error:X vector all zeros....(PCA_getFuncValue)")
        return None
    elif  np.sum(y):
        print("Error:Y vector all zeros....(PCA_getFuncValue)")
        return None

    Ix=np.ones(len(x))
    xT1=np.sum(x)
    yT1=np.sum(y)
    xTW = x.dot(W)
    xW_1Tx=xTW*(1.0/xT1)
    term1=(1.0/sigma1)*np.multiply(W-np.outer(Ix,xTW),W-np.outer(Ix,xTW)).dot(y)
    term2=(1.0/sigma2)*np.multiply(W,W).dot(y)
    diff=term1-term2

    funcValue=diff.dot(x)

    if lambda0>0:
        ATx=x.dot(A)
        ATx_1Tx=1.0/xT1*ATx
        funcValue=funcValue - lambda0*ATx_1Tx.dot(x)

    return funcValue


"""
PCA gradient of x
    input: x,y,W, numpy arrays
    output: gradient vector,is a numpy array
"""
def PCA_gradientX(x,y,W,A,lambda0,sigma1,sigma2,adjust=0.0):
    if len(x)!=len(W) or len(y)!=len(W[0]):
        print("Error:Invalid parameter....(PCA_getFuncValue)")
        return None
    elif np.sum(x)==0:
        print("Error:X vector all zeros....(PCA_getFuncValue)")
        return None
    elif  np.sum(y):
        print("Error:Y vector all zeros....(PCA_getFuncValue)")
        return None
    gradient=np.zeros(len(x))

    non_zero_count=0.0
    for i in range(len(x)):
        if x[i]>0.0 and x[i]<1.0: non_zero_count+=1.0
    if non_zero_count>0.0:
        xTW = x.dot(W)
    else:
        xTW = median_xTW(x,W)


    Ix = np.ones(len(x))
    xT1 = np.sum(x)


    yT1 = np.sum(y)

    xW_1Tx = xTW * (1.0 / xT1)
    term1 = (1.0 / (sigma1+adjust)) * np.multiply(W - np.outer(Ix, xW_1Tx), W - np.outer(Ix, xW_1Tx)).dot(y)
    term2 = (1.0 / sigma2) * np.multiply(W, W).dot(y)
    ATx = x.dot(A)
    ATx_1Tx = 1.0 / xT1 * ATx
    diff = term1 - term2

    for i in len(gradient):
        gradient[i]=-xTWy*(term1[i]-term2)


    return gradient


"""
PCA gradient of x
    input: x,y,W, numpy arrays
    output: gradient vector,is a numpy array
"""


def PCA_gradientY(x, y, W):
    if len(x) != len(W) or len(y) != len(W[0]):
        print("Error:Invalid parameter....(PCA_getFuncValue)")
        return None
    elif np.sum(x) == 0:
        print("Error:X vector all zeros....(PCA_getFuncValue)")
        return None
    elif np.sum(y):
        print("Error:Y vector all zeros....(PCA_getFuncValue)")
        return None
    gradient = np.zeros(len(y))
    xT1 = np.sum(x)
    yT1 = np.sum(y)
    xTWy = (x.dot(W)).dot(y)
    xW = x.dot(W)
    term1 = (2.0 / (xT1 * yT1)) * xW  # (2.0/xT1*yT1)*W.Y
    term2 = xTWy / (xT1 * xT1 * yT1)
    for i in len(gradient):
        gradient[i] = -xTWy * (term1[i] - term2)

    return gradient

def PCA_multiGradientDecent4PCAScore(x0,y0,OmegaX,OmegaY,W,maxIter=1000,stepSize=0.01):
    print("Start argmin f(x,y) ...")
    indicatorX = getIndicateVector(OmegaX,len(x0))
    indicatorY = getIndicateVector(OmegaY, len(x0))
    x=copy.deepcopy(x0)
    y=copy.deepcopy(y0)
    for i in xrange(maxIter):
        gradientX=PCA_gradientX(x,y,W)
        gradientY=PCA_gradientY(x,y,W)

        xOld = copy.deepcopy(x)
        yOld = copy.deepcopy(y)
        x = updatedMinimizerX(gradientX,indicatorX,x,stepSize,5)
        y = updatedMinimizerY(gradientY, indicatorY,y,stepSize,5)

        diffNormX = np.sqrt(np.sum((x-xOld)**2))
        diffNormY = np.sqrt(np.sum((y-yOld)**2))

        if diffNormX<=1e-6  and diffNormY<=1e-6:break
        if i%100==0: print("processes: {}".format(i))



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
        print("!!!!!Warning: Sigmas1 is too large and all values in the gradient vector are nonpositive!!!")
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
        print("!!!!!Warning: Sigmas1 is too small and all values in the gradient vector are nonpositive!!!")
        for i in xrange(bound):
            normalizedY[indexes[i]]=1.0

    return normalizedY

def median_xTW(x,W):
    medians=np.zeros_like(W[0])
    S=getSupp(x)
    for i in range(len(x)):
        vals=np.zeros(len(S))
        for j in S:
            vals[j]=x[j]*W[j][i]
        medians[i]=np.percentile(vals,50)*np.sum(x)

    return medians


