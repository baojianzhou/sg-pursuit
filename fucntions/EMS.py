import numpy as np
import copy
"""
EMS score :-xWy
    input: x,y,W numpy arrays
    output: float value
"""
def EMS_getFuncValue(x,y,W):
    funcValue=0.0
    if len(x)!=len(W) or len(y)!=len(W[0]):
        print("Error:Invalid parameter....(EMS_getFuncValue)")
        return None
    elif np.sum(x)==0:
        print("Error:X vector all zeros....(EMS_getFuncValue)")
        return None
    elif  np.sum(y):
        print("Error:Y vector all zeros....(EMS_getFuncValue)")
        return None


    xT1=np.sum(x)
    yT1=np.sum(y)
    xTWy=(x.dot(W)).dot(y)
    funcValue=-xTWy*xTWy/xT1*yT1
    return funcValue


"""
EMS gradient of x
    input: x,y,W, numpy arrays
    output: gradient vector,is a numpy array
"""
def EMS_gradientX(x,y,W):
    if len(x)!=len(W) or len(y)!=len(W[0]):
        print("Error:Invalid parameter....(EMS_getFuncValue)")
        return None
    elif np.sum(x)==0:
        print("Error:X vector all zeros....(EMS_getFuncValue)")
        return None
    elif  np.sum(y):
        print("Error:Y vector all zeros....(EMS_getFuncValue)")
        return None
    gradient=np.zeros(len(x))
    xT1 = np.sum(x)
    yT1 = np.sum(y)
    xTWy = (x.dot(W)).dot(y)
    Wy=W.dot(y)
    term1=(2.0/(xT1*yT1))*Wy  #(2.0/xT1*yT1)*W.Y
    term2=xTWy/(xT1*xT1*yT1)
    for i in len(gradient):
        gradient[i]=-xTWy*(term1[i]-term2)


    return gradient


"""
EMS gradient of x
    input: x,y,W, numpy arrays
    output: gradient vector,is a numpy array
"""


def EMS_gradientY(x, y, W):
    if len(x) != len(W) or len(y) != len(W[0]):
        print("Error:Invalid parameter....(EMS_getFuncValue)")
        return None
    elif np.sum(x) == 0:
        print("Error:X vector all zeros....(EMS_getFuncValue)")
        return None
    elif np.sum(y):
        print("Error:Y vector all zeros....(EMS_getFuncValue)")
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

def EMS_multiGradientDecent4EMSScore(x0,y0,OmegaX,OmegaY,W,maxIter=1000,stepSize=0.01):
    print("Start argmin f(x,y) ...")
    indicatorX = getIndicateVector(OmegaX,len(x0))
    indicatorY = getIndicateVector(OmegaY, len(x0))
    x=copy.deepcopy(x0)
    y=copy.deepcopy(y0)
    for i in xrange(maxIter):
        gradientX=EMS_gradientX(x,y,W)
        gradientY=EMS_gradientY(x,y,W)

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

