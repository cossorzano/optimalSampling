
import numpy as np
import optimalSampling

class LinearFunction2(optimalSampling.FittingFunctionLS):
    # y = b0+b1*x+b2*x2
    def getFunction(self, xn, beta):
        return beta[0]+beta[1]*xn+beta[2]*xn*xn

    def getPartialDerivative1(self, xn, yn, beta, i):
        if i==0:
            return 1
        elif i==1:
            return xn
        elif i==2:
            return xn*xn

h=LinearFunction2()
h.sigma2=0.1
trueBeta=np.asarray([1,2,3])

X=np.asarray([[0],[1],[2]])
y=h.simulateFunctionAtMultiplePoints(X, trueBeta, True)

stepx=0.01

def CramerRaoBound1(I):
    return optimalSampling.CramerRaoBound(I,1)

evaluator0=optimalSampling.FIMEvaluator()
evaluator1=optimalSampling.FIMEvaluator(CramerRaoBound1)
evaluator2=optimalSampling.VarEvaluator()

optimalSampling.simulateProcess(h,trueBeta,X,y,np.asarray([0,0,0]),30,np.mgrid[0:2+stepx:stepx],evaluator0, verbose=True)