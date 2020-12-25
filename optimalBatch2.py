
import numpy as np
import optimalSampling

class ExponentialFunction(optimalSampling.FittingFunctionLS):
    # y = b0*exp(-b1*t)+b2
    def getFunction(self, xn, beta):
        return beta[0]*np.exp(-beta[1]*xn)+beta[2]

    def postProcess(self, y):
        return np.clip(y,0.0,None)

    def getTrueSigma2(self, xn, beta):
        y=np.abs(self.getFunction(xn,beta))
        return self.sigma2*y*y

    def getWeight(self, xn, beta):
        y=np.abs(self.getFunction(xn,beta))
        return 1/y

    def getPartialDerivative1(self, xn, yn, beta, i):
        if i==0:
            return np.exp(-beta[1]*xn)
        elif i==1:
            return -beta[0]*xn*np.exp(-beta[1]*xn)
        else:
            return 1

    def getPartialDerivative2(self, xn, yn, beta, i, j):
        if i==0 and j==1 or i==1 and j==0:
            return -xn*np.exp(-beta[1]*xn)
        elif i==1 and j==1:
            return beta[0]*xn*xn*np.exp(-beta[1]*xn)
        else:
            return 0

h=ExponentialFunction()
h.sigma2=0.1
trueBeta=np.asarray([1,0.5,0.25])

X=np.asarray([[0],[1],[2]])
y=h.simulateFunctionAtMultiplePoints(X, trueBeta, True)
evaluator=optimalSampling.FIMEvaluator(optimalSampling.CramerRaoBound)

stepx=0.01
optimalSampling.simulateProcess(h,trueBeta,X,y,np.asarray([1,0.5,0.25]),20,np.mgrid[0:30+stepx:stepx],evaluator, verbose=True)