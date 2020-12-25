
import numpy as np
import optimalSampling

class LinearFunction2(optimalSampling.FittingFunctionLSMAP):
    # y = b0+b1*x
    def getPriorMean(self):
        return [None, None, 1]

    def getPriorVar(self):
        return [None, None, 0.01*0.01]

    def getFunction(self, xn, beta):
        return beta[0]+beta[1]*xn+beta[2]*xn*xn

    def getPartialDerivative1(self, xn, yn, beta, i):
        priormu=self.getPriorMean()
        priorvar=self.getPriorVar()
        if i==0:
            return 1
        elif i==1:
            return xn
        elif i==2:
            return xn*xn-(beta[2]-priormu[2])/priorvar[2]

    def getPartialDerivative2(self, xn, yn, beta, i, j):
        priorvar=self.getPriorVar()
        if i==2:
            return -1/priorvar[2]
        else:
            return 0.0

h=LinearFunction2()
h.sigma2=0.1
trueBeta=np.asarray([-0.*0.,-2*0.,1])

X=np.asarray([[0],[0.5],[1]])
y=h.simulateFunctionAtMultiplePoints(X, trueBeta, True)

stepx=0.01

def CramerRaoBound2(I):
    return optimalSampling.CramerRaoBound(I,2)

evaluator0=optimalSampling.FIMEvaluator()
evaluator1=optimalSampling.FIMEvaluator(CramerRaoBound2)
evaluator2=optimalSampling.VarEvaluator()

optimalSampling.simulateProcess(h,trueBeta,X,y,np.asarray([0,0,0]),100,np.mgrid[0:1+stepx:stepx],evaluator2, verbose=False)