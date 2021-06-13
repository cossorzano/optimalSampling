
import numpy as np
import optimalSampling

class MichaelisFunction(optimalSampling.FittingFunctionLS):
    # y = Vmax*x/(Km+x)
    def getFunction(self, xn, beta):
        return beta[0]*xn/(beta[1]+xn)

    def getPartialDerivative1(self, xn, yn, beta, i):
        denominator = beta[1] + xn
        if i==0:
            return xn / denominator
        elif i==1:
            return -beta[0]*xn/(denominator*denominator)

h=MichaelisFunction()
h.sigma2=1
trueBeta=np.asarray([100,50])

X=np.asarray([[5],[30]])
y=h.simulateFunctionAtMultiplePoints(X, trueBeta, True)

stepx=0.1

def CramerRaoBound1(I):
    return optimalSampling.CramerRaoBound(I,1)

evaluator0=optimalSampling.FIMEvaluator()
evaluator1=optimalSampling.FIMEvaluator(CramerRaoBound1)
evaluator2=optimalSampling.VarEvaluator()

optimalSampling.simulateProcess(h,trueBeta,X,y,np.asarray([80,20]),30,np.mgrid[0:100+stepx:stepx],evaluator0, verbose=True)