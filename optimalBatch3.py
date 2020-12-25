import numpy as np
import optimalSampling
from scipy.optimize import differential_evolution

class LogisticFunction(optimalSampling.FittingFunction):
    # p = exp(b0+b1*x)/(1+exp(b0+b1*x)
    def getFunction(self, xn, beta):
        e=np.exp(beta[0]+beta[1]*xn)
        return e/(1+e)

    def simulateFunction(self, xn, beta, addNoise):
        p=self.getFunction(xn, beta)
        u=np.random.uniform()
        y=1 if u<=p else 0
        return y

    def getSigma2(self, xn, beta):
        y=self.getFunction(xn,beta)
        return y*(1-y)

    def getPartialDerivative1(self, xn, yn, beta, i):
        p=self.getFunction(xn,beta)
        if i==0:
            return yn-p
        elif i==1:
            return xn*(yn-p)

    def getPartialDerivative2(self, xn, yn, beta, i, j):
        p = self.getFunction(xn, beta)
        e = np.exp(beta[0] + beta[1] * xn)
        pe = p*e
        if i==0 and j==0:
            return -pe
        elif i==1 and j==1:
            return -xn*xn*pe
        else:
            return -xn*pe

    def updateFIM(self, I, xn, yn, beta, proposed):
        p = beta.size
        for i in range(p):
            for j in range(p):
                der2ij = self.getPartialDerivative2(xn, yn, beta, i, j)
                I[i][j] += -der2ij

    def logLikelihood(self, beta):
        L=0
        N=self.y.shape[0]
        for i in range(N):
            pi=self.getFunction(self.X[i],beta)
            yi=self.y[i]
            L+=yi*np.log(pi)+(1-yi)*np.log(1-pi)
        return -L

    def optimize(self, X, y, beta0):
        self.setXy(X, y)
        results = differential_evolution(self.logLikelihood, bounds=[(0.1,5),(0.1,5)])
        return results.x

h=LogisticFunction()
trueBeta=np.asarray([2,0.5])

X=np.asarray([[-15],[-14]])
y=h.simulateFunctionAtMultiplePoints(X, trueBeta, True)

evaluator1=optimalSampling.FIMEvaluator(optimalSampling.CramerRaoBound)
evaluator2=optimalSampling.VarPEvaluator()

stepx=0.01
optimalSampling.simulateProcess(h,trueBeta,X,y,np.asarray([3,3]),100,np.mgrid[-15:-2+stepx:stepx],evaluator2,None, 0.3333, verbose=True)