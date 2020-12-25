import numpy as np
from scipy.special import factorial
import optimalSampling
from matplotlib import pyplot as plt

class fvesFunction():

    def poissonfunction(self, x, x_average):
        return (x_average ** x * np.exp(-x_average)) / factorial(x)

    def getFunction(self, xn, beta):

        laccessible = xn

        kp = beta[0]

        W = 55.5  # water concentration
        NAvogadro = 6.022E23

        # fixed parmeters (can be input as beta parameters)
        # freeprobe = beta[1]
        # ptotal = beta[2]
        # vtotal = beta[3]
        # mu = beta[4]

        ximax = 1
        freeprobe = 0.02
        q = 1.3
        ptotal = 2.5e-8 # in M
        vtotal = 1.5e-4 # in L
        mu = 125000

        LTotal = 2 * laccessible

        xm = (kp * laccessible * ximax) / (W + kp * laccessible)
        try:
            k_average = xm * ptotal * mu / LTotal
            # print(k_average)
        except:
            print('Tengo un problema')

        nvestotal = LTotal * vtotal * NAvogadro / mu
        nprottotal = ptotal * vtotal * NAvogadro

        nprotfree = nprottotal * (1 - xm)

        max_bind = 100
        k = np.linspace(1,max_bind, max_bind)

        sump = np.sum(k ** 2 * self.poissonfunction(k, np.asarray(k_average)))

        ndye = freeprobe * nprottotal
        fves = sump * nvestotal / (ndye * q ** 2 + nprotfree + sump * nvestotal)
        return fves


class FVesFunction(optimalSampling.FittingFunctionLS):
    # Melo2011 function application
    def getFunction(self, xn, beta):
        return fvesFunction().getFunction(xn, beta)

    def postProcess(self, y):
        return np.clip(y,0.0,None)

    def getTrueSigma2(self, xn, beta):
        y=np.abs(self.getFunction(xn,beta))
        return self.sigma2*y*y

    def getWeight(self, xn, beta):
        y=np.abs(self.getFunction(xn,beta))
        return 1/y

    def getBetaStep1(self, i):
        return 1e2

    def getBetaStep1(self, j):
        return 1e2


h=FVesFunction()
h.sigma2=0.01
trueBeta=np.asarray([[5e5]])

# I put 10 sampling points to properly follow the function as an example

X=np.asarray([[1e-7], [5e-7], [1e-6], [1e-5], [5e-5],[1e-4], [5e-4], [1e-3], [5e-3], [1e-2], [5e-2]])
y=h.simulateFunctionAtMultiplePoints(X, trueBeta, True)

evaluator=optimalSampling.FIMEvaluator(optimalSampling.CramerRaoBound)

stepx = 0.1
beta0 = np.asarray([[1e5]])
N = 10

grid = np.exp(np.mgrid[-7:-2+stepx:stepx]) # x sampling in this function has to be logarithmic
optimalSampling.simulateProcess(h,trueBeta,X,y,beta0,N,grid,evaluator, verbose=True, logxscale=True)