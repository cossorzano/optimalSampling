import numpy as np
from scipy.optimize import leastsq, minimize

def CramerRaoBound(I,i=-1):
    try:
        CR=np.linalg.inv(I)
    except:
        return 1e42
    if i<0:
        retval=np.trace(CR)
    else:
        retval=CR[i,i]
    if retval<0:
        return 1e42
    else:
        return retval

class FIMEvaluator():
    def __init__(self, evaluateF=CramerRaoBound):
        self.evaluateF=evaluateF

    def prepare(self, h, X, y, n, beta):
        self.I=h.calculateFIM(X, y, beta)
        self.In=self.I*n
        self.h = h
        self.X = X
        self.y = y
        self.n = n

    def evaluate(self, xProposed, beta):
        IProposed = (self.In + self.h.calculateFutureFIM(xProposed, beta)) / (self.n + 1)
        return self.evaluateF(IProposed)

class VarPEvaluator():
    def __init__(self, Nsteps=20):
        self.Nsteps=Nsteps

    def prepare(self, h, X, y, n, beta):
        self.xmin=np.min(X)
        self.xmax=np.max(X)
        self.X = X

    def evaluate(self, xProposed, beta):
        delta=(self.xmax-self.xmin)/self.Nsteps
        Ni=np.sum(np.abs(self.X-xProposed)<delta)+1
        p=h.getFunction(xProposed, beta)
        sigma2=p*(1-p)/Ni
        return 1/sigma2

class VarEvaluator():
    def __init__(self, Nsteps=20):
        self.Nsteps=Nsteps

    def prepare(self, h, X, y, n, beta):
        self.xmin=np.min(X)
        self.xmax=np.max(X)
        self.X = X
        self.y = y
        N=self.y.size
        self.yp=np.zeros(N)
        for n in range(N):
            self.yp[n]=h.getFunction(self.X[n],beta)
        self.sigma2=np.var(self.y-self.yp)

    def evaluate(self, xProposed, beta):
        delta=(self.xmax-self.xmin)/self.Nsteps
        idx = (np.abs(self.X-xProposed)<=delta).flatten()
        Ni=np.sum(idx)+1
        if Ni<=2:
            w=1/self.sigma2
        else:
            sigma2proposed=np.var(self.y[idx]-self.yp[idx])
            w=1/(sigma2proposed/Ni)
        return w/(delta+np.min(np.abs(self.X-xProposed)))

def lookForBestSamplingPoint(h, X, y, n, beta, mesh, evaluator, ymin=None, ymax=None):
    evaluator.prepare(h, X, y, n, beta)

    bestEval = None
    bestx = None
    if len(mesh.shape) == 1:
        # x is 1D
        d = 1
        firstMin=None
        firstMax=None
        for xProposed in mesh:
            ok=True
            if ymin is not None or ymax is not None:
                y = h.getFunction(xProposed,beta)
                if ymin is not None:
                    ok=ok and y>=ymin
                    if firstMin is None and y>=ymin:
                        firstMin=xProposed
                        print("First x above ymin(%f)=%f"%(ymin,xProposed))
                if ymax is not None:
                    ok=ok and y<=ymax
                    if firstMax is None and y >= ymax:
                        firstMax = xProposed
                        print("First x above ymax(%f)=%f" % (ymax, xProposed))
                # print(xProposed, y, ymin, ymax, ok)
            if ok:
                proposedValue = evaluator.evaluate(xProposed, beta)
                #print(xProposed,proposedValue)
                if bestEval is None or proposedValue<bestEval:
                    bestEval=proposedValue
                    bestx=xProposed
    else:
        d = len(mesh.shape) - 1  # Dimensions
        # TODO: **** Unfinished
    return bestx

def prettyPrint(X,y):
    for i in range(y.size):
        toPrint=""
        if len(X.shape)==2:
            for j in range(X[i,:].size):
                toPrint+="%f "%X[i,j]
        else:
            toPrint += "%f "%X[i]
        toPrint += "%f"%y[i]
        print(toPrint)

def plot(X,y,h,beta,trueBeta, logxscale):
    import matplotlib
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    if X.shape[1] > 1:
        pass
    else:
        x=np.reshape(X, (X.shape[0]))
        if logxscale:
            xs = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)
        else:
            xs=np.arange(min(x),max(x),(max(x)-min(x))/100)
        ys=np.zeros(xs.shape)
        yt=np.zeros(xs.shape)
        for i in range(xs.shape[0]):
            ys[i]=h.getFunction(xs[i],beta)
            if trueBeta is not None:
                yt[i]=h.getFunction(xs[i],trueBeta)
        ax.scatter(x, y, c=range(0,x.shape[0]), cmap='gray')
        #for i in range(x.shape[0]):
        #    plt.annotate("%d"%i,(x[i],y[i]))
        ax.plot(xs, ys, '-b',label='Estimated')
        if trueBeta is not None:
            ax.plot(xs, yt, '-r', label='Ground truth')
        ax.legend()
        if logxscale:
            ax.set_xscale('log')
        ax.set(xlabel='x', ylabel='y', title='X-Y plot')
        ax.grid()
    plt.show()

def simulateProcess(h,trueBeta,X0,y0,beta0,N,mesh,evaluator, ymin=None, ymax=None, verbose=False, logxscale=False):
    X=X0
    y=y0
    beta=beta0
    for n in range(N):
        print("Current beta:",beta)
        bestx=lookForBestSamplingPoint(h,X,y,X.shape[0],beta,mesh,evaluator,ymin,ymax)
        print("Best sampling point %d: %f"%(n,bestx))

        # Simulate measurement
        besty=h.simulateFunction(bestx, trueBeta, True)

        if np.isscalar(bestx):
            bestx=np.reshape(np.asarray([bestx]),(1,1))
        X=np.concatenate([X,bestx])
        y=np.append(y,besty)
        if X.shape[1]>1:
            prettyPrint(X,y)
        else:
            prettyPrint(np.reshape(X,(X.shape[0])),y)

        # Refine beta
        beta=h.optimize(X,y,beta)
        print("New beta:",beta)
    if verbose:
        plot(X,y,h,beta,trueBeta,logxscale)
    return X,y,beta

class FittingFunction:
    def __init__(self):
        self.X = None
        self.y = None
        self.estimatedSigma2=1

    def simplifyXn(self, xn):
        if xn.shape[0]==1:
            return xn[0]
        else:
            return xn

    def simulateFunctionAtMultiplePoints(self, X, beta, addNoise):
        y=np.zeros(X.shape[0])
        for n in range(y.size):
            xn=self.simplifyXn(X[n,:])
            y[n]=self.simulateFunction(xn, beta, addNoise)
        return y

    def getFunction(self, xn, beta):
        return 0.0

    def postProcess(self, y):
        return y

    def simulateFunction(self, xn, beta, addNoise=False):
        pass

    def getBetaStep1(self, i):
        # Beta step for the first derivative
        return 1

    def getBetaStep2(self, i):
        # Beta step for the second derivative
        return 1

    def getPartialDerivative1(self, xn, yn, beta, i):
        # implemented 5th order diff https://en.wikipedia.org/wiki/Numerical_differentiation#Higher-order_methods
        h_der1 = self.getBetaStep1(i)

        betaf = np.copy(beta)
        betaf[i] = betaf[i] + h_der1
        f_ford1 = fvesFunction().getFunction(xn, betaf)
        betaf[i] = betaf[i] + h_der1
        f_ford2 = fvesFunction().getFunction(xn, betaf)
        betaf[i] = betaf[i] - 3*h_der1
        f_back1 = fvesFunction().getFunction(xn, betaf)
        betaf[i] = betaf[i] - h_der1
        f_back2 = fvesFunction().getFunction(xn, betaf)
        return (- f_ford2 + 8*f_ford1 - 8*f_back1 + f_back2) / (12 * h_der1)

    def getPartialDerivative2(self, xn, yn, beta, i, j):
        # implemented 5th order diff https://en.wikipedia.org/wiki/Numerical_differentiation#Higher-order_methods
        # 3rd order differentiation of the 1st derivative
        h_der2 = self.getBetaStep2(j)

        betaf = np.copy(beta)
        # 1st order second differentiation
        betaf[j] = betaf[j] + h_der2
        f_ford1 = self.getPartialDerivative1(xn, yn, betaf, i)
        betaf[j] = betaf[j] - 2*h_der2
        f_back1 = self.getPartialDerivative1(xn, yn, betaf, i)
        return (f_ford1 - f_back1) / (2*h_der2)

    ''' #5th order second differenciation (optional)
            beta[j] = beta[j] + h_der2
            f_ford1 = self.getPartialDerivative1(xn, yn, beta, i)
            beta[j] = beta[j] + h_der2
            f_ford2 = self.getPartialDerivative1(xn, yn, beta, i)
            beta[j] = beta[j] - 3*h_der2
            f_back1 = self.getPartialDerivative1(xn, yn, beta, i)
            beta[j] = beta[j] - h_der2        
            f_back2 = self.getPartialDerivative1(xn, yn, beta, i)
            return (- f_ford2 + 8*f_ford1 - 8*f_back1 + f_back2) / (12 * h_der)
    '''

    def setXy(self, X, y):
        self.X=X
        self.y=y

    def getResiduals(self, beta):
        pass

    def estimateSigma2(self, X, y, beta):
        N = y.size
        residuals = np.zeros(N)
        for n in range(N):
            xn = self.simplifyXn(X[n, :])
            residuals[n] = (y[n] - self.getFunction(xn, beta))
        self.estimatedSigma2=np.var(residuals)

    def updateFIM(self, I, xn, yn, beta, proposed):
        pass

    def calculateFIM(self, X, y, beta):
        N = X.shape[0]
        p = beta.size
        I = np.zeros((p, p))
        self.estimateSigma2(X, y, beta)
        for n in range(N):
            self.updateFIM(I, self.simplifyXn(X[n, :]), y[n], beta, False)
        return I / N

    def calculateFutureFIM(self, xProposed, beta):
        p = beta.size
        I = np.zeros((p, p))
        self.updateFIM(I, xProposed, None, beta, True)
        return I

class FittingFunctionLS(FittingFunction):
    def __init__(self):
        FittingFunction.__init__(self)
        self.sigma2=1

    def getTrueSigma2(self, xn, beta):
        # Constant noise by default
        return self.sigma2

    def getTrueSigma(self, xn, beta):
        # Constant noise by default
        return np.sqrt(self.getTrueSigma2(xn, beta))

    def getEstimatedSigma2(self, xn, beta):
        # Constant noise by default
        return self.estimatedSigma2

    def getEstimatedSigma(self, xn, beta):
        # Constant noise by default
        return np.sqrt(self.getEstimatedSigma2(xn, beta))

    def getWeight(self, xn, beta):
        return 1

    def getResiduals(self, beta):
        N = self.y.size
        residuals = np.zeros(N)
        for n in range(N):
            xn = self.simplifyXn(self.X[n, :])
            residuals[n] = (self.y[n] - self.getFunction(xn, beta)) * self.getWeight(xn, beta)
        return residuals

    def optimize(self, X, y, beta0):
        self.setXy(X, y)
        optimum, J, info, mesg, _ = leastsq(self.getResiduals, beta0, full_output=True, ftol=1e-8, xtol=1e-8)
        return optimum

    def updateFIM(self, I, xn, yn, beta, proposed):
        p = beta.size
        sigmaxn2 = self.getEstimatedSigma(xn, beta)
        if not proposed:
            residualn = yn - self.getFunction(xn, beta)
        else:
            residualn = 0.0
        der1 = np.zeros(p)
        for i in range(p):
            der1[i] = self.getPartialDerivative1(xn, yn, beta, i)
        for i in range(p):
            for j in range(p):
                if not proposed:
                    der2ij = self.getPartialDerivative2(xn, yn, beta, i, j)
                else:
                    der2ij = 0.0
                I[i][j] += (der1[i] * der1[j] - residualn * der2ij) / sigmaxn2

    def simulateFunction(self, xn, beta, addNoise=False):
        y=self.getFunction(xn,beta)
        if addNoise:
            y+=np.random.normal(0.0,self.getTrueSigma(xn, beta))
        y=self.postProcess(y)
        return y

class FittingFunctionLSMAP(FittingFunctionLS):
    def getPriorMean(self):
        return None

    def getPriorVar(self):
        return None

    def objective(self, beta):
        residuals=self.getResiduals(beta)

        penalization=0
        priorMu=self.getPriorMean()
        priorVar=self.getPriorVar()
        for i in range(beta.size):
            if priorMu[i] is not None and priorVar[i] is not None:
                diff=(priorMu[i]-beta[i])
                penalization+=0.5*diff*diff/priorVar[i]
        return np.sum(np.multiply(residuals,residuals))+penalization

    def optimize(self, X, y, beta0):
        self.setXy(X, y)
        result = minimize(self.objective, beta0, tol=1e-8)
        return result.x
