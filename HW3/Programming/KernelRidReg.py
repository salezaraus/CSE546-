# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:07:55 2020

@author: salez
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

np.random.seed(11)  

def GenerateSynData(n):
    '''
    Synthetic data is generated with a true relationship 
    f*(x) = 4sin(pi*x)cos(6pi*x^2)    
    
    Parameters
    ----------
    n : 'int'
        Number of observations 
        
    Returns
    -------
    
    x_i : 'np.array'
        array of uniformly drawn samples at random on [0,1]
    
    y_i: 'np.array'
        array of true relationship + error term
        
    y_true : 'np.array'
        array of true relationship

    '''
    
    x_i = np.random.uniform(size = n)
    
    y_i = TrueFunc(x_i) + np.random.normal(size = n) 
    
    return x_i, y_i

def TrueFunc(x_i): 
    '''
    f*(x) = 4sin(pi*x)cos(6pi*x^2)    

    Parameters
    ----------
    x_i : 'float' 
        x observation
        
    Returns
    -------
    y_true : 'float'
        f(x)
    
    '''
    
    f_x = 4*np.sin(np.pi*x_i)*np.cos(6*np.pi*x_i**2)
    
    return f_x

class KernelPoly: 
    '''
    Defines a kernel class for polynomials of degree up to d. This is defined 
    by K(x,z) = (1+ <x,z>)^d
    
    Attributes
    ----------
    
    x: `np.array`, optional
        Feature space.
        
    y: `np.array`, optional
        Labels.
        
    lambd: `float`, optional
        Regularization parameter
        
    d: `int`, optional
        hyperparameter for polynomial kernel
        
    alpha: `dict`
        Lerned predictor (learned weights).
    
    
    '''
    
    def __init__(self):
        self.x = None
        self.y = None
        self.alpha = None
        self.lambd = None
        self.d = None
        
    def updateparams(self, x_train, y_train, lambdnew, dnew):
        '''Updates parameters and hyperparameters '''
        self.x = x_train
        self.y = y_train
        self.lambd = lambdnew 
        self.d = dnew 
            
            
    def KernelFunc(self, x, z): 
        """Evaluate the kernel on given points. Kenrel is defined as
            K(x,z) = (1 + <x,z>)**d
        Parameters
        ----------
        x: `np.array`
            A column vector of features.
        z: `np.array`
            A column vector of ``n`` points at which to evaluate the kernel.
            
        Returns
        -------
        K: `np.array`
            An n-by-d matrix where each column is the kernel evaluated at a
            single point.
        """
        
        return (1 + np.outer(x, z))**self.d
    
    def fit(self): 
        """Updates the class attributes and performs ridge regression. 
        Stores the learned predictor in the alpha attribute.
        """
        
        K = self.KernelFunc(self.x, self.x)
        
        self.alpha = np.linalg.solve(K + self.lambd * np.eye(len(K)), self.y)
        
        
    def predict(self, x_pred): 
        '''Using learned weights and given features predicts the labels. Uses 
        the closed form solution of the Kernel ridge regression
        
        Parameters
        ----------
        x_pred: `np.array`
            A column vector of features used for prediction
        
        Returns
        --------
        y: `np.array`
            Predicted labels   
        '''
        
        K = self.KernelFunc(self.x, x_pred)
        
        return np.dot(self.alpha, K)
    
    def MSE(self, x_pred, y_pred):
        """Predicts the labels of the given features and compares them to the
        given truth (true labels). Associates a score to the MSE loss of the 
        prediction
        
        Parameters
        ----------
        
        x_pred: `np.array`
            A column vector of features used for prediction
       
        y_pred: `np.array`
            prediction labels for the corresponding features.
            
        Returns
        -------
        MSEloss: `float`
            Mean of the square of differences in predicitons, the score of
            the goodness of predictions.
        """
        return np.mean((self.predict(x_pred) - y_pred)**2)
    
class KernelRBF: 
    '''
    Defines a kernel class for RBF. This is defined 
    by K(x,z) = exp(-gamma norm(x-z)^2)
    
    Attributes
    ----------
    
    x: `np.array`, optional
        Feature space.
        
    y: `np.array`, optional
        Labels.
        
    lambd: `float`, optional
        Regularization parameter
        
    gamma: `int`, optional
        hyperparameter for RBF kernel 
        
    alpha: `dict`
        Lerned predictor (learned weights).
       
    '''
    
    def __init__(self):
        self.x = None
        self.y = None
        self.alpha = None
        self.lambd = None
        self.gamma = None
        
    def updateparams(self, x_train, y_train, lambdnew, gammanew):
        '''Updates parameters and hyperparameters '''
        self.x = x_train
        self.y = y_train
        self.lambd = lambdnew 
        self.gamma = gammanew
            
            
    def KernelFunc(self, x, z): 
        """Evaluate the kernel on given points. Kenrel is defined as
            K(x,z) = exp(-gamma norm(x-z)^2)
        Parameters
        ----------
        x: `np.array`
            A column vector of features.
        z: `np.array`
            A column vector of ``n`` points at which to evaluate the kernel.
        Returns
        -------
        K: `np.array`
            An n-by-d matrix where each column is the kernel evaluated at a
            single point.
        """
        
        return np.exp(-self.gamma * np.subtract.outer(x,z)**2)
    
    def fit(self): 
        """Updates the class attributes and performs ridge regression. 
        Stores the learned predictor in the alpha attribute.
        """
        
        K = self.KernelFunc(self.x, self.x)
        
        self.alpha = np.linalg.solve(K + self.lambd * np.eye(len(K)), self.y)
        
        
    def predict(self, x_pred): 
        '''Using learned weights and given features predicts the labels. Uses 
        the closed form solution of the Kernel ridge regression
        
        Parameters
        ----------
        x_pred: `np.array`
            A column vector of features used for prediction
        
        Returns
        --------
        y: `np.array`
            Predicted labels   
        '''
        
        K = self.KernelFunc(self.x, x_pred)
        
        return np.dot(self.alpha, K)
    
    def MSE(self, x_pred, y_pred):
        """Predicts the labels of the given features and compares them to the
        given truth (true labels). Associates a score to the MSE loss of the 
        prediction
        
        Parameters
        ----------
        
        x_pred: `np.array`
            A column vector of features used for prediction
       
        y_pred: `np.array`
            prediction labels for the corresponding features.
            
        Returns
        -------
        MSEloss: `float`
            Mean of the square of differences in predicitons, the score of
            the goodness of predictions.
        """
        return np.mean((self.predict(x_pred) - y_pred)**2)
    
    
def CrossValid(x, y, kfold, lambd, d = None, gamma = None):
    '''
    Performs cross validation on a set of parameters for either Kernel 
    Polynomial or RBF Kernel Regression. 
    
    NOTE: size of x,y must be divisible by kfold size
    
    Parameters
    ----------
    x: `np.array`
            A column vector of features.
            
    y: `np.array`
        Labels.
    
    kfold : int
        Choose the k for k fold cross validation. For example if kfold = n, 
        then it performs leave one out cross validation. 
        
    lambd: `float`, 
        Regularization parameter
        
    d: `int`, optional
        hyperparameter for polynomial kernel. If value specified, will perform
        polynomial kernel regression 
        
    gamma: `float`, optional
        hyperparameter for RBF kernel. If value specified, will 
        RBF kernel regression 
        
    Returns
    -------
    MSE_Avg: 'float'
        Returns average MSE score for cross validation 

    '''
    n = len(x)
    
    if n % kfold != 0: 
        print('Choose kfold that divides length of x')
        return None
    
    # randomize index for cross validation
    idxs = np.random.permutation(len(x))
    
    loss = np.zeros(kfold)
    
    # Perform polynomial kernel regression 
    if d is not None: 
        
        KPoly  = KernelPoly()
        
        for i in range(kfold): 
            lowidx = int(n/kfold * i) 
            highidx = int(n/kfold *(i+1))
            
            # validation sets 
            x_val = x[idxs[lowidx:highidx]]
            y_val = y[idxs[lowidx:highidx]]
            
            # validation sets 
            x_train = np.concatenate([x[idxs[0:lowidx]], x[idxs[highidx:]]])
            y_train = np.concatenate([y[idxs[0:lowidx]], y[idxs[highidx:]]])
            
            # update parameters 
            
            KPoly.updateparams(x_train = x_train, y_train = y_train, 
                               lambdnew = lambd, dnew = d)
            
            # Fit and compute the loss 
            KPoly.fit() 
            loss[i] = KPoly.MSE(x_val, y_val)
             
        return np.mean(loss)
    
    else: 
        
        # Perform RBF kernel regression    
        Krbf  = KernelRBF()
        
        for i in range(kfold): 
            lowidx = int(n/kfold * i) 
            highidx = int(n/kfold *(i+1))
            
            # validation sets 
            x_val = x[idxs[lowidx:highidx]]
            y_val = y[idxs[lowidx:highidx]]
            
            # validation sets 
            x_train = np.concatenate([x[idxs[0:lowidx]], x[idxs[highidx:]]])
            y_train = np.concatenate([y[idxs[0:lowidx]], y[idxs[highidx:]]])
            
            # update parameters 
            
            Krbf.updateparams(x_train = x_train, y_train = y_train, 
                               lambdnew = lambd, gammanew = gamma)
            
            # Fit and compute the loss 
            Krbf.fit() 
            loss[i] = Krbf.MSE(x_val, y_val)
             
        return np.mean(loss)
        
    
def ChooseBestParams(x, y, kfold, lambdas, hyperparameters, Kernel = 'Poly'): 
    '''
    Performs a gridsearch of the best paramters   

    Parameters
    ----------
    x: `np.array`
        A column vector of features.
            
    y: `np.array`
        Labels.
        
    kfold : 'int'
        Choose the k for k fold cross validation. For example if kfold = n, 
        then it performs leave one out cross validation. 
        
    lambdas : 'list'
        List of lambdas to iterate through for grid search
        
    hyperparameters : 'list'
        List of hyperparameters corresponding to the Kernel Type
    
    Kernel : 'str'
        Kernel type: 'Poly' for Polynomial Kernel
                     'RBF' for RBF Kernel

    Returns
    -------
    Best_Param: 'tuple'
        Minimum Loss, Lambda Value and Hyperparameter

    '''
    
    # Unique combination of hyperparameters 
    param_comb = list(itertools.product(lambdas, hyperparameters))
    MSEloss = np.zeros(len(param_comb))
    
    if Kernel == 'Poly': 
        i = 0
        for params in param_comb: 
            lambd = params[0]
            degree = params[1]
            
            # Perform Cross Validation to get average error 
            MSEloss[i] = CrossValid(x, y, kfold, lambd, d = degree, 
                                     gamma = None)
            i += 1
        # find index of smallest loss 
        idxmin = np.argmin(MSEloss)
        
        return (MSEloss[idxmin], param_comb[idxmin][0], param_comb[idxmin][1])
    
    else: 
        i = 0
        for params in param_comb: 
            lambd = params[0]
            gamma = params[1]
            
            # Perform Cross Validation to get average error 
            MSEloss[i] = CrossValid(x, y, kfold, lambd, d = None, 
                                     gamma = gamma)
            i += 1
        # find index of smallest loss 
        idxmin = np.argmin(MSEloss)
        
        return (MSEloss[idxmin], param_comb[idxmin][0], param_comb[idxmin][1])
 
def A4aBest(n = 30, kfold = 30): 
    '''
    Performs gridsearch for both Polynomial and RBF Kernels Ridge Regression 
    and returns the optimal parameters for each kernel

    Parameters
    ----------
    n : int, optional
        Number of observations to generate data . The default is 30.
    kfold : int, optional
        Fold size for k-fold corss validation. The default is 30.

    Returns
    -------
    Optimal Parameters for each kernel choice

    '''
    
    lambdas = np.arange(0.01, 20, 0.01)
    degrees = np.arange(0, 50, 1)
    
    lambdasR = np.arange(0.01, 20, 0.01)
    gammas = np.arange(0.1, 20, 0.25)
    
    x, y = GenerateSynData(n)  
    
    Poly_Best = ChooseBestParams(x, y, kfold, lambdas, degrees, 
                                 Kernel = 'Poly')
    
    RBF_Best = ChooseBestParams(x, y, kfold, lambdasR, gammas, 
                                Kernel = 'RBF')
    
    print('The optimal parameters for Polynomial Kernel Regression are')
    print('lambda = %.3f and degree = %d with loss %.3f' % (Poly_Best[1], 
                                                            Poly_Best[2], 
                                                            Poly_Best[0]))
    print('\n')
    
    print('The optimal parameters for RBF Kernel Regression are')
    print('lambda = %.3f and gamma = %d with loss %.3f' % (RBF_Best[1], 
                                                           RBF_Best[2], 
                                                            RBF_Best[0]))
    print('\n')
    
    PolyLambd = Poly_Best[1]
    PolyDegree = Poly_Best[2]
    
    RBFLambd = RBF_Best[1]
    RBFGamma = RBF_Best[2]
    
    return PolyLambd, PolyDegree, RBFLambd, RBFGamma

def PlotA4b(lambdaP = 0.14, degree = 45, lambdaRBF = 0.010, gamma = 19, n = 30): 
    '''
    Plots the predictions for the optimal parameters of each Polynomial 
    Kernel and RBF Kernel

    Parameters
    ----------
    
    lambdaP : 'float'
        optimal Lambda for Polynomial Kernel 
    degree : 'int'
        optimal degree for Polynimal Kernel 
    lambdaRBF : 'float'
        optimal Lambda for RBF Kernel 
    gamma : 'float'
        optimal gamma for RBF Kernel .
        
    n : 'n' optional
        Number of Observations for generating data. The default is 30.

    Returns
    -------
    
    Plots of the prediction along with the true function
    
    x: `np.array`
        A column vector of features.
            
    y: `np.array`
        Labels.

    '''
    
    x, y = GenerateSynData(n)
    
    # Train Polynomial Kernel Regression 
    KPoly  = KernelPoly()
    KPoly.updateparams(x_train = x, y_train = y, 
                               lambdnew = lambdaP, dnew = degree)
    KPoly.fit() 
    
    # Train RBF Kernel Regression    
    Krbf  = KernelRBF()
    Krbf.updateparams(x_train = x, y_train = y, 
                               lambdnew = lambdaRBF, gammanew = gamma)
    Krbf.fit() 
    
    # Create a smooth function 
    xExp = np.linspace(x.min(), x.max(), n*4)
    
    y_P_pred = KPoly.predict(xExp)
    y_RBF_pred = Krbf.predict(xExp)
    
    fig, ax1 = plt.subplots()
    
    ax1.scatter(x,y, label = 'Data')
    ax1.plot(xExp,y_P_pred, label = 'Poly Kernel')
    ax1.plot(xExp,TrueFunc(xExp), label = 'True')
    
    ax1.set_title('Function Estimation plots')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Function Value')
    ax1.legend()
    
    fig, ax2 = plt.subplots()
    
    ax2.scatter(x,y, label = 'Data')
    ax2.plot(xExp,y_RBF_pred, label = 'RBF Kernel')
    ax2.plot(xExp,TrueFunc(xExp), label = 'True')
    
    ax2.set_title('Function Estimation plots')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Function Value')
    ax2.legend()
    
    return x, y, ax1, ax2

def Bootstrap(x, y, B, lambd, hyperparam, kernel = 'Poly'): 
    '''
    Performs bootstrap on dataset to create confidence find 5% and 95% 
    percentiles at each point of x. 
    
    Parameters
    ----------
    x: `np.array`
        A column vector of features.         
    y: `np.array`
        Labels.
    B : 'int'
        Num of bootstraps
        
    lambdas : 'float'
        Regularization parameter
        
    hyperparameters : 'float'
        hyperparameter corresponding to the Kernel Type
    
    Kernel : 'str'
        Kernel type: 'Poly' for Polynomial Kernel
                     'RBF' for RBF Kernel

    Returns
    -------
    perct5: 'np.array'
        Array, length of n with 5th percentile generating by the bootstraping
        
    perct95: 'np.array'
        Array, length of n with 95th percentile generating by the bootstraping    

    '''
    
    n = len(x)
    
    # Create a smooth function 
    xExp = np.linspace(x.min(), x.max(), n*4)
    
    # stoe all predictions for the bootstrap predicitions
    pred = np.zeros((B,len(xExp)))
    
    ind = np.arange(n) #sorted indices
    
    if kernel == 'Poly': 
        for i in range(B): 
            idxs = np.random.choice(ind, size = n, replace = True)
            
            xBoot = x[idxs]
            yBoot = y[idxs]
            
            # Train Polynomial Kernel 
            KPoly  = KernelPoly()
            KPoly.updateparams(x_train = xBoot, y_train = yBoot, 
                               lambdnew = lambd, dnew = hyperparam)
            KPoly.fit() 
            
            # Predict on smooth dataset
            pred[i] = KPoly.predict(xExp)
            
    else: 
        for i in range(B): 
            idxs = np.random.choice(ind, size = n, replace = True)
            
            xBoot = x[idxs]
            yBoot = y[idxs]
            
            # Train RBF Kernel 
            Krbf  = KernelRBF()
            Krbf.updateparams(x_train = xBoot, y_train = yBoot,
                               lambdnew = lambd, gammanew = hyperparam)
            Krbf.fit() 
            
            # Predict on smooth dataset
            pred[i] = Krbf.predict(xExp)
            
    perct5 = np.percentile(pred,5, axis = 0, interpolation = 'lower') 
    perct95 = np.percentile(pred,95, axis = 0, interpolation = 'higher') 
            
    return perct5, perct95        

def PlotA4c(lambdaP = 0.14, degree = 45, lambdaRBF = 0.010, gamma = 19,
            n = 30, B = 300):
    '''
    Plots the predictions for the optimal parameters of each Polynomial 
    Kernel and RBF Kernel

    Parameters
    ----------
    
    lambdaP : 'float'
        optimal Lambda for Polynomial Kernel 
    degree : 'int'
        optimal degree for Polynimal Kernel 
    lambdaRBF : 'float'
        optimal Lambda for RBF Kernel 
    gamma : 'float'
        optimal gamma for RBF Kernel .
        
    n : 'n' optional
        Number of Observations for generating data. The default is 30.

    Returns
    -------
    
    Plots of the prediction along with the true with 5% and 95% confidence 
    band 
    

    '''
    
    x, y, ax1, ax2 = PlotA4b(lambdaP, degree, lambdaRBF, gamma, n)
    
    # Create a smooth function 
    xExp = np.linspace(x.min(), x.max(), n*4)
        
    Poly5, Poly95 = Bootstrap(x, y, B, lambdaP, degree, kernel = 'Poly')
    RBF5, RBF95 = Bootstrap(x, y, B, lambdaRBF, gamma, kernel = 'RBF')
    
    ax1.fill_between(xExp, Poly5, Poly95, alpha=0.4, color="gray")
    ax1.set_ylim((y.min()-1, y.max()+1))
    ax1.set_title('Polynomial Kernel with B = 300')
    
    ax2.fill_between(xExp, RBF5, RBF95, alpha=0.4, color="gray")
    ax2.set_ylim((y.min()-1, y.max()+1))
    ax2.set_title('RBF Kernel with B = 300')
    
    return ax1, ax2

def A4d(n = 300, kfold = 10, B = 300): 
    '''
    Performs same steps as part a,b,c but now we have n = 300 and kfold = 10

    Parameters
    ----------
    n : int, optional
        Number of observations to generate data . The default is 300.
    kfold : int, optional
        Fold size for k-fold corss validation. The default is 10.

    Returns
    -------
    Plots with confidence intervals of the new parameters. 

    '''
    
    PolyLambd, PolyDegree, RBFLambd, RBFGamma = A4aBest(n, kfold)
    
    ax1, ax2 = PlotA4c(lambdaP = PolyLambd, degree = PolyDegree, 
                       lambdaRBF = RBFLambd, gamma = RBFGamma, n = n, B = B)
    
    return ax1, ax2

def A4e(PolyLambd = 1.630, PolyDegree = 49, RBFLambd = 0.010, RBFGamma = 19, 
        n = 1000, B = 300):
    '''
    Creates a confidence interval on the square error difference between 
    the polynomial kernel regression and RBF kernel regression

    Parameters
    ----------
    PolyLambd : 'float'
        optimal Lambda for Polynomial Kernel 
    PolyDegree : 'int'
        optimal degree for Polynimal Kernel 
    RBFLambd : 'float'
        optimal Lambda for RBF Kernel
    RBFGamma : 'float'
        optimal degree for RBF Kernel
    n : 'int', optional
        Number of observations. The default is 1000.
    B : 'int', optional
        Bootstrap observations. The default is 300.

    Returns
    -------
    Confidence intervals 

    '''    
    
    x, y = GenerateSynData(n)
    
    # Train Polynomial Kernel 
    KPoly  = KernelPoly()
    KPoly.updateparams(x, y, PolyLambd, PolyDegree)
    KPoly.fit() 
    
    # Train RBF Kernel
    Krbf  = KernelRBF()
    Krbf.updateparams(x, y, RBFLambd, RBFGamma)
    Krbf.fit() 
    
    Error = np.zeros(B)
    
    ind = np.arange(n) #sorted indices
    
    for i in range(B):
        idxs = np.random.choice(ind, size = n, replace = True)
        
        Polypred = KPoly.predict(x[idxs])
        Rbfpred = Krbf.predict(x[idxs])
        
        Error[i] = np.mean((y[idxs]-Polypred)**2 - (y[idxs]-Rbfpred)**2) 
    
    perc5 = np.percentile(Error, 5, interpolation="lower")
    perc95 = np.percentile(Error, 95, interpolation="higher")
    
    print('The respective 5%% and 95%% values are %.3f and %.3f' % (perc5, 
                                                                    perc95))

if __name__ == "__main__":
 
    #A4aBest(n = 30, kfold = 30)   
    #PlotA4b()
    #PlotA4c()
    #A4d()
    A4e()
    #ax1, ax2 = PlotA4c(1.630, 45, 0.010, 19, 1000,  300)
   
 
# =============================================================================
# # =============================================================================
# The optimal parameters for Polynomial Kernel Regression are
# lambda = 0.140 and degree = 45 with loss 1.154
# 
# 
# The optimal parameters for RBF Kernel Regression are
# lambda = 0.010 and gamma = 19 with loss 1.515
# #        
# # =============================================================================
# 
# =============================================================================



# =============================================================================
# The optimal parameters for Polynomial Kernel Regression are
# lambda = 1.630 and degree = 49 with loss 1.032
# 
# 
# The optimal parameters for RBF Kernel Regression are
# lambda = 0.010 and gamma = 19 with loss 1.352
# =============================================================================
