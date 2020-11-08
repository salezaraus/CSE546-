# -*- coding: utf-8 -*-
"""
This routine implements the coordinate descent via Lasso Regression on a 
synthetic data set for testing purposes. Then the defined functions are used on
real data set. 

@author: salez
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
        

def Generate_data(d,n,k): 
    '''Generates synthethic data based on d features, n observations, k 
    non-zero feature coefficients
    
    Inputs
    
    d : `integer`
        Choose the number of features for the x-values.
    n : `integer`
        Choose the number of observations you want generate
    k: `integer`
        Number of non zero coefficient for synthetic data

    Output

    X: `np.array`
        The feature space. A matrix with n rows of feature vectors, each with d
        features.
        
    Y : `np.array`
        A column vector of n model values.
    
    '''

    X = np.random.normal(0,1, size = (n,d)) # X with n observations and d features
    w = [i/k if i <= k else 0 for i in range(1,d+1)] # Create coefficients 

    w_actual= np.array(w) # convert to numpy array 

    e_i = np.random.normal(0,1, size = (1,n)) # Error is normal(0,1)

    Y = w_actual.dot(X.T) + e_i # Y with n observations
   

    return X,Y

def Max_lambda(X,Y): 
    
    '''Generates the largest lasso lambda penalty term needs to be 
    so that all feature coefficients are zero
    
    Inputs
    
    X: `np.array`
        The feature space. A matrix with n rows of feature vectors, each with d
        features.
        
    Y : `np.array`
        A column vector of n model values.
    
    Output
    
    lambda: 'float'
        Min value of Lasso lambda regularization term to drive all coefficents 
        to zero
    
    '''

    d = X.shape[1] # Number of features 

    # Calculate a max lambda
    lambda_can = np.zeros(d)

    y_temp = Y - np.mean(Y)

    for j in range(d): 
        lambda_can[j] = 2*abs(np.sum(X[:,j]* y_temp))
    
    return np.max(lambda_can)

    
def CoordinateDescent(X,Y,Lasso_lambda, epsilon, w_init = None):
    '''Performs coordinate descent lasso regression algorithm. Stops and 
    returns a w_hat upon a step improvement less than epsilon
    
    Inputs 
    
    X : `np.array`
        The feature space. A matrix with n rows of feature vectors, each with d
        features.
    Y : `np.array`
        A column vector of n model values.
    Lasso_lambda: `float`
        Regularization parameter.
    epsilon: `float`
        Convergence is achieved when the convergence criterion is smaller than
        tolerance of epsilon .
    w_init: 'np.array', optional 
        If user has an initial w values, input them here. If left as None, 
        initial w will be a np.array of zeros
    
    Output 
    
    w: `np.array`
        trained feature weights estimates.
    '''
    
    n,d = X.shape
    
    if w_init is None:
        w = np.zeros(d)
    else: 
        w = w_init # Initialize w as zero 
       
    NotConverged = True
    
    # Precalculate X_squared
    X_squared = 2*X**2
    
    w_old = np.zeros(d) 
    
    while NotConverged: 

        b = np.mean(Y - w.dot(X.T))
        
              
        for k in range(d):
            
            ak = np.sum(X_squared[:,k])
            
            w_old[k] = w[k]
            
            # Note: wj* xi,j must be formed when j does not
            # equal j. By making w[k] = 0 at this iteration, we can  
            # that w.dot(X.T) will not include the j = k term. We 
            # will update w[k] at a later stage
            
            w[k] = 0
        
            ck = 2*np.sum(X[:,k]*(Y-(b+ w.dot(X.T))))
            
            if ck < -Lasso_lambda: 
                w[k] = (ck+Lasso_lambda)/ak
            elif ck >= -Lasso_lambda and ck <= Lasso_lambda:
                w[k] = 0 
            else: 
                w[k] = (ck-Lasso_lambda)/ak
            #print(w_old[k], w[k])
                
        # Check difference from old w to new w 
        loss = np.sum((w.dot(X.T) - Y)**2) + Lasso_lambda*np.sum(np.abs(w)) 
        
        print(loss)
         
        if np.max(np.abs(w_old - w)) < epsilon:
            return w 
        


def Plot_Lambda(d,n,k,epsilon,L_tolerance): 
    '''Function solves multiple Lasso regression problems using 
    different lambda values starting from a max lambda until a 
    lambda value that nearly all zero. Plots number of non-zeros vs 
    lambda values. Also plots the true positive rate vs false discovery 
    rate
    
    Inputs 
    
    d : `integer`
        Choose the number of features for the x-values.
    n : `integer`
        Choose the number of observations you want generate
    k : `integer`
        Number of non zero coefficient for synthetic data
    epsilon: `float`
        Convergence is achieved when the convergence criterion is smaller than
        tolerance of epsilon.
    L_tolerance: 'integer'
        Chooses the number of non-zero coefficients that need to be generated 
        before the algorith terminates
        
    Outputs 
    
    Plots number of non-zeros vs 
    lambda values. Also plots the true positive rate vs false discovery 
    rate
        
    ''' 
    
    X,Y =  Generate_data(d,n,k)
    
    lambd_max = Max_lambda(X,Y)
    
    ite = 1
    
    
    lambdas = []
    numb_nonzeros = []
    fdr = []
    tpr = []
    
    while True: 
        lambd = lambd_max/(1.5**ite)
        w = CoordinateDescent(X,Y,lambd,epsilon)
        
        num_nonz = np.count_nonzero(w)
        correctNonZ = np.count_nonzero(w[:k])
        incorrectNonZ = np.count_nonzero(w[k+1:])
        
        if num_nonz != 0:
            fdr.append(incorrectNonZ/num_nonz)
        else: 
            fdr.append(0)
        tpr.append(correctNonZ/k)
        lambdas.append(lambd)
        numb_nonzeros.append(num_nonz)
        
        if num_nonz > L_tolerance: 
            break 
        
        ite += 1
    
    # Plot lambda vs number of zeros
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    
    ax[0].set_title('Lambda')
    ax[0].plot(lambdas, numb_nonzeros)
    ax[0].set_xlabel('Lambda Values')
    ax[0].set_ylabel('Number of Non-zero Coefficients')
    ax[0].set_xscale('log')
    
    ax[1].set_title('TPR vs FDR')
    ax[1].plot(fdr, tpr)
    ax[1].set_xlabel('False Discovery Rate')
    ax[1].set_ylabel('True Positive Rate')
    
    
#############################################################
##### Lasso on real data 
#############################################################
    
def Read_data(trainfile = 'crime-train.txt', testfile = 'crime-test.txt'): 
    '''Function reads train and test data from a text file. It splits 
    it into Xtrain, Ytrain, Xtest, Ytest. Note this is particular for 
    Crime test from the following link: 
        
    http://archive.ics.uci.edu/ml/datasets/communities+and+crime
    
    Input 
    
    trainfile : txt file 
        Text file that includes the training features to analyze 
    testfile : txt file 
        Text file that includes the text features to analyze 
        
    Output
    
    xtrain : 'np.array'
        Includes all the x features to train the model 
    ytrain : 'np.array'
        Includes the column of y response for training of model 
    xtest : 'np.array'
        Includes all the x features to test the model 
    ytest : 'np.array'
        Includes the column of y response for testing of model
    FeatureNames : 'np.array'
        Incldues all the feature names 
    
    '''
    df_train = pd.read_table(trainfile)
    df_test = pd.read_table(testfile)
    
    xtrain = np.array(df_train.drop("ViolentCrimesPerPop", axis=1).to_numpy())
    ytrain = np.array([df_train["ViolentCrimesPerPop"].to_numpy()])
    
    xtest = np.array(df_test.drop("ViolentCrimesPerPop", axis=1).to_numpy())
    ytest = np.array([df_test["ViolentCrimesPerPop"].to_numpy()])
    
    FeatureNames = np.array(df_train.drop("ViolentCrimesPerPop", axis=1).columns)
    
    return xtrain, ytrain, xtest, ytest, FeatureNames 

def MeanSqrError(X,Y,w): 
    '''Calculates the mean square error for a given X,Y
    
    Input 
    
    X : `np.array`
        The feature space. A matrix with n rows of feature vectors, each with d
        features.
    Y : `np.array`
        A column vector of n model values.
    w : 'np.array'
        Trained coefficient array, estimate
        
    Output
    
    MeanSE : 'float'
        Calculated mean square error    
    '''
    #a = Y - np.dot(X, w)
    #return (a.T @ a)/len(Y)
    
    return np.sum((Y - w.dot(X.T))**2)/len(Y[0])
    

def A5CoordinateDesc(epsilon = 0.0001): 
    '''Performs Lasso Regression on crime data set. Plots the effect of lambda 
    to the number of non-zero coefficients. Also plots the coefficient 
    regularization as a function of the lambda
    
    Input 
    
    epsilon: 'float'
        Choose tolerance for the coordinate descent algorithm to converge 
        
    Output
        Three plots are produced. One plot shows the number of non-zero 
        coefficients vs Lambda. Second plot shows the coefficient progression 
        for 5 chosen coefficients vs lambda. Last plots the Mean Square Error 
        of the Train and Test sets vs Lambda.   
    '''
    
    # Read Data 
    xtrain, ytrain, xtest, ytest, FeatureNames = Read_data()
    
    d = xtrain.shape[1] # number of features
    
    # Initialize lambda and w values to start routine
    lambd = Max_lambda(xtrain,ytrain)
    w_prev = np.zeros(d)
    
    # Set up values to track
    lambdas, nonZeros, MSETrain, MSETest = [], [], [], []
    wagePct12t29, wpctWSocSec, wpctUrban, wagePct65up, whouseholdsize = [], [], [], [], []
    
    # Track index of the following features 
    
    idxagePct12t29 = np.where(FeatureNames == 'agePct12t29')[0][0]
    idxpctWSocSec = np.where(FeatureNames == 'pctWSocSec')[0][0] 
    idxpctUrban = np.where(FeatureNames == 'pctUrban')[0][0] 
    idxagePct65up = np.where(FeatureNames == 'agePct65up')[0][0] 
    idxhouseholdsize = np.where(FeatureNames == 'householdsize')[0][0]
        
    while lambd > 0.01:
        w = CoordinateDescent(xtrain,ytrain,lambd,epsilon,w_init = w_prev)
        
        # Track coefficent values 
        wagePct12t29.append(w[idxagePct12t29])
        wpctWSocSec.append(w[idxpctWSocSec])
        wpctUrban.append(w[idxpctUrban])
        wagePct65up.append(w[idxagePct65up])
        whouseholdsize.append(w[idxhouseholdsize])
        
        # Track Other metrics 
        lambdas.append(lambd)
        nonZeros.append(np.count_nonzero(w))
        MSETrain.append(MeanSqrError(xtrain,ytrain,w))
        MSETest.append(MeanSqrError(xtest,ytest,w))
        
        # update values for next iteration 
        w_prev = w
        lambd = lambd /2.0 # decrease each lambda by 2 
        
    
    fig, ax1 = plt.subplots(1, 2, figsize=(7, 7))
    ax1[0].set_title('Lambda')
    ax1[0].plot(lambdas, nonZeros)
    ax1[0].set_xlabel('Lambda')
    ax1[0].set_ylabel('Number of Non-zero Coefficients')
    ax1[0].set_xscale('log')
    
    ax1[1].set_title('Mean Square Error')
    ax1[1].plot(lambdas, MSETrain, 'r-o', label='Train')
    ax1[1].set_xlabel('Lambda')
    ax1[1].set_ylabel('MSE')
    ax1[1].set_xscale('log')
    
    ax1[1].set_title('Mean Square Error')
    ax1[1].plot(lambdas, MSETest, 'b-o', label='Test')
    ax1[1].set_xlabel('Lambda')
    ax1[1].set_ylabel('MSE')
    ax1[1].set_xscale('log')
    ax1[1].legend()
    
    W_s =  [wagePct12t29, wpctWSocSec, wpctUrban, wagePct65up, whouseholdsize]
    label_names = ['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 
                   'householdsize']
    
    fig, ax2 = plt.subplots()
    ax2.set_title('Mean Square Error')
    ax2.set_xscale('log')
    ax2.set_xlabel('Lambda')
    ax2.set_ylabel('Coefficient Value')
    
    for feature, labels in zip(W_s,label_names): 
        ax2.plot(lambdas, feature, label=labels)
    ax2.legend()
    
def A5_lamb30(lambd = 30, epsilon = 0.0001): 
    '''This function explores the max and min coefficients for a lambda
    value of 30.
    
    Inputs
    
    lambd: 'float'
        Choose lasso regularization term, default of 30
    
    epsilon: 'float'
        Choose tolerance for the coordinate descent algorithm to converge
    
    Output: 
        
    Max and Min values of the coefficients
    '''
    
    xtrain, ytrain, xtest, ytest, FeatureNames = Read_data()
    
    w = CoordinateDescent(xtrain,ytrain,lambd,epsilon)
    
    Max_feature = FeatureNames[np.argmax(w)]
    Min_feature = FeatureNames[np.argmin(w)]
    
    print('The feature with the most positive coefficient is %s \n' % Max_feature)
    print('The feature with the most negative coefficient is %s \n' % Min_feature)
       
    
if __name__ == "__main__":
     Plot_Lambda(d = 1000,n = 500,k = 100 ,epsilon = 0.001,L_tolerance = 990)
     A5CoordinateDesc(epsilon = 0.0001) 
     A5_lamb30()         

        
    
    
    
    
    


    
    
    
    
    
    
    
    
    
        
