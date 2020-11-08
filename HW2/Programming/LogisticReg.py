# -*- coding: utf-8 -*-
"""
This routine implements logitisc regression using the MNIST dataset. I 
implement both traditional gradient descent and stochastic gradient 
descent to view their differences via plots. 

@author: salez
"""

import numpy as np
from mnist.loader import MNIST
import matplotlib.pyplot as plt 

# global path varible
mnistpath = r'C:\Users\salez\Documents\MISE Work\CSE 546\Homework\HW2\Programming\mnist'

def binaryData(values, data, labelsData): 
    '''Function takes in the values that the user wants to use for logistic 
    regression from the x data and the y labels. It returns only the data 
    and labels corresponding to the labels
    
    Input
    -----
    values - 'list or tuple'
        Values the user wants to use as a subset of the entire dataset 
        and labels 
        
    data - 'np.array'
        nxd array containing all independent features 
        
    labels_data - 'np.array'
        nx1 array containing the corresponding labels of the data array
        
    Output
    ------    
    Subset of data and labels_data corresponding values selected
    
    '''
    
    idxVal = np.logical_or(labelsData == values[0], 
                           labelsData == values[1])
    
    dataX = data[idxVal]
    labelY = labelsData[idxVal]
    
    return dataX, labelY

def binarycode(labels, values, codedVals): 
    '''For a labeled binary set of data, transfor the values into encoded 
    values (e.g. -1,1)
    
    Input
    -----
    labels - 'np.array'
        nx1 array containing the corresponding labels of the data array
        
    values - 'list or tuple'
        Values that need to be encoded 
        
    encodedVals - 'list or tuple'
        Desired encding that user would like to use (e.g. -1,1)
        
    Output
    ------
    
    encodedLab - 'np.array'
        list of encoded labels    
    '''    
    
    labels = labels.astype(int)
    
    for i in range(len(values)):
        labels[labels == values[i]] = codedVals[i]        
    return labels

def EncodeMnist(values = (2,7), encodingVal = (-1,1), filepath = mnistpath):
    '''Function loads the mnist data, filters out anything outside of 
    values indicated and encodes the labels with the encoded values indicated
    
    Input
    -----
    filepath - 'string'
        Currently set to a global variable where the location of the mnist 
        path
    
    values - 'list or tuple'
        Two values(0-9) to use for selecting binary values
        
    Output
    ------ 
    X_trainC - 'np.array'
        training data set for the selected values 
        
    X_testC - 'np.array'
        test data set for the selected values 
        
    Y_train_lab - 'np.array'
        encoded values for the Y Values of the training set 
        
    Y_test_lab - 'np.array'
        encoded values for the Y Values of the training set 
    '''
    
    # Load the MNIST Dataset 
    mndata = MNIST(filepath)
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0 # normalize dataset 
    X_test = X_test/255.0 
    
    XtrainC, trainLab = binaryData(values, X_train, labels_train)
    XtestC, testLab = binaryData(values, X_test, labels_test)
    
    Ytrainlab = binarycode(trainLab, values, encodingVal)
    Ytestlab = binarycode(testLab, values, encodingVal)
    
    return XtrainC, Ytrainlab, XtestC, Ytestlab

def LogitRegLoss(xData,ylabel,w,b,lambd): 
    '''Loss function for the logitic regression with regularization
    
    Input
    -----
    xData - 'np.array'
        nxd array data array depicting the features 
        
    ylabel - 'np.array'
        nx1 array depicting the labels of 1 or -1 
        
    w - 'np.array'
        current value of weights are being learning
        
    b - 'float'
        offset value
        
    lambda - 'float'
        regularization parameter
        
    Output
    ------
    
    Jw_b - 'float'
        Current loss value for the logisitic log likelihood
    '''
    
    mu = mu_i(xData,ylabel,w,b) # See mu_i function
    
    loss = np.mean(np.log(1/mu)) + lambd*w.dot(w.T)
    
    return loss

def mu_i(xData,ylabel,w,b): 
    '''Substitution for the exponential term in the logistic regression 
    loss function
    
    Input
    -----
    xData - 'np.array'
        nxd array data array depicting the features 
        
    ylabel - 'np.array'
        nx1 array depicting the labels of 1 or -1 
        
    w - 'np.array'
        current value of weights are being learning
        
    b - 'float'
        offset value
    
    Output
    ------
    mu - 'np.array'
       Current value for exponential term of logitics loss function
    '''
    
    mu = 1/(1 + np.exp(-1*ylabel*(b + xData.dot(w.T))))
    
    return mu

def grad_w(xData,ylabel,w,b,lambd): 
    '''Gradient of logisitc loss function with respect to the weights
    
    Input
    -----
    xData - 'np.array'
        nxd array data array depicting the features 
        
    ylabel - 'np.array'
        nx1 array depicting the labels of 1 or -1 
        
    w - 'np.array'
        current value of weights 
        
    b - 'float'
        offset value
        
    lambda - 'float'
        regularization parameter
        
    Output
    ------
    j_w - 'np.array'
       Gradient of logisitc loss function with respect to the weights
    '''
    n,d = xData.shape
    
    muterm = mu_i(xData,ylabel,w,b)-1
    
    # Reshape 1-D arrays to perform row wise multiplication 
    term1 = ylabel.reshape(-1,1)*xData*muterm.reshape(-1,1)
    
    term2 = 2*lambd*w
    
    j_w = np.mean(term1, axis =0) + term2 
    
    return j_w 

def grad_b(xData,ylabel,w,b): 
    '''Gradient of logisitc loss function with respect to the offset
    
    Input
    -----
    xData - 'np.array'
        nxd array data array depicting the features 
        
    ylabel - 'np.array'
        nx1 array depicting the labels of 1 or -1 
        
    w - 'np.array'
        current value of weights 
        
    b - 'float'
        offset value
        
        
    Output
    ------
    j_b - 'float'
       Gradient of logisitc loss function with respect to the offset
    '''
    
    muterm = mu_i(xData,ylabel,w,b)-1
    
    j_b = np.mean(ylabel*muterm)
    
    return j_b 

def missclass(xData, ylabel, w, b): 
    '''Function first classifies obersvations based on current 
    weights and offset. The misclassification is then computed when 
    compared to the actual variables 
    
    Input
    -----
    xData - 'np.array'
        nxd array data array depicting the features 
        
    ylabel - 'np.array'
        nx1 array depicting the labels of 1 or -1 
        
    w - 'np.array'
        current value of weights 
        
    b - 'float'
        offset 
        
    Output
    ------
    
    miss_C_error - 'float'
        Missclassification error
    ''' 
    
    classify = np.sign(b + xData.dot(w.T))
    
    C_error = (len(ylabel) - np.sum(classify == ylabel))/len(ylabel)
    
    return C_error
    

def Gradient_descent(XtrainC, Ytrainlab, XtestC, Ytestlab, lambd, step, 
                      maxiter = 200, batch = 1, stochastic = False): 
    '''Performs gradient descent (Stochastic Grad Descent Optional)
    and tracks loss function and classification error as a function 
    iteration
    
    Input
    -----
    X_trainC - 'np.array'
        training data set for the selected values 
        
    X_testC - 'np.array'
        test data set for the selected values 
        
    Y_train_lab - 'np.array'
        encoded values for the Y Values of the training set 
        
    Y_test_lab - 'np.array'
        encoded values for the Y Values of the training set 
        
    lambd - 'float'
        regularization parameter
        
    step - 'float'
        step size for gradient descent
        
    epsilon - 'float'
        Convergence is achieved when the convergence criterion is smaller than
        tolerance of epsilon.
        
    maxiter - 'int'
        max amount of iteration performed by gradient descent 
        before it terminates 
    
    stochastic - 'boolean', optional 
        Optional setting for performing stochastic gradient descent
        
    Output
    ------
    
    LossTr - 'np.array'
        Array that tracks the loss function as a function of each 
        iteration for the training set
        
    LossTe - 'np.array'
        Array that tracks the loss function as a function of each 
        iteration for the test set
        
    CurStep - 'np.array'
        Array that tracks iteration, will be used for plotting purposes
        
    error_train - 'np.array'
        Tracks the missclassification error after each gradient 
        descent iteration for the training set
        
    error_test - 'np.array'
        Tracks the missclassification error after each gradient 
        descent iteration for the test set
        
    w_step - 'np.array'
        Tracks the 
    '''
    
    n,d = XtrainC.shape # number of features 
    
    # initialize weights for w and b 
    
    w = np.zeros(d)
    b = 0 
    
    LossTr,LossTe,CurStep, error_train, error_test, w_step, b_step = [], [], [], [], [], [], []
    
    for i in range(maxiter): 
        
        LossTr.append(LogitRegLoss(XtrainC,Ytrainlab,w,b,lambd))
        LossTe.append(LogitRegLoss(XtestC,Ytestlab,w,b,lambd))
        CurStep.append(i)
        error_train.append(missclass(XtrainC, Ytrainlab, w, b))
        error_test.append(missclass(XtestC, Ytestlab, w, b))
        w_step.append(w)
        b_step.append(b)
             
        
        if stochastic: 
            idx = np.random.permutation(n)[:batch]
            w = w - step*grad_w(XtrainC[idx],Ytrainlab[idx],w,b,lambd)
            b  = b - step*grad_b(XtrainC[idx],Ytrainlab[idx],w,b)
            
        else: 
            w = w - step*grad_w(XtrainC,Ytrainlab,w,b,lambd)
            b  = b - step*grad_b(XtrainC,Ytrainlab,w,b)
            
        
    return LossTr, LossTe, CurStep, error_train, error_test
            
def Plots(lambd, step, maxiter = 200, batch = 1, stochastic = False): 
    '''Two plots are created here. 1) Plots of the loss of function for 
    both training and test sets as a function of iteration
    
    2) MissClassification Error 
    
    Input
    -----
    
    lambd - 'float'
        regularization parameter
        
    step - 'float'
        step size for gradient descent
        
        
    Output
    ------ 
    
    Plots
    
    '''
    
    XtrainC, Ytrainlab, XtestC, Ytestlab = EncodeMnist() 
    
    LossTr, LossTe, CurStep, error_train, error_test = Gradient_descent(XtrainC, Ytrainlab, XtestC, Ytestlab, lambd, step, maxiter, batch, stochastic)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    
    ax[0].set_title('Loss Function')
    ax[0].plot(CurStep, LossTr, label = 'Training Loss')
    ax[0].plot(CurStep, LossTe, label = 'Test Loss')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Log Loss')
    ax[0].legend()
    
    
    ax[1].set_title('Missclassification Error')
    ax[1].plot(CurStep, error_train, label = 'Training Error')
    ax[1].plot(CurStep, error_test, label = 'Test Error')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Missclassification Error')
    ax[1].legend()
    
    return None

# Plot parts b,c,d 

if __name__ == "__main__":
     Plots(lambd = 0.1, step = 0.01)   
     Plots(lambd=0.1, step=0.01, maxiter = 200, batch = 1, stochastic = True)
     Plots(lambd=0.1, step=0.01, maxiter = 200, batch = 100, stochastic = True)


    

    
        

    

    
        
    
        
    
    