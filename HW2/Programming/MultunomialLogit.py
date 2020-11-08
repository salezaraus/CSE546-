# -*- coding: utf-8 -*-
"""
This routine implements a multinomial logitistic regression procedure 
using pytorch. For comparison, both a softmax and ridge regression loss 
function are explored to see their differences on an MNIST data set with 
k = 10 classes. 


@author: salez
"""

import numpy as np
import matplotlib.pyplot as plt
from mnist.loader import MNIST
import torch

# global path varible
mnistpath = r'C:\Users\salez\Documents\MISE Work\CSE 546\Homework\HW2\Programming\mnist'

def MnistData(filepath = mnistpath): 
    '''Loads Mnist and one hot encodes Y variable for both train and test 
    variables
    
    Input
    -----
    
    filepath - 'string'
        Filepath for the location of the MNIST dataset
        
    Output
    ------
    
    Xtrain - 'np.array'
        Training set for the Mnist dataset 
        
    Ytrain - 'np.array'
        One-hot encoded Y trained labels
        
    Xtest - 'np.array'
        Training set for the Mnist dataset 
        
    Ytest - 'np.array'
        One-hot encoded Y trained labels
    '''
    
    # Load the MNIST Dataset 
    mndata = MNIST(filepath)
    Xtrain, labels_train = map(np.array, mndata.load_training())
    Xtest, labels_test = map(np.array, mndata.load_testing())
    Xtrain = Xtrain/255.0 # normalize dataset 
    Xtest = Xtest/255.0
    
    n,d = Xtrain.shape
    k = labels_train.max() + 1 # number of classes
    m = len(labels_test) # number of test observations
    
    Ytrain = np.zeros((n,k))
    Ytrain[np.arange(n), labels_train] = 1
    
    
    
    Ytest = np.zeros((m,k))
    Ytest[np.arange(m), labels_test] = 1
    
    return Xtrain, Ytrain, labels_train,  Xtest, Ytest, labels_test

def PredAcc(w_hat, X_data,Y_labels): 
    '''

    Parameters
    ----------
    w_hat : 'tensor'
        Current w_hat iteration
    X_data : 'tensor'
        X set for prediction
    Y_label: 'tensor'
        Y Labels 

    Returns
    -------
    Pred : 'tensor'
        List of predictions

    '''
    
    n = X_data.shape[0]
    
    E_j = torch.eye(w_hat.shape[1], dtype=torch.float64) # Identity Matrix
    
    # Predict based on argmax eJ*W_T*X_i
    Pred = torch.matmul(torch.matmul(X_data, w_hat), E_j)
    Pred = torch.argmax(Pred,dim=1)
    
    Acc = (torch.sum(Pred == Y_labels)/float(n)).item()
    
    return Acc


def SGD(epochs = 500, batch = 10, step = 0.05, maxiter = 200):
    '''
    Performs stochastic gradient descent for both ridge regression 
    and multinomial logistic regression on the MNIST Data. 
    
    
    Input
    -----
    epoch - 'int'
        Number of epochs to loop through
        
    batch - 'int'
        Batch size for SGD 
        
    step - 'float'
        Step size for SGD
        
    maxiter - 'int'
        max amount of iteration performed by gradient descent 
        before it terminates 
           
    Output
    -------
        
    AccuRidTrain  - 'list'
        Accuracy of Ridge Regression for the training data
        
    AccuMLTrain 'list'
        Accuracy of Multinomial Logistic for the training data 
        
    AccuRidTest  - 'list'
        Accuracy of Ridge Regression for the training data
        
    AccuMLTest 'list'
        Accuracy of Multinomial Logistic for the training data 

    '''
    
    # Generate Data
    Xtrain, Ytrain, labels_train, Xtest, Ytest, labels_test = MnistData() 
    
    n,d = Xtrain.shape
    
    k = Ytrain.shape[1]
    
    # convert data into tensor-form
    W_Mlogit = torch.zeros(d, k, requires_grad=True, dtype=torch.float64)
    W_MRidge = torch.zeros(d, k, requires_grad=True, dtype=torch.float64)
    
    XTrainTen = torch.from_numpy(Xtrain)
    YTrainTen = torch.from_numpy(Ytrain)
    
    XTestTen = torch.from_numpy(Xtest)
    
    Y_tr_labels = torch.tensor(labels_train, dtype=torch.int64)
    Y_te_labels = torch.tensor(labels_test, dtype=torch.int64)
    
    # Track Accuracies 
    AccMlogitTr, AccMlogitTe, AccMRidgeTr, AccMRidgeTe = [], [], [], []
    
    for epoch in range(epochs): 
        idx = np.random.permutation(n)[:batch] 
        print(epoch)
        for i in range(maxiter):
            
            X_batch, Y_batch, ylab = XTrainTen[idx], YTrainTen[idx,:], Y_tr_labels[idx]
            
            y_hat = torch.matmul(X_batch, W_Mlogit)
            y_tilde = torch.matmul(X_batch, W_MRidge)
                                   
            # cross entropy combines softmax calculation with NLLLoss
            MLogit_loss = torch.nn.functional.cross_entropy(y_hat, ylab)
            # MSE loss for ridge regression
            MRidge_loss = torch.nn.functional.mse_loss(y_tilde, Y_batch)
            
            # computes derivatives of the loss with respect to W
            MLogit_loss.backward()
            MRidge_loss.backward()
            
            # gradient descent update
            W_Mlogit.data = W_Mlogit.data - step * W_Mlogit.grad
            W_MRidge.data = W_MRidge.data - step * W_MRidge.grad
            
            # .backward() accumulates gradients into W.grad instead
            # of overwriting, so we need to zero out the weights
            W_Mlogit.grad.zero_()
            W_MRidge.grad.zero_()
            
        PredLogitTr = PredAcc(W_Mlogit, XTrainTen, Y_tr_labels)
        AccMlogitTr.append(PredLogitTr)
        
        PredLogitTe = PredAcc(W_Mlogit, XTestTen, Y_te_labels)
        AccMlogitTe.append(PredLogitTe)
        
        PredMRidgeTr = PredAcc(W_MRidge, XTrainTen, Y_tr_labels)
        AccMRidgeTr.append(PredMRidgeTr)
        
        PredMRidgeTe = PredAcc(W_MRidge, XTestTen, Y_te_labels)
        AccMRidgeTe.append(PredMRidgeTe)
        
    return AccMlogitTr, AccMlogitTe, AccMRidgeTr, AccMRidgeTe


def Plots(epochs = 500):
    '''
    Parameters
    ----------
    epochs : 'int', optional
        DESCRIPTION. The default is 500.

    Returns
    -------
    Plots for accuracy of Multinomial Logit Regression and Ridge Regression 
    as a function of epochs. 
    '''
    
    AccMlogitTr, AccMlogitTe, AccMRidgeTr, AccMRidgeTe = SGD()

    # Multinomial Logistic Regression
    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, epochs+1), AccMlogitTr, label='Train set')
    ax1.plot(range(1, epochs+1), AccMlogitTe, label='Test set')
    ax1.set_title('Multinomial Logistic Regression w/ SGD')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.legend()
            
    # Ridge Regression
    fig2, ax2 = plt.subplots()
    ax2.plot(range(1, epochs+1), AccMRidgeTr, label='Train set')
    ax2.plot(range(1, epochs+1), AccMRidgeTe, label='Test set')
    ax2.set_title('Ridge Regression w/ SGD')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.legend()
    
    return None

# Plot parts c
if __name__ == "__main__":
     Plots()

    

        
        
        
        
            
        
    
    
    
        

        
    
    