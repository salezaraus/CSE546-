# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:03:11 2020

@author: salez
"""


import numpy as np
import matplotlib.pyplot as plt
from mnist.loader import MNIST
import torch
import torch.nn as nn

# global path varible
mnistpath = r'C:\Users\salez\Documents\MISE Work\CSE 546\Homework\HW3\Programming\mnist'

def MnistData(filepath = mnistpath): 
    '''Loads Mnist and converts it into a pytorch tensor 
    
    Input
    -----
    
    filepath - 'string'
        Filepath for the location of the MNIST dataset
        
    Output
    ------
    
    Xtrain - 'torch.tensor'
        Training set for the Mnist dataset 
        
    trainLabels - 'torch.tensor'
        train labels
        
    Xtest - 'torch.tensor'
        Test set for the Mnist dataset
        
    testLabels - 'torch.tensor'
        test labels
    '''
    
    # Load the MNIST Dataset 
    mndata = MNIST(filepath)
    Xtrain, trainLabels = map(torch.tensor, mndata.load_training())
    Xtest, testLabels  = map(torch.tensor, mndata.load_testing())
    Xtrain = Xtrain/255.0 # normalize dataset 
    Xtest = Xtest/255.0
    
    Xtrain = Xtrain.type(torch.DoubleTensor)
    Xtest = Xtest.type(torch.DoubleTensor) 
    
    
    return Xtrain, trainLabels, Xtest, testLabels

def modelShallow(x, w0, b0, w1, b1): 
    '''
    Creates a 'shallow' neural network model to evaluate. Relu is used 
    as the activation function. 

    Parameters
    ----------
    x: 'torch.tensor'
        Training set for the Mnist dataset 
    w0 : 'torch.tensor'
        Initial layer weights of neural network
    b0 : 'torch.tensor'
        Initial layer weight offset 
    w1 : 'torch.tensor'
        1st layer weights of neural network 
    b1 : 'torch.tensor'
        1st layer weights of neural network 

    Returns
    -------
    fit: 'model function'
        fitted model

    '''
    
    # define activation function
    sigma = nn.ReLU()
    
    return sigma(x@w0 + b0) @ w1 + b1

def modelDeep(x, w0, b0, w1, b1, w2, b2): 
    '''
    Creates a 'deep' neural network model to evaluate. Relu is used 
    as the activation function. 

    Parameters
    ----------
    x: 'torch.tensor'
        Training set for the Mnist dataset 
    w0 : 'torch.tensor'
        Initial layer weights of neural network
    b0 : 'torch.tensor'
        Initial layer weight offset 
    w1 : 'torch.tensor'
        1st layer weights of neural network 
    b1 : 'torch.tensor'
        1st layer weights of neural network     
    w2 : 'torch.tensor'
        2nd layer weights of neural network 
    b2 : 'torch.tensor'
        2nd layer weights of neural network 

    Returns
    -------
    fit: 'model function'
        fitted model

    '''
    
    # define activation function
    sigma = nn.ReLU()
    term1 = x @ w0 + b0
    term2 = sigma(term1) @ w1 + b1
    
    return sigma(term2) @ w2 + b2

def Error(y_hat, labels):
    '''
    Calculates current error for current epoch calculation

    Parameters
    ----------
    y_hat: 'torch.tensor'
        predictions made by the model
        
    labels - 'torch.tensor'
        labels to compare against 

    Returns
    -------
    Error: 'float'
        Returns error in decimal 

    '''
    
    return np.sum((y_hat != labels).numpy())/len(labels)
    
def W_init(model = 'Shallow'):
    '''
    Initiailizes weights to begin the model training. Depending on model 
    specification, shallow or deep model parameters will be returned
    
    Parameters
    ----------
    model : 'string', optional
        Describes model type. The default is 'Shallow'.

    Returns
    -------
    wX: 'torch.tensor'
        Initial weights are returned, depending on model type 
        X = 0-1 for Shallow or X = 0-2 for Deep model 
       
    bX: 'torch.tensor'
        Initial weight offsets are returned, depending on model type 
        X = 0-1 for Shallow or X = 0-2 for Deep model 

    '''
    
    if model == 'Shallow': 
        
        n, d_in, h_out, k = 28, 28**2, 64, 10
        al_init = 1/np.sqrt(h_out)
        
        w0 = torch.zeros(d_in, h_out, dtype = torch.double)
        w0.uniform_(-al_init, al_init).requires_grad_()

        b0 = torch.zeros(1, h_out, dtype = torch.double)
        b0.uniform_(-al_init, al_init).requires_grad_()
        
        w1 = torch.zeros(h_out, k, dtype = torch.double)
        w1.uniform_(-al_init, al_init).requires_grad_()
        
        b1 = torch.zeros(1, k, dtype = torch.double)
        b1.uniform_(-al_init, al_init).requires_grad_()
        
        return w0, b0, w1, b1
    
    else: 
        
        n, d_in, h0, h1, k = 28, 28**2, 32, 32, 10

        al_init0 = 1/np.sqrt(h0)
        al_init1 = 1/np.sqrt(h1)
        
        w0 = torch.zeros(d_in, h0, dtype = torch.double)
        w0.uniform_(-al_init0, al_init0).requires_grad_()

        b0 = torch.zeros(1, h0, dtype = torch.double)
        b0.uniform_(-al_init0, al_init0).requires_grad_()
        
        w1 = torch.zeros(h0, h1, dtype = torch.double)
        w1.uniform_(-al_init0, al_init0).requires_grad_()
        
        b1 = torch.zeros(1, h1, dtype = torch.double)
        b1.uniform_(-al_init0, al_init0).requires_grad_()
        
        w2 = torch.zeros(h1, k, dtype = torch.double)
        w2.uniform_(-al_init1, al_init1).requires_grad_()
        
        b2 = torch.zeros(1, k, dtype = torch.double)
        b2.uniform_(-al_init1, al_init1).requires_grad_()
        
        return w0, b0, w1, b1, w2, b2

def trainModel(x, labels, batchSize = 10, epoch = 1000, epsilon = 0.01, 
               learnRate = 0.05, model = "Shallow"): 
    '''
    Trains the neural network model until the max number of epochs 
    are performed or the convergence criteria is achieved. Stochastic
    gradient descent is performed. 
    
    Parameters
    ----------
    x: 'torch.tensor'
        Training set for the Mnist dataset 
        
    labels: 'torch.tensor'
        labels to compare against
        
    batchSize: 'int'
        batches the data set to perform 
        
    epochs: 'int'
        Max number of epochs performed for training model 
        
    epsilon: 'float'
        Convergence criteria for missclassification error 
        
    learnRate: 'float'
        Learning rate for gradient descent
        
    model: 'str'
        Type of model performed for training (e.g. Shallow or Deep)
        
    Returns
    -------
    
    loss: 'list'
        list of loss progression of model 
        
    error: 'list'
        list of misclassification errors 
        
    weights: 'torch.tensor'
        Final weights of layers based on convergence criteria
    
    '''
    
    indx = np.arange(len(x))
    errors = [] 
    losses = [] 
    
    
    if model == 'Shallow': 
        
        # initialize weights
        w0, b0, w1, b1 = W_init() 
        
        optimizer = torch.optim.Adam([w0, b0, w1, b1], lr=learnRate)
        
        for i in range(epoch): 
            rand_idxs = np.random.permutation(indx)
            batchIndx = np.split(rand_idxs, batchSize)
            
            print(i)
            
            for batch in batchIndx: 
                xBatch = x[batch]
                labBatch = labels[batch]
                
                y_hat = modelShallow(xBatch, w0, b0, w1, b1)
                
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(y_hat, labBatch)
                loss.backward() 
                optimizer.step() 
                
                # calculate error
                
                y_all = modelShallow(x, w0, b0, w1, b1)
                pred = torch.argmax(y_all, dim=1)
                error = Error(pred, labels)
                errors.append(error)
                losses.append(float(loss))
                
                print('loss: ' ,float(loss))
                print(error)
                
                # Check for conversions
                if error < epsilon: 
                    return errors, losses, w0, b0, w1, b1
                
        return errors, losses, w0, b0, w1, b1
    
    else: 
        
        # initialize weights
        w0, b0, w1, b1, w2, b2 = W_init(model = 'Deep') 
        
        optimizer = torch.optim.Adam([w0, b0, w1, b1, w2, b2], lr=learnRate)
        
        
        for i in range(epoch): 
            rand_idxs = np.random.permutation(indx)
            batchIndx = np.split(rand_idxs, batchSize)
            
            print(i)
            
            for batch in batchIndx: 
                xBatch = x[batch]
                labBatch = labels[batch]
                
                y_hat = modelDeep(xBatch, w0, b0, w1, b1, w2, b2)
                
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(y_hat, labBatch)
                loss.backward() 
                optimizer.step() 
                
                # calculate error
                
                y_all = modelDeep(x, w0, b0, w1, b1, w2, b2)
                pred = torch.argmax(y_all, dim=1)
                error = Error(pred, labels)
                errors.append(error)
                losses.append(float(loss))
                
                print('loss: ' ,float(loss))
                print(error)
                
                # Check for conversions
                if error < epsilon: 
                    return errors, losses, w0, b0, w1, b1, w2, b2
                
        return errors, losses, w0, b0, w1, b1, w2, b2
                
def plotErrors(losses, model_title ='Shallow Network, SGD, Batch Size = 10'): 
    """Plots given losses as a function of epoch.
    Parameters
    ---------
    losses: `list`
        Loss per epoch
        
    model_title: 'str'
        Title for graph 
        
    """
    fig, axes = plt.subplots()

    x = np.arange(len(losses))

    axes.plot(x, losses)
    axes.set_ylabel("Loss (cross entropy)")
    axes.set_xlabel("Number of iterations")
    axes.set_title(model_title) 

    plt.show()          

    return None      
            
def A6a(learnRate = 0.05, batchSize = 10):
    '''
    Trains model for a shallow network and plots the loss vs epoch. 
    Train and Test accuracies are then reported

    Parameters
    ----------
    learnRate: 'float'
        Learning rate for gradient descent
        
    batchSize: 'int'
        batches the data set to perform 

    '''

    Xtrain, trainLabels, Xtest, testLabels = MnistData()

    errors, losses, w0, b0, w1, b1 = trainModel(Xtrain, trainLabels, 
                                     batchSize = batchSize, 
                                     epoch = 1000, epsilon = 0.01, 
                                     learnRate = learnRate, model = "Shallow")
    
    yTpred = modelShallow(Xtest, w0, b0, w1, b1)
    pred = torch.argmax(yTpred, dim=1)
    Tloss = torch.nn.functional.cross_entropy(yTpred, testLabels)
    TestError = Error(pred, testLabels)
    
    print(f"Training accuracy: {1 - errors[-1]}")
    print(f"Training loss: {losses[-1]}")

    print(f"Accuracy (test) {1 - TestError }")
    print(f"Loss (test): {float(Tloss)}")

    plotErrors(losses) 
    
def A6b(learnRate = 0.05, batchSize = 10):
    '''
    Trains model for a deep network and plots the loss vs epoch. 
    Train and Test accuracies are then reported

    Parameters
    ----------
    learnRate: 'float'
        Learning rate for gradient descent
        
    batchSize: 'int'
        batches the data set to perform 

    '''

    Xtrain, trainLabels, Xtest, testLabels = MnistData()

    errors, losses, w0, b0, w1, b1, w2, b2 = trainModel(Xtrain, trainLabels, 
                                     batchSize = batchSize, 
                                     epoch = 1000, epsilon = 0.01, 
                                     learnRate = learnRate, model = "Deep")
    
    yTpred = modelDeep(Xtest, w0, b0, w1, b1, w2, b2)
    pred = torch.argmax(yTpred, dim=1)
    Tloss = torch.nn.functional.cross_entropy(yTpred, testLabels)
    TestError = Error(pred, testLabels)
    
    print(f"Training accuracy: {1 - errors[-1]}")
    print(f"Training loss: {losses[-1]}")

    print(f"Accuracy (test) {1 - TestError }")
    print(f"Loss (test): {float(Tloss)}")

    plotErrors(losses, model_title ='Deep Network SGD, Batch Size = 10') 

if __name__ == "__main__":   
    #A6a()
    A6b() 