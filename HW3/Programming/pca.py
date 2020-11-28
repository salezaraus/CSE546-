# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 12:42:15 2020

@author: salez
"""

import numpy as np
import matplotlib.pyplot as plt
from mnist.loader import MNIST

# global path varible
mnistpath = r'C:\Users\salez\Documents\MISE Work\CSE 546\Homework\HW3\Programming\mnist'

def MnistData(filepath = mnistpath): 
    '''Loads Mnist and converts it into a np.array
    
    Input
    -----
    
    filepath - 'string'
        Filepath for the location of the MNIST dataset
        
    Output
    ------    
    Xtrain - 'np.array'
        Training set for the Mnist dataset      
    
    Xtest - 'np.array'
        Test set for the Mnist dataset
        
    '''
    
    # Load the MNIST Dataset 
    mndata = MNIST(filepath)
    X, Labels = map(np.array, mndata.load_training())
    
    Xtrain = X[:50000]
    Xtest = X[50000:] 
    
    Xtrain = Xtrain/255.0 # normalize dataset 
    Xtest = Xtest/255.0
    
    return Xtrain, Labels, Xtest

def pca(): 
    '''
    Performs PCA on MNIST data set. First, Eigenvalues are calcualted for the 
    SIGMA sample covariance matrix and prints out the eigenvalues of the 
    indicated ith values. 

    Returns
    -------
    None.

    '''
    
    Xtrain, Labels, Xtest = MnistData() 
    
    n, d = Xtrain.shape
    
    # Compute Mean Row mu and Sigma Sample covariance matrix
    
    mu = np.dot(Xtrain.T, np.ones((n, 1)))/n
    B = Xtrain - np.dot(np.ones((n, 1)), mu.T)
    Sigma = B.T.dot(B)/n
    
    eigenval, eigenvec = np.linalg.eigh(Sigma)
    
    # Sort Eigenvalues and Eigenvectors
    eigenval = eigenval[eigenval.argsort()[::-1]]
    eigenvec = eigenvec[:, np.argsort(eigenval)]
    TotEigenSum = np.sum(eigenval)
   
    
    # Print out part a) 
    for i in (1, 2, 10, 30, 50):
        print(f"Eigenvalue lambda_{i}: {eigenval[i-1]}")
    print(f"Sum of eigenvalues: {TotEigenSum}")
    
    prin_comp = 100 
    trainErr, testErr, EigenVar = [0]*prin_comp,[0]*prin_comp,[0]*prin_comp
    eigenSum = 0
    
    for k in np.arange(prin_comp): 
        trainErr[k] = ReconstructError(Xtrain, eigenvec, mu, k)
        testErr[k] = ReconstructError(Xtest, eigenvec, mu, k)
        eigenSum += eigenval[k]
        EigenVar[k] = 1-(eigenSum/TotEigenSum)
        
    plotErrors(trainErr, testErr, EigenVar)
    
    PlotKEigen(eigenvec)
    
    PlotDigit(2, Xtrain, Labels, eigenvec, mu)
    PlotDigit(6, Xtrain, Labels, eigenvec, mu)
    PlotDigit(7, Xtrain, Labels, eigenvec, mu)
        
    return None
    
def ReconstructError(data, eigenvectors, mu, k): 
    '''
    Calculates the mean square reconstruction error given data points 
    and the eivenvectors. 

    Parameters
    ----------
    data : 'np.array'
        Data set to calculate reconstruction error 
    eigenvectors : 'np.array'
        array containing the eigenvectors of the covariance matrix 
    mu: 'np.array'
        average of the training samples
    k: 'int'
        number of eigenvectors that the user wishes to perform for 
        error calculation

    Returns
    -------
    Error: 'float'
        mean square reconstruction error

    '''
    
    Vk = eigenvectors[:, :k]
    proj_matrix = np.dot(Vk, Vk.T)
    
    pred = mu.T + np.dot((data - mu.T),proj_matrix)
    
    return np.mean(np.linalg.norm(data-pred,axis = 1)**2)

def plotErrors(TrainError, TestError, EigenVar): 
    '''
    Plots the mean square error for reconstruction for the Training and Test 
    sets. Also plots the Eigenvalue ratio.

    Parameters
    ----------
    TrainError : 'list'
        MSE for training 
    TestError : 'list'
        MSE for test 
    EigenVar : 'list'
        Plots 1-eigen ratio

    Returns
    -------
    3 plots
    '''
    
    fig, ax = plt.subplots()
    ax.plot(TestError, label="Test error")
    ax.plot(TrainError, label="Train error")
    ax.set_xlabel("k")
    ax.set_ylabel("Mean squared reconstruction error")
    ax.set_title("MSE vs k")
    ax.legend()
    
    fig2, ax2 = plt.subplots()
    ax2.plot(EigenVar, label="Train error")
    ax2.set_xlabel("k")
    ax2.set_ylabel("1-Eigenvalue Ratio")
    ax2.set_title("EigenValue Fraction vs k")
    ax2.legend()
    
    plt.show()
    
def PlotKEigen(eigenvectors): 
    '''Plots the first 10 principal components

    Parameters
    ----------    
    eigenvectors : 'np.array'
        array containing the eigenvectors of the covariance matrix 
    
    
    '''
    
    
    fig2, axis2 = plt.subplots(2, int(10/2), figsize=(10, 25), sharex=True, sharey=True)
    k = 0
    
    for i in range(2): 
        for j in range(int(10/2)): 
            axis2[i, j].imshow(eigenvectors[:,k].reshape((28, 28)), cmap='gray')
            axis2[i,j].set_title(f"k={k+1}")
            k += 1
            
    plt.show()
    
    
def PlotDigit(Digit, data, labels, eigenvec, mu):
    '''
    Compares the actual digit indicated and compares the reconstruction 
    along several maginitudes of principle components. Plots are used for 
    comparison
    
    Parameters
    ----------
    Digit: 'int'
        Indicates the digit that you would like to examine (0-9)
    data: 'np.array'
        Data points 
    labels: 'np.array'
        Corresponding labels to data arrar
    eigvec: 'np.array'
        eigenvectors based on trained covariance matrix
    '''
    
    Act_ind = np.where(labels == Digit)[0][0]
    Actual_dig = data[Act_ind]
    
    kVals = [5,15,40,100]
    
    titles = ('Original', 'k = 5', 'k = 15','k = 40', 'k = 100')
    
    Predictions = [Actual_dig]
    
    for k in kVals:
        Vk = eigenvec[:, :k]
        proj_matrix = np.dot(Vk, Vk.T)
    
        Predictions  = Predictions + [mu.T + np.dot((Actual_dig - mu.T),proj_matrix)]
        
    fig, axes = plt.subplots(1,5, figsize = (7.5, 2))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(Predictions[i].reshape(28,28), cmap = 'gray') 
        ax.set_title(titles[i])
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":           
    pca()    
    
    
    
    
    
    
    
    
    
    

