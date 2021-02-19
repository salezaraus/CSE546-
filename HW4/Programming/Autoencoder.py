# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:38:23 2020

@author: salez
"""

import numpy as np
import matplotlib.pyplot as plt
from mnist.loader import MNIST
import torch
import torch.nn as nn

# global path varible
mnistpath = r'C:\Users\salez\Documents\MISE Work\CSE 546\Homework\HW4\Programming\mnist'

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
    
    
    return Xtrain, trainLabels, Xtest, testLabels

class SingleAutoEnc(nn.Module): 
    '''
    Defines class of the forward pass of a single layer autoencoder. The 
    forward pass includes encoding to a smaller dimension and decoding for 
    reconstruction. At the end of the reconstruction, we evaulate the 
    reconstruction error with the MSE loss. 
    '''
    
    def __init__(self, image_size = 28*28, h_in = 32):
        super().__init__()
        
        self.model = nn.Sequential(nn.Linear(image_size, h_in), 
                            nn.Linear(h_in, image_size)
                            ) 
            
    def forward(self, X): 
        '''
        forward pass method of enconding and decoding 

        Parameters
        ----------
        X : 'torch.tensor'
           Data to encode

        Returns
        -------
        X_R : 'torch.tensor'
            Reconstructed data 

        '''
        
        return self.model(X)
    
    def loss(self, X): 
        '''
        Calculates the MSE loss from the reconstructed data set

        Parameters
        ----------
        X : 'torch.tensor'
           Data to encode

        Returns
        -------
        loss : `nn.MSELoss`
            MSE Loss
        '''
    
        X_rec = self.forward(X) 
        loss = nn.MSELoss()
        return loss(X_rec, X)
    
class NonLinAutoEnc(nn.Module): 
    '''
    Defines class of the forward pass of a nonlinear layer autoencoder. The 
    forward pass includes encoding to a smaller dimension and decoding for 
    reconstruction. At the end of the reconstruction, we evaulate the 
    reconstruction error with the MSE loss. 
    '''
    
    def __init__(self, image_size = 28*28, h_in = 32):
        super().__init__()
        
        self.model = nn.Sequential(nn.Linear(image_size, h_in),
                            nn.ReLU(),
                            nn.Linear(h_in, image_size),
                            nn.ReLU()
                            ) 
            
    def forward(self, X): 
        '''
        forward pass method of enconding and decoding 

        Parameters
        ----------
        X : 'torch.tensor'
           Data to encode

        Returns
        -------
        X_R : 'torch.tensor'
            Reconstructed data 

        '''
        
        return self.model(X)
    
    def loss(self, X): 
        '''
        Calculates the MSE loss from the reconstructed data set

        Parameters
        ----------
        X : 'torch.tensor'
           Data to encode

        Returns
        -------
        loss : `nn.MSELoss`
            MSE Loss
        '''
    
        X_rec = self.forward(X) 
        loss = nn.MSELoss()
        return loss(X_rec, X)
    
    
def train(data, model, epochs, batches, optimizer):
    '''
    Performs training encoding/decoding model for 
    reconstruction of data. Batching of data will 
    occur in this function. 

    Parameters
    ----------
    data : 'torch.tensor'
        Full data set to use for training 
    model : 'object'
        One of the autoencoding models 
    epochs : 'int'
        Total number of epochs of full datasets
    batches : 'int'
        Total number of batches for the MNIST the 
        user would like to batch
    optimizer : 'object'
        Pytorch optimizer

    '''
    
    indx = np.arange(len(data))
    
    for i in range(epochs): 
        
        # split data into batches for each epoch
        rand_idxs = np.random.permutation(indx)
        batchIndx = np.split(rand_idxs, batches)
        
        for batch in batchIndx:
            
            Xdata = data[batch]
            optimizer.zero_grad()
            loss = model.loss(Xdata)
            loss.backward() 
            optimizer.step() 
        
        # print loss after each epoch 
        print(loss.item())
        
def plot_reconstruct(Xrec, Xactual):
    '''
    Plots the reconstructed images and the actual images for 
    side by side comparison

    Parameters
    ----------
    Xrec : 'torch.tensor'
        Reconstructed tensor after training 
    Xactual : 'torch.tensor'
        Original tensor of image 

    '''
    
    num_img = len(Xactual)
    
    fig, axes = plt.subplots(1, num_img, figsize = (10,3), sharex = True, 
                             sharey = True)
    
    for imgOri, imgRec, ax in zip(Xactual, Xrec, axes.ravel()): 
        
        img_stack = torch.cat([imgOri.view(28,28),imgRec.view(28,28)])
        
        ax.imshow(img_stack.detach().numpy())
        ax.set_axis_off()
        
    fig.set_tight_layout(True)
    return fig, ax
    
        
def unique_ind(trainLabels): 
    '''
    Returns index of unique labels for the training data

    Parameters
    ----------
    trainLabels : 'torch.tensor'
        Training labels

    Returns
    -------
    uniqu_idx : 'list'
        Index for the unique labels 
        
    '''
    
    unique_labels = trainLabels.unique().numpy() 
    
    return [np.where(trainLabels.numpy() == digit)[0][0]
            for digit in unique_labels]

def A2a(data, labels, epochs = 10, batches = 100, learningRate = 0.01): 
    '''
    Performs autoencoding on dataset using a single layer linear neural 
    network. Plots are then compared for the to see the original image 
    and the reconstructed image

    Parameters
    ----------
    data : 'torch.tensor'
        Full data set to use for training 
    labels : 'torch.tensor'
        labels of data set to use for training 
    epochs : 'int'
        Total number of epochs of full datasets
    batches : 'int'
        Total number of batches for the MNIST the 
        user would like to batch
    learningRate: 'float'
        Learning rate for gradient descent

    '''
    
    # Create a subset of data for reconstruction purposes 
    unq_indx = unique_ind(labels)
    Xactual = data[unq_indx]
    
    for h in (32,64,128): 
        model = SingleAutoEnc(784, h)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learningRate)
        
        train(data, model, epochs, batches, optimizer)
        
        # print final loss
        print('Final training loss error for h = %d is %.7f' % 
              (h, model.loss(data)))
        print('\n')
        
        Xrec = model.forward(Xactual)
        
        # print comparative plots
        fig, axes = plot_reconstruct(Xrec, Xactual)
        fig.suptitle("Reconstruction, linear \n"
                     f"(h={h}, Nbatch={batches}, epochs={epochs}, "
                     f"lr={learningRate})")
        fig.savefig(f"../graphics/A2a_h{h}.png")
        
    plt.show()
    
def A2b(data, labels, epochs = 50, batches = 100, learningRate = 0.001): 
    '''
    Performs autoencoding on dataset using a single layer nonlinear neural 
    network. Plots are then compared for the to see the original image 
    and the reconstructed image

    Parameters
    ----------
    data : 'torch.tensor'
        Full data set to use for training 
    labels : 'torch.tensor'
        labels of data set to use for training 
    epochs : 'int'
        Total number of epochs of full datasets
    batches : 'int'
        Total number of batches for the MNIST the 
        user would like to batch
    learningRate: 'float'
        Learning rate for gradient descent

    '''
    
    # Create a subset of data for reconstruction purposes 
    unq_indx = unique_ind(labels)
    Xactual = data[unq_indx]
    
    for h in (32,64,128): 
        model = NonLinAutoEnc(784, h)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learningRate)
        
        train(data, model, epochs, batches, optimizer)
        
        # print final loss
        print('Final training loss error for h = %d is %.7f' % 
              (h, model.loss(data)))
        print('\n')
        
        Xrec = model.forward(Xactual)
        
        # print comparative plots
        fig, axes = plot_reconstruct(Xrec, Xactual)
        fig.suptitle("Reconstruction, nonlinear \n"
                     f"(h={h}, Nbatch={batches}, epochs={epochs}, "
                     f"lr={learningRate})")
        fig.savefig(f"../graphics/A2b_h{h}.png")
        
    plt.show()
    
def A2c(Xtrain, Xtest, epochs = 50, batches = 100, learningRate = 0.001): 
    '''
    Compares both the autoencoding neural networks by assessing their 
    test errors with an encoding parameter of h = 128 

    Parameters
    ----------
    Xtrain : 'torch.tensor'
        Training set to train to neural networks
    Xtest : 'torch.tensor'
        Test set to assess the test loss of reach model
    epochs : 'int'
        Total number of epochs of full datasets
    batches : 'int'
        Total number of batches for the MNIST the 
        user would like to batch
    learningRate: 'float'
        Learning rate for gradient descent
    
    '''
    
    # Linear model
    modelLin = SingleAutoEnc(784, 128)
    optimizerLin = torch.optim.Adam(modelLin.parameters(),
                                     lr=learningRate)
    
    train(Xtrain, modelLin, epochs, batches, optimizerLin)
    
    LinLoss = modelLin.loss(Xtest)
    
    print('Test loss for Linear Model is: %.7f' % LinLoss.item())
    
    # NonLinear model
    modelNLin = NonLinAutoEnc(784, 128)
    optimizerNLin = torch.optim.Adam(modelNLin.parameters(),
                                     lr=learningRate)
    
    train(Xtrain, modelNLin, epochs, batches, optimizerNLin)
    
    NLinLoss = modelNLin.loss(Xtest)
    
    print('Test loss for Nonlinear Model is: %.7f' % NLinLoss.item())
    
    
    
    
if __name__ == "__main__":  
    Xtrain, trainLabels, Xtest, testLabels = MnistData(filepath = mnistpath)
    A2a(Xtrain, trainLabels)
    A2b(Xtrain, trainLabels)
    A2c(Xtrain, Xtest)



# =============================================================================
# model_single = SingleAutoEnc(784, 64)
# optimizer = torch.optim.Adam(model_single.parameters(), lr=0.05)
# 
# train(Xtrain, model_single, 100, 100, optimizer)    
# 
#   
# loss = nn.MSELoss()
# 
# X_rec = model_single.forward(Xtest)
# 
# p = loss(X_rec, Xtest)
# 
# print(p.item())
# =============================================================================


# =============================================================================
# image_size = 28*28
# h_in = 32
# 
# Xtrain, trainLabels, Xtest, testLabels = MnistData(filepath = mnistpath)
# 
# X = Xtrain[:200]
# 
# model = nn.Sequential(nn.Linear(image_size, h_in), 
#                              nn.Linear(h_in, image_size))
# 
# x_rec = model(X)
# 
# loss = nn.MSELoss()
# 
# p = loss(x_rec, X)
# 
# model2 = SingleAutoEnc(784, 32).forward() 
# =============================================================================


