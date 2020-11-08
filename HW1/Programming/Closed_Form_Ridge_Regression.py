# -*- coding: utf-8 -*-
"""
This routine uses the MNIST data set to test the closed formed solution of 
ridge regression. First, a the raw input of the MNIST used to test the accuracy 
of the closed formed solution with a fixed ridge regression regularization term

Next, we explore the use of an affine transformation of the pixel values from 
d features to p features in an effor to increase our accuracy. We finally see 
performance of choosing a p and its test accuracy. 

@author: Christopher Salazar
"""
import numpy as np
from mnist.loader import MNIST
import matplotlib.pyplot as plt 

# Load the MNIST Dataset 
mndata = MNIST(r'C:\Users\salez\Documents\MISE Work\CSE 546\Homework\HW1\Programming\mnist')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())
X_train = X_train/255.0 # normalize dataset 
X_test = X_test/255.0

# Transform X_train and X_test into corresponding Y form 
d = len(X_train[0])
# Number of outputs 
k = 10 

# Construct Y_train array per the form required for closed form solution
n_train = len(X_train)
Y_train = np.zeros((n_train,k))

for i in range(n_train):
    Y_train[i][labels_train[i]] = 1
    
# Construct Y_test 
n_test = len(X_test)
Y_test = np.zeros((n_test,k))

for i in range(n_test):
    Y_test[i][labels_test[i]] = 1
    
    
# Function uses closed form solution of ridge regression to train on 
# training set and returns a W_hat matrix used in predictions
def train(X, Y, reg_lambda, d):
    '''Closed form solution to recover W_hat estimate for least squares 
    Regression. 
    
    Input:  X - Training Matrix X 
            Y - Training Matrix Y 
            reg_lambda - regularization term 
            d - number of features for each X
            
    Output: W_hat matrix
    '''          
    reg_matrix = np.eye(d)
    return np.linalg.pinv(X.T.dot(X) + reg_matrix).dot(X.T).dot(Y)


def predict(W_hat, X_inst):
    # Creat predictions based on calculated W_hat from our fit function 
    E_j = np.eye(W_hat.shape[1])
    
    # Prediction based on e_j*W_hat-T *x_i
    pred_vals = np.zeros(X_inst.shape[0])
    for i in range(X_inst.shape[0]):
        pred_vals[i] = np.argmax(E_j.dot(W_hat.T).dot(X_inst[i]))
    
    return pred_vals

# Choose a fixed regularization term 
reg_lambda = 10E-4

# Train the function 
W_hat = train(X_train, Y_train, reg_lambda, d)

m_train = predict(W_hat, X_train)
m_test = predict(W_hat, X_test)

# Calculate Errors 

error_train = 1-(np.sum(m_train == labels_train)/n_train)
error_test = 1-(np.sum(m_test == labels_test)/n_test)

print('Training Error is %.3f' % error_train)
print('Test Error is %.3f' % error_test)


####################################################
# Transformation of X from d features to p features 
####################################################

# Paramater values 
p_values = np.linspace(500,6000, 12)
mu = 0
sigma_2 = 0.1



def H_trans(X, d, p, mu, sigma_2):
    ''' Transforms X- nxd matrix into H(X) -nxp matrix
    where H(x) = cos(Gx + b). G is a random matrix sampled 
    iid from a Gaussian distribution with mean 0 and sigma_sq = 1. b 
    is an iid vector ampled i.i.d. from the uniform distribution on [0 2π]
    
    Input: 
        X - Original Matrix - nxd
        d - original features 
        p - expanded feature length 
        mu - mu value for G matrix 
        sigma_2 - sigma_2 value for G matrix
        
    Output: 
        H(X) matrix
        
    '''
    H_x = np.zeros((len(X),p))
    
    G = np.random.normal(mu, sigma_2, size=(p, d))
    b = np.random.uniform(0,2*np.pi, size=(1,p))
    
    H_x = np.cos(X.dot(G.T)+b)
    
    return H_x

# Split X_train, Y_train into training and validation sets for relevant measures 
Train_sample = X_train
YTrain_sample = Y_train

labels_train_sample = labels_train

indices = np.random.permutation(Train_sample.shape[0])
training_idx, valid_idx = indices[:int(0.8*Train_sample.shape[0])], indices[int(0.8*Train_sample.shape[0]):]
labels_training, labels_validation = labels_train[training_idx], labels_train[valid_idx]
trainingY, validationY = YTrain_sample[training_idx,:], YTrain_sample[valid_idx,:]
labels_training, labels_validation = labels_train[training_idx], labels_train[valid_idx]

error_train = np.zeros(len(p_values))
error_valid = np.zeros(len(p_values))

i = 0 

for p in p_values:
    
    # Transform Training and Validation from d to p dimensions
    H_train_sample = H_trans(Train_sample, d, int(p), mu, sigma_2)
    trainingX, validationX = H_train_sample[training_idx,:], H_train_sample[valid_idx,:]
   
    # Train to achieve W_hat 
    W_hat = train(trainingX, trainingY, reg_lambda, int(p))
    
    # Find predictor values using training and validation sets 
    m_train = predict(W_hat, trainingX)
    m_val = predict(W_hat, validationX)
    
    # Calculate error
    error_train[i] = 1-(np.sum(m_train == labels_training)/labels_training.shape[0])
    error_valid[i] = 1-(np.sum(m_val == labels_validation)/labels_validation.shape[0])
    
    print(error_train[i])
    print(error_valid[i])
    i += 1 
    
print(error_train)
print(error_valid)
    
# Plot Training, Validation Error as a function of p 
plt.figure()
plt.plot(p_values,error_train, label = 'Training Error')
plt.plot(p_values,error_valid, label = 'Validation Error')
plt.xlabel('p')
plt.ylabel('Error')
plt.title('p value graph')
plt.legend()   


# Check how model will do on unseen data.

p = 5000
mu = 0
sigma_2 = 0.1
reg_lambda = 10E-4

# Preset G and b matrix so that we use the same transformation on the 
# test set 
G = np.random.normal(mu, sigma_2, size=(p, d))
b = np.random.uniform(0,2*np.pi, size=(1,p))

def H_trans_fixed(X, G, b):
    ''' Transforms X- nxd matrix into H(X) -nxp matrix
    
    Input: 
        X - Original Matrix - nxd
        G - Gaussian Matrix with each value iid with N(mu,sigma_squared)
        b - iid uniform distribution vector
        
    Output: 
        H(X) matrix
        
    '''
    H_x = np.zeros((len(X),p))
    
    
    H_x = np.cos(X.dot(G.T)+b)
    
    return H_x

H_train = H_trans_fixed(X_train,  G, b)
H_test = H_trans_fixed(X_test, G, b)

W_hat = train(H_train, Y_train, reg_lambda, int(p))
    
m_test = predict(W_hat, H_test)

error_test = 1-(np.sum(m_test == labels_test)/labels_test.shape[0])

print(error_test)


  
