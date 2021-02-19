# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 21:32:53 2020

@author: salez
"""

import csv
import numpy as np
from scipy.sparse.linalg import svds
import torch
import matplotlib.pyplot as plt
from numpy.linalg import solve

data = []
with open('data/ml-100k/u.data') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        data.append([int(row[0])-1, int(row[1])-1, int(row[2])])
data = np.array(data)

num_observations = len(data) # num_observations = 100,000
num_users = max(data[:,0])+1 # num_users = 943, indexed 0,...,942
num_items = max(data[:,1])+1 # num_items = 1682 indexed 0,...,1681

np.random.seed(1)
num_train = int(0.80*num_observations)
perm = np.random.permutation(data.shape[0])
train = data[perm[0:num_train],:]
test = data[perm[num_train::],:]

# Split Validation
num_train_split = int(0.85*train.shape[0])
permV = np.random.permutation(train.shape[0])

trainV = train[permV[0:num_train_split],:]
val = train[permV[num_train_split:],:]

def AvgPool(train_set): 
    '''
    Takes the training data and returns a vector 
    of average ratings for each movie from each user
    in the training set
    
    Parameters
    ----------
    
    train : 'np.array'
        Array with all user, moves and ratings 
        
    Returns
    -------
    mu_mov : 'np.array'
        Sorted average movie rating
        
    '''
    
    # Based on vectorization described here: 
    # https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
    
    mov = train_set[:, [1]] # extract only movies column
    
    # creates an array of indices, sorted by unique element
    idx_sort = np.argsort(mov, axis = 0)

    # sorts records array so all unique elements are together 
    sorted_movies_array = mov[idx_sort]
    
    # returns the unique values, the index of the first occurrence of a value, and the count for each element
    vals, idx_start, count = np.unique(sorted_movies_array, 
                                       return_counts=True, return_index=True)
    
    # splits the indices into separate arrays
    movie_index = np.split(idx_sort, idx_start[1:]) 
    
    movie_avg = np.zeros(1682)
    
   
    for movie in movie_index:
        indx = train_set[movie[0],[1]]
        movie_avg[indx] = np.mean(train_set[movie, [2]])   
       
    return movie_avg

def TestAvg(movie_avg, test_set): 
    '''
    Calculates average square error on test based 
    on averaging training movie rating in training set

    Parameters
    ----------
    movie_avg : 'np.array'
        Sorted average movie rating
        
    test_set : 'np.array'
        Test array with all user, moves and ratings 
        

    Returns
    -------
    Avg_sq_error: 'float'
        Average square error 

    '''
    
    num_test = 0
    square_error = 0
    
    
    for entry in test_set: 
        
        movID = entry[1]
        rating = entry[2]
        
        if movie_avg[movID] == 0: 
            continue
        
        #print((movie_avg[movID]-rating)**2)
        
        square_error += (movie_avg[movID]-rating)**2 
        num_test += 1 
        
    return square_error/num_test

movavg = AvgPool(train)

TestAvg(movavg, test)

def R_tilde(data): 
    '''
    Constructs estimated mxn matrix that represents the full movie/user rating 
    for each movie user combination. For any missing entries, 0 will be used.  

    Parameters
    ----------
    data : 'np.array'
        Array with all user, moves and ratings 

    Returns
    -------
    R_tile: 'np.array'
        Filled array with availale user/movie ratings 

    '''
    
    m = 1682 # num movies
    n = 943 # num users 
    
    R_tilde = np.zeros((m,n))
    
    for entry in data: 
        i = entry[1]
        j = entry[0]
        
        R_tilde[i,j] = entry[2]
        
    return R_tilde

def svdRecon(R_tilde, d): 
    '''
    Performs SVD on R_tilde and reconstructs it using the 
    a rank(d) approximation 

    Parameters
    ----------
    R_tilde : 'np.array'
        Sparse matrix that includes all etries 
    d : 'int'
        parameter to return the best rank d approximation 

    Returns
    -------
    R_d : 'np.array'
        Rank d matrix approximation 

    '''
    
    u, s, vt = svds(R_tilde, d)
    
    R_d = u.dot(np.diag(s)).dot(vt)
    
    return R_d 

def ErrorB(data, R_d): 
    
    '''
    Performs average squared error on training and test set 
    using the reconstructed R matrix 
    
    Parameters
    ----------
    data : 'np.array'
        data array for known ratings of movies/users
   
    Returns
    -------
    Err : 'float'
        average square error 
        
    '''
    
    n_data = len(data)
    Error = 0 
     
    for entry in data:
        i = entry[1]
        j = entry[0]
        
        Error += (R_d[i,j] - entry[2])**2
        
    return Error/n_data

def B1b(data, train, test): 
    '''
    Performs analysis on reconstruction of best Rank d 
    SVD. Training and Test Errors are calculated and plotted

    Parameters
    ----------
    data : 'np.array'
        data array for known ratings of movies/users
        
    train : 'np.array'
        training data array
        
    test : 'np.array'
        test data array
        

    '''
    
    d_s = [1,2,5,10,20,50]
    
    trainE = []
    testE = []
    
    R_t = R_tilde(train)
    
    for d in d_s: 
        R_d = svdRecon(R_t, d)
        
        trainE.append(ErrorB(train, R_d))
        testE.append(ErrorB(test, R_d))
        
   

    plt.plot(d_s, trainE, label='Training Error')
    plt.plot(d_s, testE, label='Test Error')
    plt.legend()
    plt.title('SVD')
    plt.xlabel('d')
    plt.ylabel('Average Square Error')
    plt.show()
        
    
    
    
        
B1b(data, train, test)




def Altloss(data, U_s, V_s, lambd): 
    '''

    Parameters
    ----------
    data : 'np.array'
        DESCRIPTION.
    U_s : TYPE
        DESCRIPTION.
    V_s : TYPE
        DESCRIPTION.
    lambd : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    loss = 0
        
    for entry in data: 
        u_idx = entry[1]
        v_idx = entry[0]
        
        u_i = U_s[u_idx]
        v_i = V_s[v_idx]
    
        
        loss += (np.dot(u_i, v_i) - entry[2])**2 
        
    # Regularization term u
    reg_u = lambd*np.sum(np.linalg.norm(U_s, axis =1)**2)
    
    # Regularization term v
    reg_v = lambd*np.sum(np.linalg.norm(V_s, axis =1)**2)
    
    loss = loss + reg_u + reg_v
    
    return loss

def Alt_train(train_dat, U_s, V_s, lambd, max_epochs = 200,
              epsilon = 1): 
    '''
    Performs training of model using alternate minimization terminates 
    either by converging to a small epsilon or reaching the maximum epochs

    Parameters
    ----------
    train_dat : 'np.array'
        Training data for which both user and movie have a rating
    u_s : TYPE
        DESCRIPTION.
    v_s : TYPE
        DESCRIPTION.
    lambd : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    trainR = R_tilde(train_dat)
    
    NumMov = U_s.shape[0]
    NumUse = V_s.shape[0]
    d = U_s.shape[1]
    LambaI = np.eye(d) * lambd
    
    for epoch in range(max_epochs): 
        
        # Minimize Loss with respect to Movies 
        
        # Calculate costant terms
        YTY = V_s.T.dot(V_s)
        
        for m in range(NumMov): 
            U_s[m,:] = solve((YTY + LambaI),trainR[m,:].T.dot(V_s))
            
        # Minimize Loss with respect to Users 
        
        XTX = U_s.T.dot(U_s)
        
        for n in range(NumUse): 
            V_s[n,:] = solve((XTX + LambaI),trainR[:,n].dot(U_s))
         
        loss = Altloss(train_dat, U_s, V_s, lambd)
        
        print('Loss is: ',loss)
        
        # Check if converged
        if epoch == 0: 
            l_old = Altloss(train_dat, U_s, V_s, lambd)
            continue 
        else:
            per_change = abs(l_old - loss)
            print(per_change)
            
            if per_change < epsilon: 
                return U_s, V_s
            
            l_old = loss
        
    return U_s, V_s


def ErrorC(data, U_s, V_s): 
    '''
    Calculates Average square error based on the 
    usage of latent vectors U_s, V_s

    Parameters
    ----------
    data : 'np.array'
        data to calculate average error 
    U_s : 'np.array'
        Movie latent vector of size m,d
    V_S : 'np.array'
        User latent vector of size n,d

    Returns
    -------
    Error: 'float'
        Average squared error

    '''
    
    n_data = len(data)
    Error = 0 
    
    for entry in data: 
        u_idx = entry[1]
        v_idx = entry[0]
        
        u_i = U_s[u_idx]
        v_i = V_s[v_idx]
    
        
        Error += (np.dot(u_i, v_i) - entry[2])**2
    
    return Error/n_data


def HypSearch(train, val): 
    '''
    Performs Alternating Minimization algorithm to approximate the best 
    hyperparameters for each d. 

    Parameters
    ----------
        
    train : 'np.array'
        training data array
        
    val : 'np.array'
        validation data array
        
    Returns
    -------
    
    BestHP : 'list'
        Best combination of lambda and sigma 
    '''
    
    np.random.seed(seed = 30)
    
    # Define hyperparameters to tune
    d_s = [1,2,5,10,20,50]
    lambdas = [0.1,1,10,100,1000]
    sigmas = [0.01, 0.1, 1, 10]   
    
    m = 1682
    n = 943
              
    BestHP = [] # Find best hyperparameters for each d
    
    
    for d in d_s: 
        
        valE = [] # Store Validation Error
        hyperparams = [] # Store HyperParameter pairs 
        titles = [] # Store title of parameters
        
        for lambd in lambdas: 
            for sig in sigmas: 
                title = f"d = {d}, Lambda = {lambd}, sigma = {sig}"
                
                titles.append(title)
                hyperparams.append([lambd, sig])
                
                # movies 
                U_s = sig*np.random.rand(m,d)

                # users
                V_s = sig*np.random.rand(n,d)
                
                U_new, V_new = Alt_train(train, U_s, V_s, lambd)
                
                ValError = ErrorC(val, U_new, V_new)
                
                valE.append(ValError)
                
        idxmin = np.argmin(valE)
        
        BestHP.append(hyperparams[idxmin])
        
        print('Best Hyperparameters are ' + titles[idxmin] + f" with loss {valE[idxmin]}")
        
        
    return BestHP
                

def B1cd(lambd, train, test): 
    '''
    Plots the training and test average square error best 
    on the best hyperparameters

    Parameters
    ----------
    lambd : 'float'
        regularization term     
    train : 'np.array'
        training data to construct latent matrices
    test : 'np.array'
        test data to perform test error
    '''
    
    # Define hyperparameters to tune
    d_s = [1,2,5,10,20,50]
    sig = 2 # based on best hyperparmeter
    
    
    m = 1682
    n = 943
    
    trainE = []
    testE = []
    
    
    for d in d_s:
        # movies 
        U_s = sig*np.random.rand(m,d)

        # users
        V_s = sig*np.random.rand(n,d)
        
        U_new, V_new = Alt_train(train, U_s, V_s, lambd)
        
        
        TrainEr = ErrorC(train, U_new, V_new)
        TestEr = ErrorC(test, U_new, V_new)
        
        trainE.append(TrainEr)
        testE.append(TestEr)
        

    plt.plot(d_s, trainE, label='Training Error')
    plt.plot(d_s, testE, label='Test Error')
    plt.legend()
    plt.title('Average Square Error')
    plt.xlabel('d')
    plt.ylabel('Average Square Error')
    plt.show()
    
B1cd(0.4, train, test)
        
    
   

# =============================================================================
#     plt.plot(d_s, trainE, label='Training Error')
#     plt.plot(d_s, testE, label='Test Error')
#     plt.legend()
#     plt.title('Average Square Error')
#     plt.xlabel('d')
#     plt.ylabel('Average Square Error')
#     plt.show()
# =============================================================================



    
#u_train, v_train = Alt_train(train_dat, u_s, v_s, lambd, max_epochs = 50, step_size = 0.001, 
 #           epsilon = 0.001)




#print(TestAvg(AvgPool(train), test)) 
        
    