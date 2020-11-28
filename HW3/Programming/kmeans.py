# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 12:35:17 2020

@author: salez
"""


import numpy as np
import matplotlib.pyplot as plt
from mnist.loader import MNIST

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
    Xtrain, trainLabels = map(np.array, mndata.load_training())
    Xtest, testLabels  = map(np.array, mndata.load_testing())
    Xtrain = Xtrain/255.0 # normalize dataset 
    Xtest = Xtest/255.0
    
    return Xtrain, trainLabels, Xtest, testLabels

def Clust_init(k_num, Num_data): 
    '''
    Initializes k cluster centers at random based on indeces for the number 
    of data points

    Parameters
    ----------
    k_num : 'int'
        Number of clusters to initiailze 
    Num_data : 'int'
        Number of total data points
    Returns
    -------
    
    k_init: 'np.array'
        Array of length k_num of random index of training data 

    '''
    
    return np.random.choice(Num_data, k_num, replace = False)

def cent_to_clust(centers, data_points): 
    '''
    Given the center of each cluster, assign the remaining data points 
    to the nearest center 
    
    Parameters
    ----------
    centers: 'np.array'
        k_num x d array of cluster centers. 
        
    data_points: 'np.array'
        data points to be assigned to cluster centers
        
    Returns
    -------
    
    clusters : 'np.array'
        array where subarrays are stored according to each cluster 
    '''
    
    distance = np.zeros([len(centers),len(data_points)])
    
    for i in range(len(centers)): 
        distance[i] = np.linalg.norm(data_points - centers[i], axis = 1)
        
    closepoints = np.argmin(distance, axis = 0)
    
    clusters = np.array([data_points[np.where(closepoints == i)] for 
                         i in range(len(centers))])
    
    return clusters

def new_centers(clusters): 
    '''
    Calculates new centers after the data points have been assigned to
    each cluster

    Parameters
    ----------
    clusters : 'np.array'
        array where subarrays are stored according to each cluster 

    Returns
    -------
    new_center: 'np.array'
        New centers are calcuated

    '''
    new_centers = np.zeros([len(clusters), clusters[0].shape[1]])
    
    for i in range(len(clusters)): 
        new_centers[i] = np.mean(clusters[i], axis = 0)
        
    return new_centers

def K_means_objective(centers, clusters): 
    '''
    Calculats the k means objective function from the homework specification. 

    Parameters
    ----------
    centers : 'np.array'
        coordinates the relate the center of each cluster
    clusters : 'np.array'
        array where subarrays are stored according to each cluster 

    Returns
    -------
    
    k_obj: 'float' 
        Calculates the ojective function of the current cluster
    
    '''
    
    int_step = np.zeros(len(centers))
    
    for i in range(len(centers)): 
        int_step[i] = np.sum(np.linalg.norm(clusters[i]-centers[i], axis = 1)**2)
        
    k_obj = np.sum(int_step)
    
    return k_obj

def Lloyds_alg(data, k_num, maxIter = 100, epsilon = 0.01): 
    '''
    Implements Lloyds algorithm based k clusters and converges based on 
    a tolerance between the center changes
    
    Parameters
    ----------
    data: 'np.array'
        data to be used on to per k means clustering     
    k_num : 'int'
        Number of clusters to initiailze     
    epsilon: 'float', optional
        tolerance defined to terminate algorithm
        
    Returns
    -------
    centers: 'np.array'
        final cluster centers 
    clusters: 'np.array'
        final clusters of Lloyds Algorithm 
    K_obj: 'list'
        List of k-means objective as algorithm progresses
        
    '''
    
    n = len(data)
    
    # Initialize centers and clusters
    clus_ind = Clust_init(k_num, n)
    centers = data[clus_ind]
    
    previousC = centers

    clusters = cent_to_clust(centers, data)
    
    
    Converged = False
    Dis_centers_old = 0 
    niter = 0 
    
    k_means_obj = [] 
    
    while not Converged:
        
        # new centers and clusters
        centers = new_centers(clusters)
        clusters = cent_to_clust(centers, data)
        
        Dist_Centers = np.linalg.norm(centers - previousC)
        
        # if only on first iteration, update and skip while loop
        if niter == 0:
            Dist_cent_prev = Dist_Centers
            k_means_obj.append(K_means_objective(centers, clusters))
            niter +=1 
            continue
        
        print(np.abs(Dist_Centers - Dist_cent_prev))
        
        # Convergence criteria 
        if np.abs(Dist_Centers - Dist_cent_prev) < epsilon: 
            Converged = True
            
        else: 
            Dist_cent_prev = Dist_Centers 
            
        previousC = centers
        k_means_obj.append(K_means_objective(centers, clusters))
        print(K_means_objective(centers, clusters))
        niter +=1 
        
        if niter > maxIter: 
            Converged = True
        
    return centers, clusters, k_means_obj

def A5c(k_num = 10, maxIter = 100, epsilon = 0.001): 
    '''
    Performs k means clustering Lloyd Algorithm for problem A5c. This uses 10 
    clusters (for each class) and plots the loss function vs number of 
    iterations. 
    
    Parameters
    ----------
    k_num : 'int'
        Number of clusters to initiailze     
    epsilon: 'float', optional
        tolerance defined to terminate algorithm
    
    '''
    
    Xtrain, trainLabels, Xtest, testLabels = MnistData() 
    
    centers, clusters, k_means_obj = Lloyds_alg(Xtrain, k_num, maxIter, epsilon)
    
    fig1, axis1 = plt.subplots()
    
    axis1.plot(k_means_obj)
    axis1.set_xlabel('Number of Iterations')
    axis1.set_ylabel('K-means objective')
    axis1.set_title('K-means loss function vs Number of Iterations')
    plt.show()
    
    fig2, axis2 = plt.subplots(2, int(k_num/2), figsize=(10, 25), sharex=True, sharey=True)
    c_num = 0
    
    for i in range(2): 
        for j in range(int(k_num/2)): 
            axis2[i, j].imshow(centers[c_num].reshape((28, 28)), cmap='gray')
            c_num += 1
            
    plt.axis("off")
    plt.show()
    
def errors(centers, data_points): 
    '''
    Calculates Error based on the minimum number of error for each cluster 

    Parameters
    ----------
    centers: 'np.array'
        final cluster centers 
    data_points: 'np.array'
        data to be used on to per k means clustering    

    Returns
    -------
    Error: 'float'
        Calculates training or test error 

    '''
    num_centers = len(centers)
    Err = np.zeros([num_centers, data_points.shape[0]])
    
    for i in range(num_centers):
        Err[i] = np.linalg.norm(centers[i] - data_points, axis=1)**2
        
    minError = np.min(Err, axis = 0)
    
    return np.mean(minError)

def A5d(k_num = [2, 4, 6, 8, 16, 32, 64], maxIter = 100, epsilon = 0.01): 
    '''
    plots training and test errors based on several k means clusters

    Parameters
    ----------
    k_num : list, optional
        list of k cluster to perform. The default is [2, 4, 6, 8, 16, 32, 64].
    maxIter : 'int', optional
        Max number of iteration for the Lloyds algo. The default is 100.
    epsilon : 'float', optional
        tolerance for convergence of Lloyds algo. The default is 0.01.

    '''
    
    Xtrain, trainLabels, Xtest, testLabels = MnistData() 
    
    Err_Train, Err_Test = [0]*len(k_num), [0]*len(k_num)
    
    for i in range(len(k_num)):        
        centers, clusters, k_means_obj = Lloyds_alg(Xtrain, k_num[i], 
                                                    maxIter, epsilon)
        Err_Train[i] = errors(centers, Xtrain)
        Err_Test[i] = errors(centers, Xtest)
        
    plt.plot(k_num, Err_Test, label="test errors")
    plt.plot(k_num, Err_Train, label="train errors")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
    
    
if __name__ == "__main__":   
    A5c()    
    A5d()
    
        
        
        
        
        
        
        
    
        
    
    
    
