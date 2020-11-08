'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree=1, reg_lambda=1E-8):
        """
        Constructor
        """
        #TODO
        
        self.degree = degree
        self.regLambda = reg_lambda
        
    def polyfeatures(self, X, degree):
        """
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        """
        #TODO
        
        for d in range(2,degree+1):
            X = np.append(X,X[:,[0]]**d,1)
            
        return X

    def fit(self, X, y):
        """
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        """
        #TODO
        
        # Perform polynomial expansion on X 
        X = self.polyfeatures(X, self.degree)
        
        n = len(X)

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), X]

        n, d = X_.shape
        d = d-1  # remove 1 for the extra column of ones we added to get the original num features

        # construct reg matrix
        reg_matrix = self.regLambda * np.eye(d + 1)
        reg_matrix[0, 0] = 0
        
        mus = []
        stds = []
        
        # Standardize all columns 
        for j in range(1,d+1): 
            mu = np.average(X_[:,[j]])
            std = np.std(X_[:,[j]])
            X_[:,[j]] = (X_[:,[j]]-mu)/std
            
            # Store mu's and std's 
            
            mus.append(mu)
            stds.append(std)
            
        # analytical solution (X'X + regMatrix)^-1 X' y
        self.theta = np.linalg.pinv(X_.T.dot(X_) + reg_matrix).dot(X_.T).dot(y)
        
        # store standardization values 
        self.mus = mus
        self.stds = stds

    def predict(self, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """
        # TODO
        
        # Perform polynomial expansion on X 
        X = self.polyfeatures(X, self.degree)
        
        n = len(X)

        # add 1s column
        X_ = np.c_[np.ones([n, 1]), X]
        
        n, d = X_.shape
        d = d-1  # remove 1 for the extra column of ones we added to get the original num features
        
        # Standardize all columns by the same standardization of from the fit 
        # function 
        for j in range(1,d+1): 
            X_[:,[j]] = (X_[:,[j]]-self.mus[j-1])/self.stds[j-1]
            
        # predict
        return X_.dot(self.theta)
      
        

#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------



def learningCurve(Xtrain, Ytrain, Xtest, Ytest, reg_lambda, degree):
    """
    Compute learning curve

    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree

    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]

    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """

    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    #TODO -- complete rest of method; errorTrain and errorTest are already the correct shape
    
    m = len(Xtest)
    
    for i in range(1,n): 
        # Increment by each instance 
        X_train_instance = Xtrain[0:i+1,[0]]
        y_train_instance = Ytrain[0:i+1,[0]]
        
        # Call the PolynomialRegression Class to create model with degree 
        # and Reg_Lambda paramenters
        model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
        model.fit(X_train_instance, y_train_instance)
        
        # Predict for each instance and the subsequent test sets. 
        yTrain_predict_instance = model.predict(X_train_instance)
        yTest_predict_instance = model.predict(Xtest)
        
        # Calculate mean squared error 
        errorTrain[i] = np.sum((yTrain_predict_instance - y_train_instance)**2)/len(X_train_instance)
        errorTest[i] = np.sum((yTest_predict_instance - Ytest)**2)/m
        

    return errorTrain, errorTest
