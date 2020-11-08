# -*- coding: utf-8 -*-
"""
This routine explores the bias variance tradeoff for a synthetic data set. 
Here we calculate estimators for bias, variance and the sum of their errors 
using a step function estimator. 

@author: salez
"""

# Homework 1 B.1 part d

import numpy as np
import matplotlib.pyplot as plt 

# Initiate paramenters for step function
n = 256 
m = np.array([1,2,4,8,16,32])
sigma_2 = 1 

# Create x_i and y_i pairs 
x = np.linspace(1/n,1,n)
y = np.zeros(n)
y_true = np.zeros(n)

# Defines true function for any x
def f_true(x): 
    return 4*np.sin(np.pi*x)*np.cos(6*np.pi*x**2)
    
# Defines our f_hat step function 
def f_m(y,m,n): 
    y_split = np.array_split(y,n/m)
    f_hat = np.array([])
    for k in range(int(n/m)): 
        f_hat = np.append(f_hat,np.ones(m)*y_split[k].mean())
    return f_hat 
        
# create y_i and y_true values for evaluation 
for i in range(len(x)): 
    y[i] = f_true(x[i]) + np.random.normal()
    y_true[i] = f_true(x[i])
    
# Determine Emperical Error 
emp_error = np.zeros(len(m))
for i in range(len(m)):
    y_est = f_m(y,m[i],n)
    emp_error[i] = ((y_est-y_true)**2).mean()
    
# Determine Average Bias Error 
avg_bias = np.zeros(len(m))
for i in range(len(m)):
    y_true_avg = f_m(y_true,m[i],n)
    avg_bias[i] = ((y_true_avg-y_true)**2).mean()
    
# Determine Average Bias Squared Error 
variance = np.zeros(len(m))
for i in range(len(m)):
    variance[i] = (1/m[i])
    
# Sum of errors
sum_errors = avg_bias + variance
    
        
# plot curve
plt.figure()
plt.plot(m,emp_error, label = 'Emperical Error')
plt.plot(m,avg_bias, label = 'Average Bias Squared')
plt.plot(m,variance, label = 'Variance')
plt.plot(m,sum_errors, label = 'Sum of Errors')
plt.xlabel('m')
plt.ylabel('Error')
plt.legend() 




