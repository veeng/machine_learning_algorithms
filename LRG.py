# Logistic Regression implementation for COMP4900A1
import numpy as np
import time
import pandas as pd




class MLE:

    def __init__(self):
        pass
    

    #https://www.kaggle.com/jeppbautista/logistic-regression-from-scratch-python
    # e = Euler's number which is 2.71828.
    # x0 = the value of the sigmoid's midpoint on the x-axis.
    # L = the maximum value.
    # k = steepness of the curve.

    def sigmoid(self,X,weight):
        z = np.dot(X,weight)
        return 1/(1+np.exp(-z))



    # Two ways to optimize the regression
    # One is through loss minimizing with the use of gradient descent
    # the other is with the use of Maximum Likelihood Estimation.

    # The goal is to minimize the loss by means of increasing or decreasing the weights, 
    # which is commonly called fitting. Which weights should be bigger and which should be smaller? 
    # This can be decided by a function called Gradient descent. 
    # The Gradient descent is just the derivative of the loss function with respect to its weight

    def loss(h,y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def gradient_descent(X,h,y):
        return np.dot(X.T, (h - y)) / y.shape[0]

    def update_weight_loss(weight, learning_rate, gradient):
        return weight - learning_rate * gradient



    # the goal here is to maximize the likelihood we can achieve this through Gradient ascent, 
    # not to be mistaken from gradient descent. 
    # Gradient ascent is the same as gradient descent, 
    # except its goal is to maximize a function rather than minimizing it.



    def gradient_ascent(X, h, y):
        return np.dot(X.T, y - h)

    def update_weight_mle(weight, learning_rate, gradient):
        return weight + learning_rate * gradient

    #along with training accuracy deprend your approach
    def fit(type,X,label,num_iter,lr):
        start_time = time.time()
        intercept = np.ones((X.shape[0],1))
        X = np.concatenate((intercept,X),axis=1)
        theta = np.zeros(X.shape[1])

        if (type == "mle"):
            for i in range(num_iter):
                h = sigmoid(X,theta)
                gradient = gradient_ascent(X,h,label)
                theta = update_weight_mle(theta,lr,gradient)

        return theta


    #TODO: need to implment if test data different dim than weight, more research
    def prediction(test,weight):
        intercept = np.ones((test.shape[0],1))
        test = np.concatenate((intercept,test),axis=1)
        result = sigmoid(test,weight)
        f = pd.DataFrame(result)
        f[0] = f[0].apply(lambda x: 1 if x > 0.4999 or x == 0.4999 else 0)
        return result
        
    
def accuracy_checking(X,label,label_frame,num_iter,lr):
    start_time = time.time()

    lll_weight = fit('lll',X,label,num_iter,lr)
    lll_train_time = time.time() - start_time
    predLLL = prediction(X,lll_weight)
    lll_test_time = time.time() - lll_train_time



    mle_weight = fit('mle',X,label,num_iter,lr)
    mle_train_time = time.time() - lll_test_time
    predMLE = prediction(X,mle_weight)
    mle_test_time = time.time() - mle_train_time


    #append lll and mle data side by side with dataset:
    label_ = pd.DataFrame(data=label)
    label_[0] = label_[0].apply(lambda x: 1 if x > 0.4999 or x == 0.4999 else 0)
    predMLE_ = pd.DataFrame(data=predMLE)
    predMLE_[0] = predMLE_[0].apply(lambda x: 1 if x > 0.4999 or x == 0.4999 else 0)
    predLLL_ = pd.DataFrame(data=predLLL)
    predLLL_[0] = predLLL_[0].apply(lambda x: 1 if x > 0.4999 or x == 0.4999 else 0)


    
    final_frame = pd.concat([label_,predMLE_,predLLL_],axis=1)
    col = ['actual','mle','lll']
    final_frame.columns = col
    
    acc_lll = final_frame.loc[final_frame['lll']==final_frame['actual']].shape[0]/final_frame.shape[0] * 100
    acc_mle = final_frame.loc[final_frame['mle']==final_frame['actual']].shape[0]/final_frame.shape[0] * 100
    
    mle_result = (acc_mle,mle_train_time,mle_test_time)
    lll_result = (acc_lll,lll_train_time,lll_test_time)
    
    return [mle_result,lll_result]


        
        
        
    
    
    
        
    
    

