#!/bin/python

import numpy as np
class Regression:
 
    def __init__(self,X,y,tolerance=1e-5,l1=0.,l2=0.):
        self.tolerance = tolerance
        self.labels = y.reshape(y.size,1)
        self.w = np.zeros((X.shape[1]+1,1))
        self.features = np.ones((X.shape[0],X.shape[1]+1))
        self.features[:,1:] = X
        self.shuffled_features = self.features
        self.shuffled_labels = self.labels
        self.l1=l1
        self.l2=l2
        self.likelihood_history = []
    def log_likelihood(self):
    	p = self.probability()
    	loglikelihood = self.labels*np.log(p+1e-24) + (1-self.labels)*np.log(1-p+1e-24)
    	loglikelihood = -1*loglikelihood.sum()
    	loglikelihood += self.l1*np.abs(self.w).sum()
    	loglikelihood += self.l2*np.power(self.w, 2).sum()/2
    	return loglikelihood
 
    def probability(self):

        return 1/(1+np.exp(-self.features.dot(self.w)))
 
    def log_likelihood_gradient(self):
    	error = self.labels-self.probability()
    	product = error*self.features
    	grad = product.sum(axis=0).reshape(self.w.shape)
    	grad -= self.l1*np.sign(self.w)
    	grad -= self.l2*self.w
    	return grad
 
    def gradient_decent(self,alpha=1e-7,max_iterations=1e4):

        previous_likelihood = self.log_likelihood()
        difference = self.tolerance+1
        iteration = 0
        self.likelihood_history = [previous_likelihood]
        while (difference > self.tolerance) and (iteration < max_iterations):
            self.w = self.w + alpha*self.log_likelihood_gradient()
            temp = self.log_likelihood()
            difference = np.abs(temp-previous_likelihood)
            previous_likelihood = temp
            self.likelihood_history.append(previous_likelihood)
            iteration += 1
 
    #def predict_probabilty(self,X):
    #    features = np.ones((X.shape[0],X.shape[1]+1))
    #    features[:,1:] = (X-self.mean_x)/self.std_x
    #    return 1/(1+np.exp(-features.dot(self.w)))
 
    def get_coefficients(self):
        return self.w.T[0]

    def fit(self, data, truth, alpha):
    	self.gradient_decent(apha)
    	return self.get_coefficients()

    def predict(self, test):
    	return np.transpose(self.w)*np.array(test)
 