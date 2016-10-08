#!/bin/python

import numpy as np
class LogisticRegression:
 
    def __init__(self,X,y,tolerance=1e-5,l1=0.,l2=0.):
        """Initializes Class for Logistic Regression"""
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
 