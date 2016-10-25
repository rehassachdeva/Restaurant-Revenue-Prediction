#!/usr/bin/python

import numpy as np
#import matplotlib.pyplot as plt

	
    #Theta = (X'X + G'G)^-1 X'y


def fit(X, y, alpha=0):
      # y: dependent variable vector for m examples
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        G = alpha * np.eye(X.shape[1])
        G[0, 0] = 0  # Don't regularize bias
        params = np.dot(np.linalg.inv(np.dot(X.T, X) + np.dot(G.T, G)),np.dot(X.T, y))
        return params

def predict(params, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.dot(X, params)

