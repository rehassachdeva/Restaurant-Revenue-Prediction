#!/bin/python
#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class RidgeRegressor(object):
	
    #Theta = (X'X + G'G)^-1 X'y


    def fit(self, X, y, alpha=0):
      # y: dependent variable vector for m examples
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        G = alpha * np.eye(X.shape[1])
        G[0, 0] = 0  # Don't regularize bias
        self.params = np.dot(np.linalg.inv(np.dot(X.T, X) + np.dot(G.T, G)),
                             np.dot(X.T, y))

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return np.dot(X, self.params)


if __name__ == '__main__':
    X = np.linspace(0, 6, 100)
    y = 1 + 2 * np.sin(X)
    yhat = y + .5 * np.random.normal(size=len(X))

    plt.plot(X, y, 'g', label='y = 1 + 2 * sin(x)')
    plt.plot(X, yhat, 'rx', label='noisy samples')

    tX = np.array([X]).T
    tX = np.hstack((tX, np.power(tX, 2), np.power(tX, 3)))

    r = RidgeRegressor()
    r.fit(tX, y)
    plt.plot(X, r.predict(tX), 'b', label=u'ŷ (alpha=0.0)')
    alpha = 3.0
    r.fit(tX, y, alpha)
    plt.plot(X, r.predict(tX), 'y', label=u'ŷ (alpha=%.1f)' % alpha)

    plt.legend()
    plt.show()
