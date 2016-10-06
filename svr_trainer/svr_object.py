#!/bin/python
import numpy as np

class SVR:
	"SVR Object"
	def __init__(self, data):
		self.alpha=[]
		self.b=0
		self.train_data=data

	def kernel_function(self, x, y):
		x = np.array(x)
		y = np.array(y)
		lmbda = 1
		val = np.square(np.linalg.norm(x-y))
		return np.exp(-lmbda*val)


	def predict(self, test_data):
		n_train = len(train_data)
		f = 0
		for i in range(n_train):
			f = f + alpha[i]*kernel_function(test_data, train_data[i])
		f = f + b
		f = f/2
		return f
