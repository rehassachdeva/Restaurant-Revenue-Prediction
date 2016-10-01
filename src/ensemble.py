import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from math import log

from datetime import datetime

from sklearn.neighbors import KNeighborsClassifier

def read_data(filename):
	dataset = pd.read_csv(filename)
	return dataset

def plot_histogram(arr, xlabel, ylabel, title):
	plt.hist(arr, facecolor='green', alpha=0.75, bins=50)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.grid(True)
	plt.show()

def parse_date(train_data):
	latest_date = datetime.strptime("01/01/2015", '%m/%d/%Y')
	
	open_num_days = []
	open_month = []
	open_year = []

	for date in train_data['Open Date']:
		cur_date = datetime.strptime(date, '%m/%d/%Y')
		open_num_days.append((latest_date - cur_date).days)
		open_month.append(cur_date.month)
		open_year.append(cur_date.year)

	train_data['Days'] = open_num_days
	train_data['Month'] = open_month
	train_data['Year'] = open_year

def svm_regression(train_arr, train_target_arr, query):
	
#one vs many regresion classifier
lin_clf = svm.LinearSVR()
lin_clf.fit(trainArr, trainClassArr.ravel())	#ravel,to make numpy.ndarray to 1D



def adjust_type(test_data):
	query_matrix = test_data.query('Type == "MB"')
	search_matrix = test_data.query('Type != "MB"')

	features = test_data.columns.values[5:]

	clf = KNeighborsClassifier(n_neighbors=5)
	clf.fit(search_matrix[features], search_matrix['Type'])

	query_matrix['Type'] = clf.predict(query_matrix[features])

	test_data = search_matrix.append(query_matrix)
	test_data = test_data.sort_values(['Id'])

	return test_data

if __name__ == "__main__":

	train_data = read_data("train.csv")
	test_data = read_data("test.csv")

	train_fields = train_data.columns.values

	attributes = train_fields[:-1]
	target = [ train_fields[-1] ]

	train_arr = train_data.as_matrix(attributes)
	train_target_arr = train_data.as_matrix(target)

	# x = list(train_data['revenue'])

	# y = [log(i) for i in x]

	# plot_histogram(x, "Revenue", "Frequency", "Histogram of Revenue")
	# plot_histogram(y, "Log(Revenue)", "Frequency", "Histogram of Log(Revenue)")

	# parse_date(test_data)

	# print len(train_data.columns.values)

	test_data = adjust_type(test_data)

	print test_data

	# print train_data

