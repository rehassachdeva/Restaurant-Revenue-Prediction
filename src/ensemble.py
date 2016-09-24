import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from math import log

from datetime import datetime

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestRegressor

from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder

from sklearn import svm 

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

def adjust_type(test_data):
	query_matrix = test_data.query('Type == "MB"')
	search_matrix = test_data.query('Type != "MB"')

	features = test_data.columns.values[5:]

	clf = KNeighborsClassifier(n_neighbors=5)
	clf.fit(search_matrix[features], search_matrix['Type'])

	query_matrix['Type'] = clf.predict(query_matrix[features])

	test_data = search_matrix.append(query_matrix)
	#test_data = test_data.sort_values(['Id'])
	test_data = test_data.sort(['Id'])

	return test_data

def adjust_city(train_data,test_data):
	data=train_data.append(test_data)
	features = [ 'P1', 'P2', 'P11', 'P19', 'P20', 'P23', 'P30' ]
	kmeans = KMeans(n_clusters=20).fit(data[features])
	train_data['City']=kmeans.predict(train_data[features].values)
	test_data['City']=kmeans.predict(test_data[features].values)
	return train_data,test_data

def encode_transform(train_data, test_data, attributes):
	train_arr = train_data.as_matrix(attributes)
	test_arr = test_data.as_matrix(attributes)
	M = train_arr
	M = np.concatenate((M,test_arr), axis=0)

	# To ensure test and train get the same encoding, also encode
	# first 3 attributes
	for i in range(3):
		M[:, i] = LabelEncoder().fit_transform(M[:,i])

	# Separate the train and test class
	train_arr = M[:len(train_arr)]
	test_arr = M[len(train_arr):]

	return train_arr, test_arr

def pca_transform(train_data, test_data):
	pca = PCA(n_components=30)
	pca_train_data = pca.fit_transform(train_data)
	pca_test_data = pca.fit_transform(test_data)
	return pca_train_data, pca_train_data

def random_forest(train_arr, test_arr, train_class_arr):
	rf = RandomForestRegressor()
	rf.fit(train_arr, train_class_arr.ravel())
	return rf.predict(test_arr)

def svm_regressor(train_arr, test_data, train_class_arr):
	# one vs many regresion classifier
	lin_clf = svm.LinearSVR()
	# ravel,to make numpy.ndarray to 1D
	lin_clf.fit(train_arr, train_class_arr.ravel())
	return lin_clf.predict(test_arr)

def calc_final_predictions(rf_predictions, svm_predictions):
	return [0.5 * (x + y) for x, y in zip(rf_predictions, svm_predictions)]

if __name__ == "__main__":

	train_data = read_data("../data/train.csv")
	test_data = read_data("../data/test.csv")

	attributes = ['City', 'City Group', 'Type',
			 'P1','P2','P3','P4','P5','P6',	'P7','P8','P9','P10','P11','P12',
			 'P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23',
			 'P24','P25','P26','P27','P28','P29','P30','P31','P32','P33','P34',
			 'P35','P36','P37', 'Days', 'Month', 'Year'
	]

	# x = list(train_data['revenue'])

	# y = [log(i) for i in x]

	# plot_histogram(x, "Revenue", "Frequency", "Histogram of Revenue")
	# plot_histogram(y, "Log(Revenue)", "Frequency", "Histogram of Log(Revenue)")

	#Pre Processing
	#insert 2 new features, month and year that  can potentially help proxy seasonality differences since restaurant revenues are highly cylical.
	parse_date(train_data) 
	parse_date(test_data)
	#  disparity between the features for the training set and test set
	test_data = adjust_type(test_data)
	train_data, test_data = adjust_city(train_data, test_data)

	# train_data, test_data = pca_transform(train_data, test_data)

	train_arr, test_arr = encode_transform(train_data, test_data, attributes)

	train_class_arr = train_data.as_matrix(['revenue'])

	rf_predictions = random_forest(train_arr, test_arr, train_class_arr)
	
	svm_predictions = svm_regressor(train_arr, test_arr, train_class_arr)

	final_predictions = calc_final_predictions(rf_predictions, svm_predictions)

	print final_predictions