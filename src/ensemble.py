import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from math import log

from datetime import datetime

def read_data(filename):
	dataset = pd.read_csv(filename)
	fields = list(dataset.columns)
	return fields, dataset

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

	train_data['No. of days open'] = open_num_days
	train_data['Month of opening'] = open_month
	train_data['Year of opening'] = open_year

if __name__ == "__main__":

	train_fields, train_data = read_data("../data/train.csv")
	attributes = train_fields[:-1]
	target = [ train_fields[-1] ]

	train_arr = train_data.as_matrix(attributes)
	train_target_arr = train_data.as_matrix(target)

	x = list(train_data['revenue'])

	y = [log(i) for i in x]

	# plot_histogram(x, "Revenue", "Frequency", "Histogram of Revenue")
	# plot_histogram(y, "Log(Revenue)", "Frequency", "Histogram of Log(Revenue)")

	parse_date(train_data)

	print train_data

