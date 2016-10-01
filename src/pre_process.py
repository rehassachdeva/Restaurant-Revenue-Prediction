#!/bin/python
import csv
from datetime import datetime
import csv
import random
import math
import operator
 
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((float(instance1[x]) - float(instance2[x])), 2)
	return math.sqrt(distance)
 
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(1,len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((x, dist,x))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
 
def getResponse(neighbors,r_search):
	classVotes = [0]*3
	for x in range(len(neighbors)):
		if r_search[neighbors[x] ][5]=='FC':
			classVotes[0]=classVotes[0]+1
		elif r_search[neighbors[x] ][5]=='IL':
			classVotes[1]=classVotes[1]+1
		elif r_search[neighbors[x] ][5]=='DT':
			classVotes[2]=classVotes[2]+1
	index, value = max(enumerate(classVotes), key=operator.itemgetter(1))
	return index, value
 
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
#Load Dataset
train = []
with open('../data/train.csv', 'rb') as csvfile:
	datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in datareader:
		train.append(', '.join(row).split(',')) #Every row in train corresponds to a row in the train.csv file
		#print ', '.join(row)

train[0].insert(2,'Open_month')
train[0].insert(3,'Open_year')
train[0].remove(' Group')
train[0][6]='City Group'

test = []
with open('../data/test.csv', 'rb') as csvfile:
	datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in datareader:
		test.append(', '.join(row).split(',')) #Every row in train corresponds to a row in the test.csv file
		#print ', '.join(row)
test[0].remove(' Group')
test[0][4]='City Group'

#Parsing open date 
for k in range(1,len(train)):
	datestring=train[k][1]
	dt = datetime.strptime(datestring, '%m/%d/%Y')
	train[k].insert(2,dt.month) #insert 2 new features, month and year that  can potentially help proxy seasonality differences since restaurant revenues are highly cylical.
	train[k].insert(3,dt.year)

#The unaccounted problem
##  disparity between the features for the training set and test set

#Query matrix for mobile type restaurants in the test set that aren't present in training set
query = [i[5:] for i in test if i[4] == 'MB'] # Remove cateogrical variables from query
query=['']+query
r_search = [i for i in test if i[4] != 'MB'] 
search=[] # Excludes cateogrical variables from query to run KNN
for i in test:
	if i[4]!='MB':
		if(len(i)==42):
			search.append(i[5:])
		else:
			search.append(i[6:])
predictions=[]
k = 5
for x in range(1,len(test)):
	neighbors = getNeighbors(query, search[x], k)
	index,value = getResponse(neighbors,r_search)
	test[  test.index(r_search[x]) ][5] = value #After obtaining K nearest neighbours for each row of Q, the mode is taken for the restaurant type and the mobile type is replaced with the mode. 
	#predictions.append(result)
	#print('> predicted=' + repr(result) + ', actual=' + repr(test[x][-1]))
#accuracy = getAccuracy(testSet, predictions)
#print('Accuracy: ' + repr(accuracy) + '%')
	
#Write the new test and train files. 
with open('PreProcessedTrain.csv', 'w') as mycsvfile:
    thedatawriter = csv.writer(mycsvfile, dialect='mydialect')
    for row in train:
        thedatawriter.writerow(row)

with open('PreProcessedTest.csv', 'w') as mycsvfile:
    thedatawriter = csv.writer(mycsvfile, dialect='mydialect')
    for row in test:
        thedatawriter.writerow(row)

