#/bin/python
#From Rehas
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder  
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")


from sklearn import svm
attributes = ['Open Date', 'City', 'City Group', 'Type',
			 'P1','P2','P3','P4','P5','P6',	'P7','P8','P9','P10','P11','P12',
			 'P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23',
			 'P24','P25','P26','P27','P28','P29','P30','P31','P32','P33','P34',
			 'P35','P36','P37'
]
classes = ['revenue']

trainArr = train.as_matrix(attributes)
testArr = test.as_matrix(attributes)

M=trainArr
M=np.concatenate((M,testArr), axis=0)

#To ensure test and train get the same encoding, also encode first 4 attributes
for i in range(4):
	M[:, i] = LabelEncoder().fit_transform(M[:,i])

trainClassArr = train.as_matrix(classes)

#Separate the train and test class
trainArr=M[:len(trainArr)]
testArr=M[len(trainArr):]

#one vs many regresion classifier
lin_clf = svm.LinearSVR()
lin_clf.fit(trainArr, trainClassArr.ravel())	#ravel,to make numpy.ndarray to 1D

#Predict
print lin_clf.predict([testArr[0]])