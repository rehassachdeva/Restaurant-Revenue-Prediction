#/bin/python
#From Rehas
import pandas as pd

train = pd.read_csv("../data/train.csv")
#test = pd.read_csv("../data/testIris.csv")
from sklearn.preprocessing import LabelEncoder  

from sklearn import svm
attributes = ['P1','P2','P3','P4','P5','P6']
classes = ['revenue']

trainArr = train.as_matrix(attributes)
trainClassArr = train.as_matrix(classes)


#one vs many regresion classifier
lin_clf = svm.LinearSVR()
lin_clf.fit(trainArr, trainClassArr.ravel())	#ravel,to make numpy.ndarray to 1D

#Predict
print lin_clf.predict([[6, 4.5,	6, 7.5, 6, 4]])