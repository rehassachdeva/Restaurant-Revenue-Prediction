#/bin/python
#From Rehas
import pandas as pd

train = pd.read_csv("../data/trainIris.csv")
test = pd.read_csv("../data/testIris.csv")

from sklearn import svm
attributes = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
classes = ['class']

trainArr = train.as_matrix(attributes)
trainClassArr = train.as_matrix(classes)


#one vs many classifier
lin_clf = svm.LinearSVC()
lin_clf.fit(trainArr, trainClassArr.ravel())	#ravel,to make numpy.ndarray to 1D

#Predict
print lin_clf.predict([[5.5, 1.8, 6.4, 3.1]])