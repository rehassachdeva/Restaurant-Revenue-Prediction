# First let's import the dataset, using Pandas.
import pandas as pd

train = pd.read_csv("../data/trainIris.csv")
test = pd.read_csv("../data/testIris.csv")

from sklearn.ensemble import RandomForestClassifier

# the data has to be in a numpy array in order for
# the random forest algorithm to accept it.
# Also, output must be separated.

attributes = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
classes = ['class']

trainArr = train.as_matrix(attributes)
trainClassArr = train.as_matrix(classes)

# Training

# n_estimators is the number of trees in the forest.
# n_jobs is the number of jobs to run in parallel for both fit and predict.
# If -1, then the number of jobs is set to the number of cores.

rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
rf.fit(trainArr, trainClassArr)

# Testing
# put the test data in the same format.

testArr = test.as_matrix(attributes)
test['predictions'] = rf.predict(testArr)

print test