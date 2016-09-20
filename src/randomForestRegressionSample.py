import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
from math import sqrt

Boston = load_boston()

bosScaled = scale(Boston.data)

bosData = pd.DataFrame(bosScaled)
bosData.columns = Boston.feature_names
bosData['TARGET'] = Boston.target

train = bosData[:300]
test = bosData[300:]

rf = RandomForestRegressor()
rf.fit(bosScaled[:300], Boston.target[:300])

test.loc[:,'PREDICTION'] = rf.predict(bosScaled[300:])

print sqrt(mean_squared_error(test['TARGET'], test['PREDICTION']))