import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from math import log

train = pd.read_csv("../data/train.csv")
fields = list(train.columns)
attributes = fields[:-1]
target = [ fields[-1] ]

trainArr = train.as_matrix(attributes)
trainTargetArr = train.as_matrix(target)

x = list(train['revenue'])

y = [log(i) for i in x]

plt.hist(y, facecolor='green', alpha=0.75, bins=50)
plt.xlabel('Revenue')
plt.ylabel('Frequency')
plt.title(r'Histogram of Revenue')
plt.grid(True)

plt.show()