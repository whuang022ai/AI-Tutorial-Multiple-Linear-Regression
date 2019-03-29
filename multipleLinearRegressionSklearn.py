import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
data = pd.read_csv('stu.csv')
math = data['Math'].values
read = data['Reading'].values
write = data['Writing'].values
X = np.array([math, read]).T
Y = np.array(write)
model = LinearRegression()
model = model.fit(X, Y)
print(model.intercept_)
print(model.coef_)