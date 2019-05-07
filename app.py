import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# features matrix
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
# y vector (class, label, value)
y = [4, 5, 20, 14, 32, 22, 38, 43]
# making x and y a numpy arrays in order to make some math operations on them
x, y = np.array(x), np.array(y)

# print a nice visual matrix with help of pandas
col1 = ['X1', 'X2']
df = pd.DataFrame(x, columns=col1)
print(df)

# fit the features and their classes to make a model
model = LinearRegression().fit(x, y)

# r_sq = model.score(x, y)
# print('coefficient of determination:', r_sq)
# print('intercept:', model.intercept_)
# print('slope:', model.coef_)

# TODO:
# predict the y for the x matrix, this is only for calculating the
# error and i haven't done this yet
y_pred = model.predict(x)
y_pred = np.array(y_pred)
print('predicted response:', y_pred.round(), sep='\n')
print('actual Y values:', y, sep='\n')

# new inputs
x_new = np.arange(10).reshape((-1, 2))
df = pd.DataFrame(x_new, columns=col1)
print("New Inputs: ", df, sep='\n')
# predicted output
y_new = model.predict(x_new)
print("new input predicted values: ", y_new)
