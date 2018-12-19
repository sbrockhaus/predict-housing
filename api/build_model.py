
# import python libraries 
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt   #Data visualisation libraries 
#import seaborn as sns
#matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.externals import joblib


######## data wrangling

# load data 
data = pd.read_csv('housing.csv')
data = pd.DataFrame(data)
print("Housing data loaded.")

# create data frame with covariates and vector with response values
x = data.loc[: , data.columns != 'house_value']
y = data['house_value']

######### do a split into train and test data to evaluate RMSE out-of-bag 
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=101)
#
#lm = LinearRegression()
#lm.fit(x_train,y_train)
#
## compute MSE for training and test data 
#mse_train = np.mean( (y_train - lm.predict(x_train)) ** 2 )
#mse_test = np.mean( (y_test - lm.predict(x_test)) ** 2 )
#print("MSE train: ", round(mse_train, 2), "; MSE test: ", round(mse_test, 2))



######## model fitting 

# create and fit mode 
lm = LinearRegression()
lm.fit(x,y)
print("Model fitted.")

# save the model 
joblib.dump(lm, 'model.pkl')
print("Model dumped.")

# load model into workspace 
# lm = joblib.load('model.pkl')


######## compute standard deviation 

# compute residual sum of squares 
epsilon2 = (y - lm.predict(x)) ** 2
sqr = np.sum(epsilon2)

# compute degrees of freedom of lm as n - p 
df = (x.shape[0] + 1) - (x.shape[1] + 2)

# compute standard deviation of regression errors
std = (sqr/df) ** (0.5)
std
joblib.dump(std, 'std.pkl')
print("std dumped.")
print("std:", str(round(std, 3)))




