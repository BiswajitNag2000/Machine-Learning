import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet
import statsmodels.api as sm
df = pd.read_csv("/content/housing_price.csv")
df.head()
df.isna().sum()
x = df.drop(columns = "median_house_value").values 
y = df.median_house_value.values
print("independent data\n",x)
print("\ndependent data\n",y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state = 10)
print("x_train and x_test dataset shape",x_train.shape,x_test.shape)
print("y_train and y_test dataset shape",y_train.shape,y_test.shape)
regressor = LinearRegression()  
regressor.fit(x_train, y_train)
regressor.score(x_test,y_test)
print("R2 value:",regressor.score(x_test,y_test))
print("\ncoefficient: \n ",regressor.coef_)
print("\nintercept:",regressor.intercept_)
y_pred = regressor.predict(x_test)
pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = pred.head(10)
print(df1)
mse = np.mean((df1.Predicted-df1.Actual)**2)
print( "coefficient :",regressor.coef_)
print("\n intercepter :",regressor.intercept_)
print("\n mse :",mse)
print("\n final score",regressor.score(x_test,y_test))
X2 = sm.add_constant(x)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
