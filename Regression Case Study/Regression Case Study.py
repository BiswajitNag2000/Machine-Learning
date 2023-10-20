import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.api import add_constant, OLS
from sklearn.linear_model import LinearRegression
df = pd.read_csv("insurance.csv")
df.head(10)
df.describe()
df.isna().sum()
df.info()
new_df = df.copy()
sns.boxplot(new_df['charges'])
hp = sorted(new_df['charges'])
q1, q3= np.percentile(hp,[25,75])
lower_bound = q1 -(1.5 * (q3-q1)) 
upper_bound = q3 + (1.5 * (q3-q1))
below = new_df['charges'] > lower_bound
above = new_df['charges'] < upper_bound
new_df = new_df[below & above]
new_df.shape
new_df.describe()
sns.distplot(new_df['charges'])
new_df.describe().transpose()
fullRaw2 = pd.get_dummies(new_df).copy() 
print(fullRaw2.shape)
fullRaw2.head()
fullRaw2.shape
x = fullRaw2.drop(["charges"], axis = 1).copy()
y = fullRaw2["charges"].copy()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state=100) 
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
model = LinearRegression().fit(x_train,y_train)
pred = model.predict(x_test)
score1 = model.score(x_test,y_test)
score1
1 - (1-model.score(x_test, y_test))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
import seaborn as sns

residuals_linear = y_test - model.predict(x_test)
sns.distplot(residuals_linear)
plt.title('Linear')
predictors = x_train.columns

coef = pd.Series(model.coef_,predictors).sort_values()

coef.plot(kind='bar', title='Model Coefficients')
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif= pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif["features"] = x.columns
vif["VIF"] 
x = add_constant(x)
tempMaxVIF = 5
maxVIF = 5
trainXCopy = x.copy()
counter = 1
highVIFColumnNames = []

while (tempMaxVIF >= maxVIF):
    tempVIFDf = pd.DataFrame()
    tempVIFDf['VIF'] = [variance_inflation_factor(trainXCopy.values, i) for i in range(trainXCopy.shape[1])]
    tempVIFDf['Column_Name'] = trainXCopy.columns
    tempVIFDf.dropna(inplace=True)
    tempColumnName = tempVIFDf.sort_values(["VIF"])[-1:]["Column_Name"].values[0]
    tempMaxVIF = tempVIFDf.sort_values(["VIF"])[-1:]["VIF"].values[0]

if (tempMaxVIF >= maxVIF):
     print(counter)
     print(tempColumnName)
     trainXCopy = trainXCopy.drop(tempColumnName, axis = 1)    
     highVIFColumnNames.append(tempColumnName)
counter = counter + 1
print(highVIFColumnNames)

highVIFColumnNames.remove('const')
print(highVIFColumnNames)
print(len(highVIFColumnNames))

x_new = x.drop(highVIFColumnNames, axis = 1)
print(x.shape)
x_new
x_train2, x_test2, y_train2, y_test2 = train_test_split(x_new,y,test_size = 0.20,random_state=10) 
m1ModelDef = OLS(y_train2,x_train2) 
m1ModelBuild = m1ModelDef.fit()
m1ModelBuild.summary()
score3 =  m1ModelBuild.rsquared
print (score3)
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
ridgeReg = Ridge(alpha=0.00001, normalize=True)
x3 = fullRaw2.drop(["charges"], axis = 1).copy()
y3 = fullRaw2["charges"].copy()
x_train3,x_test3,y_train3,y_test3 = train_test_split(x3,y3,test_size = 0.20,random_state=150) 

ridgeReg.fit(x_train3,y_train3)
pred = ridgeReg.predict(x_test3)
score4 = ridgeReg.score(x_test3,y_test3)
predictors = x_train.columns
coef = pd.Series(ridgeReg.coef_,predictors).sort_values()
coef.plot(kind='bar', title='Model Coefficients')
lassoReg = Lasso(alpha=0.0001)
lassoReg.fit(x_train3,y_train3)
pred = lassoReg.predict(x_test3)
score5 = lassoReg.score(x_test3,y_test3)
print (score5)
predictors = x_train.columns
coef = pd.Series(lassoReg.coef_,predictors).sort_values()
coef.plot(kind='bar', title='Model Coefficients')
print("all model score is:")
print("simple linear regression:          ",score1)
print("After VIF simple linear regression:",score3)
print("ridge regression:                  ",score4)
print("lasso regression:                  ",score5)
    