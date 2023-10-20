import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv("/content/housing.csv")
df.describe()
df.isnull().sum()
mean_value=df['total_bedrooms'].mean()
df=df.fillna(mean_value)
print("Replacing null values with mean:")
df.isnull().sum()
df.duplicated().sum()
y= df.median_house_value
x=df.drop('median_house_value',axis=1)
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
print("shape of original dataset :", df.shape)
print("shape of input - training set", X_train.shape)
print("shape of output - training set", y_train.shape)
print("shape of input - testing set", X_test.shape)
print("shape of output - testing set", y_test.shape)