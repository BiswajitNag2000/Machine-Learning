import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.api import add_constant, OLS
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from io import StringIO
df = pd.read_csv("TaxiFare.csv")
df.head(10)
df = df.drop("unique_id",axis = 1)
df.describe()
df.isna().sum()
df.info()
df["date_time_of_pickup"] = pd.to_datetime(df["date_time_of_pickup"])
new_df = df.assign(hour = df["date_time_of_pickup"].dt.hour, 
                  dayOfTheMonth = df["date_time_of_pickup"].dt.day,
                  month = df["date_time_of_pickup"].dt.month, 
                  dayOfTheWeek = df["date_time_of_pickup"].dt.dayofweek)
new_df.drop("date_time_of_pickup", axis = 1, inplace = True)

new_df.head()
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c 
    return km

new_df["distance"] = haversine_np(new_df["longitude_of_pickup"], new_df["latitude_of_pickup"],
                                   new_df["longitude_of_dropoff"], new_df["latitude_of_dropoff"])

new_df.head()
new_df.describe().transpose()
print(new_df["amount"].describe())
fullRaw = new_df[new_df["amount"] >= 2.5]
print(new_df["amount"].describe())

print(new_df["distance"].describe())
new_df = new_df[(new_df["distance"] >= 1) & (new_df["distance"] <= 130)]
print(new_df["distance"].describe())
new_df.shape


x = new_df.drop(["amount"], axis = 1).copy()
y = new_df["amount"].copy()


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state=100) 


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


from sklearn.ensemble import RandomForestRegressor
M1 = RandomForestRegressor(random_state=123)
M1 = M1.fit(x_train,y_train)
varImpDf = pd.DataFrame()
varImpDf["Importance"] = M1.feature_importances_
varImpDf["Variable"] = x_train.columns
varImpDf.sort_values("Importance", ascending = False, inplace = True)

varImpDf.head()
testPredDf = pd.DataFrame()

testPredDf["Prediction"] = M1.predict(x_test)
testPredDf["Actual"] = y_test.values
testPredDf.head()
print("RMSE",np.sqrt(np.mean((testPredDf["Actual"] - testPredDf["Prediction"])**2)))
print("MAPE",(np.mean(np.abs(((testPredDf["Actual"] - testPredDf["Prediction"])/testPredDf["Actual"]))))*100)