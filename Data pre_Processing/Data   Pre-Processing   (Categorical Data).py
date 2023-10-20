import pandas as pd
import numpy as np
hotel_df = pd.read_csv("C:\\Users\\nagbi\\OneDrive\\Desktop\\hotel_bookings.csv")
np.random.seed(0)
hotel_df.info()
hotel_df.head(5)
hotel_df.isnull().sum()
hotel_df =hotel_df.drop(['company','agent'],  axis = 1)
hotel_df.head()
cat_df_hotel = hotel_df.select_dtypes(include=['object']).copy()
cat_df_hotel.head()
print(cat_df_hotel.isnull().sum())
cat_df_hotel = cat_df_hotel.fillna(cat_df_hotel['country'].value_counts().index[0])
print("After replacing null values with mode:")
print(cat_df_hotel.isnull().sum())
cat_df_hotel.describe()
print(cat_df_hotel).unique().sample(10)
cat_df_hotel_specific = cat_df_hotel.copy()
cat_df_hotel_specific['hotel'] = np.where(cat_df_hotel_specific['hotel'].str.contains('City Hotel'), 1, 0)

cat_df_hotel_specific.sample(10)
cat_df_hotel_sklearn = cat_df_hotel.copy()

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
cat_df_hotel_sklearn['country_code'] = lb_make.fit_transform(cat_df_hotel['country'])

cat_df_hotel_sklearn.sample(5)
cat_df_hotel_onehot = cat_df_hotel.copy()
cat_df_hotel_onehot = pd.get_dummies(cat_df_hotel_onehot, columns=['arrival_date_month'])

print(cat_df_hotel_onehot.head())



