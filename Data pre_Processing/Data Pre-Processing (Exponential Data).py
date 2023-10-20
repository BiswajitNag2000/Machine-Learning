from mlxtend.preprocessing import minmax_scaling
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df2 = pd.read_csv("C:\\Users\\nagbi\\OneDrive\\Desktop\\hotel_bookings.csv")
original_data = np.random.exponential(size = 1000)
df2= pd.DataFrame(original_data)
df2.head()
scaled_data = minmax_scaling(df2, columns = [0])
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
from scipy import stats
from sklearn import preprocessing
normalized_data = preprocessing.normalize(df2)   
fig, ax=plt.subplots(1,2)
sns.distplot(df2[0], ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data, ax=ax[1])
ax[1].set_title("Normalized data")