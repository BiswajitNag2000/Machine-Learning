import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
heart_df=pd.read_csv("Heart_Disease.csv")
heart_df.head()
heart_df.describe()
heart_df.info()
heart_df.isnull().sum()
heart_df = heart_df.apply(lambda x: x.fillna(x.mean()),axis=0)
heart_df.isnull().sum()
heart_df.nunique()/heart_df.shape[0]

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report,accuracy_score
from sklearn.metrics import roc_curve , auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
X= heart_df.drop("TenYearCHD",axis=1)
y=heart_df["TenYearCHD"]
print("Columns in X :",X.columns)
print("y :",y)
print("shape of X:",X.shape)
print("shape of y:",y.shape[0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
import sys
print("After splitting the data-")
print("size of input train data is:", sys.getsizeof(X_train))
print("sizeof input test data is:", sys.getsizeof(X_test))
print("size of output train data is:", sys.getsizeof(y_train))
print("size of output test data is:", sys.getsizeof(y_test))
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)
X_train=pd.DataFrame(X_train,columns=X.columns)
X_test=pd.DataFrame(X_test,columns=X.columns)
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(classification_report(y_test,y_pred))
conf_mat=confusion_matrix(y_test,y_pred)
print("Confusion matrix is \n",conf_mat)
plt.figure(figsize=(7,7))
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(conf_mat, square=True,cmap="BuPu",annot=True,fmt='d')
plt.xlabel('true label')
plt.ylabel('predicted label')

print("accuracy score : ",accuracy_score(y_test,y_pred))
print("accuracy:", round(100*accuracy_score(y_test,y_pred)),"%")
print(model.intercept_ )
print(model.coef_ )