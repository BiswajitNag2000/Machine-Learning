import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv("Breast_cancer_data.csv")
corr = df.corr()
fig = plt.figure(figsize=(15, 12))
r = sns.heatmap(corr, cmap='Blues')
r.set_title("Data Correlation")
y = df["diagnosis"].values
X = df.drop(["diagnosis"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
svc_diagnosis = SVC(C=10, kernel='linear')
svc_diagnosis.fit(X_train, y_train)
pred = svc_diagnosis.predict(X_test)
acc_svc_diagnosis = accuracy_score(y_test, pred)
svc_diagnosis = SVC(C=10, kernel='rbf', gamma=2)
svc_diagnosis.fit(X_train, y_train)
pred = svc_diagnosis.predict(X_test)
acc_svc_diagnosis = accuracy_score(y_test, pred)

print(df.sample(5), '\n', df.describe())
plt.show()
print("After splitting the data-", '\n'

                                   "size of input train data is:", sys.getsizeof(X_train), '\n',

      "sizeof input test data is:",
      sys.getsizeof(X_test), '\n',
      "size of output train data is:", sys.getsizeof(y_train), '\n',
      "size of output test data is:", sys.getsizeof(y_test), '\n',
      'Accuracy Score of Linear Model: ', acc_svc_diagnosis, '\n',
      'Accuracy Score of Gaussian Model: ', acc_svc_diagnosis)
