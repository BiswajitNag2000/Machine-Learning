import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('diabetes.csv')

X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
test_accuracy[i] = knn.score(X_test, y_test)
knn2 = KNeighborsClassifier(n_neighbors=2)
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train, y_train)
knn2.fit(X_train, y_train)
pred2 = knn2.predict(X_test)
acc_gnb = accuracy_score(y_test, pred2)
pred3 = knn3.predict(X_test)
acc_gnb = accuracy_score(y_test, pred3)

print("Target/Output variables:", X, '\n', "Feature Vectotrs:", y)
print("After splitting the data-", '\n' "size of input train data is:", sys.getsizeof(X_train),
      '\n'"sizeof input test data is:", sys.getsizeof(X_test), '\n'
                                                               "size of output train data is:", sys.getsizeof(y_train),
      '\n'
      "size of output test data is:", sys.getsizeof(y_test))
plt.show()

print("Accuracy score for 2 neighbours-", '\n''Accuracy Score: ', acc_gnb, '\n'
                                                                           "Accuracy score for 3 neighbours-",
      '\n''Accuracy Score: ', acc_gnb)

y_pred_proba = knn2.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Knn')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Knn(n_neighbors=2) ROC curve')
plt.show()

y_pred_proba = knn3.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Knn')
plt.xlabel('False Positve Rate')
plt.ylabel('True Positive Rate')
plt.title('Knn(n_neighbors=3) ROC curve')
plt.show()
