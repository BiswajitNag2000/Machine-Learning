import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing  import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from matplotlib.colors import ListedColormap
from sklearn import metrics
import seaborn as sns
df1 = pd.read_csv("C:\\Users\\nagbi\\OneDrive\\Desktop\\TaxiFare.csv")
df1.h1n1_vaccine.value_counts()
df1.shape
print(df1)
df1.isnull().sum()/len(df1)*100
df1 = df1.drop("has_health_insur",axis = 1)
df1 = df1.dropna()
df1.shape
purchased=df1[df1.h1n1_vaccine==0].h1n1_vaccine.count()
notpurchased=df1[df1.h1n1_vaccine==1].h1n1_vaccine.count()
plt.bar(0,purchased,label='no')
plt.bar(1,notpurchased,label='yes')
plt.xticks([])
plt.ylabel('Count')
plt.legend()
plt.show()
df1.info()
from sklearn import preprocessing
df1.age_bracket.unique() 
le = preprocessing.LabelEncoder()
df1['age_bracket'] = le.fit_transform(df1.age_bracket.values)
df1['age_bracket']
df1.qualification = le.fit_transform(df1.qualification.values)
df1.qualification
df1.race = le.fit_transform(df1.race.values)
df1.sex = le.fit_transform(df1.sex.values)
df1.income_level = le.fit_transform(df1.income_level.values)
df1.marital_status = le.fit_transform(df1.marital_status.values)
df1.housing_status = le.fit_transform(df1.housing_status.values)
df1.employment = le.fit_transform(df1.employment.values)
df1.census_msa = le.fit_transform(df1.census_msa.values)


print(df1)
x= df1.drop("h1n1_vaccine",axis = 1)
y= df1['h1n1_vaccine'] 

x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=0.20)
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(x_train, y_train)
 

y_pred = clf.predict(x_test)
 
from sklearn import metrics 
print()
score1 = metrics.accuracy_score(y_test, y_pred)

print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
x_smot, y_smot = oversample.fit_resample(x, y)

purchased=y_smot[y_smot==0].count()
notpurchased=y_smot[y_smot==1].count()
plt.bar(0,purchased,label='no')
plt.bar(1,notpurchased,label='yes')
plt.xticks([])
plt.ylabel('Count')
plt.legend()
plt.show()
x_train,x_test,y_train,y_test=train_test_split(x_smot, y_smot,test_size=0.20)
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(x_train, y_train)
 

y_pred = clf.predict(x_test)
 
from sklearn import metrics 
print()
score2 =  metrics.accuracy_score(y_test, y_pred)

print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))
print("before smot accuracy is:",score1)
print("after smot accuracy is:",score2)
