import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv('/content/car_evaluation.csv', header=None)
df.head()
df.shape
df.describe()
col_names = ['buying', 'maintainance', 'doors', 'persons', 'luggage_capacity', 'safety', 'class']
df.columns = col_names
col_names
df.head()
col_names = ['buying', 'maintainance', 'doors', 'persons', 'luggage_capacity', 'safety', 'class']
print("Counting the frequency of each categorical variable in the dataset")
for col in col_names:
    
    print(df[col].value_counts())  
    print("Frequency of each ordinal data in the target column - class:")
df['class'].value_counts()
df.isnull().sum()
X = df.drop(['class'], axis=1)

y = df['class']
print("Feature vectors are:")
X.head()
print("Target column is:")
y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.shape, X_test.shape
import category_encoders as ce
encoder = ce.OrdinalEncoder(cols=['buying', 'maintainance', 'doors', 'persons', 'luggage_capacity', 'safety'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)
X_train.head()
X_test.head()
from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)
y_pred_gini
from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))
import graphviz 
from sklearn import tree
dot_data = tree.export_graphviz(clf_gini, out_file=None, 
                              feature_names=X_train.columns,  
                              class_names=y_train,  
                              filled=True, rounded=True,  
                              special_characters=True)

graph = graphviz.Source(dot_data) 

print(graph)
features=pd.DataFrame({'Features':X_train.columns,'Importance':np.round(clf_gini.feature_importances_,3)})
features=features.sort_values('Importance',ascending=False)
print (features)