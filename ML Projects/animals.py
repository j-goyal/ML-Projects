import numpy as np
import pandas as pd


from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



dataset = pd.read_csv('animals.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, [16]].values

print(dataset.describe())                   # statistical summary
print(dataset.groupby('type').size())       # class distribution

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

obj = ColumnTransformer([('transformer', OneHotEncoder(), [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15])], remainder='passthrough')
X = obj.fit_transform(X)

# AVOID DUMMY TRAP
X = np.delete(X, np.s_[0:29:2], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# ***************************** KNN ()********************************
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('Using KNN', accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# **************************** Logistic Regression ***********************
logistic = LogisticRegression(random_state=0)
logistic.fit(X_train, y_train)
logistic_pred = logistic.predict(X_test)
print('Using Logistic Regression', accuracy_score(y_test, logistic_pred))
print(confusion_matrix(y_test, logistic_pred))

# **************************** Decision Tree ***************************
decision = DecisionTreeClassifier(criterion='gini', random_state=0)
decision.fit(X_train, y_train)
decision_tree_pred = decision.predict(X_test)
print('Using Decision Tree', accuracy_score(y_test, decision_tree_pred))
print(confusion_matrix(y_test, decision_tree_pred))

# **************************** Random Forest Tree ***********************
rndom = RandomForestClassifier(n_estimators=10, criterion="gini", random_state=0)
rndom.fit(X_train, y_train)
rndom_forest_pred = rndom.predict(X_test)
print('Using Random Forest Tree', accuracy_score(y_test, rndom_forest_pred))
print(confusion_matrix(y_test, rndom_forest_pred))

# ***************************** Naive Bayes **************************
naive = GaussianNB()
naive.fit(X_train, y_train)
pred_naive = naive.predict(X_test)
print('Using Naive Bayes', accuracy_score(y_test, pred_naive))
print(confusion_matrix(y_test, pred_naive))

# **************************** SVM (linear kernel) ******************************************
svm = SVC(kernel='linear', random_state=0)
svm.fit(X_train, y_train)
predict_svm = svm.predict(X_test)
print('Using SVM kernel', accuracy_score(y_test, predict_svm))
print(confusion_matrix(y_test, predict_svm))



# print(logistic_pred)
# print(y_test)

# k-fold validation (for better model)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(svm, X=X_train, y=y_train, cv=10)
print(accuracies.mean(), accuracies.std())

