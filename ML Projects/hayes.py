import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
import seaborn as sns
dataset = pd.read_csv("hayes-roth.txt")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5].values


# *************************** Visualizing Data ****************************
sns.pairplot(dataset, hue="Outcome")
plt.show()

scatter_matrix(dataset)
plt.show()

# **************************************************************************

print(dataset.describe())                   # statistical summary
print(dataset.groupby('Outcome').size())    # class distribution

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""
algorithms = []
scores = []
names = []

algorithms.append(('Logisitic Regression', LogisticRegression()))
algorithms.append(('K-Nearest Neighbours', KNeighborsClassifier()))
algorithms.append(('Decision Tree Classifier', DecisionTreeClassifier()))
algorithms.append(('Naive Bayes Classifier', GaussianNB()))
algorithms.append(('Random Tree Classifier', RandomForestClassifier()))
algorithms.append(('SVM', SVC(gamma='auto')))

for name, algo in algorithms:
    k_fold = model_selection.KFold(n_splits=10, random_state=0)

    # Applying k-cross validation
    cvResults = model_selection.cross_val_score(algo, X_train, y_train,
                                                cv=k_fold, scoring='accuracy')

    scores.append(cvResults)
    names.append(name)
    print(str(name) + ' : ' + str(cvResults.mean()))
"""

"""           Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""


# ***************************** KNN ********************************
knn = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('Using KNN', accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# ***************************** Naive Bayes **************************
naive = GaussianNB()
naive.fit(X_train, y_train)
pred_naive = naive.predict(X_test)
print('Using Naive Bayes', accuracy_score(y_test, pred_naive))
print(confusion_matrix(y_test, pred_naive))

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

# **************************** Logistic Regression ***********************
logistic = LogisticRegression(random_state=0)
logistic.fit(X_train, y_train)
logistic_pred = logistic.predict(X_test)
print('Using Logistic Regression', accuracy_score(y_test, logistic_pred))
print(confusion_matrix(y_test, logistic_pred))

# **************************** SVM ******************************************
svm = SVC(kernel='rbf', random_state=0)
svm.fit(X_train, y_train)
predict_svm = svm.predict(X_test)
print('Using SVM kernel', accuracy_score(y_test, predict_svm))
print(confusion_matrix(y_test, predict_svm))
