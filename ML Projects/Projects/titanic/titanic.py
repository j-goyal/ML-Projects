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

# ********************************* TRAIN DATASET ********************************************
dataset_train = pd.read_csv("train.csv")
X_train = dataset_train.iloc[:, :].values
y_train = dataset_train.iloc[:, 1].values

X_train = np.delete(X_train, np.s_[0, 1, 3, 8, 10], axis=1)


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X_train[:, [2]])
X_train[:, [2]] = imputer.transform(X_train[:, [2]])

X_train = np.delete(X_train, (61,829), axis=0)
y_train = np.delete(y_train, (61,829), axis=0)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("OneHot",         # Just a name
         OneHotEncoder(),  # The transformer class
         [6]               # The column(s) to be applied on.
         )
    ],
    remainder='passthrough'  # do not apply anything to the remaining columns
)
X_train = transformer.fit_transform(X_train)

X_train = np.delete(X_train, np.s_[0], axis=1)

labelencoder_X = LabelEncoder()
X_train[:, 3] = labelencoder_X.fit_transform(X_train[:, 3])

#print(dataset.describe())                   # statistical summary
#print(dataset.groupby('Outcome').size())    # class distribution

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ********************************* TEST DATASET ********************************************

dataset_test = pd.read_csv("test.csv")
X_test = dataset_test.iloc[:, :].values
#y_test = dataset_test.iloc[:, 1].values

X_test = np.delete(X_test, np.s_[0,2,7,9], axis=1)

imputer1 = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer1 = imputer1.fit(X_test[:, [2,5]])
X_test[:, [2,5]] = imputer1.transform(X_test[:, [2,5]])


transformer = ColumnTransformer(
    transformers=[
        ("OneHot",         # Just a name
         OneHotEncoder(),  # The transformer class
         [6]               # The column(s) to be applied on.
         )
    ],
    remainder='passthrough'  # do not apply anything to the remaining columns
)
X_test = transformer.fit_transform(X_test)

X_test = np.delete(X_test, np.s_[0], axis=1)

labelencoder_X = LabelEncoder()
X_test[:, 3] = labelencoder_X.fit_transform(X_test[:, 3])


dataset = pd.read_csv("gender_submission.csv")
y_test = dataset.iloc[:, 1].values

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

# k-fold validation (for better model)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(logistic, X=X_train, y=y_train, cv=10)
print(accuracies.mean(), accuracies.std())