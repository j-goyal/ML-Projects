import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


dataset_train = pd.read_csv("loan_train.csv")


X_train = dataset_train.iloc[:, 1:-1].values
y_train = dataset_train.iloc[:, -1].values


imputer1 = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer1 = imputer1.fit(X_train[:, [7, 8]])
X_train[:, [7, 8]] = imputer1.transform(X_train[:, [7, 8]])

imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer2 = imputer2.fit(X_train[:, [0, 1, 2, 4, 9]])
X_train[:, [0, 1, 2, 4, 9]] = imputer2.transform(X_train[:, [0, 1, 2, 4, 9]])


labelencoder_X = LabelEncoder()
X_train[:, 0] = labelencoder_X.fit_transform(X_train[:, 0])
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1])
X_train[:, 2] = labelencoder_X.fit_transform(X_train[:, 2])
X_train[:, 3] = labelencoder_X.fit_transform(X_train[:, 3])
X_train[:, 4] = labelencoder_X.fit_transform(X_train[:, 4])


transformer = ColumnTransformer(
    transformers=[
        ("OneHot",         # Just a name
         OneHotEncoder(),  # The transformer class
         [10]               # The column(s) to be applied on.
         )
    ],
    remainder='passthrough'  # do not apply anything to the remaining columns
)
X_train = transformer.fit_transform(X_train)

X_train = np.delete(X_train, np.s_[0], axis=1)


"""
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",         # Just a name
         OneHotEncoder(),  # The transformer class
         [4]               # The column(s) to be applied on.
         )
    ],
    remainder='passthrough'  # do not apply anything to the remaining columns
)
X_train = transformer.fit_transform(X_train)

X_train = np.delete(X_train, np.s_[0], axis=1)

"""


dataset_test = pd.read_csv("loan_test.csv")
X_test = dataset_test.iloc[:, 1:12].values


imputer3 = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer3 = imputer3.fit(X_test[:, [7, 8]])
X_test[:, [7, 8]] = imputer3.transform(X_test[:, [7, 8]])

imputer4 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer4 = imputer4.fit(X_test[:, [0, 2, 4, 9]])
X_test[:, [0, 2, 4, 9]] = imputer4.transform(X_test[:, [0, 2, 4, 9]])

labelencoder_X = LabelEncoder()
X_test[:, 0] = labelencoder_X.fit_transform(X_test[:, 0])
X_test[:, 1] = labelencoder_X.fit_transform(X_test[:, 1])
X_test[:, 2] = labelencoder_X.fit_transform(X_test[:, 2])
X_test[:, 3] = labelencoder_X.fit_transform(X_test[:, 3])
X_test[:, 4] = labelencoder_X.fit_transform(X_test[:, 4])


transformer2 = ColumnTransformer(
    transformers=[
        ("OneHot",         # Just a name
         OneHotEncoder(),  # The transformer class
         [10]               # The column(s) to be applied on.
         )
    ],
    remainder='passthrough'  # do not apply anything to the remaining columns
)
X_test = transformer2.fit_transform(X_test)

X_test = np.delete(X_test, np.s_[0], axis=1)

"""

transformer2 = ColumnTransformer(
    transformers=[
        ("OneHot",         # Just a name
         OneHotEncoder(),  # The transformer class
         [4]               # The column(s) to be applied on.
         )
    ],
    remainder='passthrough'  # do not apply anything to the remaining columns
)
X_test = transformer2.fit_transform(X_test)

X_test = np.delete(X_test, np.s_[0], axis=1)

# print(dataset_train.describe())             # statistical summary
# print(dataset.groupby('Outcome').size())    # class distribution
"""


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ***************************** KNN ********************************
knn = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

dataset1 = pd.DataFrame({'Loan_Status': y_pred[:, ]})  #converting array to dataframe
# export dataframe to csv file for submission
export_csv = dataset1.to_csv(r'C:\Users\windows 10\Desktop\ML Project\knn.csv', index = None, header=True)


# ***************************** Naive Bayes **************************
naive = GaussianNB()
naive.fit(X_train, y_train)
pred_naive = naive.predict(X_test)

dataset1 = pd.DataFrame({'Loan_Status': pred_naive[:, ]})  #converting array to dataframe
# export dataframe to csv file for submission
export_csv = dataset1.to_csv(r'C:\Users\windows 10\Desktop\ML Project\naive.csv', index = None, header=True)



# **************************** Decision Tree ***************************
decision = DecisionTreeClassifier(criterion='gini', random_state=0)
decision.fit(X_train, y_train)
decision_tree_pred = decision.predict(X_test)

dataset1 = pd.DataFrame({'Loan_Status': decision_tree_pred[:, ]})  #converting array to dataframe
# export dataframe to csv file for submission
export_csv = dataset1.to_csv(r'C:\Users\windows 10\Desktop\ML Project\decision.csv', index = None, header=True)



# **************************** Random Forest Tree ***********************
rndom = RandomForestClassifier(n_estimators=10, criterion="gini", random_state=0)
rndom.fit(X_train, y_train)
rndom_forest_pred = rndom.predict(X_test)

dataset1 = pd.DataFrame({'Loan_Status': rndom_forest_pred[:, ]})  #converting array to dataframe
# export dataframe to csv file for submission
export_csv = dataset1.to_csv(r'C:\Users\windows 10\Desktop\ML Project\random_forest.csv', index = None, header=True)



# **************************** Logistic Regression ***********************
logistic = LogisticRegression(solver='lbfgs', random_state=0)
logistic.fit(X_train, y_train)
logistic_pred = logistic.predict(X_test)

dataset1 = pd.DataFrame({'Loan_Status': logistic_pred[:, ]})  #converting array to dataframe
# export dataframe to csv file for submission
export_csv = dataset1.to_csv(r'C:\Users\windows 10\Desktop\ML Project\logistic.csv', index = None, header=True)



# **************************** SVM ******************************************
svm = SVC(kernel='linear', random_state=0)
svm.fit(X_train, y_train)
predict_svm = svm.predict(X_test)

dataset1 = pd.DataFrame({'Loan_Status': predict_svm[:, ]})  #converting array to dataframe
# export dataframe to csv file for submission
export_csv = dataset1.to_csv(r'C:\Users\windows 10\Desktop\ML Project\SVM(linear).csv', index=None, header=True)



"""
# k-fold validation (for better model)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(logistic, X=X_train, y=y_train, cv=10)
print(accuracies.mean(), accuracies.std())
"""