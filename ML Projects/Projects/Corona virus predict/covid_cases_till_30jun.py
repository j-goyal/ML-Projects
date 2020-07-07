# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:09:12 2020

@author: windows 10
"""

from datetime import date
import numpy 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression

from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("india_covid.csv")
x = [i for i in range(1,107)]

y = dataset.iloc[:, 1].values
y = list(y)

# apply polynomial regression with 5 features
mymodel = numpy.poly1d(numpy.polyfit(x, y, 5))

myline = numpy.linspace(1, 106)

plt.scatter(x, y, color='green')
plt.plot(myline, mymodel(myline), color='red')
plt.show()

#predict future values for next 31 days
case=list()
for i in range(107,138):
    case.append(mymodel(i))


from datetime import timedelta

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

li = list()
start_dt = date(2020, 5, 31)
end_dt = date(2020, 6, 30)
for dt in daterange(start_dt, end_dt):
    li.append(dt.strftime("%Y-%m-%d"))


future = dict()
for i in range(31):
    future[li[i]]=case[i]
    
"""    
for i in range(len(case)-1):
    print(case[i+1]-case[i])
"""