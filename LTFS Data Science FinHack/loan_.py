# -*- coding: utf-8 -*-
#Import Necessary Libraries
import numpy as np
import pandas as pd

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


#Read in and Explore the Data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#Read in and Explore the Data
train = train.head(1000)
test = test.head(250)

features_details = train.describe(include = "all");
nan_value_detail = pd.isnull(train).sum();
print(features_details)
print(nan_value_detail)

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(train['PERFORM_CNS.SCORE'], train['loan_default'])
ax.set_xlabel('Credit score')
ax.set_ylabel('load defaulter')
plt.show()

#test credit score
from sklearn.model_selection import train_test_split
x = train[["PERFORM_CNS.SCORE"]]
x_test = test[["PERFORM_CNS.SCORE"]]
y = train[["loan_default"]]
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.22, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_val = sc_X.transform(x_val)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

from sklearn.metrics import accuracy_score

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_pred)




# Visualising the Training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, logreg.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()