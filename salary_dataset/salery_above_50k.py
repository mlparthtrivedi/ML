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
sns.set(rc={'figure.figsize':(11.7,8.27)})
#sns.set(rc={'figure.figsize':(31.7,28.27)})



#Read in and Explore the Data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#Data Analysis
features_details = train.describe(include = "all");
nan_value_detail = pd.isnull(train).sum();

#Data Visualization

#single features with output
sns.barplot(x="sex", y="output", data=train)
sns.barplot(x="race", y="output", data=train)
sns.barplot(x="native-country", y="output", data=train)
sns.barplot(x="workclass", y="output", data=train)
sns.barplot(x="education", y="output", data=train)
sns.barplot(x="marital-status", y="output", data=train,linewidth=6)
sns.barplot(x="occupation", y="output", data=train,linewidth=6)
sns.barplot(x="relationship", y="output", data=train,linewidth=6)

#compare two value with output
sns.barplot(x="race", y="output", hue="sex", data=train)
sns.barplot(x="workclass", y="output", hue="sex", data=train)
sns.barplot(x="education", y="output", hue="sex", data=train)
sns.barplot(x="marital-status", y="output", hue="sex", data=train)
sns.barplot(x="native-country", y="output", hue="sex", data=train)
sns.regplot(x="hours-per-week", y="output", data=train)


sns.factorplot(x="output", y="hours-per-week", data=train)
sns.FacetGrid(train, col="output").map(sns.distplot, "hours-per-week")

#data manipulation

#sort the ages into logical categories
train["age"] = train["age"].fillna(-0.5)
test["age"] = test["age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["age"], bins, labels = labels)
#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="output", data=train)
sns.barplot(x="AgeGroup", y="output", hue="sex", data=train)


train = train.drop(['age'], axis = 1)
test = test.drop(['age'], axis = 1)

#remove nan values
pd.value_counts(train['workclass'].values, sort=False)
train = train.fillna({"workclass": " Private"})
test = test.fillna({"workclass": " Private"})


pd.value_counts(train['occupation'].values, sort=False)
train = train.fillna({"occupation": " Prof-specialty"})
test = test.fillna({"occupation": " Prof-specialty"})


pd.value_counts(train['native-country'].values, sort=False)
train = train.fillna({"native-country": " United-States"})
test = test.fillna({"native-country": " United-States"})


pd.value_counts(train['marital-status'].values, sort=False)


#convert to numbers
#map each WorkClass value to a numerical value
workclass_mapping = {" Never-worked": 1, " Without-pay": 2, " Private": 3,
                     " Federal-gov": 3, " Self-emp-not-inc": 4, " State-gov": 5, " Local-gov": 6,
                     " Self-emp-inc": 7}
train['workclass'] = train['workclass'].map(workclass_mapping)
test['workclass'] = test['workclass'].map(workclass_mapping)

train = train.drop(['education'], axis = 1)
test = test.drop(['education'], axis = 1)

#map each maritalStatus value to a numerical value
maritalStatus_mapping = {" Married-spouse-absent": 1, " Widowed": 2, " Never-married": 3,
                     " Married-AF-spouse": 3, " Separated": 4, " Married-civ-spouse": 5, " Divorced": 6}
train['marital-status'] = train['marital-status'].map(maritalStatus_mapping)
test['marital-status'] = test['marital-status'].map(maritalStatus_mapping)


#map each occupation value to a numerical value
unique_occupation = train['occupation'].unique()
occupation_mapping = {}
i = 1;
for x in unique_occupation:
  occupation_mapping[x] = i
  i = i + 1;
train['occupation'] = train['occupation'].map(occupation_mapping)
test['occupation'] = test['occupation'].map(occupation_mapping)

#map each relationship
unique_relationship = train['relationship'].unique()
relationship_mapping = {}
i = 1;
for x in unique_relationship:
  relationship_mapping[x] = i
  i = i + 1;
train['relationship'] = train['relationship'].map(relationship_mapping)
test['relationship'] = test['relationship'].map(relationship_mapping)

#map each race
unique_race = train['race'].unique()
race_mapping = {}
i = 1;
for x in unique_race:
  race_mapping[x] = i
  i = i + 1;
train['race'] = train['race'].map(race_mapping)
test['race'] = test['race'].map(race_mapping)

#map each sex
unique_sex = train['sex'].unique()
sex_mapping = {}
i = 1;
for x in unique_sex:
  sex_mapping[x] = i
  i = i + 1;
train['sex'] = train['sex'].map(sex_mapping)
test['sex'] = test['sex'].map(sex_mapping)

#map each native country
unique_country = train['native-country'].unique()
country_mapping = {}
i = 1;
for x in unique_country:
  country_mapping[x] = i
  i = i + 1;
train['native-country'] = train['native-country'].map(country_mapping)
test['native-country'] = test['native-country'].map(country_mapping)

#map each native AgeGroup
unique_AgeGroup = train['AgeGroup'].unique()
AgeGroup_mapping = {}
i = 1;
for x in unique_AgeGroup:
  AgeGroup_mapping[x] = i
  i = i + 1;
train['AgeGroup'] = train['AgeGroup'].map(AgeGroup_mapping)
test['AgeGroup'] = test['AgeGroup'].map(AgeGroup_mapping)


#Choosing the Best Model
from sklearn.model_selection import train_test_split
x = train.drop(['output'], axis=1)
y = train["output"]
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.22, random_state = 0)


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)

# Support Vector Machines
'''from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)'''

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)

# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)

models = pd.DataFrame({
    'Model': [ 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [ acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)


'''
#model optimizing
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


learning_rates = list(range(1,train.shape[1]))
train_results = []
test_results = []
for eta in learning_rates:
   model = GradientBoostingClassifier(max_features=eta)
   model.fit(x_train, y_train)
   train_pred = model.predict(x_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = model.predict(x_val)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(learning_rates, train_results, 'b', label="Train AUC")
line2, = plt.plot(learning_rates, test_results, 'r', label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel('learning rate')
plt.show()

#kfold cross validation
from sklearn.model_selection import cross_val_score
gbx = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10)
scores = cross_val_score(gbx, x_train, y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
#86.54!!
#print("oob score:", round(gbx.oob_score_, 4)*100, "%")

#confusion metrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(gbx, x_train, y_train, cv=10)
CM = confusion_matrix(y_train, predictions)
#precision and recall
from sklearn.metrics import precision_score, recall_score
print("Precision:", precision_score(y_train, predictions))
print("Recall:",recall_score(y_train, predictions))
#f1 score
from sklearn.metrics import f1_score
f1_score(y_train, predictions)

#features minify
importances = pd.DataFrame({'feature':x_train.columns,'importance':np.round(randomforest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)
importances.plot.bar()

'''

########################### TEST start ###################
#Choosing the Best Model
from sklearn.model_selection import train_test_split
a = test.drop(['output'], axis=1)
b = test["output"]

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
bpred = gbk.predict(a)
acc_gbk = round(accuracy_score(bpred, b) * 100, 2)
print(acc_gbk)




