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
data = pd.read_csv("new.csv")
train=data.sample(frac=0.8,random_state=200)
test=data.drop(train.index)

features_details = train.describe(include = "all");
nan_value_detail = pd.isnull(train).sum();

#Data Visualization

#draw a bar plot of output by pregnant_count
sns.barplot(x="pregnant_count", y="output", data=train)
for i in range(1,16):
  print("Percentage of pregnant_count who has ",i," output:", train["output"][train["pregnant_count"] == i].value_counts(normalize = True)[1]*100)

#draw a bar plot of output byPlasma glucose concentration
sns.factorplot(x="output", y="Plasma glucose concentration", data=train)
sns.FacetGrid(train, col="output").map(sns.distplot, "Plasma glucose concentration")

#draw a bar plot of output BP
sns.factorplot(x="output", y="BP", data=train)
sns.FacetGrid(train, col="output").map(sns.distplot, "BP")

#draw a bar plot of output Triceps skinfold thickness
sns.factorplot(x="output", y="Triceps skinfold thickness", data=train)
sns.FacetGrid(train, col="output").map(sns.distplot, "Triceps skinfold thickness")

#2-Hour serum insulin
#draw a bar plot of output Triceps skinfold thickness
sns.factorplot(x="output", y="2-Hour serum insulin", data=train)
sns.FacetGrid(train, col="output").map(sns.distplot, "2-Hour serum insulin")

#draw a bar plot of output BMI
sns.factorplot(x="output", y="BMI", data=train)
sns.FacetGrid(train, col="output").map(sns.distplot, "BMI")

#draw a bar plot of output Diabetes pedigree function
sns.factorplot(x="output", y="Diabetes pedigree function", data=train)
sns.FacetGrid(train, col="output").set(xticks=[0,0.2,0.4,0.6,0.8,1], yticks=[0,1]).map(sns.distplot, "Diabetes pedigree function")

#draw a bar plot of output BMI
sns.factorplot(x="output", y="Age", data=train)
sns.FacetGrid(train, col="output").map(sns.distplot, "Age")
#sort the ages into logical categories
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)
#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="output", data=train)
plt.show()

#saleprice correlation matrix
k = 10 #number of variables for heatmap
corrmat = data.corr()
cols = corrmat.nlargest(k, 'output')['output'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()