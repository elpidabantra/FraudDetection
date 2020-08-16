""" Import libraries and packages essential for the code """

import numpy as np
import pandas as pd
import csv
import sys
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn import preprocessing
from imblearn.under_sampling import RandomUnderSampler 
from sklearn.model_selection import train_test_split
import imblearn


# read the data using the pandas library
dataset = pd.read_csv('creditcard.csv', header = 0, comment='\t', sep = ",")


"""
# data exploration
print(dataset.shape)
print(dataset.info()) # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.info.html
# interesting info (memory_usage, null_counts)
# we see that we have only numerical values so no need to transform categorical ones into dummy variables and also non-null values

# As most of the columns V1, V2,... V28 are transformed using PCA so neither features make much sense and nor will the descriptive statistics so we will leave them and consider only Time and Amount which makes sense. (cols 0 and 29)
print(dataset[['Time', 'Amount']].describe())
"""



# Data cleaning
# Data Standardizing - Standardizing tends to make the training process well behaved because the numerical condition of the optimization problems is improved
def norm(x):
	return (x - dataset.min()) / (dataset.max() - dataset.min())
normed_dataset = norm(dataset)   # Now all the values of the X dataset are between 0 and 1



# Split labels and features
yold = normed_dataset["Class"]
Xold = normed_dataset
Xold.pop("Class")

undersample = RandomUnderSampler(sampling_strategy=0.5)
X_over, y_over = undersample.fit_resample(Xold, yold)


# *** Note *** without undersampling we have 100% accuracy which is suspicious enough and sign of an imbalanced dataset
normed_dataset = pd.concat([X_over, y_over.reindex(X_over.index)], axis=1)
# Data Split for training
train, test = train_test_split(normed_dataset, test_size=0.2)

train_label = train["Class"]
test_label = test["Class"]
train.pop("Class")
test.pop("Class")



# Feature selection according to their importance - instead of correlation we do this to reduce the data volume
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html

X = train
Y = train_label
model = ExtraTreesClassifier()
model.fit(X, Y)
#print(train.columns.values)
#print(model.feature_importances_) 


train.pop("Time")
train.pop("V1")
train.pop("V2")
train.pop("V5")
train.pop("V6")
train.pop("V8")
train.pop("V13")
train.pop("V15")
train.pop("V19")
train.pop("V20")
train.pop("V21")
train.pop("V22")
train.pop("V23")
train.pop("V24")
train.pop("V25")
train.pop("V26")
train.pop("V27")
train.pop("V28")
train.pop("Amount")



test.pop("Time")
test.pop("V1")
test.pop("V2")
test.pop("V5")
test.pop("V6")
test.pop("V8")
test.pop("V13")
test.pop("V15")
test.pop("V19")
test.pop("V20")
test.pop("V21")
test.pop("V22")
test.pop("V23")
test.pop("V24")
test.pop("V25")
test.pop("V26")
test.pop("V27")
test.pop("V28")
test.pop("Amount")

print(train.columns.values) # only these features were kept


# Select Model

# Build the logistic regression model 

logreg = LogisticRegression()
logreg.fit(train, train_label)
y_predLR = logreg.predict(test)


# Performance Metrics
logreg.score(train, train_label)
print('Score of Logistic Regression classifier on test set: {:.2f}'.format(logreg.score(test, test_label)))
acc_LG = accuracy_score(test_label, y_predLR)
print('Accuracy of Logistic Regression classifier on test set: {:.2f}'.format(acc_LG))
confusion_matrix1 = confusion_matrix(test_label, y_predLR)
print("	", "pred no", "pred yes")
print("actual no", confusion_matrix1[0])  
print("actual yes", confusion_matrix1[1])


sys.exit()
############# After here you can find the results of the other 3 ML Algorithms

# Build RF classifier

clfRF = RandomForestClassifier(n_estimators=100)
clfRF.fit(train, train_label) 
y_predRF = clfRF.predict(test)


# Performance Metrics
clfRF.score(train, train_label)
print('Score of Random Forest classifier on test set: {:.2f}'.format(clfRF.score(test, test_label)))
acc_rf = accuracy_score(test_label, y_predRF)
print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(acc_rf))
confusion_matrix2 = confusion_matrix(test_label, y_predRF)
print("	", "pred no", "pred yes")
print("actual no", confusion_matrix2[0])  
print("actual yes", confusion_matrix2[1])









# Built NB classifier

clfNB = GaussianNB()
clfNB.fit(train, train_label) 
y_predNB = clfNB.predict(test)


# Performance Metrics
clfNB.score(train, train_label)
print('Score of Naive Bayes classifier on test set: {:.2f}'.format(clfNB.score(test, test_label)))
acc_nb = accuracy_score(test_label, y_predNB)
print('Accuracy of Naive Bayes classifier on test set: {:.2f}'.format(acc_nb))
confusion_matrix3 = confusion_matrix(test_label, y_predNB)
print("	", "pred no", "pred yes")
print("actual no", confusion_matrix3[0])  
print("actual yes", confusion_matrix3[1])





# Build the svm model 

clfSVM = svm.SVC(gamma = 'scale')
clfSVM.fit(train, train_label)  
y_predSVM = clfSVM.predict(test)


# Performance Metrics
clfSVM.score(train, train_label)
print('Score of SVM classifier on test set: {:.2f}'.format(clfSVM.score(test, test_label)))
acc_svm = accuracy_score(test_label, y_predSVM)
print('Accuracy of SVM classifier on test set: {:.2f}'.format(acc_svm))
confusion_matrix4 = confusion_matrix(test_label, y_predSVM)
print("	", "pred no", "pred yes")
print("actual no", confusion_matrix4[0])  
print("actual yes", confusion_matrix4[1])




