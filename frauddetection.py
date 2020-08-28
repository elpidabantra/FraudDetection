# Main Libraries for analysis

import numpy as np # linear algebra
import pandas as pd # data  processing, CSV file
import csv # file reading and Writing
import sys # system-spesific parameters and functions
from sklearn import preprocessing # Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn; they might behave badly if the individual features do not more or less look like standard normally distributed data: Gaussian with zero mean and unit variance. The sklearn.preprocessing package provides several common utility functions and transformer classes to change raw feature vectors into a representation that is more suitable for the downstream estimators.
from sklearn.preprocessing import StandardScaler, RobustScaler # Scale the feautres
import imblearn # Handling Imballance data
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler # Resampling Technique


# Classifiers and Modeling Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier   # isws na mn ton valw 


# Features Importances - Selection Libraries
from sklearn.ensemble import ExtraTreesClassifier # This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
from sklearn.feature_selection import SelectFromModel # Meta-transformer for selecting features based on importance weights.


# Performance Metrics and Visualisations
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


import matplotlib.pyplot as plt
#matplotlib inline

import seaborn as sns # visualise random distributions. It uses matplotlb





# read the data using the pandas library
dataset = pd.read_csv('creditcard.csv', header = 0, comment='\t', sep = ",")



### Data Exploration ###



# read the first five rows
dataset.head() 
#check out the dimension of the dataset
dataset.shape 

# Obervations:
    # Interesting info (memory_usage, null_counts = 0)
    # We see that we have only numerical values so no need to transform categorical ones into dummy variables and also non-null values

# Print the full summary and the columns 
dataset.info() 
dataset.columns




## Desrriptive Statistics


# Summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
dataset.describe()
# As most of the columns V1, V2,... V28 are transformed using PCA so neither features make much sense and nor will the descriptive statistics so we will leave them and consider only Time and Amount which makes sense. 
dataset[['Time', 'Amount']].describe()

# Observations:
    # Mean transaction is somewhere is 88 and standard deviation is around 250.
    # The median is 22 which is very less as compared to mean which signifies that there are outliers or our data is highly positive skewed which is effecting the amount and thus the mean. 
    # The maximum transaction that was done is of 25,691 and minimum is 0.


# Check the percentages of fraudulent and non-fraudulent transactions
majority, minority = np.bincount(dataset['Class'])
total = majority + minority
print('Examples:\n    Total: {}\n    Minority: {} ({:.2f}% of total)\n'.format(
    total, minority, 100 * minority / total))
print(f'Percent of Non-Fraudulent Transactions(Majority) = {round(dataset["Class"].value_counts()[0]/len(dataset) * 100,2)}%') # 
print(f'Percent of Fraudulent Transactions(Minority) = {round(dataset["Class"].value_counts()[1]/len(dataset) * 100,2)}%')

# Observations:
    # Only 492 (or 0.173%) of transaction are fraudulent. That means the data is highly unbalanced with respect with target variable Class.
    # Most of the transactions are legitimate. In case we use this data to predtict the frauds, our algorithms will overfit. There will be a bias towards the majority class and the accuracy of the models will be misleading. 
    # So, later on, we will balance the data to make the algorithms to produce reliable results.



# Feature Correlation with Response to the label(Class)
corr = dataset.corrwith(dataset['Class']).reset_index()
corr.columns = ['Index','Correlations']
corr = corr.set_index('Index')
corr = corr.sort_values(by=['Correlations'], ascending = True)
plt.figure(figsize=(9, 12))
fig = sns.heatmap(corr, annot=True, fmt="g", cmap='Set3', linewidths=0.3, linecolor='black')
plt.title("Feature Correlation with Class", fontsize=18)
plt.show()

# Observations:
    # V17, V14, V12 and V10 are negatively correlated. Notice how the lower these values are, the more likely the end result will be a fraud transaction.
    # V2, V4, V11, and V19 are positively correlated. Notice how the higher these values are, the more likely the end result will be a fraud transaction.
    # For some of the features we can observe a good selectivity in terms of distribution for the two values of Class: V4, V11 have clearly separated distributions for Class values 0 and 1,
    # V12, V14, V18 are partially separated, V1, V2, V3, V10 have a quite distinct profile, whilst V20-V28 have similar profiles for the two values of Class and thus not very useful in differentiation of both the classes.
    # In general, with just few exceptions (Time and Amount), the features distribution for legitimate transactions (values of Class = 0) is centered around 0, sometime with a long queue at one of the extremities. 
    # In the same time, the fraudulent transactions (values of Class = 1) have a skewed (asymmetric) distributio.




# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)
# RobustScaler is less prone to outliers.

traindataset = dataset.copy()

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

traindataset['scaled_amount'] = rob_scaler.fit_transform(traindataset['Amount'].values.reshape(-1,1))
traindataset['scaled_time'] = rob_scaler.fit_transform(traindataset['Time'].values.reshape(-1,1))

traindataset.drop(['Time','Amount'], axis=1, inplace=True)
scaled_amount = traindataset['scaled_amount']
scaled_time = traindataset['scaled_time']

traindataset.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
traindataset.insert(0, 'scaled_amount', scaled_amount)
traindataset.insert(1, 'scaled_time', scaled_time)
print(traindataset.describe())




###############################################################################





###############################################################################



### Data Manipulation ###



# Before proceeding with the Random UnderSampling technique we have to separate the orginal dataframe.
# We do this because we want to test our models on the original testing set and not on the testing set created by the Random UnderSampling technique.
# Also, the resampling technique should be done only on the training set. 



# Data Split for training 80:20
X = traindataset.drop(['Class'], axis=1) # Features
Y = traindataset['Class'] # Labels
# The tes_size is being chosen by general rule
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Check the shape
print(X_train.shape, X_test.shape)



# Resampling Technique - Balance the data - Handling imbalanced data Process


# We need ratio = 1 between the two classes
undersample = RandomUnderSampler(sampling_strategy=1) 
X_trainundersam, y_trainundersam = undersample.fit_resample(X_train, y_train)
# Returning to new training set # Concat. # Concatenate pandas objects along a particular axis with optional set logic along the other axes. Can also add a layer of hierarchical indexing on the concatenation axis, which may be useful if the labels are the same (or overlapping) on the passed axis number.
undersamdataset = pd.concat([X_trainundersam, y_trainundersam.reindex(X_trainundersam.index)], axis=1)

# equally distributed
print('Distribution of the Classes in the subsample dataset')
print(undersamdataset['Class'].value_counts()/len(undersamdataset))

# Check the difference
print(undersamdataset) 
print(dataset) 



# Separate undersampled data into X and y sets - split features and labels 
Xnew = undersamdataset.drop(['Class'], axis=1)  # Features
print(Xnew)
Ynew = undersamdataset["Class"] # Mono ta lables
print(Ynew)



"""
Mporw edw na kanw standardize an kanw
"""


##############################################################################


##############################################################################



### Feature Selection ###



# Selecting features with the ExtraTressClassifier and SelectFromModel.
# Note: ExtraTreesClassifier tends to be biased. But This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

etcmodel = ExtraTreesClassifier(n_estimators=100, random_state=42)
etcmodel.fit(Xnew, Ynew)
feat_labels = Xnew.columns.values
print(feat_labels)
feat_import = etcmodel.feature_importances_
print(feat_import)

importances = etcmodel.feature_importances_
std = np.std([tree.feature_importances_ for tree in etcmodel.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]


# Print the feature ranking
print("Feature ranking:")
for f in range(Xnew.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# Plot the impurity-based feature importances of the ExtraTreesClassifier
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# Select the most important Values
# We will use SelectFromModel, using a threshold to extract the most important features
# Setting the threshold for which variables to keep based on their variance

sfm = SelectFromModel(etcmodel, threshold=0.03, prefit=True)
print('Number of features before selection: {}'.format(Xnew.shape[1]))
# Number of features before selection: 30

# Throwing away all the variables which fall below the threshold level␣,→specified above
n_features = sfm.transform(Xnew).shape[1]
print('Number of features after selection: {}'.format(n_features))
# Number of features after selection: 9

selected_features = list(feat_labels[sfm.get_support()])

# split features and labels adding only the selected features
Xnew_2 = undersamdataset[selected_features]
X_test2 = X_test[selected_features]

#check the difference
print(Xnew_2)
print(Xnew)

# The training and testing should have the same features
print(Xnew_2.columns)
print(X_test2.columns)






##############################################################################





##############################################################################

# ta test dataset legontai X_test2, y_test, 
# ta train dataset legontai Xnew_2, Ynew




### Data Modeling ###



# Build the Logistic Regression model

logreg = LogisticRegression()
logreg.fit(Xnew_2, Ynew)
y_predLR = logreg.predict(X_test2)


# Performance Metrics
acc_LR = accuracy_score(y_test, y_predLR)
print('Accuracy of Logistic Regression classifier on test set: {:.2f}'.format(acc_LR))
confusion_matrix1 = confusion_matrix(y_test, y_predLR)
print("	", "pred no", "pred yes")
print("actual no", confusion_matrix1[0])  
print("actual yes", confusion_matrix1[1])


# Classification report
labels = ['No Fraud', 'Fraud']
print ('LogisticRegression:')
print(classification_report(y_test, y_predLR, target_names=labels))

# ROC_AUC
aucLR = roc_auc_score(y_test, y_predLR)
print('Logistic Regression ROC_AUC Score: ', roc_auc_score(y_test, y_predLR))





# Build RandomForest model

clfRF = RandomForestClassifier(n_estimators=100, random_state=42)
clfRF.fit(Xnew_2, Ynew) 
y_predRF = clfRF.predict(X_test2)


# Performance Metrics
acc_rf = accuracy_score(y_test, y_predRF)
print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(acc_rf))
confusion_matrix2 = confusion_matrix(y_test, y_predRF)
print("	", "pred no", "pred yes")
print("actual no", confusion_matrix2[0])  
print("actual yes", confusion_matrix2[1])

# Classification report
labels = ['No Fraud', 'Fraud']
print ('RandomForest:')
print(classification_report(y_test, y_predRF, target_names=labels))

# ROC AUC
aucRF = roc_auc_score(y_test, y_predRF)
print('RandomForests ROC_AUC Score: ', roc_auc_score(y_test, y_predRF))





# Built Naive Bayes model

clfNB = GaussianNB()
clfNB.fit(Xnew_2, Ynew) 
y_predNB = clfNB.predict(X_test2)


# Performance Metrics
acc_nb = accuracy_score(y_test, y_predNB)
print('Accuracy of Naive Bayes classifier on test set: {:.2f}'.format(acc_nb))
confusion_matrix3 = confusion_matrix(y_test, y_predNB)
print("	", "pred no", "pred yes")
print("actual no", confusion_matrix3[0])  
print("actual yes", confusion_matrix3[1])

# Classification report
labels = ['No Fraud', 'Fraud']
print ('Naive Bayes:')
print(classification_report(y_test, y_predNB, target_names=labels))

# ROC AUC
aucNB = roc_auc_score(y_test, y_predNB)
print('Naive Bayes ROC_AUC Score: ', roc_auc_score(y_test, y_predNB))





# Build the Support Vector Machines model

clfSVM = svm.SVC()
clfSVM.fit(Xnew_2, Ynew)  
y_predSVM = clfSVM.predict(X_test2)


# Performance Metrics
acc_svm = accuracy_score(y_test, y_predSVM)
print('Accuracy of SVM classifier on test set: {:.2f}'.format(acc_svm))
confusion_matrix4 = confusion_matrix(y_test, y_predSVM)
print("	", "pred no", "pred yes")
print("actual no", confusion_matrix4[0])  
print("actual yes", confusion_matrix4[1])

# Classification report
labels = ['No Fraud', 'Fraud']
print ('SVM:')
print(classification_report(y_test, y_predSVM, target_names=labels))

# ROC AUC
aucSVM = roc_auc_score(y_test, y_predSVM)
print('Support Vestor Machines ROC_AUC Score: ', roc_auc_score(y_test, y_predSVM))





# Build the AdaBoost model

ABclf = AdaBoostClassifier(n_estimators=100, random_state=42)
ABclf.fit(Xnew_2, Ynew)
y_predAB = ABclf.predict(X_test2)


# Performance Metrics
acc_AB = accuracy_score(y_test, y_predAB)
print('Accuracy of AdaBoost classifier on test set: {:.2f}'.format(acc_AB))
confusion_matrix4 = confusion_matrix(y_test, y_predAB)
print("	", "pred no", "pred yes")
print("actual no", confusion_matrix4[0])  
print("actual yes", confusion_matrix4[1])

# Classification report
labels = ['No Fraud', 'Fraud']
print ('AdaBoost:')
print(classification_report(y_test, y_predAB, target_names=labels))

# ROC AUC
aucAB = roc_auc_score(y_test, y_predAB)
print('AdaBoost Classifier ROC_AUC Score: ', roc_auc_score(y_test, y_predAB))





# All ROC_AUC 

print('Logistic Regression ROC_AUC Score: ', roc_auc_score(y_test, y_predLR))
print('RandomForests ROC_AUC Score: ', roc_auc_score(y_test, y_predRF))
print('Naive Bayes ROC_AUC Score : ', roc_auc_score(y_test, y_predNB))
print('Support Vestor Machines ROC_AUC Score: ', roc_auc_score(y_test, y_predSVM))
print('AdaBoost Classifier ROC_AUC Score: ', roc_auc_score(y_test, y_predAB))

log_fpr, log_tpr, log_thresold = roc_curve(y_test, y_predLR)
rf_fpr, rf_tpr, rf_threshold = roc_curve(y_test, y_predRF)
nb_fpr, nb_tpr, nb_threshold = roc_curve(y_test, y_predNB)
svm_fpr, svm_tpr, svm_threshold = roc_curve(y_test, y_predSVM)
ab_fpr, ab_tpr, ab_threshold = roc_curve(y_test, y_predAB)

def graph_roc_curve_multiple(log_fpr, log_tpr, rf_fpr, rf_tpr, nb_fpr, nb_tpr, svm_fpr, svm_tpr, ab_fpr, ab_tpr):
    plt.figure(figsize=(16,8))
    plt.title('ROC Curve \n Logistic Regression has the highest score', fontsize=18)
    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_test, y_predLR)))
    plt.plot(rf_fpr, rf_tpr, label='Random Forests Score: {:.4f}'.format(roc_auc_score(y_test, y_predRF)))
    plt.plot(nb_fpr, nb_tpr, label='Naive Bayes Score: {:.4f}'.format(roc_auc_score(y_test, y_predNB)))
    plt.plot(svm_fpr, svm_tpr, label='Support Vector MAchines Score: {:.4f}'.format(roc_auc_score(y_test, y_predSVM)))
    plt.plot(ab_fpr, ab_tpr, label='AdaBoost Classifier Score: {:.4f}'.format(roc_auc_score(y_test, y_predAB)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()

graph_roc_curve_multiple(log_fpr, log_tpr, rf_fpr, rf_tpr, nb_fpr, nb_tpr, svm_fpr, svm_tpr, ab_fpr, ab_tpr)
plt.show()
