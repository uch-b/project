# This project seeks to address the classification problem of whether a browsing of an online shop results in a purchase or not. I will fit a simple logistic regression, gradient boosting classifier and a nonlinear support vector machine classifier to the data to try to predict if an online browsing results in a purchase. Nowadays many people choose to shop online than go to stores because it is convenient. For many online shops, identifying the relevant factors to when a customer makes a purchase online and the relative importance of each factor is a very useful information that can help choose what to put more resource in such as advertising. In my project, I will use the time of the year and whether a customer is a new or a returning visitor. For this project, I will use the Online Shoppers Purchasing Intention Dataset Data Set found at 
# https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset# 
# and the citation is 
# Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Real-time prediction of online shoppers’ purchasing intention using multilayer perceptron and LSTM recurrent neural networks. Neural Comput & Applic 31, 6893–6908 (2019). https://doi.org/10.1007/s00521-018-3523-0 .
# The dataset is from UCI Machine Learning Repository and it contains 12330 different data points from different users in a one year time period.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV


data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv")
data.info()


# This tabulated data has 12330 rows and 18 columns, the memory usage is 1.5+ MB, 8 categorical and 10 numeric features, "SpecialDay" is a measure of how close the time is to a celebration day, "VisitorType" indicates whether a customer is a new or a returning visitor and "Revenue" is the variable I want to predict.

# For this project, I am only interested in some features, so I will drop the features that I am not interested in.

delete = []
for i in range(len(data.columns)):
    if i < 9:
        delete.append(data.columns[i])
data.drop(columns=delete, inplace=True)
data.info()


# Now I inspect each feature to check if all of them are important to my analysis and drop if any feature is not.

data['SpecialDay']

data['Month'].value_counts()

data['OperatingSystems']
data.drop(columns=['OperatingSystems'], inplace=True)

data['Browser']
data.drop(columns=['Browser'], inplace=True)

data['Region']
data.drop(columns=['Region'], inplace=True)

data['TrafficType']
data.drop(columns=['TrafficType'], inplace=True)

data['VisitorType']

data['Weekend']

data['Revenue']

data.info()


# Now that I have the features I want in my analysis, I check each of them for potential issues before starting my analysis.

for col in data.columns:
    print(col, data[col].unique())


# I see that VisitorType has 'Other' and let's inspect how many observations have 'Other'.

data['VisitorType'].value_counts()


# Since the number of observations with 'Other' is not many and there is no clear way to impute values for these observations, I will drop rows with 'Other' as the visitor type.

indexes = []
for i in range(len(data)):
    if data['VisitorType'][i] == 'Other':
        indexes.append(i)
data.drop(index=indexes, inplace=True)
data.info()


# To make later analysis easier, I will change the Dtypes of some of the features to int.

data['Weekend'] = data['Weekend'].astype(int)
data['Revenue'] = data['Revenue'].astype(int)
data.loc[data['VisitorType'] == 'Returning_Visitor', 'VisitorType'] = 1
data.loc[data['VisitorType'] == 'New_Visitor', 'VisitorType'] = 0
data.loc[data['Month'] == 'Feb', 'Month'] = 2
data.loc[data['Month'] == 'Mar', 'Month'] = 3
data.loc[data['Month'] == 'May', 'Month'] = 5
data.loc[data['Month'] == 'Oct', 'Month'] = 10
data.loc[data['Month'] == 'June', 'Month'] = 6
data.loc[data['Month'] == 'Jul', 'Month'] = 7
data.loc[data['Month'] == 'Aug', 'Month'] = 8
data.loc[data['Month'] == 'Nov', 'Month'] = 11
data.loc[data['Month'] == 'Sep', 'Month'] = 9
data.loc[data['Month'] == 'Dec', 'Month'] = 12
data['VisitorType'] = data['VisitorType'].astype(int)
data['Month'] = data['Month'].astype(int)

data.info()
for col in data.columns:
    print(col, data[col].unique())


# To have a better idea of the dataset, let's visualize each column.


data['SpecialDay'].hist()
plt.title('SpecialDay')


# From the histogram, I conclude that not many days are close to a special day which makes sense because most days in a year are not close to a special day.


data['Month'].hist()
plt.title('Month')

data['Month'].value_counts()


# The months of January and April are missing in this dataset but I think this dataset is still useful to my analysis because all the other months are in the dataset.

data['VisitorType'].value_counts()


# Most visitors are returning visitors.

data['Weekend'].value_counts()


# Most observations did not happen during the weekends.


data['Revenue'].value_counts()


# Most observations did not result in a purchase.


corr = data.corr()
print(corr)
sns.heatmap(corr)
sns.pairplot(data, vars = data.columns[:4], diag_kind = "kde")


# From the correlation matrix, 'SpecialDay' has a negative correlation with 'Revenue'. 'SpecialDay' and 'Revenue' having a negative correlation is not expected, so I will do a linear regression and look at the significance of the coefficient.

model = smf.ols(formula = 'Revenue ~ SpecialDay', data = data).fit()
print(model.summary())


# From the regression result, 'SpecialDay' has a statistically significant negative coefficient which I did not expect. This could be because a polynomial regression might work better since the adjusted r-squared is 0.007 in the result above.

# # Models

# ## Logistic Regression Model


# prepare the data for a logistic regression model
X, y = data.loc[:,data.columns[:4]], data.loc[:,data.columns[4]]
print(X,y)

# split the data into training and test sets with 5 percent of the data in the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

# fit a logistic regression model on training set
logisticRegModel = LogisticRegression().fit(X_train, y_train)

# get the coefficients for the logistic regression model
print(logisticRegModel.coef_)
print(X_train.columns)


# Looking at the coefficients, I see that SpecialDay has a negative coefficient which agrees with the previous linear regression result.

# checking for multicollinearity in the logistic regression model
vif = [(variance_inflation_factor(X,i), X.columns[i]) for i in range(len(X.columns))]
print(vif)


# The VIF is higher for the variables 'VisitorType' and 'Month', so I will do a linear regression on both to check if any of them should be dropped.


model = smf.ols(formula = 'VisitorType ~ SpecialDay + Weekend + Month', data = data).fit()
print(model.summary())


# Since the adjusted r-squared is low, I decide to keep the variable 'VisitorType' in my analysis.


model = smf.ols(formula = 'Month ~ SpecialDay + Weekend + VisitorType', data = data).fit()
print(model.summary())


# I choose to keep the variable 'Month' in my analysis because the adjusted r-squared is low.


# finding the mean accuracy of the model
logisticRegModel.score(X_test, y_test)


# My logistic regression model has a mean accuracy of 0.835.

# ## Gradient Boosting Classifier


# fitting a gradient boosting classifier to the training data
gradientBoost = GradientBoostingClassifier(learning_rate = 0.05, max_depth = 10, max_features = 'sqrt', subsample = 0.5)
print(gradientBoost.get_params())
gradientBoost.fit(X_train, y_train)


# finding the mean accuracy of the gradient boosting classifier
gradientBoost.score(X_test, y_test)


# My gradient boosting classifier has a mean accuracy of 0.835 which is the same as my logistic regression model. Now I will try changing the 'max_features' to 'None' and decreasing the learning rate to 0.01.

gradientBoost = GradientBoostingClassifier(learning_rate = 0.01, max_depth = 10, subsample = 0.5)
print(gradientBoost.get_params())
gradientBoost.fit(X_train, y_train)

gradientBoost.score(X_test, y_test)


# My gradient boosting classifier still has a mean accuracy of 0.835 which is the same as my logistic regression model.

print(gradientBoost.feature_importances_)
print(X_test.columns)


# ## Nonlinear Support Vector Machine Classifier


# fitting a nonlinear support vector machine classifier to the training data
# searching for parameter values with the highest accuracy using K-Folds cross validation

kernel = ['linear', 'poly', 'rbf', 'sigmoid']
degree = [i for i in range(3, 6)]
parameters = {"kernel":kernel, "degree":degree}
searchSVC = GridSearchCV(estimator = SVC(probability=True), param_grid=parameters, scoring = 'accuracy', cv = 5)
searchSVC.fit(X_train, y_train)
print(searchSVC.best_params_)
bestSVC = searchSVC.best_estimator_


# Using these results, I now know that linear kernel might work best.

# finding the mean accuracy of the nonlinear support vector machine classifier
bestSVC.score(X_test, y_test)

c = [np.log(x) for x in np.arange(np.exp(2**(-5)), np.exp(1)+1)]
gamma = [x for x in np.arange(2**(-5), 2)]
parameters = {"C":c, "gamma":gamma}
searchSVC = GridSearchCV(estimator = SVC(kernel='linear', probability=True), param_grid=parameters, scoring = 'accuracy', cv = 5)
searchSVC.fit(X_train, y_train)
print(searchSVC.best_params_)
bestSVC = searchSVC.best_estimator_

# finding the mean accuracy of the nonlinear support vector machine classifier
bestSVC.score(X_test, y_test)

# weights for the features
print(bestSVC.coef_)
print(X_test.columns)


# Again, I get the same mean accuracy for the nonlinear support vector machine classifier, so I will use different evaluation metrics and visualizations for these different models.

# In summary, I trained three different models on the data and got the same mean accuracy score for all three. Further evaluation metrics will show the differences between my logistic regression model, gradient boosting classifier and nonlinear support vector machine classifier. Looking at the feature importances for the gradient boosting classifier, month is the most important and I think the month variable seems to be the most important one because of the variation in the variable. While 'VisitorType' and 'Weekend' have only 0 and 1, 'Month' has 10 different values. The second most important feature is 'VisitorType'.

# ## Evaluation metrics and visualizations

# For the evaluation metrics, I will evaluate my models using ROC curve, precision-recall curve, confusion matrix and cross validation score.


# ROC curve for the logistic regression model
RocCurveDisplay.from_estimator(logisticRegModel, X_test, y_test)
plt.title('ROC curve for the logistic regression model')
plt.show()


# ROC curve for the gradient boosting classifier
RocCurveDisplay.from_estimator(gradientBoost, X_test, y_test)
plt.title('ROC curve for the gradient boosting classifier')
plt.show()


# ROC curve for the nonlinear support vector machine classifier
RocCurveDisplay.from_estimator(bestSVC, X_test, y_test)
plt.title('ROC curve for the nonlinear support vector machine classifier')
plt.show()


# precision-recall curve for the logistic regression model
PrecisionRecallDisplay.from_estimator(logisticRegModel, X_test, y_test)
plt.title('precision-recall curve for the logistic regression model')
plt.show()


# precision-recall curve for the gradient boosting classifier
PrecisionRecallDisplay.from_estimator(gradientBoost, X_test, y_test)
plt.title('precision-recall curve for the gradient boosting classifier')
plt.show()


# precision-recall curve for the nonlinear support vector machine classifier
PrecisionRecallDisplay.from_estimator(bestSVC, X_test, y_test)
plt.title('precision-recall curve for the nonlinear support vector machine classifier')
plt.show()


# confusion matrix for the logistic regression model
ConfusionMatrixDisplay.from_estimator(logisticRegModel, X_test, y_test)
plt.title('confusion matrix for the logistic regression model')
plt.show()


# confusion_matrix for the gradient boosting classifier
ConfusionMatrixDisplay.from_estimator(gradientBoost, X_test, y_test)
plt.title('confusion matrix for the gradient boosting classifier')
plt.show()


# confusion_matrix for the nonlinear support vector machine classifier
ConfusionMatrixDisplay.from_estimator(bestSVC, X_test, y_test)
plt.title('confusion matrix for the nonlinear support vector machine classifier')
plt.show()


# cross validation score for the logistic regression model
print(cross_val_score(logisticRegModel, X, y, cv=10))
# cross validation score for the gradient boosting classifier
print(cross_val_score(gradientBoost, X, y, cv=10))
# cross validation score for the nonlinear support vector machine classifier
print(cross_val_score(bestSVC, X, y, cv=10))


# From the evaluation metrics, my gradient boosting classifier has the best performance out of the three, followed by my logistic regression model. I will try to improve the performance of my gradient boosting classifier by decreasing the learning rate to 0.005 and setting subsample to 0.1 and evaluate the new model.

# ## Gradient Boosting Classifier


gradientBoost = GradientBoostingClassifier(learning_rate = 0.005, max_depth = 10, subsample = 0.1)
gradientBoost.fit(X_train, y_train)
print(gradientBoost.score(X_test, y_test))
print(gradientBoost.feature_importances_)
print(X_test.columns)


# evaluation metrics and visualizations for the gradient boosting classifier
# ROC curve for the gradient boosting classifier
RocCurveDisplay.from_estimator(gradientBoost, X_test, y_test)
plt.title('ROC curve for the gradient boosting classifier')
plt.show()
# precision-recall curve for the gradient boosting classifier
PrecisionRecallDisplay.from_estimator(gradientBoost, X_test, y_test)
plt.title('precision-recall curve for the gradient boosting classifier')
plt.show()
# confusion_matrix for the gradient boosting classifier
ConfusionMatrixDisplay.from_estimator(gradientBoost, X_test, y_test)
plt.title('confusion matrix for the gradient boosting classifier')
plt.show()
# cross validation score for the gradient boosting classifier
print(cross_val_score(gradientBoost, X, y, cv=10))


# Looking at the ROC curves, the previous gradient boosting classifier had an AUC of 0.65 and the new gradient boosting classifier has an AUC of 0.66 which is a slight improvement in the model performance.

# # Discussion and Conclusion

# Results of my models do not support the idea that special days have the biggest effect on whether a customer makes a purchase online or not. The most important features to consider are the month and whether the customer is a new or returning visitor. All three of my models had the same mean accuracy score. For my gradient boosting classifier, changing the learning rate and the subsample did not result in a considerable improvement in my original gradient boosting classifier and this may be because the dataset was missing the months of January and April. Including the months of January and April may improve my model performances.
