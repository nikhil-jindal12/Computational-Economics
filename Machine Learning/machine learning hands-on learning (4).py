"""
In this practice, we will use the dataset we have prepared to build machine
learning models

@author: sw
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


# ==========================================================================
# Step 1. import and prepare the dataset
# ==========================================================================

# Task 1.1. Import the dataset
# Note: if you followed our previous hands-on practice and complted project 10,
# You shall have generated the exact same dataset by yourselves.
# In case you are still working on project 10, these dataset are available on
# canvas for download
train_set_allnum = pd.read_csv("train_set_allnum.csv")
test_set_allnum = pd.read_csv("test_set_allnum.csv")

# Task 1.2. set the variable "Id" as the index of your train data
train_set_allnum = train_set_allnum.set_index("Id")

# Task 1.3. separete data label from the features in the train set 
train_label = train_set_allnum["SalePrice"] # this is log_SalePrice
train_features = train_set_allnum.drop("SalePrice", axis=1)

# Task 1.4. replicate 1.2 and 1.3 on the test set
test_set_allnum = test_set_allnum.set_index("Id")
test_label = test_set_allnum["SalePrice"] # this is log_SalePrice
test_features = test_set_allnum.drop("SalePrice", axis=1)

# ==========================================================================
# Step 2. Machine Learning Models
# ==========================================================================

# Task 2.1. use linear regression to train your model
lin_reg = LinearRegression()
lin_reg.fit(train_features, train_label)

# Task 2.2.with the trained model, use your the train set to make prediction
logSalePrice_train_linear = lin_reg.predict(train_features)
logSalePrice_train_linear = pd.Series(logSalePrice_train_linear,
                                      name="logSalePrice_train_linear")

# Task 2.3. report the mse score in your linear regression (train)
lin_mse_train = mean_squared_error(train_label, logSalePrice_train_linear)
lin_r2_train = r2_score(train_label, logSalePrice_train_linear)
print(f"The mean squared error: {lin_mse_train}")
print(f"The R^2 value is: {lin_r2_train}")

# Task 2.4. convert the predicted logPrice back to Price (train)
predict_price_train_linear = np.expm1(logSalePrice_train_linear)

# Task 2.5. convert the observed logPrice back to Price (train)
observed_price_train = np.expm1(train_label)

# Task 2.6. Use the function below to see if your prediction is close (train).
def obs_vs_pred_train(observation, prediction):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(observation, prediction)
    ax.plot([0, max(observation)], [0, max(observation)], color='red')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.set_ylim(0, prediction.max())
    ax.set_xlabel("observed Sale Price")
    ax.set_ylabel("predicted Sale Price")
    ax.set_title("Train Set")
    plt.show()
    
obs_vs_pred_train(observed_price_train, predict_price_train_linear)

# Task 2.7. with the trained model, use the test set to make prediction
test_features = test_features[train_features.columns]
logSalePrice_test_linear = lin_reg.predict(test_features)

# Task 2.8. report the mse score in your linear regression (test)
lin_mse_test = mean_squared_error(test_label, logSalePrice_test_linear)
lin_r2_test = r2_score(test_label, logSalePrice_test_linear)
print(f"The mean squared error: {lin_mse_test}")
print(f"The R^2 value is: {lin_r2_test}")

# Task 2.9. convert the predicted logPrice back to Price (test)
predict_price_test_linear = np.expm1(logSalePrice_test_linear)

# Task 2.10 convert the observed logPrice back to Price (test)
observed_price_test = np.expm1(test_label)

# Task 2.11. Use the function below to see if your prediction is close (test).
def obs_vs_pred_test(observation, prediction):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(observation, prediction)
    ax.plot([0, max(observation)], [0, max(observation)], color='red')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    # ax.set_ylim(0, prediction.max())
    ax.set_xlabel("observed Sale Price")
    ax.set_ylabel("predicted Sale Price")
    ax.set_title("Test Set")
    plt.show()

obs_vs_pred_test(observed_price_test, predict_price_test_linear)

# Task 2.12. replicate task 2.1 to 2.11 with LASSO, set alpha = 0.3
lasso_reg = Lasso(alpha=0.3)
lasso_reg.fit(train_features, train_label)
logSalePrice_train_lasso = lasso_reg.predict(train_features)
logSalePrice_train_lasso = pd.Series(logSalePrice_train_lasso,
                                     name="logSalePrice_train_lasso")
lasso_mse_train = mean_squared_error(train_label, logSalePrice_train_lasso)
lasso_r2_train = r2_score(train_label, logSalePrice_train_lasso)
print(f"The mean squared error: {lasso_mse_train}")
print(f"The R^2 value is: {lasso_r2_train}")
predict_price_train_lasso = np.expm1(logSalePrice_train_lasso)
obs_vs_pred_train(observed_price_train, predict_price_train_lasso)


# Task 2.13. replicate task 2.1 to 2.11 with regression tree (manual para tune)
tree_reg = DecisionTreeRegressor(max_depth=3)
tree_reg.fit(train_features, train_label)
logSalePrice_train_tree = tree_reg.predict(train_features)
logSalePrice_train_tree = pd.Series(logSalePrice_train_tree,
                                    name="logSalePrice_train_tree")
tree_mse_train = mean_squared_error(train_label, logSalePrice_train_tree)
tree_r2_train = r2_score(train_label, logSalePrice_train_tree)
print(f"The mean squared error: {tree_mse_train}")
print(f"The R^2 value is: {tree_r2_train}")
predict_price_train_tree = np.expm1(logSalePrice_train_tree)
obs_vs_pred_train(observed_price_train, predict_price_train_tree)

# Task 2.14. replicate task 2.1 to 2.11 with random forest (manual para tune)

# Task 2.15. replicate task 2.1 to 2.11 with GradientBoost (manual para tune)
gboost_reg = GradientBoostingRegressor(max_depth=4, n_estimators=100,
                                   learning_rate=0.08, max_features=10,
                                   subsample=1, random_state=42)
gboost_reg.fit(train_features, train_label)
logSalePrice_train_gboost = gboost_reg.predict(train_features)
logSalePrice_train_gboost = pd.Series(logSalePrice_train_gboost,
                                      name="logSalePrice_train_gboost")
gboost_mse_train = mean_squared_error(train_label, logSalePrice_train_gboost)
gboost_r2_train = r2_score(train_label, logSalePrice_train_gboost)
print(f"The mean squared error: {gboost_mse_train}")
print(f"The R^2 value is: {gboost_r2_train}")

predict_price_train_gboost = np.expm1(logSalePrice_train_gboost)
observed_price_train = np.expm1(train_label)
obs_vs_pred_train(observed_price_train, predict_price_train_gboost)

logSalePrice_test_gboost = gboost_reg.predict(test_features)
logSalePrice_test_gboost = pd.Series(logSalePrice_test_gboost,
                                      name="logSalePrice_test_gboost")
gboost_mse_test = mean_squared_error(test_label, logSalePrice_test_gboost)
gboost_r2_test = r2_score(test_label, logSalePrice_test_gboost)
print(f"The mean squared error: {gboost_mse_test}")
print(f"The R^2 value is: {gboost_r2_test}")
predict_price_test_gboost = np.expm1(logSalePrice_test_gboost)
observed_price_test = np.expm1(test_label)
obs_vs_pred_test(observed_price_test, predict_price_test_gboost)

# define hyperparameter space
parameters = {
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.05, 0.08, 0.1, 0.15],
    "subsample": [0.8, 1],
    "random_state": [42],
    "max_leaf_nodes": [8, 16, 32, 64]
}

# use grid search and cross validation to tune the parameters of the model
cv_gboost = GridSearchCV(GradientBoostingRegressor(), parameters, cv=10)
cv_gboost.fit(train_features, train_label)

# print the best results
cv_gboost_result = pd.DataFrame(cv_gboost.cv_results_)
print(cv_gboost.best_params_)

# cv_gboost train set
logSalePrice_train_gboost_cv = cv_gboost.predict(train_features)
logSalePrice_train_gboost_cv = pd.Series(logSalePrice_train_gboost_cv,
                                      name="logSalePrice_train_gboost_cv")
gboost_mse_train_cv = mean_squared_error(train_label, logSalePrice_train_gboost_cv)
gboost_r2_train_cv = r2_score(train_label, logSalePrice_train_gboost_cv)
print(f"The mean squared error: {gboost_mse_train_cv}")
print(f"The R^2 value is: {gboost_r2_train_cv}")

logSalePrice_test_gboost_cv = cv_gboost.predict(test_features)
logSalePrice_test_gboost_cv = pd.Series(logSalePrice_train_gboost_cv,
                                      name="logSalePrice_train_gboost_cv")
gboost_mse_test_cv = mean_squared_error(test_label, logSalePrice_test_gboost_cv)
gboost_r2_test_cv = r2_score(test_label, logSalePrice_test_gboost_cv)
print(f"The mean squared error: {gboost_mse_test_cv}")
print(f"The R^2 value is: {gboost_r2_test_cv}")
