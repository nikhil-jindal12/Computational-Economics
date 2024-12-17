import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
# ==========================================================================
# Step 1. define goal
# ==========================================================================

# In this project, we will use the given dataset to build ML models
# In particular, we will use regression algorithms to build models 
# that can be used to predict House Sale Price

# Task 1. Review the machine learning introduction lecture (no coding needed)

# ==========================================================================
# Step 2. Data collection
# ==========================================================================
# Read the data description, see what information is available

# Task 2. import the datasets
test = pd.read_csv('test-1.csv')
train = pd.read_csv('train.csv')

# ==========================================================================
# Step 3. Exploratory Data Analysis, Getting familiar with your data
# ==========================================================================

# Task 3.1. get descriptive statistics of your dataset


# Task 3.2. visualize the distribution of your label "SalePrice"


# Task 3.3 visualize the association between overall quality and sale price


# Task 3.4. visualize the association between YearBuilt and SalePrice

# Task 3.5. visualize the association between three variables:
# (1) overall quality 
# (2) SalePrice
# (3) YearBuilt 
# hint: use OverallQual as X, SalePrice as y, and YearBuilt as color scale

# Task 3.6. Display all distributions of numerical variables


# Task 3.7.visualize the associations of a set of variables of interest


# Task 3.8. check how each variable is correlated with SalePrice


# Task 3.9. Use a heatmap to visualize the correlation among variables.

