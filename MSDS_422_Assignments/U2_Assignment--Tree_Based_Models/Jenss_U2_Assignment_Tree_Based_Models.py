"""
Assignment 2: Tree Based Models

In this assignment, we will continue to use Python to develop predictive models. We will use three different tree based techniques to analyze the data: DECISION TREES, RANDOM FORESTS, and GRADIENT BOOSTING.

Assignment Requirements

I. Decision Trees:
1. Develop a decision tree to predict the probability of default
    - Calculate the accuracy of the model on both the training and test data set.
    - Create a graph that shows the ROC curves for both the training and test data set. Clearly label each curve and display the Area Under the ROC curve.
    - Display the Decision Tree using a Graphviz program
    - List the variables included in the decision tree that predict loan default.
2. Develop a decision tree to predict the loss amount assuming that the loan defaults
    - Calculate the RMSE for both the training data set and test data set.
    - Display the Decision Tree using a Graphviz program
    - List the variables included in the decision tree that predeict loss amount.
    
II. Random Forests:
1. Develop a Random Forest to predict the probability of default
    - Calculate the accuracy of the model on both the training and test data set.
    - Create a graph that shows the ROC curves for both the training and test data set. Clearly label each curve and display the Area Under the ROC curve.
    - List the variables included in the Random Forest that predict loan default.
2. Develop a Random Forest to predict the loss amount assuming that the loan defaults
    - Calculate the RMSE for both the training data set and the test data set
    - List the variables included in the Random Forest that predict loss amount.
    
III. Radient Boosting:
1. Develop a Gradient Boosting model to predict the probability of default
    - Calculate the accuracy of the model on both the training and test data set
    - Create a graph that shows the ROC curves for both the training and test data set. Clearly label each curve and display the Area Under the ROC curve.
    - List the variables included in the Gradient Boosting that predict loan default.
2. Develop a Gradient Boosting to predict the loss amount assuming that the loan defaults
    - Calculate the RMSE for both the training data set and the test data set
    - List the variables included in the Gradient Boosting that predict loss amount.

IV. ROC Curves
- Generate a ROC curve for the Decision Tree, Random Forest, and Gradient Boosting models using the Test Data Set.
- Use different colors for each curve and clearly label them.
- Include the Area under the ROC Curve on the graph.
"""

#%%
"""
ASSIGNMENT 1: DATA PREPARATION CODE
    - Previous code for setup for Assignment 2
"""

"""
PHASE I: ENVIRONMENT SETUP & DATA IMPORT
    1. Download the HMEQ Data Set
    2. Read the data into Python
"""

"""
Set up the environment
"""
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Setting display options
sns.set()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Import the data
INFILE = "/Users/stefanjenss/Documents/DataScience/Data_Science_Masters/MSDS_422_Machine_Learning/MSDS_422_Assignments/HMEQ_Loss.csv"
df = pd.read_csv(INFILE)

# Define the target variables
TARGET_F = "TARGET_BAD_FLAG"
TARGET_A = "TARGET_LOSS_AMT"

"""
Inspect the data to see if it was imported correctly and to get a feel for what is in the data.
"""
# Print a transpose of the data so that it will fit on the screen.
# print(df.head(3).T)

# Print the data info to check for missing values and data types.
print(df.info())

# %%
"""
PHASE II: DATA EXPLORATION
    1. Explore both the inputs and target variables using statistical techniques.
    2. Explore both the inputs and target variables using graphs and other visualizations.
    3. Look for relationships between the input variables and the targets.
"""

"""
1. Explore both the inputs and target variables using statistical techniques.
"""

"""
Generate a statistical description of the data transposed (so that it is easier to read).
"""
x = df.describe()
# print(x.T)

#%%
"""
2. Explore both the inputs and target variables using graphs and other visualizations.
"""

"""
Firstly, find the variables that are objects (strings), integers, and floats and sort them 
into their appropriate lists.
"""
dt = df.dtypes
objList = []
intList = []
floatList = []

# Sort the variables into their appropriate lists
for i in dt.index:
    # print("here is i .....", i, "..... and here is the type," dt[i])
    if i in (["TARGET_BAD_FLAG", "TARGET_LOSS_AMT"]) : continue
    if dt[i] in (["object"]) : objList.append(i)
    if dt[i] in (["int64"]) : intList.append(i)
    if dt[i] in (["float64"]) : floatList.append(i)
    
# Check the lists
print("OBJECT")
print("------")
for i in objList:
    print(i)
print("\n")

print("INTEGER")
print("-------")
for i in intList:
    print(i)
print("\n")

print("FLOAT")
print("-----")
for i in floatList:
    print(i)
print("\n")

#%%
"""
PIE CHARTS - CATEGORICAL VARIABLES

For the categorical variables, create pie charts to show the distribution of the data.
"""
for i in objList:
    x = df[i].value_counts(dropna=False)    # Get the value counts for each class
    # print(x)
    theLabels = x.axes[0].tolist()          # Get the labels from the value_counts
    # print(theLabels)
    theSlices = list(x)                     # Get the counts from the value_counts 
    # print(theSlices)
    plt.pie(theSlices,
            labels=theLabels,
            autopct='%1.1f%%',
            #shadow=True,
            #startangle=90)
            )
    plt.title(f'Pie Chart of {i}')
    plt.show()
    print("================\n")
    
# Create a pie chart for the target variable `TARGET_BAD_FLAG` since this is a binary 
# indicator of whether or not the loan was bad.
x = df[TARGET_F].value_counts(dropna=False)
theLabels = x.axes[0].tolist()
theSlices = list(x)
plt.pie(theSlices,
        labels=theLabels,
        autopct='%1.1f%%',
        #shadow=True,
        #startangle=90)
        )
plt.title(f'Pie Chart of {TARGET_F}')
plt.show()

#%%
"""
HISTOGRAMS - CONTINUOUS VARIABLES

For the continuous variables, create histograms to show the distribution of the data.
"""

# Create histograms for the integer variables
for i in intList:
    plt.hist(df[i])
    plt.xlabel(i)
    plt.ylabel("Count")
    plt.title(f'Histogram of {i}')
    plt.show()
    
# Create histograms for the float variables

# Calculate the number of rows and columns for the grid
num_plots = len(floatList)
num_cols = 2
num_rows = 5

# Create a grid of subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 18))

# Iterate over the float variables and plot histograms in the grid
for i, var in enumerate(floatList):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col] if num_rows > 1 else axes[col]
    ax.hist(df[var])
    ax.set_xlabel(var)
    ax.set_ylabel("Count")
    ax.set_title(f'Histogram of {var}')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the grid of histograms
plt.show()

#%%
"""
3. Look for relationships between the input variables and the targets.
"""

"""
Explore the relationship between the target variables and the categorical variables
"""
for i in objList:
    print("Class = ", i)
    g = df.groupby(i)
    # print(g[i].count())
    x = g[TARGET_F].mean()
    print("Probability of Loan Default", x)
    print("..........")
    x = g[TARGET_A].mean()
    print("Average Loss Amount", x)
    print("____________________\n")

#%%
"""
Explore the relationship between the target variables and the continuous variables
"""

# Integer variables
print("INTEGER VARIABLES")
print("\n")
for i in intList:
    print("Variable = ", i)
    g = df.groupby(TARGET_F)
    x = g[i].mean()
    print("Value of Loan", x)
    c = df[i].corr(df[TARGET_A])
    c = round(100 * c, 2)
    print("Correlation with Loss Amount = ", c, "%")
    print("____________________\n")
    
# Float variables
print("FLOAT VARIABLES")
print("\n")
for i in floatList:
    print("Variable = ", i)
    g = df.groupby(TARGET_F)
    x = g[i].mean()
    print("Variable Value", x)
    c = df[i].corr(df[TARGET_A])
    c = round(100 * c, 2)
    print("Correlation with Loss Amount = ", c, "%")
    print("____________________\n")

# %%
"""
PHASE III: DATA PREPARATION
    1. Fix (imput) all missing data.
        - Note: For numerical data, create a glaf variable to indicate if the value was missing
    2. Convert all categorical variables into numerical variables (using one hot encoding)
"""

"""
FILLING IN MISSING DATA.

We will use the second method of imputation discussed in the lectures (filling in missing with 
the category "MISSING").
"""

# Firstly, quickly explore the object variables by looping through the list of object variables.
# Looking at the unique values, the most common value, the number of missing values, and the number 
# of rows in each group.
for i in objList:
    print(i)
    print(df[i].unique())           # print the unique values
    g = df.groupby(i)               # group by the object variable
    print(g[i].count())             # count the number of rows in each group
    print("MOST COMMON = ", df[i].mode()[0])        # print the most common value
    print("MISSING = ", df[i].isna().sum())         # print the number of missing values
    print("\n")
    
# The `REASON` class has 252 missing values, and the `JOB` class has 279 missing values.

#%%
"""
IMPUTATION METHOD: FILL IN MISSING WITH THE CATEGORY "MISSING"
"""
for i in objList:
    if df[i].isna().sum() == 0 : continue           # skip if there are no missing values
    print(i)                                        # print the variable name
    print("HAS MISSING")                            # print that it has missing values
    FLAG = "M_" + i                                 # create new flag variable name **NEW
    NAME = "IMP_" + i                               # create a new variable name 
    print(NAME)                                     # print the new variable name
    print(FLAG)                                     # print the new flag variable name **NEW
    df[FLAG] = df[i].isna() + 0                     # populate the new flag variable **NEW
    df[NAME] = df[i].fillna("MISSING")              # fill in the missing values with the category "MISSING"
    print(f'Variable {i} has this many missing {df[i].isna().sum()}')       # print the number of missing values for the old variable
    print(f'Variable {NAME} has this many missing {df[NAME].isna().sum()}') # print the number of missing values for the new variable
    g = df.groupby(NAME)                            # group by the new variable
    print(g[NAME].count())                          # count the number of rows in each group
    print("\n")
    df = df.drop(i, axis = 1)                       # drop the old variable

#%%
"""
MISSING VALUE IMPUTATION FOR NUMERICAL VARIABLES

It is my hunch that for the numerical variables, "VALUE", "LOAN", and "DEBTINC" it would be best to fill in
the missing values with the median of the group. The group would be the "JOB" class. This is because
the value of a person's home is likely to be related to how much they make and what their job is. 
Therefore, it would make sense to fill in the missing values with the median of the group that the 
individual belongs to. The same logic applies to the "LOAN" variable--the more a person makes at 
their job, the more they will likely be able to barrow.

For the rest of the numerical variables, I will fill in the missing values with the median of the 
entire dataset.
"""

# Check the median value of "VALUE" and "LOAN" for each "JOB" class to see if this is a good idea.
# For "VALUE":
g = df.groupby("IMP_JOB")
i = "VALUE"
print(g[i].median())
print("____________________\n")

# For "LOAN":
g = df.groupby("IMP_JOB")
i = "LOAN"
print(g[i].median())

# Check to see if we should be using the "JOB" class to impute the missing value for "DEBTINC"
g = df.groupby("IMP_JOB")
i = "DEBTINC"
print(g[i].median())
print("____________________\n")

#%%

# PERFORM MISSING VALUE IMPUTATION FOR "VALUE" BASED ON "JOB" CLASS
# Name: VALUE, dtype: float64
# IMP_JOB
# MISSING     78227.0
# Mgr        101258.0
# Office      89094.5
# Other       76599.5
# ProfExe    110007.0
# Sales       84473.5
# Self       130631.0

i = "VALUE"
FLAG = "M_" + i
IMP = "IMP_" + i

print(i)
print(FLAG)
print(IMP)

df[FLAG] = df[i].isna() + 0
df[IMP] = df[i]
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["MISSING"]), IMP] = 78227
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["Mgr"]), IMP] = 101258
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["Office"]), IMP] = 89094.5
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["Other"]), IMP] = 76599.5
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["ProfExe"]), IMP] = 110007
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["Sales"]), IMP] = 84473.5
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["Self"]), IMP] = 130631
df.loc[df[IMP].isna(), IMP] = df[i].median()                # fill in the rest with the median
df = df.drop(i, axis=1)                                     # drop the old variable

#%%

# PERFORM MISSING VALUE IMPUTATION FOR "LOAN" BASED ON "JOB" CLASS
# Name: LOAN, dtype: float64
# IMP_JOB
# MISSING    13400.0
# Mgr        18100.0
# Office     16200.0
# Other      15650.0
# ProfExe    17300.0
# Sales      14300.0
# Self       24000.0

i = "LOAN"
FLAG = "M_" + i
IMP = "IMP_" + i

print(i)
print(FLAG)
print(IMP)

df[FLAG] = df[i].isna() + 0
df[IMP] = df[i]
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["MISSING"]), IMP] = 13400
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["Mgr"]), IMP] = 18100
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["Office"]), IMP] = 16200
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["Other"]), IMP] = 15650
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["ProfExe"]), IMP] = 17300
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["Sales"]), IMP] = 14300
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["Self"]), IMP] = 24000
df.loc[df[IMP].isna(), IMP] = df[i].median()                  # fill in the rest with the median
df = df.drop(i, axis=1)                                       # drop the old variable

#%%

# PERFORM MISSING VALUE IMPUTATION FOR "DEBTINC" BASED ON "JOB" CLASS
# Name: DEBTINC, dtype: float64
# IMP_JOB
# MISSING    30.311902
# Mgr        35.661118
# Office     36.158718
# Other      35.247328
# ProfExe    33.378041
# Sales      35.764058
# Self       34.830194

i = "DEBTINC"
FLAG = "M_" + i
IMP = "IMP_" + i

print(i)
print(FLAG)
print(IMP)

df[FLAG] = df[i].isna() + 0
df[IMP] = df[i]
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["MISSING"]), IMP] = 30.311902
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["Mgr"]), IMP] = 35.661118
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["Office"]), IMP] = 36.158718
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["Other"]), IMP] = 35.247328
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["ProfExe"]), IMP] = 33.378041
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["Sales"]), IMP] = 35.764058
df.loc[df[IMP].isna() & df["IMP_JOB"].isin(["Self"]), IMP] = 34.830194
df.loc[df[IMP].isna(), IMP] = df[i].median()                  # fill in the rest with the median
df = df.drop(i, axis=1)                                       # drop the old variable

#%%

# PERFORM MISSING VALUE IMPUTATION FOR THE REST OF THE NUMERICAL VARIABLES
floatList = []
dt = df.dtypes
for i in dt.index :
    #print(" here is i .....", i , " ..... and here is the type", dt[i] )
    if i in (["TARGET_BAD_FLAG", "TARGET_LOSS_AMT"]) : continue
    if dt[i] in (["float64"]) : floatList.append(i)

for i in floatList:
    if df[i].isna().sum() == 0: continue
    FLAG = "M_" + i
    IMP = "IMP_" + i
    df[FLAG] = (df[i].isna() + 0)                         # create a flag variable
    df[IMP] = df[i]                                       # create a new variable
    df.loc[df[IMP].isna(), IMP] = df[i].median()          # fill in the missing values with the median
    df = df.drop(i, axis=1)                               # drop the old variable

# Check to see if there are any missing values left
df.info()

# %%
"""
PERFORM ONE HOT ENCODING ON THE CATEGORICAL VARIABLES
"""
# Create new objList with the new variables
dt = df.dtypes
objList = []
for i in dt.index :
    #print(" here is i .....", i , " ..... and here is the type", dt[i] )
    if i in (["TARGET_BAD_FLAG", "TARGET_LOSS_AMT"]) : continue
    if dt[i] in (["object"]) : objList.append(i)

# Perform one hot encoding on the categorical variables
for i in objList:
    # print(f'Class = {i}')                   # print the class
    thePrefix = "z_" + i                    # create the prefix for the new variables
    # print(f'Prefix = {thePrefix}')          # print the prefix
    y = pd.get_dummies(df[i], prefix=thePrefix, drop_first=False, dtype="int64") # perform one hot encoding
    df = pd.concat([df, y], axis=1)         # concatenate the new variables to the data frame
    df = df.drop(i, axis=1)                 # drop the old variable


# %%
"""
==================================================
!!!START OF ASSIGNMENT 2 CODE!!!
==================================================
"""

# Import necessary libraries for assignment 2 - Tree Based Models
from operator import itemgetter                         # for sorting dictionaries
from sklearn.model_selection import train_test_split    # for splitting the data
import sklearn.metrics as metrics                       # for model evaluation
# For Decision Tree
from sklearn import tree                                
from sklearn.tree import _tree                          # for tree structure    
# For Random Forest
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import RandomForestClassifier
# For Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
import math

#%%
"""
PRE-PHASE: PREPARING THE DATA FOR MODELING
    1. Split the data into training and testing sets
    2. Handle outliers
"""

""" SPLIT THE DATA INTO TRAINING AND TESTING SETS """
X = df.copy()
X = X.drop(TARGET_F, axis=1)
X = X.drop(TARGET_A, axis=1)

Y = df[[TARGET_F, TARGET_A]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.80, test_size=0.20, random_state=1)  # 80% training, 20% testing

print('FLAG DATA')
print("TRAINING = ", X_train.shape)
print("TESTING = ", X_test.shape)

#%%
""" HANDLE OUTLIERS """
# Outliers will be considered to be those entries that have a `TARGET_LOSS_AMT` value that is greater than $60,000.

F = ~ Y_train[TARGET_A].isna()      # filter out the missing values from the training data
W_train = X_train[F].copy()         # create a working copy of the training data
Z_train = Y_train[F].copy()         # create a working copy of the target data

F = ~ Y_test[TARGET_A].isna()       # filter out the missing values from the testing data
W_test = X_test[F].copy()           # create a working copy of the testing data
Z_test = Y_test[F].copy()           # create a working copy of the target data

# Show description of the target variables before handling the outliers
print(Z_train.describe())
print(Z_test.describe())
print("\n\n")

# Cap the loss amount at $60,000
F = Z_train[TARGET_A] > 60000
Z_train.loc[F, TARGET_A] = 60000

F = Z_test[TARGET_A] > 60000
Z_test.loc[F, TARGET_A] = 60000

# Show description of the target variables after handling the outliers
print(Z_train.describe())
print(Z_test.describe())
print("\n\n")

print("======")
print("AMOUNT DATA")
print("TRAINING = ", W_train.shape)
print("TESTING = ", W_test.shape)


# %%
"""
PHASE I: DECISION TREES
    1. Develop a decision tree to predict the probability of default
    2. Develop a decision tree to predict the loss amount assuming that the loan defaults
"""

"""
DECISION TREE TO PREDICT THE PROBABILITY OF DEFAULT [FLAG]
"""

# Define the range of max_depth values
max_depth_range = range(3, 9)

# Initialize lists to store the accuracy scores
accuracy_train = []
accuracy_test = []

# Iterate over the max_depth values
for max_depth in max_depth_range:
    # Create the decision tree model
    fm01_Tree = tree.DecisionTreeClassifier(max_depth=max_depth)
    fm01_Tree = fm01_Tree.fit(X_train, Y_train[TARGET_F])

    # Calculate the accuracy of the model on the training and test data set
    Y_Pred_train = fm01_Tree.predict(X_train)
    Y_Pred_test = fm01_Tree.predict(X_test)

    # Append the accuracy scores to the lists
    accuracy_train.append(metrics.accuracy_score(Y_train[TARGET_F], Y_Pred_train))
    accuracy_test.append(metrics.accuracy_score(Y_test[TARGET_F], Y_Pred_test))

# Display the accuracy scores for each max_depth value
print("\n=================\n")
print("Decision Tree | Probability of Default\n")
for i, max_depth in enumerate(max_depth_range):
    print(f"Max Depth: {max_depth}")
    print("Accuracy (Training) |", accuracy_train[i])
    print("Accuracy (Testing)  |", accuracy_test[i])
    print("-----------------")

#%%

""" Create a graph that shows the ROC curves for both the training and test data set. 
    Clearly label each curve and display the Area Under the ROC curve. """
    
# Finalize the fm01_Tree model with the best max_depth value (6)
fm01_Tree = tree.DecisionTreeClassifier(max_depth=6)
fm01_Tree = fm01_Tree.fit(X_train, Y_train[TARGET_F])
    
# Calculate the ROC curve for the training data set
probs = fm01_Tree.predict_proba(X_train)
p1 = probs[:,1]
fpr_train, tpr_train, threshold = metrics.roc_curve(Y_train[TARGET_F], p1)
roc_auc_train = metrics.auc(fpr_train, tpr_train)

# Calculate the ROC curve for the test data set
probs = fm01_Tree.predict_proba(X_test)
p1 = probs[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve(Y_test[TARGET_F], p1)
roc_auc_test = metrics.auc(fpr_test, tpr_test)

# Save the test data for later use
fpr_TREE = fpr_test
tpr_TREE = tpr_test
auc_TREE = roc_auc_test

# Plot the ROC curves
plt.title('ROC Curve | Decision Tree | Probability of Default')
plt.plot(fpr_train, tpr_train, 'b', label = 'AUC (Training) = %0.2f' % roc_auc_train)
plt.plot(fpr_test, tpr_test, 'g', label = 'AUC (Testing) = %0.2f' % roc_auc_test)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate | Sensitivity')
plt.xlabel('False Positive Rate | 1 - Specificity')
plt.show()

#%%
""" Display the Decision Tree using a Graphviz program. """
feature_cols = list(X.columns.values)
tree.export_graphviz(fm01_Tree, out_file='tree_DT.txt', filled=True, rounded=True,
                     feature_names = feature_cols, class_names = ["Good", "Bad"])

#%%
""" List the variables included in the decision tree that predict loan default. """
# Define the function to get the variable names used in the decision tree
def getTreeVars(TREE, varNames):
    tree_ = TREE.tree_                       # get the tree structure
    varName = [varNames[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]  # get the variable names
    
    nameSet = set()
    for i in tree_.feature:
        if i != _tree.TREE_UNDEFINED:
            nameSet.add(i)
    nameList = list(nameSet)
    parameter_list = list()
    for i in nameList:
        parameter_list.append(varNames[i])
    return parameter_list

# Call the function to identify the variables used in the decision tree
vars_tree_flag = getTreeVars(fm01_Tree, feature_cols)

# Display the variables used in the decision tree
for i in vars_tree_flag:
    print(i)
    
# Save the variables used in the Decision Tree to a list for later use
vars_TREE = vars_tree_flag


# %%
"""
DECISION TREE TO PREDICT THE AMOUNT LOST ASSUMING THAT THE LOAN DEFAULTS [AMOUNT]
"""

# Define the range of max_depth values
max_depths = range(3, 8)

# Initialize lists to store the RMSE values
RMSE_TRAIN = []
RMSE_TEST = []
RMSE_RATIO = []

# Loop through each max_depth value
for DEPTH in max_depths:
    # Create the decision tree model for the loss amount
    amt_m01_Tree = tree.DecisionTreeRegressor(max_depth=DEPTH, random_state=1)
    amt_m01_Tree = amt_m01_Tree.fit(W_train, Z_train[TARGET_A])

    # Predict the loss amount for the training and test data
    Z_Pred_train = amt_m01_Tree.predict(W_train)
    Z_Pred_test = amt_m01_Tree.predict(W_test)

    # Calculate the RMSE for the training and test data
    rmse_train = math.sqrt(metrics.mean_squared_error(Z_train[TARGET_A], Z_Pred_train))
    rmse_test = math.sqrt(metrics.mean_squared_error(Z_test[TARGET_A], Z_Pred_test))

    # Calculate the ratio between the RMSE for the test data and the RMSE for the training data
    rmse_ratio = rmse_test / rmse_train

    # Append the RMSE values to the lists
    RMSE_TRAIN.append(rmse_train)
    RMSE_TEST.append(rmse_test)
    RMSE_RATIO.append(rmse_ratio)

    # Display the RMSE values for the current max_depth
    print("Max Depth =", DEPTH)
    print("TREE RMSE Train:", rmse_train)
    print("TREE RMSE Test:", rmse_test)
    print("RMSE Ratio:", rmse_ratio)
    print()

# Save the test data for later use
RMSE_TREE = min(RMSE_TEST)

#%%
"""Display the Decision Tree using a Graphviz program."""
# Finalize the amt_m01_Tree model with the best max_depth value (6)
amt_m01_Tree = tree.DecisionTreeRegressor(max_depth=6, random_state=1)
amt_m01_Tree = amt_m01_Tree.fit(W_train, Z_train[TARGET_A])

# Display the Decision Tree using a Graphviz program
feature_cols = list(X.columns.values)
vars_tree_amt = getTreeVars(amt_m01_Tree, feature_cols)
tree.export_graphviz(amt_m01_Tree, out_file='tree_DT_AMT.txt', filled=True, rounded=True,
                     feature_names = feature_cols, impurity=False, precision=0)

#%%
"""List the variables included in the decision tree that predict loss amount."""
for i in vars_tree_amt:
    print(i)
    
# Save the variables used in the Decision Tree to a list for later use
vars_TREE_amt = vars_tree_amt

# %%
"""
PHASE II: RANDOM FOREST
    1. Develop a Random Forest to predict the probability of default
    2. Develop a Random Forest to predict the loss amount assuming that the loan defaults
"""

"""
RANDOM FOREST TO PREDICT THE PROBABILITY OF DEFAULT [FLAG]
"""

""" Evaluate the accuracy of the model on both the training and test data sets. """
# Create the Random Forest model for the probability of default (FLAG)
fm01_RF = RandomForestClassifier(n_estimators=100, random_state=1)
fm01_RF = fm01_RF.fit(X_train, Y_train[TARGET_F])

# Check the accuracy of the model on the training and test data sets
Y_Pred_train = fm01_RF.predict(X_train)
Y_Pred_test = fm01_RF.predict(X_test)

# Display the accuracy of the model on the training and test data sets
print("Random Forest | Probability of Default\n")
print("Accuracy (Training) |", metrics.accuracy_score(Y_train[TARGET_F], Y_Pred_train))
print("Accuracy (Testing)  |", metrics.accuracy_score(Y_test[TARGET_F], Y_Pred_test))
print("\n")

#%%
""" Graph the ROC curves for both the training and test data sets. """
# Calculate the ROC curve for the training data set
probs = fm01_RF.predict_proba(X_train)
p1 = probs[:,1]
fpr_train, tpr_train, threshold = metrics.roc_curve(Y_train[TARGET_F], p1)
roc_auc_train = metrics.auc(fpr_train, tpr_train)

# Calculate the ROC curve for the test data set
probs = fm01_RF.predict_proba(X_test)                                       # Calculate the probability of default
p1 = probs[:,1]                                                             # Save the probability of default
fpr_test, tpr_test, threshold = metrics.roc_curve(Y_test[TARGET_F], p1)     # Calculate the ROC curve for the test data set
roc_auc_test = metrics.auc(fpr_test, tpr_test)                              # Calculate the AUC for the test data set

# Save the test data for later use
fpr_RF = fpr_test
tpr_RF = tpr_test
roc_auc_RF = roc_auc_test

# View the ROC curves
plt.title('ROC Curve | Random Forest | Probability of Default')
plt.plot(fpr_train, tpr_train, 'b', label = 'AUC (Training) = %0.2f' % roc_auc_train)
plt.plot(fpr_test, tpr_test, 'g', label = 'AUC (Testing) = %0.2f' % roc_auc_test)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.025, 1.025])
plt.ylim([-0.025, 1.025])
plt.ylabel('True Positive Rate | Sensitivity')
plt.xlabel('False Positive Rate | 1 - Specificity')
plt.show()

#%%
""" List the variables included in the Random Forest that predict loan default. """
# Identify the important variables in the Random Forest Model
def getEnsembleTreeVars(ENSTREE, varNames):                                 # Define the function to get the variable names used in the ensemble tree
    importance = ENSTREE.feature_importances_                               # Get the importance of the variables
    index = np.argsort(importance)                                          # Sort the variables by importance
    theList = []                                                            # Initialize the list
    for i in index:                                                         # Loop through the variables
        imp_val = importance[i]                                             # Get the importance value
        if imp_val > np.average(ENSTREE.feature_importances_):              # If the importance value is greater than the average importance
            v = int(imp_val / np.max(ENSTREE.feature_importances_) * 100)   # Calculate the importance value by dividing by the maximum importance
            theList.append((varNames[i], v))                                # Append the variable name and importance value to the list
    theList = sorted(theList, key=itemgetter(1), reverse=True)              # Sort the list; 'itemgetter(1)' specifies to sort by the second element in the tuple
    return theList                                                          # Return the list    

# Call the function to identify the important variables in the Random Forest Model
feature_cols = list(X.columns.values)
vars_RF_flag = getEnsembleTreeVars(fm01_RF, feature_cols)

# Display the important variables in the Random Forest Model
for i in vars_RF_flag:
    print(i)
    
# Save the important variables for the Random Forest Model into a list for later use
vars_RF = list()
for i in vars_RF_flag:
    vars_RF.append(i[0])

# %%
"""
RANDOM FOREST TO PREDICT LOSS AMOUNT ASSUMING THAT THE LOAN DEFAULTS [AMOUNT]
"""

""" Calculate the RMSE for both the training and test data sets. """
# Create the Random Forest model for the loss amount
amt_m01_RF = RandomForestRegressor(n_estimators=100, random_state=1)
amt_m01_RF = amt_m01_RF.fit(W_train, Z_train[TARGET_A])

# Predict the loss amount for the training and test data
Z_Pred_train = amt_m01_RF.predict(W_train)
Z_Pred_test = amt_m01_RF.predict(W_test)

# Calculate the RMSE for the training and test data
RMSE_TRAIN = math.sqrt(metrics.mean_squared_error(Z_train[TARGET_A], Z_Pred_train))
RMSE_TEST = math.sqrt(metrics.mean_squared_error(Z_test[TARGET_A], Z_Pred_test))

# Display the RMSE for the training and test data
print("Random Forest | Loss Amount\n")
print("RMSE (Training) |", RMSE_TRAIN)
print("RMSE (Testing)  |", RMSE_TEST)

# Save the test data for later use
RMSE_RF = RMSE_TEST

#%%
""" List the variables included in the Random Forest that predict loss amount. """
# Call the function to identify the important variables in the Random Forest Model
feature_cols = list(X.columns.values)
vars_RF_amt = getEnsembleTreeVars(amt_m01_RF, feature_cols)

# Display the important variables in the Random Forest Model
for i in vars_RF_amt:
    print(i)
   
# %%
"""
PHASE III: GRADIENT BOOSTING
    1. Develop a Gradient Boosting model to predict the probability of default
    2. Develop a Gradient Boosting model to predict the loss amount assuming that the loan defaults
"""

"""
GRADIENT BOOSTING TO PREDICT THE PROBABILITY OF DEFAULT [FLAG]
"""

""" Evaluate the accuracy of the model on both the training and test data sets. """
# Create the Gradient Boosting model for the probability of default (FLAG)
fm01_GB = GradientBoostingClassifier(n_estimators=100, random_state=1)
fm01_GB = fm01_GB.fit(X_train, Y_train[TARGET_F])

# Check the accuracy of the model on the training and test data sets
Y_Pred_train = fm01_GB.predict(X_train)
Y_Pred_test = fm01_GB.predict(X_test)

# Display the accuracy of the model on the training and test data sets
print("Gradient Boosting | Probability of Default\n")
print("Accuracy (Training) |", metrics.accuracy_score(Y_train[TARGET_F], Y_Pred_train))
print("Accuracy (Testing)  |", metrics.accuracy_score(Y_test[TARGET_F], Y_Pred_test))
print("\n")

#%%
""" Graph the ROC curves for both the training and test data sets. """
# Calculate the ROC curve for the training data set
probs = fm01_GB.predict_proba(X_train)
p1 = probs[:,1]
fpr_train, tpr_train, threshold = metrics.roc_curve(Y_train[TARGET_F], p1)
roc_auc_train = metrics.auc(fpr_train, tpr_train)

# Calculate the ROC curve for the test data set
probs = fm01_GB.predict_proba(X_test)
p1 = probs[:,1]
fpr_test, tpr_test, threshold = metrics.roc_curve(Y_test[TARGET_F], p1)
roc_auc_test = metrics.auc(fpr_test, tpr_test)

# Save the test data for later use
fpr_GB = fpr_test
tpr_GB = tpr_test
roc_auc_GB = roc_auc_test

# View the ROC curves
plt.title('ROC Curve | Gradient Boosting | Probability of Default')
plt.plot(fpr_train, tpr_train, 'b', label = 'AUC (Training) = %0.2f' % roc_auc_train)
plt.plot(fpr_test, tpr_test, 'g', label = 'AUC (Testing) = %0.2f' % roc_auc_test)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.025, 1.025])
plt.ylim([-0.025, 1.025])
plt.ylabel('True Positive Rate | Sensitivity')
plt.xlabel('False Positive Rate | 1 - Specificity')
plt.show()

#%%
""" List the variables included in the Gradient Boosting model that predict loan default. """
# Call the function to identify the important variables in the Gradient Boosting Model
feature_cols = list(X.columns.values)
vars_GB_flag = getEnsembleTreeVars(fm01_GB, feature_cols)

# Display the important variables in the Gradient Boosting Model
for i in vars_GB_flag:
    print(i)

# %%
"""
GRADIENT BOOSTING TO PREDICT LOSS AMOUNT ASSUMING THAT THE LOAN DEFAULTS [AMOUNT]
"""

""" Calculate the RMSE for both the training and test data sets. """
# Create the Gradient Boosting model for the loss amount
amt_m01_GB = GradientBoostingRegressor(n_estimators=100, random_state=1)
amt_m01_GB = amt_m01_GB.fit(W_train, Z_train[TARGET_A])

# Predict the loss amount for the training and test data
Z_Pred_train = amt_m01_GB.predict(W_train)
Z_Pred_test = amt_m01_GB.predict(W_test)

# Calculate the RMSE for the training and test data
RMSE_TRAIN = math.sqrt(metrics.mean_squared_error(Z_train[TARGET_A], Z_Pred_train))
RMSE_TEST = math.sqrt(metrics.mean_squared_error(Z_test[TARGET_A], Z_Pred_test))

# Display the RMSE for the training and test data
print("Gradient Boosting | Loss Amount\n")
print("RMSE (Training) |", RMSE_TRAIN)
print("RMSE (Testing)  |", RMSE_TEST)

# Save the test data for later use
RMSE_GB = RMSE_TEST

#%%
""" List the variables included in the Gradient Boosting model that predict loss amount. """
# Call the function to identify the important variables in the Gradient Boosting Model
feature_cols = list(X.columns.values)
vars_GB_amt = getEnsembleTreeVars(amt_m01_GB, feature_cols)

# Display the important variables in the Gradient Boosting Model
for i in vars_GB_amt:
    print(i)
    
# %%
"""
PHASE IV: ROC CURVES AND MODEL COMPARISON
    1. Create a graph that shows the ROC curves for all of the models developed in this assignment.
    2. Compare the RMSE values for the loss amount for all of the models developed in this assignment.
    3. Compare the important variables for the probability of default for all of the models developed in this assignment.
    4. Compare the important variables for the loss amount for all of the models developed in this assignment.
"""

""" Create a graph that shows the ROC curves for all of the models developed in this assignment. """
plt.title('ROC Curves | Probability of Default')
plt.plot(fpr_TREE, tpr_TREE, 'b', label = 'Decision Tree | AUC = %0.2f' % auc_TREE, color = 'red')
plt.plot(fpr_RF, tpr_RF, 'b', label = 'Random Forest | AUC = %0.2f' % roc_auc_RF, color = 'green')
plt.plot(fpr_GB, tpr_GB, 'b', label = 'Gradient Boosting | AUC = %0.2f' % roc_auc_GB, color = 'blue')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.025, 1.025])
plt.ylim([-0.025, 1.025])
plt.ylabel('True Positive Rate | Sensitivity')
plt.xlabel('False Positive Rate | 1 - Specificity')
plt.show()

#%%
""" Compare the RMSE values for the loss amount for all of the models developed in this assignment. """
print("RMSE Values | Loss Amount")
print("Decision Tree |", RMSE_TREE)
print("Random Forest |", RMSE_RF)
print("Gradient Boosting |", RMSE_GB)

#%%
""" Compare the important variables for the probability of default for all of the models developed in this assignment. """
# Print the important variables for the probability of default for each model
print(f'| Model | Important Variables |')
print(f'| Vars_TREE | {vars_TREE} |')
print(f'| Vars_RF | {vars_RF} |')
print(f'| Vars_GB_flag | {vars_GB_flag} |')
print("\n")

# For the `vars_GB_flag` list, each tuple contains the variable name and the importance value. Create a list of just the variable names.
vars_GB_flag_2 = list()
for i in vars_GB_flag:
    vars_GB_flag_2.append(i[0])

# Create an empty dictionary to store the variable counts
variable_counts = {}

# Loop through the important variables for each model and update the counts
for i in vars_TREE:
    variable_counts[i] = variable_counts.get(i, 0) + 1

for i in vars_RF:
    variable_counts[i] = variable_counts.get(i, 0) + 1

for i in vars_GB_flag_2:
    variable_counts[i] = variable_counts.get(i, 0) + 1

# Sort the variable counts by count in descending order
sorted_counts = sorted(variable_counts.items(), key=lambda x: x[1], reverse=True)

# Print the variable counts
print("Important Variables | Probability of Default")
print("| Variable | Appearance Count |")
for variable, count in sorted_counts:
    print(f"| {variable} | {count} |")

# %%
""" Compare the important variables for the loss amount for all of the models developed in this assignment. """

# Print the important variables for the loss amount for each model
print(f'| Model | Important Variables |')
print(f'| Vars_TREE_AMT | {vars_TREE_amt} |')
print(f'| Vars_RF_AMT | {vars_RF_amt} |')
print(f'| Vars_GB_AMT | {vars_GB_amt} |')
print("\n")

# For the `vars_RF_amt` list, each tuple contains the variable name and the importance value. Create a list of just the variable names.
vars_RF_amt_2 = list()
for i in vars_RF_amt:
    vars_RF_amt_2.append(i[0])

# For the `vars_GB_amt` list, each tuple contains the variable name and the importance value. Create a list of just the variable names.
vars_GB_amt_2 = list()
for i in vars_GB_amt:
    vars_GB_amt_2.append(i[0])

# Create an empty dictionary to store the variable counts
variable_counts = {}

# Loop through the important variables for each model and update the counts
for i in vars_TREE_amt:
    variable_counts[i] = variable_counts.get(i, 0) + 1
    
for i in vars_RF_amt_2:
    variable_counts[i] = variable_counts.get(i, 0) + 1

for i in vars_GB_amt_2:
    variable_counts[i] = variable_counts.get(i, 0) + 1

# Sort the variable counts by count in descending order
sorted_counts = sorted(variable_counts.items(), key=lambda x: x[1], reverse=True)

# Print the variable counts
print("Important Variables | Loss Amount")
print("| Variable | Appearance Count |")
for variable, count in sorted_counts:
    print(f"| {variable} | {count} |")

# %%
"""
Final Observations/Conclusions About Comparing the Models

- The Random Forest model for predicting the probability of default had the highest AUC value (0.97) compared to the Decision Tree
(0.83) and the Gradient Boosting model (0.94). This indicates that the Random Forest model is the best model for predicting the
probability of default.

- The Gradient Boosting model for predicting the loss amount had the lowest RMSE value (2,272) compared to the Decision Tree (4,212)
and the Random Forest model (2,725). This indicates that the Gradient Boosting model is the best model for predicting the loss amount.

- The important variables for the probability of default varied between the models. However, the following variables appeared in the 
important variable lists for all three models: `M_VALUE`, `M_DEBTINC`, `IMP_DEBTINC`, `IMP_DEROG`, `IMP_DELINQ`, `IMP_CLAGE`. Based
on the importance scores for the Gradient Boosting model, the most important of these variables appears to be `M_DEBTINC`.

- Similarly, the important variables for the loss amount varied between the models. However, the following variables appeared in the
important variable lists for all three models: `IMP_LOAN`, `M_DEBTINC`, `IMP_CLNO`. Based on the importance score for both the Random
Forest and Gradient Boosting models, the most important of these variables appears to be `IMP_LOAN`.

"""
