"""
# Assignment 3: Regression Based Models

In this assignment, we will continue to use Python to develop predictive models. We will use two different types of regression: Linear and Logistic. We will use Logistic regression to determine the probability of a default. Linear regression will be used to calculate the loss amount assuming that default occurs.

### Assignment Requirements
#### I. Create a Training and Test Data Set
#### II. Logistic Regression:
1. Develop a logistic regression model to determine the probability of loan default. Use all of the variables.
2. Develop a logistic regression model to determine the probability of a loan default. Use the variables that were selected by a DECISION TREE.
3. Develop a logistic regression model to determine the probability of a loan default. Use the variables that were selected by a RANDOM FOREST.
4. Develop a logistic regression model to determine the probability of a loan default. Use the variables that were selected by a GRADIENT BOOSTING model.
5. Develop a logistic regression model to determine the probability of a loan default. Use the variables that were selected by STEPWISE SELECTION.
- For each of the models:
    - Calculate the accuracy of the model on both the training and test data set
    - Create a graph that shows the ROC curves for both the training and test data set. Clearly label each curve and display the Area Under the ROC curve.
    - Display a ROC curve for the test data with all your models on the same graph (tree based and regression). Discuss which one is the most accurate. Which one would you recommend using?
    - For one of the Regression Models, print the coefficients. Do the variables make sense? If not, what would you recommend?
#### III. Linear Regression:
1. Develop a linear regression model to determine the expected loss if the loan defaults. Use all of the variables.
2. Develop a linear regression model to determine the expected loss if the loan defaults. Use the variables that were selected by a DECISION TREE.
3. Develop a linear regression model to determine the expected loss if the loan defaults. Use the variables that were selected by a RANDOM FOREST.
4. Develop a linear regression model to determine the expected loss if the loan defaults. Use the variables that were selected by a GRADIENT BOOSTING model.
5. Develop a linear regression model to determine the expected loss if the loan defaults. Use the variables that were selected by STEPWISE SELECTION.
- For each of the models
    - Calculate the RMSE for both the training data set and the test data set
    - List the RMSE for the test data set for all of the models created (tree based and regression). Discuss which one is the most accurate. Which one would you recommend using?
    - For one of the Regression Models, print the coefficients. Do the variables make sense? If not, what would you recommend?
"""
#%%
"""
SETUP THE ENVIRONMENT
    1. Import Libraries
    2. Setup Display Options
    3. Load Data and Define Target Variables
"""
""" Import Libraries """
# Data Manipulation Libraries
import math
import pandas as pd
import numpy as np
from operator import itemgetter
# Data Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Machine Learning Libraries
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
# -----
from sklearn import tree
from sklearn.tree import _tree
# -----
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
# -----
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
# -----
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
# -----
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

""" Setup Display Options """
# Suppress Warnings
import warnings
warnings.filterwarnings('ignore')

# Display Options
sns.set()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

""" Load Data and Define Target Variables """
# Import the data
INFILE = "/Users/stefanjenss/Documents/DataScience/Data_Science_Masters/MSDS_422_Machine_Learning/MSDS_422_Assignments/HMEQ_Loss.csv"
df = pd.read_csv(INFILE)

# Define the target variables
TARGET_F = "TARGET_BAD_FLAG"
TARGET_A = "TARGET_LOSS_AMT"

# Define the data types of the variables
dt = df.dtypes

# %%
"""
---------------------------------------
CODE FROM PREVIOUS ASSIGNMENT FOR SETUP
--------------------------------------- 
"""
# Firstly, create a list of object and numeric variables
objList = []
numList = []
for i in dt.index:
    #print("here is i .....", i, "..... and here is the type", dt[i])
    if i in ([TARGET_F, TARGET_A]): continue
    if dt[i] in (["object"]): objList.append(i)
    if dt[i] in (["int64", "float64"]): numList.append(i)

#%%
"""
FILLING IN MISSING DATA.

IMPUTATION METHOD: FILL IN MISSING WITH THE CATEGORY "MISSING"
"""
for i in objList:
    if df[i].isna().sum() == 0 : continue           # skip if there are no missing values
    FLAG = "M_" + i                                 # create new flag variable name **NEW
    NAME = "IMP_" + i                               # create a new variable name 
    df[FLAG] = df[i].isna() + 0                     # populate the new flag variable **NEW
    df[NAME] = df[i].fillna("MISSING")              # fill in the missing values with the category "MISSING"
    g = df.groupby(NAME)                            # group by the new variable
    df = df.drop(i, axis = 1)                       # drop the old variable

# Get the data types to include the new variables
dt = df.dtypes
objList = []
for i in dt.index:
    if i in ([TARGET_F, TARGET_A]): continue
    if dt[i] in (["object"]): objList.append(i)

# %%
"""
MISSING VALUE IMPUTATION FOR NUMERIC VARIABLES

IMPUTATION METHOD 1: PERFORM MISSING VALUE IMPUTATION FOR "VALUE" BASED ON "JOB" CLASS
"""
i = "VALUE"
FLAG = "M_" + i
IMP = "IMP_" + i

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

# %%
"""
IMPUTATION METHOD 2: PERFORM MISSING VLUAE IMPUTATION FOR "LOAN" BASED ON "JOB" CLASS
"""
i = "LOAN"
FLAG = "M_" + i
IMP = "IMP_" + i

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

# %%
"""
IMPUTATION METHOD 3: PERFORM MISSING VALUE IMPUTATION FOR "DEBTINC" BASED ON "JOB" CLASS
"""
i = "DEBTINC"
FLAG = "M_" + i
IMP = "IMP_" + i

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

# %%
"""
IMPUTATION METHOD 4: PERFORM A GENERAL MISSING VALUE IMPUTATION FOR THE REST OF THE NUMERIC VARIABLES BASED ON THE MEDIAN
"""
# Create a list of numeric variables
floatList = []
dt = df.dtypes
for i in dt.index :
    #print(" here is i .....", i , " ..... and here is the type", dt[i] )
    if i in (["TARGET_BAD_FLAG", "TARGET_LOSS_AMT"]) : continue
    if dt[i] in (["float64"]) : floatList.append(i)
    
# Perform the missing value imputation on the remaining non-imputed numeric variables
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
ONE-HOT ENCODING: PERFORM ONE-HOT ENCODING FOR THE CATEGORICAL VARIABLES
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
    thePrefix = "z_" + i                      # create the prefix for the new variables
    # print(f'Prefix = {thePrefix}')          # print the prefix
    y = pd.get_dummies(df[i], prefix=thePrefix, drop_first=False, dtype="int64") # perform one hot encoding
    df = pd.concat([df, y], axis=1)         # concatenate the new variables to the data frame
    df = df.drop(i, axis=1)                 # drop the old variable
    
# Check to see if the one-hot encoding was successful
df.info()

# %%
"""
==================================
!!! START OF ASSIGNMENT 3 CODE !!!
==================================
"""

"""
PHASE I: CREATE A TRAINING AND TEST DATA SET
    - Additionally, handle outliers
"""

""" Split the data into training and test sets """
X = df.copy()                   # create a copy of the data frame
X = X.drop(TARGET_F, axis=1)    # drop the target flag variable
X = X.drop(TARGET_A, axis=1)    # drop the target amount variable
Y = df[[TARGET_F, TARGET_A]]    # create a data frame with the target variables
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.80, test_size=0.20, random_state=1)  # 80% training, 20% testing

# Display the shape of the training and test data
print('FLAG DATA')
print("TRAINING = ", X_train.shape)
print("TESTING = ", X_test.shape)

# %%
"""
Handle Outliers
    - Outliers will be considered to be those entries that have a `TARGET_LOSS_AMT` value that is greater than $60,000.
    - Cap the loss amount at $60,000
"""

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
PHASE II: CREATE FUNCTIONS TO RETRIEVE MODEL ACCURACY, ROC CURVES, AND RMSE SCORES FOR THE MODELS
"""

""" Define a function to calculate the accuracy score of the model """
def getProbAccuracyScores(NAME, MODEL, X, Y):
    pred = MODEL.predict(X)
    probs = MODEL.predict_proba(X)
    acc_score  = metrics.accuracy_score(Y, pred)
    p1 = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(Y, p1)
    auc = metrics.auc(fpr, tpr)
    return [NAME, acc_score, fpr, tpr, auc]

""" Define a function to plot the ROC curve """
def print_ROC_Curve(TITLE, LIST):
    fig = plt.figure(figsize=(6, 4))
    plt.title(TITLE)
    for theResults in LIST:
        NAME = theResults[0]
        fpr = theResults[2]
        tpr = theResults[3]
        auc = theResults[4]
        theLabel = "AUC " + NAME + " %0.2f" % auc
        plt.plot(fpr, tpr, label = theLabel)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

""" Define a function to print the accuracy scores """
def print_Accuracy(TITLE, LIST):
    print(TITLE)
    print("======")
    for theResults in LIST:
        NAME = theResults[0]
        ACC = theResults[1]
        print(NAME, "Accuracy Score = ", ACC)
    print("------\n\n")

""" Define a function to calculate the RMSE scores """
def getAmtAccuracyScores(NAME, MODEL, X, Y):
    pred = MODEL.predict(X)
    MEAN = Y.mean()
    RMSE = math.sqrt(metrics.mean_squared_error(Y, pred))
    return [NAME, RMSE, MEAN]

# %%
"""
PHASE III: EVALUATE THE PREVIOUSLY CREATED MODELS USING THE FUNCTIONS DEFINED IN PHASE II
    These models include:
        - Decision Tree
        - Random Forest
        - Gradient Boosting
        
We want to evaluate the previously created M.L. modules from Assignment 2 to compare their results to the Logistic Regression and
Linear Regression modules that we will create in this assignment.
"""

"""
DECISION TREE
"""
WHO = "Decision Tree" # Define the name of the model
# Define the function to get the tree variables
def getTreeVars(TREE, varNames):
    tree_ = TREE.tree_
    varName = [varNames[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

    nameSet = set()
    for i in tree_.feature:
        if i != _tree.TREE_UNDEFINED:
            nameSet.add(i)
    nameList = list(nameSet)
    parameter_list = list()
    for i in nameList:
        parameter_list.append(varNames[i])
    return parameter_list

"""
DEFAULT PROBABILITY - DECISION TREE
"""
# Define the decision tree model
CLM = tree.DecisionTreeClassifier( max_depth=3 )
CLM = CLM.fit( X_train, Y_train[ TARGET_F ] )

# Call the functions to calculate the accuracy scores
TRAIN_CLM = getProbAccuracyScores( WHO + "_Train", CLM, X_train, Y_train[ TARGET_F ] )
TEST_CLM = getProbAccuracyScores( WHO, CLM, X_test, Y_test[ TARGET_F ] )

# Call the functions to print the results
print_ROC_Curve( WHO, [ TRAIN_CLM, TEST_CLM ] ) 
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )

# Get the tree variables and print the tree
feature_cols = list( X.columns.values )
tree.export_graphviz(CLM,out_file='tree_f.txt',filled=True, rounded=True, feature_names = feature_cols, impurity=False, class_names=["Good","Bad"]  )
vars_tree_flag = getTreeVars( CLM, feature_cols )

# %%
"""
AMOUNT LOST ASSUMING DEFAULT - DECISION TREE
"""
# Define the decision tree model
AMT = tree.DecisionTreeRegressor( max_depth= 4 )
AMT = AMT.fit( W_train, Z_train[TARGET_A] )

# Call the functions to calculate the accuracy scores
TRAIN_AMT = getAmtAccuracyScores( WHO + "_Train", AMT, W_train, Z_train[TARGET_A] )
TEST_AMT = getAmtAccuracyScores( WHO, AMT, W_test, Z_test[TARGET_A] )
print_Accuracy( WHO + " RMSE ACCURACY", [ TRAIN_AMT, TEST_AMT ] )

# Get the tree variables and print the tree
feature_cols = list( X.columns.values )
vars_tree_amt = getTreeVars( AMT, feature_cols ) 
tree.export_graphviz(AMT,out_file='tree_a.txt',filled=True, rounded=True, feature_names = feature_cols, impurity=False, precision=0  )

# Save the results
TREE_CLM = TEST_CLM.copy()
TREE_AMT = TEST_AMT.copy()

# %%
"""
RANDOM FOREST
"""
WHO = "Random Forest" # Define the name of the model
# Define the function to get the ensemble tree variables
def getEnsembleTreeVars(ENSTREE, varNames):
    importance = ENSTREE.feature_importances_           # get the importance of the variables
    index = np.argsort(importance)                      # get the index of the importance
    theList = []                                        # create a list for the variables
    for i in index:                                     # loop through the index
        imp_val = importance[i]                                 # get the importance value
        if imp_val > np.average(ENSTREE.feature_importances_):  # if the importance value is greater than the average
            v = int(imp_val/np.max(ENSTREE.feature_importances_)*100)   # calculate the importance value
            theList.append((varNames[i], v))                            # append the variable name and importance value to the list
    theList = sorted(theList, key=itemgetter(1), reverse=True)  # sort the list
    return theList                                              # return the list

"""
DEFAULT PROBABILITY - RANDOM FOREST
"""
# Define the random forest model
CLM = RandomForestClassifier(n_estimators=25, random_state=1)
CLM = CLM.fit(X_train, Y_train[TARGET_F])

# Call the functions to calculate the accuracy scores
TRAIN_CLM = getProbAccuracyScores(WHO + "_Train", CLM, X_train, Y_train[TARGET_F])
TEST_CLM = getProbAccuracyScores(WHO, CLM, X_test, Y_test[TARGET_F])

# Call the functions to print the results
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

# Get the ensemble tree variables
feature_cols = list(X.columns.values)
vars_RF_flag = getEnsembleTreeVars(CLM, feature_cols)

# %%
"""
AMOUNT LOST ASSUMING DEFAULT - RANDOM FOREST
"""
# Define the random forest model
AMT = RandomForestRegressor(n_estimators=100, random_state=1)
AMT = AMT.fit(W_train, Z_train[TARGET_A])

# Call the functions to calculate the accuracy scores
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, W_test, Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Get the ensemble tree variables
feature_cols = list(X.columns.values)
vars_RF_amt = getEnsembleTreeVars(AMT, feature_cols)

# Print the results
for i in vars_RF_amt:
    print(i)

# Save the results
RF_CLM = TEST_CLM.copy()
RF_AMT = TEST_AMT.copy()

# %%
"""
GRADIENT BOOSTING
"""
WHO = "Gradient Boosting" # Define the name of the model
"""
DEFAULT PROBABILITY - GRADIENT BOOSTING
"""
# Define the function to get the ensemble tree variables
CLM = GradientBoostingClassifier(random_state=1)
CLM = CLM.fit(X_train, Y_train[TARGET_F])

# Call the functions to calculate the accuracy scores
TRAIN_CLM = getProbAccuracyScores(WHO + "_Train", CLM, X_train, Y_train[TARGET_F])
TEST_CLM = getProbAccuracyScores(WHO, CLM, X_test, Y_test[TARGET_F])

# Call the functions to print the results
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

# Get the ensemble tree variables
feature_cols = list(X.columns.values)
vars_GB_flag = getEnsembleTreeVars(CLM, feature_cols)

# %%
"""
AMOUNT LOST ASSUMING DEFAULT - GRADIENT BOOSTING
"""
# Define the gradient boosting model
AMT = GradientBoostingRegressor(random_state=1)
AMT = AMT.fit(W_train, Z_train[TARGET_A])

# Call the functions to calculate the accuracy scores
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, W_test, Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Get the ensemble tree variables
feature_cols = list(X.columns.values)
vars_GB_amt = getEnsembleTreeVars(AMT, feature_cols)

# Print the results
for i in vars_GB_amt:
    print(i)

# Save the results
GB_CLM = TEST_CLM.copy()
GB_AMT = TEST_AMT.copy()

# %%
"""
PHASE IV: LOGISTIC REGRESSION AND LINEAR REGRESSION MODELS FOR THE PROBABILITY OF A LOAN DEFAULT AND THE EXPECTED LOSS AMOUNT
Logistic and linear regression models will be created using the following variable selection methods:
    - All variables
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - Stepwise Selection
"""
"""
First, define functions to get the coefficients of the logistic and linear regression model
"""
# Define a function to get the coefficients of the logistic regression model
def getCoefLogit(MODEL, TRAIN_DATA):
    varNames = list(TRAIN_DATA.columns.values)          # get the variable names
    coef_dict = {}                                      # create a dictionary for the coefficients
    coef_dict["INTERCEPT"] = MODEL.intercept_[0]        # get the intercept
    for coef, feat in zip(MODEL.coef_[0], varNames):    # loop through the coefficients and variable names
        coef_dict[feat] = coef                              # add the coefficient to the dictionary
    print("\nDEFAULT")
    print("----------")
    print("Total Variables: ", len(coef_dict))          # print the total number of variables
    for i in coef_dict:                                 # loop through the dictionary
        print(i, "=", coef_dict[i])                         # print the variable and coefficient

# Define a function to get the coefficients of the linear regression model
def getCoefLinear(MODEL, TRAIN_DATA):
    varNames = list(TRAIN_DATA.columns.values)          # get the variable names
    coef_dict = {}                                      # create a dictionary for the coefficients
    coef_dict["INTERCEPT"] = MODEL.intercept_           # get the intercept
    for coef, feat in zip(MODEL.coef_, varNames):       # loop through the coefficients and variable names
        coef_dict[feat] = coef                              # add the coefficient to the dictionary
    print("\nAMOUNT")
    print("----------")
    print("Total Variables: ", len(coef_dict))          # print the total number of variables
    for i in coef_dict:                                 # loop through the dictionary
        print(i, "=", coef_dict[i])                         # print the variable and coefficient
    
# %%
"""
REGRESSION FOR ALL VARIABLES
"""
WHO = "Regression - All Variables" # Define the name of the model
"""
DEFAULT PROBABILITY - LOGISTIC REGRESSION - ALL VARIABLES
"""
# Define the logistic regression model
CLM = LogisticRegression(solver='newton-cg', max_iter=1000)
CLM = CLM.fit(X_train, Y_train[TARGET_F])

# Call the functions to calculate the accuracy scores
TRAIN_CLM = getProbAccuracyScores(WHO + "_Train", CLM, X_train, Y_train[TARGET_F])
TEST_CLM = getProbAccuracyScores(WHO, CLM, X_test, Y_test[TARGET_F])

# Call the functions to print the results
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

#%%
"""
AMOUNT - LINEAR REGRESSION - ALL VARIABLES
"""
# Define the linear regression model
AMT = LinearRegression()
AMT = AMT.fit(W_train, Z_train[TARGET_A])

# Call the functions to calculate the accuracy scores
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, W_test, Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Get the list of variables
varNames = list(X_train.columns.values)

# Call the functions to get the coefficients
REG_ALL_CLM_COEF = getCoefLogit(CLM, X_train)
REG_ALL_AMT_COEF = getCoefLinear(AMT, W_train)

# Save the results
REG_ALL_CLM = TEST_CLM.copy()
REG_ALL_AMT = TEST_AMT.copy()

# %%
"""
REGRESSION FOR DECISION TREE VARIABLES
"""
WHO = "Regression - Decision Tree" # Define the name of the model
"""
DEFAULT PROBABILITY - LOGISTIC REGRESSION - DECISION TREE VARIABLES
"""
# Define the logistic regression model
CLM = LogisticRegression(solver='newton-cg', max_iter=1000)
CLM = CLM.fit(X_train[vars_tree_flag], Y_train[TARGET_F])

# Call the functions to calculate the accuracy scores
TRAIN_CLM = getProbAccuracyScores(WHO + "_Train", CLM, X_train[vars_tree_flag], Y_train[TARGET_F])
TEST_CLM = getProbAccuracyScores(WHO, CLM, X_test[vars_tree_flag], Y_test[TARGET_F])

# Call the functions to print the results
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

# %%
"""
AMOUNT - LINEAR REGRESSION - DECISION TREE VARIABLES
"""
# Define the linear regression model
AMT = LinearRegression()
AMT = AMT.fit(W_train[vars_tree_amt], Z_train[TARGET_A])

# Call the functions to calculate the accuracy scores
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train[vars_tree_amt], Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, W_test[vars_tree_amt], Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Get the list of variables
varNames = list(X_train.columns.values)

# Call the functions to get the coefficients
REG_TREE_CLM_COEF = getCoefLogit(CLM, X_train[vars_tree_flag])
REG_TREE_AMT_COEF = getCoefLinear(AMT, W_train[vars_tree_amt])

# Save the results
REG_TREE_CLM = TEST_CLM.copy()
REG_TREE_AMT = TEST_AMT.copy()

# %%
"""
REGRESSION FOR RANDOM FOREST VARIABLES
"""
WHO = "Regression - Random Forest" # Define the name of the model
"""
First, define the random forest variables for the logistic and linear regression model mnodels
"""
# Define and print the random forest variables for the logistic regression model for the probability of default
print("\n\n")
RF_flag = []
for i in vars_RF_flag:
    print(i)
    theVar = i[0]
    RF_flag.append(theVar)

# Define and print the random forest variables for the linear regression model for the amount lost assuming default
print("\n\n")
RF_amt = []
for i in vars_RF_amt:
    print(i)
    theVar = i[0]
    RF_amt.append(theVar)

# %%
"""
DEFAULT PROBABILITY - LOGISTIC REGRESSION - RANDOM FOREST VARIABLES
"""
# Define the logistic regression model
CLM = LogisticRegression(solver='newton-cg', max_iter=1000)
CLM = CLM.fit(X_train[RF_flag], Y_train[TARGET_F])

# Call the functions to calculate the accuracy scores
TRAIN_CLM = getProbAccuracyScores(WHO + "_Train", CLM, X_train[RF_flag], Y_train[TARGET_F])
TEST_CLM = getProbAccuracyScores(WHO, CLM, X_test[RF_flag], Y_test[TARGET_F])

# Call the functions to print the results
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

# %%
"""
AMOUNT - LINEAR REGRESSION - RANDOM FOREST VARIABLES
"""
# Define the linear regression model
AMT = LinearRegression()
AMT = AMT.fit(W_train[RF_amt], Z_train[TARGET_A])

# Call the functions to calculate the accuracy scores
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train[RF_amt], Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, W_test[RF_amt], Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Get the coefficients of the logistic and linear regression models
REG_RF_CLM_COEF = getCoefLogit(CLM, X_train[RF_flag])
REG_RF_AMT_COEF = getCoefLinear(AMT, W_train[RF_amt])

# Save the results
REG_RF_CLM = TEST_CLM.copy()
REG_RF_AMT = TEST_AMT.copy()

# %%
"""
REGRESSION FOR GRADIENT BOOSTING VARIABLES
"""
WHO = "Regression - Gradient Boosting" # Define the name of the model
"""
Firstly, define the gradient boosting variables for the logistic and linear regression models
"""
# Define and print the gradient boosting variables for the logistic regression model for the probability of default
print("\n\n")
GB_flag = []
for i in vars_GB_flag:
    print(i)
    theVar = i[0]
    GB_flag.append(theVar)
    
# Define and print the gradient boosting variables for the linear regression model for the amount lost assuming default
print("\n\n")
GB_amt = []
for i in vars_GB_amt:
    print(i)
    theVar = i[0]
    GB_amt.append(theVar)

# %%
"""
DEFAULT PROBABILITY - LOGISTIC REGRESSION - GRADIENT BOOSTING VARIABLES
"""
# Define the logistic regression model
CLM = LogisticRegression(solver='newton-cg', max_iter=1000)
CLM = CLM.fit(X_train[GB_flag], Y_train[TARGET_F])

# Call the functions to calculate the accuracy scores
TRAIN_CLM = getProbAccuracyScores(WHO + "_Train", CLM, X_train[GB_flag], Y_train[TARGET_F])
TEST_CLM = getProbAccuracyScores(WHO, CLM, X_test[GB_flag], Y_test[TARGET_F])

# Call the functions to print the results
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

# %%
"""
AMOUNT - LINEAR REGRESSION - GRADIENT BOOSTING VARIABLES
"""
# Define the linear regression model
AMT = LinearRegression()
AMT = AMT.fit(W_train[GB_amt], Z_train[TARGET_A])

# Call the functions to calculate the accuracy scores
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, W_train[GB_amt], Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, W_test[GB_amt], Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Get the coefficients of the logistic and linear regression models
REG_GB_CLM_COEF = getCoefLogit(CLM, X_train[GB_flag])
REG_GB_AMT_COEF = getCoefLinear(AMT, W_train[GB_amt])

# Save the results
REG_GB_CLM = TEST_CLM.copy()
REG_GB_AMT = TEST_AMT.copy()

# %%
"""
REGRESSION FOR STEPWISE SELECTION VARIABLES
"""
WHO = "Regression - Stepwise Selection" # Define the name of the model
"""
DEFAULT PROBABILITY - STEPWISE SELECTION VARIABLES
"""
U_train = X_train[vars_tree_flag].copy()    # create a working copy of the training data
stepVarNames = list(U_train.columns.values) # get the variable names
maxCols = U_train.shape[1]                  # get the maximum number of columns

# Defin the sequential forward selection model
sfs = SFS(LogisticRegression(solver='newton-cg', max_iter=100),
          k_features=(1, maxCols),
          forward=True,
          floating=False,
          cv=3)
sfs.fit(U_train.values, Y_train[TARGET_F].values)

# Plot the results
theFigure = plot_sfs(sfs.get_metric_dict(), kind=None)
plt.title('Default Probability Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()

#  
dfm = pd.DataFrame.from_dict(sfs.get_metric_dict()).T       # create a data frame from the dictionary 
dfm = dfm[['feature_names', 'avg_score']]                   # get the feature names and average score
dfm.ave_score = dfm.avg_score.astype(float)                 # convert the average score to a float


print(" ................... ")
maxIndex = dfm.avg_score.argmax()                           # get the index of the maximum average score
print("argmax")
print(dfm.iloc[maxIndex, ])                                 # print the maximum average score
print(" ................... ")

stepVars = dfm.iloc[maxIndex]                              # get the variables from the maximum average score
stepVars = stepVars.feature_names                          # get the feature names from the maximum average score
print(stepVars)                                            # print the feature names from the maximum average score

finalStepVars = []                                         # create a list for the final step variables
for i in stepVars:                                         # loop through the step variables
    index = int(i)                                         # get the index of the variable
    try:
        theName = stepVarNames[index]                      # get the name of the variable
        finalStepVars.append(theName)                      # append the name of the variable to the final step variables
    except:
        pass

for i in finalStepVars:                                    # loop through the final step variables
    print(i)                                               # print the variable

U_train = X_train[finalStepVars] 
U_test = X_test[finalStepVars]

# %%
"""
AMOUNT - STEPWISE SELECTION VARIABLES
"""
V_train = W_train[GB_amt].copy()                            # create a working copy of the training data
stepVarNames = list(V_train.columns.values)                 # get the variable names
maxCols = V_train.shape[1]                                  # get the maximum number of columns

# Define the sequential forward selection model
sfs = SFS(LinearRegression(),
          k_features=(1, maxCols),
          forward=True,
          floating=False,
          scoring='r2',
          cv=5)
sfs.fit(V_train.values, Z_train[TARGET_A].values)

# Plot the results
theFigure = plot_sfs(sfs.get_metric_dict(), kind=None)
plt.title('Amount Sequential Forward Selection (w. R^2)')
plt.grid()
plt.show()

# Create a data frame from the dictionary
dfm = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
dfm = dfm[['feature_names', 'avg_score']]
dfm.ave_score = dfm.avg_score.astype(float)

# Print the maximum average score
print(" ................... ")
maxIndex = dfm.avg_score.argmax()
print("argmax")
print(dfm.iloc[maxIndex, ])
print(" ................... ")

# Get the variables from the maximum average score
stepVars = dfm.iloc[maxIndex]
stepVars = stepVars.feature_names
print(stepVars)

finalStepVars = []
for i in stepVars:
    index = int(i)
    try:
        theName = stepVarNames[index]
        finalStepVars.append(theName)
    except:
        pass
    
for i in finalStepVars:
    print(i)

V_train = W_train[finalStepVars]
V_test = W_test[finalStepVars]

# %%
"""
DEFAULT PROBABILITY - LOGISTIC REGRESSION - STEPWISE SELECTION VARIABLES
"""
# Define the logistic regression model
CLM = LogisticRegression(solver='newton-cg', max_iter=1000)
CLM = CLM.fit(U_train, Y_train[TARGET_F])

# Call the functions to calculate the accuracy scores
TRAIN_CLM = getProbAccuracyScores(WHO + "_Train", CLM, U_train, Y_train[TARGET_F])
TEST_CLM = getProbAccuracyScores(WHO, CLM, U_test, Y_test[TARGET_F])

# Call the functions to print the results
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

# %%
"""
AMOUNT - LINEAR REGRESSION - STEPWISE SELECTION VARIABLES
"""
# Define the linear regression model
AMT = LinearRegression()
AMT = AMT.fit(V_train, Z_train[TARGET_A])

# Call the functions to calculate the accuracy scores
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, V_train, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, V_test, Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Get the coefficients of the logistic and linear regression models
REG_STEP_CLM_COEF = getCoefLogit(CLM, U_train)
REG_STEP_AMT_COEF = getCoefLinear(AMT, V_train)

# Save the results
REG_STEP_CLM = TEST_CLM.copy()
REG_STEP_AMT = TEST_AMT.copy()

# %%
"""
RUN ALL OF THE MODELS AND PRINT THE RESULTS
"""
# Create a list of all of the classification models
ALL_CLM = [TREE_CLM, RF_CLM, GB_CLM, REG_ALL_CLM, REG_TREE_CLM, REG_RF_CLM, REG_GB_CLM, REG_STEP_CLM]

# Sort the results by the AUC and print the ROC curve
ALL_CLM = sorted(ALL_CLM, key=lambda x: x[4], reverse=True)
print_ROC_Curve("ALL CLASSIFICATION MODELS ACCURACY", ALL_CLM)

# Create a list of all of the regression models
ALL_AMT = [TREE_AMT, RF_AMT, GB_AMT, REG_ALL_AMT, REG_TREE_AMT, REG_RF_AMT, REG_GB_AMT, REG_STEP_AMT]

# Sort the results by the RMSE and print the RMSE scores
ALL_AMT = sorted(ALL_AMT, key=lambda x: x[1])
print_Accuracy("ALL AMOUNT LOST RMSE MODELS ACCURACY", ALL_AMT)

# %%
