"""
# Assignment 4: Neural Networks

### Assignment Requirements
#### I. Create a Training and Test Data Set

#### II. Tensor Flow Model to Predict Loan Default
1. Develop a model using Tensor Flow that will predict Loan Default
2. For your model, do the following
    - Try at least three different Activation Functions
    - Try one and two hidden layers
    - Try using a Dropout Layer
3. Explore using a variable selection technique
4. For each of the models
    - Calculate the accuracy of the model on both the training and test data set
    - Create a graph that shows the ROC curves for both the training and test data set. Clearly label each curve and display
      the Area Under the ROC curve.
    - Display the ROC curve for the test data with all your models on the same graph (tree based, regression, and TF). 
      Discuss which one is the most accurate. Which one would you recommend using?

#### III. Tensor Flow Model to Predict Loss Given Default
1. Develop a model using Tensor Flow that will predict Loan Default.
2. For your model, do the following:
    - Try at least three different Activation Functions
    - Try one and two hidden layers
    - Try using a Dropout Layer
3. Explore using a variable selection technique
4. For each of the models:
    - Calculate the RMSE for both the training data set and the test data set
    - List the RMSE for the test data set for all the models created (tree based, regression, and TF). Discuss which one is 
      the most accurate. Which one would you recommend using?
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
sns.set_style()
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
Create a DataFrame of the variable data types and create lists of the categorical and numeric variable names
"""
# Create a DataFrame of the variable data types
dt = df.dtypes

# Create lists of the categorical and numeric variable names
objList = []
numList = []
for i in dt.index :
    #print(" here is i .....", i , " ..... and here is the type", dt[i] )
    if i in ( [ TARGET_F, TARGET_A ] ) : continue
    if dt[i] in (["object"]) : objList.append( i )
    if dt[i] in (["float64","int64"]) : numList.append( i )
    
# %%
"""
PREPHASE: DATA PREPARATION
    1. Fill in missing with the category "MISSING"
    2. Perform missing value imputation for the numeric variables
    3. Perform one-hot encoding of the categorical variables
    4. Remove outliers
"""
"""
(1) Fill in missing with the category "MISSING"
"""
for i in objList:
    if df[i].isna().sum() == 0 : continue           # skip if there are no missing values
    NAME = "IMP_" + i                               # create a new variable name 
    df[NAME] = df[i]                                # copy the original variable
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
(2) Missing value imputation for the numeric variables
"""
"""
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
(3) Perform one-hot encoding of the categorical variables
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

# Create a new list of the objList (categorical variables)
objList = []
dt = df.dtypes
for i in dt.index:
    if i in ([TARGET_F, TARGET_A]): continue
    if dt[i] in (["object"]): objList.append(i)

# %%
"""
(4) Remove outliers
"""
# Create a new list of the numeric variables
dt = df.dtypes
numList = []
for i in dt.index:
    if i in ([TARGET_F, TARGET_A]): continue
    if dt[i] in (["float64", "int64"]): numList.append(i)
    
# Remove the outliers
#   - Create a function to remove the outliers
#   - The cutoff for the outliers is 3 standard deviations from the mean
#   - The function will create a flag variable for each variable that has outliers
#   - Additionally, the function will create a new variable that replaces the outlier with the cutoff value
for i in numList:
    theMean = df[i].mean()
    theSD = df[i].std()
    theMax = df[i].max()
    theCutoff = round(theMean + 3 * theSD)
    if theMax < theCutoff: continue
    FLAG = "O_" + i
    TRUNC = "TRUNC_" + i
    df[FLAG] = (df[i] > theCutoff) + 0      # create a flag variable
    df[TRUNC] = df[i]                       # create a new variable
    df.loc[df[TRUNC] > theCutoff, TRUNC] = theCutoff # replace the outlier with the cutoff value
    df = df.drop(i, axis=1)                 # drop the old variable

# Drop the original categorical variables
for i in objList:               
    df = df.drop(i, axis=1)

# %%
"""
SPLIT THE DATA INTO TRAINING AND TEST SETS 
    1. Create the X and Y data
    2. Split the data into training and test sets
    3. Create the training and test data for the amount lost upon default model
        - Set a maximum loss amount of $60,000
    
"""
""" (1) Create the X and Y data, where X includes all the columns expect the target variables and Y includes the target variables """
X = df.copy()
X = X.drop([TARGET_F, TARGET_A], axis=1)
Y = df[[TARGET_F, TARGET_A]]

""" (2) Split the data into training and test sets """
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=2)

# Display the shape of the training and test dat 
print("FLAG DATA")
print("TRAINING = ", X_train.shape)
print("TEST = ", X_test.shape)

""" (3) Create the training and test data for the damages model and set a maximum loss amount of $60,000 """
F = ~ Y_train[TARGET_A].isna()
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
CREATE MODEL ACCURACY METRIC FUNCTIONS
    1. Function to calculate the accuracy score of the model
    2. Function to plot the ROC curve of the model
    3. Function to print the accuracy score of the model
    4. Function to calculate the RMSE score of the model
"""
""" (1) Define a function to calculate the accuracy score of the model """
def getProbAccuracyScores(NAME, MODEL, X, Y):
    pred = MODEL.predict(X)
    probs = MODEL.predict_proba(X)
    acc_score  = metrics.accuracy_score(Y, pred)
    p1 = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(Y, p1)
    auc = metrics.auc(fpr, tpr)
    return [NAME, acc_score, fpr, tpr, auc]

""" (2) Define a function to plot the ROC curve """
def print_ROC_Curve(TITLE, LIST):
    fig = plt.figure(figsize=(10, 8))
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

""" (3) Define a function to print the accuracy scores """
def print_Accuracy(TITLE, LIST):
    print(TITLE)
    print("======")
    for theResults in LIST:
        NAME = theResults[0]
        ACC = theResults[1]
        print(NAME, "Accuracy Score = ", ACC)
    print("------\n\n")

""" (4) Define a function to calculate the RMSE scores """
def getAmtAccuracyScores(NAME, MODEL, X, Y):
    pred = MODEL.predict(X)
    MEAN = Y.mean()
    RMSE = math.sqrt(metrics.mean_squared_error(Y, pred))
    return [NAME, RMSE, MEAN]

# %%
"""
DEVELOP ALL PREVIOUS MODELS FOR THE PROBABILIY OF DEFAULT AND THE AMOUNT OF LOSS
    1. Decision Tree Model
    2. Random Forest Model
    3. Gradient Boosting Model
    4. Linear and Logistic Regression Models
        a. Regression - All Variables
        b. Regression - Decision Tree Variables
        c. Regression - Random Forest Variables
        d. Regression - Gradient Boosting Variables
        e. Regression - Stepwise Selection Variables
"""
"""
(1) DECISION TREE
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

""" DEFAULT PROBABILITY - DECISION TREE """
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

""" AMOUNT LOST ASSUMING DEFAULT - DECISION TREE """
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

"""
(2) RANDOM FOREST
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

""" DEFAULT PROBABILITY - RANDOM FOREST """
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

""" AMOUNT LOST ASSUMING DEFAULT - RANDOM FOREST """
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

"""
(3) GRADIENT BOOSTING
"""
WHO = "Gradient Boosting" # Define the name of the model
""" DEFAULT PROBABILITY - GRADIENT BOOSTING """
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

""" AMOUNT LOST ASSUMING DEFAULT - GRADIENT BOOSTING """
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

"""
(4) LINEAR AND LOGISTIC REGRESSION MODELS
"""
""" First, define functions to get the coefficients of the logistic and linear regression model """
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
    
"""
(4.a) REGRESSION FOR ALL VARIABLES
"""
WHO = "Regression - All Variables" # Define the name of the model
""" DEFAULT PROBABILITY - LOGISTIC REGRESSION - ALL VARIABLES """
# Define the logistic regression model
CLM = LogisticRegression(solver='newton-cg', max_iter=1000)
CLM = CLM.fit(X_train, Y_train[TARGET_F])

# Call the functions to calculate the accuracy scores
TRAIN_CLM = getProbAccuracyScores(WHO + "_Train", CLM, X_train, Y_train[TARGET_F])
TEST_CLM = getProbAccuracyScores(WHO, CLM, X_test, Y_test[TARGET_F])

# Call the functions to print the results
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

""" AMOUNT - LINEAR REGRESSION - ALL VARIABLES """
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

"""
(4.b) REGRESSION FOR DECISION TREE VARIABLES
"""
WHO = "Regression - Decision Tree" # Define the name of the model
""" DEFAULT PROBABILITY - LOGISTIC REGRESSION - DECISION TREE VARIABLES """ 
# Define the logistic regression model
CLM = LogisticRegression(solver='newton-cg', max_iter=1000)
CLM = CLM.fit(X_train[vars_tree_flag], Y_train[TARGET_F])

# Call the functions to calculate the accuracy scores
TRAIN_CLM = getProbAccuracyScores(WHO + "_Train", CLM, X_train[vars_tree_flag], Y_train[TARGET_F])
TEST_CLM = getProbAccuracyScores(WHO, CLM, X_test[vars_tree_flag], Y_test[TARGET_F])

# Call the functions to print the results
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

""" AMOUNT - LINEAR REGRESSION - DECISION TREE VARIABLES """
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

"""
(4.c) REGRESSION FOR RANDOM FOREST VARIABLES
"""
WHO = "Regression - Random Forest" # Define the name of the model
""" First, define the random forest variables for the logistic and linear regression model mnodels """
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

""" DEFAULT PROBABILITY - LOGISTIC REGRESSION - RANDOM FOREST VARIABLES """
# Define the logistic regression model
CLM = LogisticRegression(solver='newton-cg', max_iter=1000)
CLM = CLM.fit(X_train[RF_flag], Y_train[TARGET_F])

# Call the functions to calculate the accuracy scores
TRAIN_CLM = getProbAccuracyScores(WHO + "_Train", CLM, X_train[RF_flag], Y_train[TARGET_F])
TEST_CLM = getProbAccuracyScores(WHO, CLM, X_test[RF_flag], Y_test[TARGET_F])

# Call the functions to print the results
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

""" AMOUNT - LINEAR REGRESSION - RANDOM FOREST VARIABLES """
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

"""
(4.d) REGRESSION FOR GRADIENT BOOSTING VARIABLES
"""
WHO = "Regression - Gradient Boosting" # Define the name of the model
""" Firstly, define the gradient boosting variables for the logistic and linear regression models """
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

""" DEFAULT PROBABILITY - LOGISTIC REGRESSION - GRADIENT BOOSTING VARIABLES """
# Define the logistic regression model
CLM = LogisticRegression(solver='newton-cg', max_iter=1000)
CLM = CLM.fit(X_train[GB_flag], Y_train[TARGET_F])

# Call the functions to calculate the accuracy scores
TRAIN_CLM = getProbAccuracyScores(WHO + "_Train", CLM, X_train[GB_flag], Y_train[TARGET_F])
TEST_CLM = getProbAccuracyScores(WHO, CLM, X_test[GB_flag], Y_test[TARGET_F])

# Call the functions to print the results
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

""" AMOUNT - LINEAR REGRESSION - GRADIENT BOOSTING VARIABLES """
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

"""
(4.e) REGRESSION FOR STEPWISE SELECTION VARIABLES
"""
WHO = "Regression - Stepwise Selection" # Define the name of the model
""" DEFAULT PROBABILITY - STEPWISE SELECTION VARIABLES """
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

# Create a data frame from the dictionary
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

""" AMOUNT - STEPWISE SELECTION VARIABLES """
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

""" DEFAULT PROBABILITY - LOGISTIC REGRESSION - STEPWISE SELECTION VARIABLES """
# Define the logistic regression model
CLM = LogisticRegression(solver='newton-cg', max_iter=1000)
CLM = CLM.fit(U_train, Y_train[TARGET_F])

# Call the functions to calculate the accuracy scores
TRAIN_CLM = getProbAccuracyScores(WHO + "_Train", CLM, U_train, Y_train[TARGET_F])
TEST_CLM = getProbAccuracyScores(WHO, CLM, U_test, Y_test[TARGET_F])

# Call the functions to print the results
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

""" AMOUNT - LINEAR REGRESSION - STEPWISE SELECTION VARIABLES """
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

""" RUN ALL OF THE MODELS AND PRINT THE RESULTS """
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
"""
UNIT 4 ASSIGNMENT WORK - NEURAL NETWORKS WITH TENSOR FLOW

### II. Tensor Flow Model to Predict Loan Default
1. Develop a model using Tensor Flow that will predict Loan Default
2. For your model, do the following
    - Try at least three different Activation Functions
    - Try one and two hidden layers
    - Try using a Dropout Layer
3. Explore using a variable selection technique
4. For each of the models
    - Calculate the accuracy of the model on both the training and test data set
    - Create a graph that shows the ROC curves for both the training and test data set. Clearly label each curve and display
      the Area Under the ROC curve.
    - Display the ROC curve for the test data with all your models on the same graph (tree based, regression, and TF). 
      Discuss which one is the most accurate. Which one would you recommend using?
"""
"""
PRE-PHASE II: PREPARE THE ENVIRONMENT FOR USING TENSORFLOW, CREATE METRICS FUNCTION, & PREPARE THE DATA
"""
# Set up the environment for using TensorFlow
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
theScaler = MinMaxScaler()
theScaler.fit(X_train)

# Define a function to get the probability and accuracy scores for the TensorFlow model
def get_TF_ProbAccuracyScores(NAME, MODEL, X, Y):
    probs = MODEL.predict(X)
    pred_list = []
    for p in probs:
        pred_list.append(np.argmax(p))
    pred = np.array(pred_list)
    acc_score = metrics.accuracy_score(Y, pred)
    p1 = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(Y, p1)
    auc = metrics.auc(fpr, tpr)
    return [NAME, acc_score, fpr, tpr, auc]

# Set up the training and test data for the classification model
WHO = "Tensor_Flow"

# Transform the training and test data
U_train = theScaler.transform(X_train)
U_test = theScaler.transform(X_test)

# Create a dataframe from the transformed data
U_train = pd.DataFrame(U_train)
U_test = pd.DataFrame(U_test)

# Set the column names for the transformed data
U_train.columns = list(X_train.columns.values)
U_test.columns = list(X_test.columns.values)

# # Define the columns to use for the classification model
# U_train = U_train[GB_flag]
# U_test = U_test[GB_flag]

# %%
"""
(1) Develop a model using Tensor Flow that will predict Loan Default (CLASSIFICATION MODEL)
(2) For your model, do the following
    - Try at least three different Activation Functions
    - Try one and two hidden layers
    - Try using a Dropout Layer
(4) For each of the models
    - Calculate the accuracy of the model on both the training and test data set
    - Create a graph that shows the ROC curves for both the training and test data set. Clearly label each curve and display
      the Area Under the ROC curve.
    - Display the ROC curve for the test data with all your models on the same graph (tree based, regression, and TF). 
      Discuss which one is the most accurate. Which one would you recommend using?
"""
""" Model #1: One Hidden Layer with ReLU Activation Function """
# Define the shape size, activation function, lose metric, optimizer, and epochs for the classification model
F_theShapeSize = U_train.shape[1]               # Shape size
F_theActivation = tf.keras.activations.relu     # ReLU Activation Function
F_theLossMetric = tf.keras.losses.SparseCategoricalCrossentropy() # Loss Metric = Sparse Categorical Crossentropy
F_theOptimizer = tf.keras.optimizers.Adam()     # Optimizer = Adam = Adaptive Moment Estimation
F_theEpochs = 100                               # Epochs = 100 iterations

# Define the number of units for the classification model
F_theUnits = int(2*F_theShapeSize / 3)          # Number of Units = 2/3 of the Shape Size

# Define the layers for the classification model
F_LAYER_01 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation, input_dim=F_theShapeSize)
F_LAYER_OUTPUT = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)

# Create the classification model
CLM = tf.keras.Sequential()
CLM.add(F_LAYER_01)
CLM.add(F_LAYER_OUTPUT)
CLM.compile(loss=F_theLossMetric, optimizer=F_theOptimizer)
CLM.fit(U_train, Y_train[TARGET_F], epochs=F_theEpochs, verbose=False)

# Get the probability and accuracy scores for the classification model
TRAIN_CLM = get_TF_ProbAccuracyScores(WHO + "_Train", CLM, U_train, Y_train[TARGET_F])
TEST_CLM = get_TF_ProbAccuracyScores(WHO, CLM, U_test, Y_test[TARGET_F])

# Print the accuracy scores for the classification model
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

""" AUC = 0.92  Accuracy = 0.913 """
# Save the results
TF_CLM_M1 = TEST_CLM.copy()

# %%
""" Model #2: One Hidden Layer with Sigmoid Activation Function """
# Define the new activation function for the classification model (Sigmoid Activation Function), everything else remains the same
F_theShapeSize = U_train.shape[1]               # Shape size
F_theActivation = tf.keras.activations.sigmoid  # Sigmoid Activation Function
F_theLossMetric = tf.keras.losses.SparseCategoricalCrossentropy() # Loss Metric = Sparse Categorical Crossentropy
F_theOptimizer = tf.keras.optimizers.Adam()     # Optimizer = Adam = Adaptive Moment Estimation
F_theEpochs = 100                               # Epochs = 100 iterations

# Define the layers for the classification model
F_LAYER_01 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation, input_dim=F_theShapeSize)
F_LAYER_OUTPUT = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)

# Create the classification model
CLM = tf.keras.Sequential()
CLM.add(F_LAYER_01)
CLM.add(F_LAYER_OUTPUT)
CLM.compile(loss=F_theLossMetric, optimizer=F_theOptimizer)
CLM.fit(U_train, Y_train[TARGET_F], epochs=F_theEpochs, verbose=False)

# Get the probability and accuracy scores for the classification model
TRAIN_CLM = get_TF_ProbAccuracyScores(WHO + "_Train", CLM, U_train, Y_train[TARGET_F])
TEST_CLM = get_TF_ProbAccuracyScores(WHO, CLM, U_test, Y_test[TARGET_F])

# Print the accuracy scores for the classification model
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

""" AUC = 0.90 | Accuracy = 0.883 """
# Save the results
TF_CLM_M2 = TEST_CLM.copy()

# %%
""" Model #3: One Hidden Layer with Tanh Activation Function """
# Define the new activation function for the classification model (Tahn Activation Function), everything else remains the same
F_theShapeSize = U_train.shape[1]               # Shape size
F_theActivation = tf.keras.activations.tanh     # Tanh Activation Function
F_theLossMetric = tf.keras.losses.SparseCategoricalCrossentropy() # Loss Metric = Sparse Categorical Crossentropy
F_theOptimizer = tf.keras.optimizers.Adam()     # Optimizer = Adam = Adaptive Moment Estimation
F_theEpochs = 100                               # Epochs = 100 iterations

# Define the layers for the classification model
F_LAYER_01 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation, input_dim=F_theShapeSize)
F_LAYER_OUTPUT = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)

# Create the classification model
CLM = tf.keras.Sequential()
CLM.add(F_LAYER_01)
CLM.add(F_LAYER_OUTPUT)
CLM.compile(loss=F_theLossMetric, optimizer=F_theOptimizer)
CLM.fit(U_train, Y_train[TARGET_F], epochs=F_theEpochs, verbose=False)

# Get the probability and accuracy scores for the classification model
TRAIN_CLM = get_TF_ProbAccuracyScores(WHO + "_Train", CLM, U_train, Y_train[TARGET_F])
TEST_CLM = get_TF_ProbAccuracyScores(WHO, CLM, U_test, Y_test[TARGET_F])

# Print the accuracy scores for the classification model
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

""" AUC = 0.91 | Accuracy = 0.898 """
# Save the results
TF_CLM_M3 = TEST_CLM.copy()

# %%
""" Model #4: Two Hidden Layers with ReLU Activation Function """
# Define the new activation function for the classification model (ReLU Activation Function), everything else remains the same
F_theShapeSize = U_train.shape[1]               # Shape size
F_theActivation = tf.keras.activations.relu     # ReLU Activation Function
F_theLossMetric = tf.keras.losses.SparseCategoricalCrossentropy() # Loss Metric = Sparse Categorical Crossentropy
F_theOptimizer = tf.keras.optimizers.Adam()     # Optimizer = Adam = Adaptive Moment Estimation
F_theEpochs = 100                               # Epochs = 100 iterations

# Define the layers for the classification model
F_LAYER_01 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation, input_dim=F_theShapeSize)
F_LAYER_02 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation)
F_LAYER_OUTPUT = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)

# Create the classification model
CLM = tf.keras.Sequential()
CLM.add(F_LAYER_01)
CLM.add(F_LAYER_02)
CLM.add(F_LAYER_OUTPUT)
CLM.compile(loss=F_theLossMetric, optimizer=F_theOptimizer)
CLM.fit(U_train, Y_train[TARGET_F], epochs=F_theEpochs, verbose=False)

# Get the probability and accuracy scores for the classification model
TRAIN_CLM = get_TF_ProbAccuracyScores(WHO + "_Train", CLM, U_train, Y_train[TARGET_F])
TEST_CLM = get_TF_ProbAccuracyScores(WHO, CLM, U_test, Y_test[TARGET_F])

# Print the accuracy scores for the classification model
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

""" AUC = 0.92 | Accuracy = 0.915 """
# Save the results
TF_CLM_M4 = TEST_CLM.copy()

# %%
""" Model #5: Using a Dropout Layer with Two Hidden Layers with ReLU Activation Function """
# Define the new activation function for the classification model (ReLU Activation Function), everything else remains the same
F_theShapeSize = U_train.shape[1]               # Shape size
F_theActivation = tf.keras.activations.relu     # ReLU Activation Function
F_theLossMetric = tf.keras.losses.SparseCategoricalCrossentropy() # Loss Metric = Sparse Categorical Crossentropy
F_theOptimizer = tf.keras.optimizers.Adam()     # Optimizer = Adam = Adaptive Moment Estimation
F_theEpochs = 100                               # Epochs = 100 iterations

# Define the layers for the classification model
F_LAYER_01 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation, input_dim=F_theShapeSize)
F_LAYER_DROP = tf.keras.layers.Dropout(0.2)
F_LAYER_02 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation)
F_LAYER_OUTPUT = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)

# Create the classification model
CLM = tf.keras.Sequential()
CLM.add(F_LAYER_01)
CLM.add(F_LAYER_DROP)
CLM.add(F_LAYER_02)
CLM.add(F_LAYER_OUTPUT)
CLM.compile(loss=F_theLossMetric, optimizer=F_theOptimizer)
CLM.fit(U_train, Y_train[TARGET_F], epochs=F_theEpochs, verbose=False)

# Get the probability and accuracy scores for the classification model
TRAIN_CLM = get_TF_ProbAccuracyScores(WHO + "_Train", CLM, U_train, Y_train[TARGET_F])
TEST_CLM = get_TF_ProbAccuracyScores(WHO, CLM, U_test, Y_test[TARGET_F])

# Print the accuracy scores for the classification model
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

""" AUC = 0.94 | Accuracy = 0.923  {Best so far} """
# Save the results
TF_CLM_M5 = TEST_CLM.copy()

# %%
"""
Bring together all of the 5 TensorFlow Neural Network models thus far and compare them
"""
# Create a list of all of the classification TensorFlow models
TensorFlow_CLM_Models = [TF_CLM_M1, TF_CLM_M2, TF_CLM_M3, TF_CLM_M4, TF_CLM_M5]

# Change the model names
for i, model in enumerate(TensorFlow_CLM_Models):
    model[0] = f"Model {i+1}"  # Change the name to "Model 1", "Model 2", etc.

# Sort the classification models by the AUC and print the ROC curve
TensorFlow_CLM_Models = sorted(TensorFlow_CLM_Models, key=lambda x: x[4], reverse=True)
print_ROC_Curve("All TensorFlow Classification Models Accuracy (All Variables)", TensorFlow_CLM_Models)

# %%
"""
(3) Explore using a variable selection technique
    The variable selections that we will use include:
        1. Decision Tree Variables
        2. Random Forest Variables
        3. Gradient Boosting Variables
(4) For each of the models
    - Calculate the accuracy of the model on both the training and test data set
    - Create a graph that shows the ROC curves for both the training and test data set. Clearly label each curve and display
      the Area Under the ROC curve.
    - Display the ROC curve for the test data with all your models on the same graph (tree based, regression, and TF). 
      Discuss which one is the most accurate. Which one would you recommend using?
"""
""" Variable Selection #1: Decision Tree Variables (2 Hidden Layers with ReLU Activation Function) """
# Define the variables to be used for the classification model
U_train_DF = U_train[vars_tree_flag]
U_test_DF = U_test[vars_tree_flag]

# Define the classification model with the Decision Tree Variables
F_theShapeSize = U_train_DF.shape[1]               # Shape size
F_theActivation = tf.keras.activations.relu     # ReLU Activation Function
F_theLossMetric = tf.keras.losses.SparseCategoricalCrossentropy() # Loss Metric = Sparse Categorical Crossentropy
F_theOptimizer = tf.keras.optimizers.Adam()     # Optimizer = Adam = Adaptive Moment Estimation
F_theEpochs = 100                               # Epochs = 100 iterations

# Define the layers for the classification model
F_LAYER_01 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation, input_dim=F_theShapeSize)
F_LAYER_DROP = tf.keras.layers.Dropout(0.2)
F_LAYER_02 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation)
F_LAYER_OUTPUT = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)

# Create the classification model
CLM = tf.keras.Sequential()
CLM.add(F_LAYER_01)
CLM.add(F_LAYER_DROP)
CLM.add(F_LAYER_02)
CLM.add(F_LAYER_OUTPUT)
CLM.compile(loss=F_theLossMetric, optimizer=F_theOptimizer)
CLM.fit(U_train_DF, Y_train[TARGET_F], epochs=F_theEpochs, verbose=False)

# Get the probability and accuracy scores for the classification model
TRAIN_CLM = get_TF_ProbAccuracyScores(WHO + "_Train", CLM, U_train_DF, Y_train[TARGET_F])
TEST_CLM = get_TF_ProbAccuracyScores(WHO, CLM, U_test_DF, Y_test[TARGET_F])

# Print the accuracy scores for the classification model
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

""" AUC = 0.90 | Accuracy = 0.883 """
# Save the results
TF_CLM_DT = TEST_CLM.copy()

# %%
""" Variable Selection #2: Random Forest Variables (2 Hidden Layers with ReLU Activation Function) """
# Define the variables to be used for the classification model
U_train_RF = U_train[RF_flag]
U_test_RF = U_test[RF_flag]

# Define the classification model with the Random Forest Variables
F_theShapeSize = U_train_RF.shape[1]               # Shape size
F_theActivation = tf.keras.activations.relu     # ReLU Activation Function
F_theLossMetric = tf.keras.losses.SparseCategoricalCrossentropy() # Loss Metric = Sparse Categorical Crossentropy
F_theOptimizer = tf.keras.optimizers.Adam()     # Optimizer = Adam = Adaptive Moment Estimation
F_theEpochs = 100                               # Epochs = 100 iterations

# Define the layers for the classification model
F_LAYER_01 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation, input_dim=F_theShapeSize)
F_LAYER_DROP = tf.keras.layers.Dropout(0.2)
F_LAYER_02 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation)
F_LAYER_OUTPUT = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)

# Create the classification model
CLM = tf.keras.Sequential()
CLM.add(F_LAYER_01)
CLM.add(F_LAYER_DROP)
CLM.add(F_LAYER_02)
CLM.add(F_LAYER_OUTPUT)
CLM.compile(loss=F_theLossMetric, optimizer=F_theOptimizer)
CLM.fit(U_train_RF, Y_train[TARGET_F], epochs=F_theEpochs, verbose=False)

# Get the probability and accuracy scores for the classification model
TRAIN_CLM = get_TF_ProbAccuracyScores(WHO + "_Train", CLM, U_train_RF, Y_train[TARGET_F])
TEST_CLM = get_TF_ProbAccuracyScores(WHO, CLM, U_test_RF, Y_test[TARGET_F])

# Print the accuracy scores for the classification model
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

""" AUC = 0.91 | Accuracy = 0.901 """
# Save the results
TF_CLM_RF = TEST_CLM.copy()

# %%
""" Variable Selection #3: Gradient Boosting Variables (2 Hidden Layers with ReLU Activation Function) """
# Define the variables to be used for the classification model
U_train_GB = U_train[GB_flag]
U_test_GB = U_test[GB_flag]

# Define the classification model with the Gradient Boosting Variables
F_theShapeSize = U_train_GB.shape[1]               # Shape size
F_theActivation = tf.keras.activations.relu     # ReLU Activation Function
F_theLossMetric = tf.keras.losses.SparseCategoricalCrossentropy() # Loss Metric = Sparse Categorical Crossentropy
F_theOptimizer = tf.keras.optimizers.Adam()     # Optimizer = Adam = Adaptive Moment Estimation
F_theEpochs = 100                               # Epochs = 100 iterations

# Define the layers for the classification model
F_LAYER_01 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation, input_dim=F_theShapeSize)
F_LAYER_DROP = tf.keras.layers.Dropout(0.2)
F_LAYER_02 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation)
F_LAYER_OUTPUT = tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)

# Create the classification model
CLM = tf.keras.Sequential()
CLM.add(F_LAYER_01)
CLM.add(F_LAYER_DROP)
CLM.add(F_LAYER_02)
CLM.add(F_LAYER_OUTPUT)
CLM.compile(loss=F_theLossMetric, optimizer=F_theOptimizer)
CLM.fit(U_train_GB, Y_train[TARGET_F], epochs=F_theEpochs, verbose=False)

# Get the probability and accuracy scores for the classification model
TRAIN_CLM = get_TF_ProbAccuracyScores(WHO + "_Train", CLM, U_train_GB, Y_train[TARGET_F])
TEST_CLM = get_TF_ProbAccuracyScores(WHO, CLM, U_test_GB, Y_test[TARGET_F])

# Print the accuracy scores for the classification model
print_ROC_Curve(WHO, [TRAIN_CLM, TEST_CLM])
print_Accuracy(WHO + " CLASSIFICATION ACCURACY", [TRAIN_CLM, TEST_CLM])

""" AUC = 0.91 | Accuracy = 0.899 """
# Save the results
TF_CLM_GB = TEST_CLM.copy()

# %%
"""
Bring together the TensorFlow Neural Network models with the Decision Tree, Random Forest, and Gradient Boosting variables. Plus 
the original Model #5 (Two Hidden Layers with ReLU Activation Function and Dropout Layer)
"""
# Create a list of all of the classification TensorFlow variable selection models
TensorFlow_CLM_VSMs = [TF_CLM_DT, TF_CLM_RF, TF_CLM_GB, TF_CLM_M5]

# Change the model names to reflect the variable selection technique (i.e., Decision Tree, Random Forest, Gradient Boosting, All Variables)
TensorFlow_CLM_VSMs[0][0] = "Decision Tree Variables"
TensorFlow_CLM_VSMs[1][0] = "Random Forest Variables"
TensorFlow_CLM_VSMs[2][0] = "Gradient Boosting Variables"
TensorFlow_CLM_VSMs[3][0] = "All Variables"

# Sort the classification models by the AUC and print the ROC curve
TensorFlow_CLM_VSMs = sorted(TensorFlow_CLM_VSMs, key=lambda x: x[4], reverse=True)
print_ROC_Curve("TensorFlow Classification Models Accuracy (Variable Selection)", TensorFlow_CLM_VSMs)

# %%
"""
Phase III. Tensor Flow Model to Predict Loss Given Default
1. Develop a model using Tensor Flow that will predict Loan Default.
2. For your model, do the following:
    - Try at least three different Activation Functions
    - Try one and two hidden layers
    - Try using a Dropout Layer
3. Explore using a variable selection technique
4. For each of the models:
    - Calculate the RMSE for both the training data set and the test data set
    - List the RMSE for the test data set for all the models created (tree based, regression, and TF). Discuss which one is 
      the most accurate. Which one would you recommend using?
"""
"""
Pre-Phase III: Prepare the Data for the Regression Tensor Flow Model to Predict Loss Given Default
"""
# Transform the training and test data
V_train = theScaler.transform(W_train)
V_test = theScaler.transform(W_test)

# Create a DataFrame from the transformed data
V_train = pd.DataFrame(V_train)
V_test = pd.DataFrame(V_test)

# Set the column names for the transformed data
V_train.columns = list(W_train.columns.values)
V_test.columns = list(W_test.columns.values)

# %%
"""
(1) Develop a model using Tensor Flow that will predict Loan Default (REGRESSION MODEL)
(2) For your model, do the following
    - Try at least three different Activation Functions
    - Try one and two hidden layers
    - Try using a Dropout Layer
(4) For each of the models
    - Calculate the RMSE for both the training data set and the test data set
    - List the RMSE for the test data set for all the models created (tree based, regression, and TF). Discuss which one is 
      the most accurate. Which one would you recommend using?
"""
""" Model #1: One Hidden Layer with ReLU Activation Function """
# Define the shape size, activation function, lose metric, optimizer, and epochs for the amount lost regression model
A_theShapeSize = V_train.shape[1]               # Shape size
A_theActivation = tf.keras.activations.relu     # ReLU Activation Function
A_theLossMetric = tf.keras.losses.MeanSquaredError() # Loss Metric = Mean Squared Error
A_theOptimizer = tf.keras.optimizers.Adam() # Optimizer = Adam = Adaptive Moment Estimation
A_theEpochs = 100                               # Epochs = 100 iterations

# Define the number of units for the regression model
A_theUnits = int(2*A_theShapeSize)

# Define the layers for the regression model
A_LAYER_01 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation, input_dim=A_theShapeSize)
A_LAYER_OUTPUT = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear)

# Create the amount lost regression model
AMT = tf.keras.Sequential()
AMT.add(A_LAYER_01)
AMT.add(A_LAYER_OUTPUT)
AMT.compile(loss=A_theLossMetric, optimizer=A_theOptimizer)
AMT.fit(V_train, Z_train[TARGET_A], epochs=A_theEpochs, verbose=False)

# Get the probability and accuracy scores for the regression model
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, V_train, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, V_test, Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Save the results
TF_AMT_M1 = TEST_AMT.copy()

""" Train RMSE = 11324.9 | Test RMSE = 10721.3 """

# %%
""" Model #2: One Hidden Layer with Sigmoid Activation Function """
# Define the shape size, activation function, lose metric, optimizer, and epochs for the amount lost regression model
A_theShapeSize = V_train.shape[1]               # Shape size
A_theActivation = tf.keras.activations.sigmoid  # Sigmoid Activation Function
A_theLossMetric = tf.keras.losses.MeanSquaredError() # Loss Metric = Mean Squared Error
A_theOptimizer = tf.keras.optimizers.Adam() # Optimizer = Adam = Adaptive Moment Estimation
A_theEpochs = 100                               # Epochs = 100 iterations

# Define the number of units for the regression model
A_theUnits = int(2*A_theShapeSize)

# Define the layers for the regression model
A_LAYER_01 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation, input_dim=A_theShapeSize)
A_LAYER_OUTPUT = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear)

# Create the amount lost regression model
AMT = tf.keras.Sequential()
AMT.add(A_LAYER_01)
AMT.add(A_LAYER_OUTPUT)
AMT.compile(loss=A_theLossMetric, optimizer=A_theOptimizer)
AMT.fit(V_train, Z_train[TARGET_A], epochs=A_theEpochs, verbose=False)

# Get the probability and accuracy scores for the regression model
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, V_train, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, V_test, Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Save the results 
TF_AMT_M2 = TEST_AMT.copy()

""" Train RMSE = 16976.5 | Test RMSE = 16080.2 """

# %%
""" Model #3: One Hidden Layer with Tanh Activation Function """
# Define the shape size, activation function, lose metric, optimizer, and epochs for the amount lost regression model
A_theShapeSize = V_train.shape[1]               # Shape size
A_theActivation = tf.keras.activations.tanh     # Tanh Activation Function
A_theLossMetric = tf.keras.losses.MeanSquaredError() # Loss Metric = Mean Squared Error
A_theOptimizer = tf.keras.optimizers.Adam() # Optimizer = Adam = Adaptive Moment Estimation
A_theEpochs = 100                               # Epochs = 100 iterations

# Define the number of units for the regression model
A_theUnits = int(2*A_theShapeSize)

# Define the layers for the regression model
A_LAYER_01 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation, input_dim=A_theShapeSize)
A_LAYER_OUTPUT = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear)

# Create the amount lost regression model
AMT = tf.keras.Sequential()
AMT.add(A_LAYER_01)
AMT.add(A_LAYER_OUTPUT)
AMT.compile(loss=A_theLossMetric, optimizer=A_theOptimizer)
AMT.fit(V_train, Z_train[TARGET_A], epochs=A_theEpochs, verbose=False)

# Get the probability and accuracy scores for the regression model
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, V_train, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, V_test, Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Save the results 
TF_AMT_M3 = TEST_AMT.copy()

""" Train RMSE = 16970.6 | Test RMSE = 16074.5 """

# %%
""" Model #4: Two Hidden Layers with ReLU Activation Function """
# Define the shape size, activation function, lose metric, optimizer, and epochs for the amount lost regression model
A_theShapeSize = V_train.shape[1]               # Shape size
A_theActivation = tf.keras.activations.relu     # ReLU Activation Function
A_theLossMetric = tf.keras.losses.MeanSquaredError() # Loss Metric = Mean Squared Error
A_theOptimizer = tf.keras.optimizers.Adam() # Optimizer = Adam = Adaptive Moment Estimation
A_theEpochs = 100                               # Epochs = 100 iterations

# Define the number of units for the regression model
A_theUnits = int(2*A_theShapeSize)

# Define the layers for the regression model
A_LAYER_01 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation, input_dim=A_theShapeSize)
A_LAYER_02 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation)
A_LAYER_OUTPUT = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear)

# Create the amount lost regression model
AMT = tf.keras.Sequential()
AMT.add(A_LAYER_01)
AMT.add(A_LAYER_02)
AMT.add(A_LAYER_OUTPUT)
AMT.compile(loss=A_theLossMetric, optimizer=A_theOptimizer)
AMT.fit(V_train, Z_train[TARGET_A], epochs=A_theEpochs, verbose=False)

# Get the probability and accuracy scores for the regression model
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, V_train, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, V_test, Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Save the results 
TF_AMT_M4 = TEST_AMT.copy()

""" Train RMSE = 4910.7 | Test RMSE = 6285.4 """

# %%
""" Model #5: Using a Dropout Layer with Two Hidden Layers with ReLU Activation Function """
# Define the shape size, activation function, lose metric, optimizer, and epochs for the amount lost regression model
A_theShapeSize = V_train.shape[1]               # Shape size
A_theActivation = tf.keras.activations.relu     # ReLU Activation Function
A_theLossMetric = tf.keras.losses.MeanSquaredError() # Loss Metric = Mean Squared Error
A_theOptimizer = tf.keras.optimizers.Adam() # Optimizer = Adam = Adaptive Moment Estimation
A_theEpochs = 100                               # Epochs = 100 iterations

# Define the number of units for the regression model
A_theUnits = int(2*A_theShapeSize)

# Define the layers for the regression model
A_LAYER_01 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation, input_dim=A_theShapeSize)
A_LAYER_DROP = tf.keras.layers.Dropout(0.2)
A_LAYER_02 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation)
A_LAYER_OUTPUT = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear)

# Create the amount lost regression model
AMT = tf.keras.Sequential()
AMT.add(A_LAYER_01)
AMT.add(A_LAYER_DROP)
AMT.add(A_LAYER_02)
AMT.add(A_LAYER_OUTPUT)
AMT.compile(loss=A_theLossMetric, optimizer=A_theOptimizer)
AMT.fit(V_train, Z_train[TARGET_A], epochs=A_theEpochs, verbose=False)

# Get the probability and accuracy scores for the regression model
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, V_train, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, V_test, Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Save the results 
TF_AMT_M5 = TEST_AMT.copy()

""" Train RMSE = 4935.0 | Test RMSE = 6283.9 {Best Regression Model so Far - Slightly better with the drop out layer} """

# %%
"""
Bring together all of the 5 TensorFlow Neural Network Regression models thus far and compare them
"""
# Create a list of all of the regression TensorFlow models
TensorFlow_AMT_Models = [TF_AMT_M1, TF_AMT_M2, TF_AMT_M3, TF_AMT_M4, TF_AMT_M5]

# Change the model names
for i, model in enumerate(TensorFlow_AMT_Models):
    model[0] = f"Model {i+1}" # Change the name to "Model 1", "Model 2", etc.
    
# Sort the regression models by the RMSE and print the RMSE scores
TensorFlow_AMT_Models = sorted(TensorFlow_AMT_Models, key=lambda x: x[1])
print_Accuracy("All TensorFlow Regression Models RMSE (All Variables)", TensorFlow_AMT_Models)

# %%
"""
(3) Explore using a variable selection technique
    The variable selections that we will use include:
        1. Decision Tree Variables
        2. Random Forest Variables
        3. Gradient Boosting Variables
(4) For each of the models
    - Calculate the RMSE for both the training data set and the test data set
    - List the RMSE for the test data set for all the models created (tree based, regression, and TF). Discuss which one is 
      the most accurate. Which one would you recommend using?
"""
""" Variable Selection #1: Decision Tree Variables (Two Hidden Layers with ReLU Activation Function) """
# Define the variables to be used for the regression model
V_train_DT = V_train[vars_tree_flag]
V_test_DT = V_test[vars_tree_flag]

# Define the regression model with the Decision Tree Variables
A_theShapeSize = V_train_DT.shape[1]               # Shape size
A_theActivation = tf.keras.activations.relu     # ReLU Activation Function
A_theLossMetric = tf.keras.losses.MeanSquaredError() # Loss Metric = Mean Squared Error
A_theOptimizer = tf.keras.optimizers.Adam() # Optimizer = Adam = Adaptive Moment Estimation
A_theEpochs = 100                               # Epochs = 100 iterations

# Define the number of units for the regression model
A_theUnits = int(2*A_theShapeSize)

# Define the layers for the regression model
A_LAYER_01 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation, input_dim=A_theShapeSize)
A_LAYER_02 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation)
A_LAYER_OUTPUT = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear)

# Create the amount lost regression model
AMT = tf.keras.Sequential()
AMT.add(A_LAYER_01)
AMT.add(A_LAYER_02)
AMT.add(A_LAYER_OUTPUT)
AMT.compile(loss=A_theLossMetric, optimizer=A_theOptimizer)
AMT.fit(V_train_DT, Z_train[TARGET_A], epochs=A_theEpochs, verbose=False)

# Get the probability and accuracy scores for the regression model
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, V_train_DT, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, V_test_DT, Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Save the results 
TF_AMT_DT = TEST_AMT.copy()

""" Train RMSE = 10291.0 | Test RMSE = 10091.9 """

# %%
""" Variable Selection #2: Random Forest Variables (Two Hidden Layers with ReLU Activation Function) """
# Define the variables to be used for the regression model
V_train_RF = V_train[RF_flag]
V_test_RF = V_test[RF_flag]

# Define the regression model with the Random Forest Variables
A_theShapeSize = V_train_RF.shape[1]               # Shape size
A_theActivation = tf.keras.activations.relu     # ReLU Activation Function
A_theLossMetric = tf.keras.losses.MeanSquaredError() # Loss Metric = Mean Squared Error
A_theOptimizer = tf.keras.optimizers.Adam() # Optimizer = Adam = Adaptive Moment Estimation
A_theEpochs = 100                               # Epochs = 100 iterations

# Define the number of units for the regression model
A_theUnits = int(2*A_theShapeSize)

# Define the layers for the regression model
A_LAYER_01 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation, input_dim=A_theShapeSize)
A_LAYER_02 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation)
A_LAYER_OUTPUT = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear)

# Create the amount lost regression model
AMT = tf.keras.Sequential()
AMT.add(A_LAYER_01)
AMT.add(A_LAYER_02)
AMT.add(A_LAYER_OUTPUT)
AMT.compile(loss=A_theLossMetric, optimizer=A_theOptimizer)
AMT.fit(V_train_RF, Z_train[TARGET_A], epochs=A_theEpochs, verbose=False)

# Get the probability and accuracy scores for the regression model
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, V_train_RF, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, V_test_RF, Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Save the results 
TF_AMT_RF = TEST_AMT.copy()

""" Train RMSE = 8368.3 | Test RMSE = 8386.0 """

# %%
""" Variable Selection #3: Gradient Boosting Variables (Two Hidden Layers with ReLU Activation Function) """
# Define the variables to be used for the regression model
V_train_GB = V_train[GB_flag]
V_test_GB = V_test[GB_flag]

# Define the regression model with the Gradient Boosting Variables
A_theShapeSize = V_train_GB.shape[1]               # Shape size
A_theActivation = tf.keras.activations.relu     # ReLU Activation Function
A_theLossMetric = tf.keras.losses.MeanSquaredError() # Loss Metric = Mean Squared Error
A_theOptimizer = tf.keras.optimizers.Adam() # Optimizer = Adam = Adaptive Moment Estimation
A_theEpochs = 100                               # Epochs = 100 iterations

# Define the number of units for the regression model
A_theUnits = int(2*A_theShapeSize)

# Define the layers for the regression model
A_LAYER_01 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation, input_dim=A_theShapeSize)
A_LAYER_02 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation)
A_LAYER_OUTPUT = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear)

# Create the amount lost regression model
AMT = tf.keras.Sequential()
AMT.add(A_LAYER_01)
AMT.add(A_LAYER_02)
AMT.add(A_LAYER_OUTPUT)
AMT.compile(loss=A_theLossMetric, optimizer=A_theOptimizer)
AMT.fit(V_train_GB, Z_train[TARGET_A], epochs=A_theEpochs, verbose=False)

# Get the probability and accuracy scores for the regression model
TRAIN_AMT = getAmtAccuracyScores(WHO + "_Train", AMT, V_train_GB, Z_train[TARGET_A])
TEST_AMT = getAmtAccuracyScores(WHO, AMT, V_test_GB, Z_test[TARGET_A])
print_Accuracy(WHO + " RMSE ACCURACY", [TRAIN_AMT, TEST_AMT])

# Save the results 
TF_AMT_GB = TEST_AMT.copy()

""" Train RMSE = 8681.1 | Test RMSE = 8655.1 """

# %%
"""
Bring together the TensorFlow Neural Network Regression models with the Decision Tree, Random Forest, and Gradient Boosting variables. 
Plus the original Model #5 (Two Hidden Layers with ReLU Activation Function and Dropout Layer)
"""
# Create a list of all the regression TensorFlow variable selection models
TensorFlow_AMT_VSMs = [TF_AMT_DT, TF_AMT_RF, TF_AMT_GB, TF_AMT_M4]

# Change the model names to reflect the variable selection technique (i.e., Decision Tree, Random Forest, Gradient Boosting, 
# All Variables)
TensorFlow_AMT_VSMs[0][0] = "Decision Tree Variables"
TensorFlow_AMT_VSMs[1][0] = "Random Forest Variables"
TensorFlow_AMT_VSMs[2][0] = "Gradient Boosting Variables"
TensorFlow_AMT_VSMs[3][0] = "All Variables"

# Sort the regression models by the RMSE and print the RMSE scores
TensorFlow_AMT_VSMs = sorted(TensorFlow_AMT_VSMs, key=lambda x: x[1])
print_Accuracy("TensorFlow Regression Models RMSE (Variable Selection)", TensorFlow_AMT_VSMs)

""" The best model not using all of the variables is the Random Forest Variables model with a Test RMSE of 8386.0 """

# %%
"""
BRING IT ALL TOGETHER - ALL MODELS
    1. Compare all the previous classification models to the best TensorFlow classification model (Model #5 and Model #5 with Gradient
       Boosting Variables)
    2. Compare all the previous regression models to the best TensorFlow regression model (Model #5 and Model #5 with Random Forest) 
"""
""" (1) Compare all the previous classification models to the best TensorFlow classification model """
# Create a list of all the classification models
All_CLM_Models = [TREE_CLM, RF_CLM, GB_CLM, REG_ALL_CLM, REG_TREE_CLM, REG_RF_CLM, REG_GB_CLM, TF_CLM_M5, TF_CLM_GB]

# Change the model names
All_CLM_Models[0][0] = "Decision Tree"
All_CLM_Models[1][0] = "Random Forest"
All_CLM_Models[2][0] = "Gradient Boosting"
All_CLM_Models[3][0] = "Regression - All Variables"
All_CLM_Models[4][0] = "Regression - Decision Tree Variables"
All_CLM_Models[5][0] = "Regression - Random Forest Variables"
All_CLM_Models[6][0] = "Regression - Gradient Boosting Variables"
All_CLM_Models[7][0] = "TensorFlow - Model #5 (All Variables)"
All_CLM_Models[8][0] = "TensorFlow - Model #5 (Gradient Boosting Variables)"

# Sort the classification models by the AUC and print the ROC curve
All_CLM_Models = sorted(All_CLM_Models, key=lambda x: x[4], reverse=True)
print_ROC_Curve("All Classification Models Accuracy", All_CLM_Models)

# %%
""" (2) Compare all the previous regression models to the best TensorFlow regression model """
# Create a list of all the regression models
All_AMT_Models = [TREE_AMT, RF_AMT, GB_AMT, REG_ALL_AMT, REG_TREE_AMT, REG_RF_AMT, REG_GB_AMT, TF_AMT_M4, TF_AMT_RF]

# Change the model names
All_AMT_Models[0][0] = "Decision Tree"
All_AMT_Models[1][0] = "Random Forest"
All_AMT_Models[2][0] = "Gradient Boosting"
All_AMT_Models[3][0] = "Regression - All Variables"
All_AMT_Models[4][0] = "Regression - Decision Tree Variables"
All_AMT_Models[5][0] = "Regression - Random Forest Variables"
All_AMT_Models[6][0] = "Regression - Gradient Boosting Variables"
All_AMT_Models[7][0] = "TensorFlow - Model #4 (All Variables)"
All_AMT_Models[8][0] = "TensorFlow - Model #4 (Random Forest Variables)"

# Sort the regression models by the RMSE and print the RMSE scores
All_AMT_Models = sorted(All_AMT_Models, key=lambda x: x[1])
print_Accuracy("All Regression Models RMSE", All_AMT_Models)

# %%
