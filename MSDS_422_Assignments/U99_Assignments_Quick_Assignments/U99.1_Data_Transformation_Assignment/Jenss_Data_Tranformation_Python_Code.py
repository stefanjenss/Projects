"""
# Assignment 99: Data Transformation

In this assignment, we will explore the data and begin to prepare the data set so that it can be used in predictive models.

### Assignment Requirements
#### I. Assignment Set-Up
1. Download the HMEQ Data set and read the data into Python
2. Create a new Data Frame that only has numeric input variables. In other words, remove the Target variables and the categorical variables.
#### II. Normalize the Data
1. Use MinMaxScaler to Normalize the numeric data.
2. Save the results into a DataFrame
3. Name the columns of the DataFrame using the prefix "nor_"
4. Append the Targets Variables and the Categorical Variables
5. Select one numeric variable and normalize it without using MinMaxScaler. Show that the results are similar to MinMaxScaler.
#### III. Standardize the Data
1. Use StandardScaler to standardize the numeric data.
2. Save the results into a DataFrame
3. Name the columns of the DataFrame using the prefix "std_"
4. Append the Targets Variables and the Categorical Variables
5. Select one numeric variable and normalize it without using StandardScaler. Show that the results are similar to StandardScaler.
"""

# %%
"""
PHASE I: ASSIGNMENT SET-UP
    1. Download the HMEQ Data set and read the data into Python
    2. Create a new Data Frame that only has numeric input variables. In other words, remove the Target variables and the categorical 
    variables.
"""
# Import the required libraries
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler  # This puts the data on a scale of 0 to 1
from sklearn.preprocessing import StandardScaler 

# Set the display settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

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
Create a new Data Frame that only has numeric input variables. In other words, remove the Target variables and the categorical 
variables.
"""
# Create lists of the numeric and categorical variables
objList = []
numList = []
for i in dt.index:
    if i in [TARGET_F, TARGET_A]:
        continue
    if dt[i] in ['object']:
        objList.append(i)
    if dt[i] in ['int64', 'float64']:
        numList.append(i)

# Create a new data frame with only the numeric variables
X = df[numList].copy()                  # Make a copy of the data
varNames = X.columns                    # Get the variable names

# Print the first few rows of the data frame
print(X.head())
print(X.describe())
print("\n\n")

# %%
"""
PHASE II: NORMALIZE THE DATA
    1. Use MinMaxScaler to Normalize the numeric data.
    2. Save the results into a DataFrame
    3. Name the columns of the DataFrame using the prefix "nor_"
    4. Append the Targets Variables and the Categorical Variables
    5. Select one numeric variable and normalize it without using MinMaxScaler. Show that the results are similar to MinMaxScaler
"""
"""
Use MinMaxScaler to Normalize the numeric data and save the results into a DataFrame
"""
print("NORMALIZING THE DATA \n\n\n")
theScaler = MinMaxScaler()              # Create and define the scaler
theScaler.fit(X)                        # Fit the scaler to the data

# Transform the data
X_MINMAX = theScaler.transform(X)       # Transform the data
X_MINMAX = pd.DataFrame(X_MINMAX)       # Convert the data to a dataframe
# Print the first few rows of the data frame
print(X_MINMAX.head())
print("\n\n")

# %%
""""
Name the columns of the DataFrame using the prefix "nor_"
"""
# Create a new list of variable names
varNames_minmax = []
for i in varNames:
    newName = "nor_" + i
    varNames_minmax.append(newName)
print(varNames_minmax)
print("\n\n")

# Rename the columns and check the results with .head() and .describe()
X_MINMAX.columns = varNames_minmax
print(X_MINMAX.head())
print("\n\n")

print(X_MINMAX.describe())
print("\n\n")

# %%
"""
Append the Targets Variables and the Categorical Variables Back to the DataFrame
"""
# Add back the target variables
X_MINMAX[TARGET_F] = df[TARGET_F]
X_MINMAX[TARGET_A] = df[TARGET_A]

# Add back the categorical variables
X_MINMAX[objList] = df[objList]

# Concatenate the original data with the normalized data
X_NEW = pd.concat([X, X_MINMAX], axis=1)

# Print the first few rows of the data frame to check the results
print(X_NEW.head())
print("\n\n")

# %%
"""
Calculate the MinMaxScaler for `nor_LOAN` and compare the results
"""
# Create a test data frame with only the variable `LOAN` and its normalized value to check the scaler
X_LOAN = X_NEW[["LOAN", "nor_LOAN"]].copy()
print(X_LOAN.head())
print("\n\n")
print(X_LOAN['LOAN'].describe())
print("\n\n")

# Calculate the normalized value for `LOAN` without using MinMaxScaler
LOAN_TEST = (X_LOAN['LOAN'] - X_LOAN['LOAN'].min()) / (X_LOAN['LOAN'].max() - X_LOAN['LOAN'].min())
X_LOAN = X_LOAN.assign(calc_Loan = LOAN_TEST.values)
print(X_LOAN.head())
print("\n\n")

# %%
"""
PHASE III: STANDARDIZE THE DATA
    1. Use StandardScaler to standardize the numeric data.
    2. Save the results into a DataFrame
    3. Name the columns of the DataFrame using the prefix "std_"
    4. Append the Targets Variables and the Categorical Variables
    5. Select one numeric variable and normalize it without using StandardScaler. Show that the results are similar to StandardScaler
"""
"""
Use StandardScaler to standardize the numeric data and save the results into a DataFrame
"""
print(" STANDARDIZING THE DATA \n\n\n")

# Create and define the scaler
theScaler = StandardScaler()
theScaler.fit(X)

# Transform the data
Y_STD = theScaler.transform(X)
Y_STD = pd.DataFrame(Y_STD)

# Print the first few rows of the data frame
print(Y_STD.head())
print("\n\n")

# %%
"""
Name the columns of the DataFrame using the prefix "std_"
"""
varNames_std = []
for i in varNames:                 # Create a new list of variable names
    newName = "std_" + i
    varNames_std.append(newName)

# Rename the columns and check the results with .head() and .describe()    
Y_STD.columns = varNames_std
print(Y_STD.head())
print("\n\n")
print(Y_STD.describe())
print("\n\n")

# %%
"""
Append the Targets Variables and the Categorical Variables Back to the DataFrame
"""
# Add back the target variables
Y_STD[TARGET_F] = df[TARGET_F]
Y_STD[TARGET_A] = df[TARGET_A]

# Add back the categorical variables
Y_STD[objList] = df[objList]

# Concatenate the original data with the standardized data
Y_NEW = pd.concat([X, Y_STD], axis=1)
print(Y_NEW.head())
print("\n\n")

# %%
"""
Manually calculare the StandardScaler for `std_LOAN` and compare the results
"""
# Create a test dataframe to check the scaler
Y_LOAN = Y_NEW[["LOAN", "std_LOAN"]].copy()
print(Y_LOAN.head())
print("\n\n")
print(Y_LOAN['LOAN'].describe())
print("\n\n")

# Calculate the standardized value for `LOAN` without using StandardScaler
LOAN_TEST = (Y_LOAN['LOAN'] - Y_LOAN['LOAN'].mean()) / Y_LOAN['LOAN'].std()
Y_LOAN = Y_LOAN.assign(calc_Loan = LOAN_TEST.values)
print(Y_LOAN.head())
print("\n\n")
# %%
