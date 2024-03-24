"""
Assignment 1: Data Preparation

In this assignment, we will explore the data and begin to prepare the data set so that it can be used 
in predictive models.

### Assignment Requirements
1. Download the HMEQ Data Set
2. Read the data into Python
3. Explore both the inputs and target variables using statistical techniques.
4. Explore both the inputs and target variables using graphs and other visualizations.
5. Look for relationships between the input variables and the targets.
6. Fix (imput) all missing data.
    - Note: For numerical data, create a glaf variable to indicate if the value was missing
7. Convert all categorical variables into numerical variables (using one hot encoding)
"""

#%%
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
sns.set()

# Setting display options
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
