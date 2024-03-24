"""
# Assignment 99.2: Principal Component Analaysis

### Assignment Requirements
#### I. Assignment Set-Up
1. Download the HMEQ Data set
2. Read the data into Python
3. **Impute any missing numeric values. You may use a simple approach such as the mean or median.**
4. Create a new Data Frame that only has numeric input variables. In other words, remove the Target variables and the categorical variables.

#### II. Standardize the Data
1. Use StandardScaler to standardize the numeric data.

#### III. Principal Component Analysis
1. Calculate the Principal Components for your numeric data.
2. Use a Scree Plot (or some other approach) to determine the appropriate number of Principal Components. **Note that you will need at LEAST 2 Principal Components to complete this assignment. So choose a value of at least 2.**
3. Convert your Principal Component data set into a Data Frame
4. Reduce the size of your Data Frame to the number of PC's that you decided upon
Name the columns PC_1, PC_2, etc.
5. Append the Targets Variables and the Categorical Variables

#### IV. Interpret the Principal Component
1. Select at least one of the principal components, print it's weight values
2. Attempt to interpret the coefficients. In other words, does it tell a story?

#### V. Plot the Principal Components
1. Create a scatter plot with PC_1 as the X-Axis and PC_2 as the Y_Axis.
2. Using one of the categorical variables (or the Target Flag), color the dots in the scatter plot.
3. Note whether or not the Principal Components predict the categorical variables.
4. If you selected more than 2 PC's, feel free to experiment by using different PC's and determining if they predict the target.
"""

# %%
"""
PHASE I: ASSIGNMENT SET-UP
    1. Download the HMEQ Data set
    2. Read the data into Python
    3. Impute any missing numeric values. You may use a simple approach such as the mean or median.
    4. Create a new Data Frame that only has numeric input variables. In other words, remove the Target variables and the categorical 
    variables.
"""
# Import the necessary libraries
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# Set the display settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

""" 1 & 2) Load Data and Define Target Variables """
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
3) Impute any missing numeric values. You may use a simple approach such as the mean or median. 
    - I have chosen to use the imputation method that I have been using in the past assignments for mental consistency.
"""
# Firstly, create a list of object and numeric variables
objList = []
numList = []
for i in dt.index:
    #print("here is i .....", i, "..... and here is the type", dt[i])
    if i in ([TARGET_F, TARGET_A]): continue
    if dt[i] in (["object"]): objList.append(i)
    if dt[i] in (["int64", "float64"]): numList.append(i)

"""
FILLING IN MISSING DATA.

IMPUTATION METHOD: FILL IN MISSING WITH THE CATEGORY "MISSING"
"""
for i in objList:
    if df[i].isna().sum() == 0 : continue           # skip if there are no missing values
    # FLAG = "M_" + i                                 # create new flag variable name **NEW
    NAME = "IMP_" + i                               # create a new variable name 
    # df[FLAG] = df[i].isna() + 0                     # populate the new flag variable **NEW
    df[NAME] = df[i].fillna("MISSING")              # fill in the missing values with the category "MISSING"
    g = df.groupby(NAME)                            # group by the new variable
    df = df.drop(i, axis = 1)                       # drop the old variable

# Get the data types to include the new variables
dt = df.dtypes
objList = []
for i in dt.index:
    if i in ([TARGET_F, TARGET_A]): continue
    if dt[i] in (["object"]): objList.append(i)

"""
MISSING VALUE IMPUTATION FOR NUMERIC VARIABLES

IMPUTATION METHOD 1: PERFORM MISSING VALUE IMPUTATION FOR "VALUE" BASED ON "JOB" CLASS
"""
i = "VALUE"
# FLAG = "M_" + i
IMP = "IMP_" + i

# df[FLAG] = df[i].isna() + 0
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
# FLAG = "M_" + i
IMP = "IMP_" + i

# df[FLAG] = df[i].isna() + 0
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
# FLAG = "M_" + i
IMP = "IMP_" + i

# df[FLAG] = df[i].isna() + 0
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
    # FLAG = "M_" + i
    IMP = "IMP_" + i
    # df[FLAG] = (df[i].isna() + 0)                         # create a flag variable
    df[IMP] = df[i]                                       # create a new variable
    df.loc[df[IMP].isna(), IMP] = df[i].median()          # fill in the missing values with the median
    df = df.drop(i, axis=1)                               # drop the old variable
    
# Check to see if there are any missing values left
df.info()

# %%
"""
4) Create a new Data Frame that only has numeric input variables. In other words, remove the Target variables and the categorical
variables.
"""
# Define the data types of the variables
dt = df.dtypes

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
PHASE II: STANDARDIZE THE DATA
    1. Use StandardScaler to standardize the numeric data.
"""
# Create a standard scaler and fit it to the data
theScaler = StandardScaler()
theScaler.fit(X)

# Transform the data
X_STD = theScaler.transform(X)

# %%
"""
PHASE III: PRINCIPAL COMPONENT ANALYSIS
    1. Calculate the Principal Components for your numeric data.
    2. Use a Scree Plot (or some other approach) to determine the appropriate number of Principal Components. **Note that you will need at LEAST 2 Principal Components to complete this assignment. So choose a value of at least 2.**
    3. Convert your Principal Component data set into a Data Frame
    4. Reduce the size of your Data Frame to the number of PC's that you decided upon
"""

""" 1) Create a principal component analysis (PCA) model and fit it to the data """
MAX_N = X_STD.shape[1]          # Get the number of columns
pca = PCA(n_components=MAX_N)   # Create a principal component analysis model
pca.fit(X_STD)                  # Fit the model to the data

# %%
""" 
2) Use a Scree Plot (or some other approach) to determine the appropriate number of Principal Components. 
        - **Note that you will need at LEAST 2 Principal Components to complete this assignment. So choose a value of at least 2.** 
"""
# Evaluate the eigenvalues, which are the variance of the data
ev = pca.explained_variance_
print("Eigen Values")
print(ev)
print("\n\n")

# %%
# Summarize the variance and the total variance
varPCT = []
totPCT = []
total = 0
for i in ev:
    total = total + i
    VAR = int( i / len(ev) * 100)
    PCT = int( total / len(ev) * 100)
    varPCT.append(VAR)
    totPCT.append( PCT )
    print( round(i,2), "variation=", VAR,"%"," total=", PCT,"%")
    
# %%
# Plot the eigenvalues using a scree plot
PC_NUM = np.arange( MAX_N ) + 1
plt.plot( PC_NUM , ev, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.show()

# %%
# Plot the variance explained
PC_NUM = np.arange( MAX_N ) + 1
plt.plot( PC_NUM , varPCT, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

# %%
# Plot the total variance explained
PC_NUM = np.arange( MAX_N ) + 1
plt.plot( PC_NUM , totPCT, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Total Variance Explained')
plt.show()

"""
Principal Component Selection:
    - Based on the scree plot, I will choose 4 principal components for the assignment.
    - This is because the eigenvalues and variance explained by the principal components levels off after the 4th principal component.    
    - Additionally, the eigenvalues of the first 4 principal components are greater than 1.0, which is a common rule of thumb for
    selecting principal components.
"""

# %%
"""
3) Convert your Principal Component data set into a Data Frame
"""
# Create a data frame of the principal components
dfc = pd.DataFrame(pca.components_)
dfc.columns = list(X.columns)
print(dfc)

# %%
"""
4) Reduce the size of your Data Frame to the number of PC's that you decided upon
"""
# Create a PCA object with 4 principal components and fit it to the data
pca = PCA(n_components=4)
pca.fit(X_STD)

# Transform the PCA data to the stardardized data and convert it to a data frame
X_PCA = pca.transform(X_STD)
X_PCA = pd.DataFrame(X_PCA)
X_PCA = X_PCA.iloc[:, 0:4]      # Reduce the data frame to the first 4 principal components

""" Name the columns of the Principal Components """
# Define the names of the Principal Components
colNames = X_PCA.columns
pcaNames = []
for i in colNames:
    index = int(i) + 1              # Convert the index to an integer starting at 1
    pcaName = "PC_" + str(index)    # Create the name of the principal component
    pcaNames.append(pcaName)        # Append the name to the list

# Rename the columns of the data frame
X_PCA.columns = pcaNames

# View the results of the data frame
print(X_PCA.head())
print("\n\n")

print(df.head())
print("\n\n")

# %%
"""
PHASE IV: INTERPRET THE PRINCIPAL COMPONENT
    1. Select at least one of the principal components, print it's weight values
    2. Attempt to interpret the coefficients. In other words, does it tell a story?
"""

"""
1) Select at least one of the principal components, print it's weight values
"""
# Print the weight values of the first principal component
print("Results (Principal Component 1):")
print(dfc.iloc[0])
print("\n\n")

"""
Results (Principal Component 1):
IMP_VALUE      0.584874
IMP_LOAN       0.312647
IMP_DEBTINC    0.175836
IMP_MORTDUE    0.573248
IMP_YOJ        0.035622
IMP_DEROG     -0.025728
IMP_DELINQ     0.052885
IMP_CLAGE      0.233939
IMP_NINQ       0.045930
IMP_CLNO       0.372932
Name: 0, dtype: float64
"""

# %%
"""
#### V. Plot the Principal Components
1. Create a scatter plot with PC_1 as the X-Axis and PC_2 as the Y_Axis.
2. Using one of the categorical variables (or the Target Flag), color the dots in the scatter plot.
3. Note whether or not the Principal Components predict the categorical variables.
4. If you selected more than 2 PC's, feel free to experiment by using different PC's and determining if they predict the target.


None of the principal components plotted against each other predict the target variables well. However, out of the principal components plotted against each other,
PC_1 and PC_2 were the best at predicting the target variables with the least amount of overlap between the two target variables.

"""

""" Re-attach the target variable to the principal components """
X_PCA["TARGET_BAD_FLAG"] = df["TARGET_BAD_FLAG"]
X_PCA["TARGET_LOSS_AMT"] = df["TARGET_LOSS_AMT"]
print(X_PCA.head())
print("\n\n")

# %%
"""
PLOT 1: PC_1 vs. PC_2
    - Using the target flag as the color
"""
# Group the data by the target variable and print the first few rows
for Name, Group in X_PCA.groupby("TARGET_BAD_FLAG"):
    print(Group.head())
    print("\n")

# Plot the principal components by target variable
for Name, Group in X_PCA.groupby("TARGET_BAD_FLAG"):
    plt.scatter(Group.PC_1, Group.PC_2, label=Name)
plt.xlabel("PC_1")
plt.ylabel("PC_2")
plt.legend(["Good", "Default"])
plt.show()

# %%
"""
PLOT 2: PC_1 vs. PC_3
    - Using the target flag as the color
"""
# Group the data by the target variable and print the first few rows
for Name, Group in X_PCA.groupby("TARGET_BAD_FLAG"):
    print(Group.head())
    print("\n")

# Plot the principal components by target variable
for Name, Group in X_PCA.groupby("TARGET_BAD_FLAG"):
    plt.scatter(Group.PC_1, Group.PC_3, label=Name)
plt.xlabel("PC_1")
plt.ylabel("PC_3")
plt.legend(["Good", "Default"])
plt.show()

# %%
"""
PLOT 3: PC_1 vs. PC_4
    - Using the target flag as the color
"""
for Name, Group in X_PCA.groupby("TARGET_BAD_FLAG"):
    print(Group.head())
    print("\n")

# Plot the principal components by target variable
for Name, Group in X_PCA.groupby("TARGET_BAD_FLAG"):
    plt.scatter(Group.PC_1, Group.PC_4, label=Name)
plt.xlabel("PC_1")
plt.ylabel("PC_4")
plt.legend(["Good", "Default"])

# %%
"""
PLOT 4: PC_2 vs. PC_3
    - Using the target flag as the color
"""
for Name, Group in X_PCA.groupby("TARGET_BAD_FLAG"):
    print(Group.head())
    print("\n")

# Plot the principal components by target variable
for Name, Group in X_PCA.groupby("TARGET_BAD_FLAG"):
    plt.scatter(Group.PC_2, Group.PC_3, label=Name)
plt.xlabel("PC_2")
plt.ylabel("PC_3")
plt.legend(["Good", "Default"])
plt.show()

# %%
"""
PLOT 5: PC_2 vs. PC_4
    - Using the target flag as the color
"""
for Name, Group in X_PCA.groupby("TARGET_BAD_FLAG"):
    print(Group.head())
    print("\n")

# Plot the principal components by target variable
for Name, Group in X_PCA.groupby("TARGET_BAD_FLAG"):
    plt.scatter(Group.PC_2, Group.PC_4, label=Name)
plt.xlabel("PC_2")
plt.ylabel("PC_4")
plt.legend(["Good", "Default"])
plt.show()

# %%
"""
PLOT 6: PC_3 vs. PC_4
    - Using the target flag as the color
"""
for Name, Group in X_PCA.groupby("TARGET_BAD_FLAG"):
    print(Group.head())
    print("\n")

# Plot the principal components by target variable
for Name, Group in X_PCA.groupby("TARGET_BAD_FLAG"):
    plt.scatter(Group.PC_3, Group.PC_4, label=Name)
plt.xlabel("PC_3")
plt.ylabel("PC_4")
plt.legend(["Good", "Default"])
plt.show()
# %%
