"""
# Assignment 99.3: KMeans Clustering

### Assignment Requirements
#### I. Assignment Set-Up
1. Download the HMEQ Data set
2. Read the data into Python
3. **Impute any missing numeric values. You may use a simple approach such as the mean or median.**
4. Create a new Data Frame that only has numeric input variables. In other words, remove the Target variables and the 
categorical variables.

#### II. Tranform the Data
1. Use StandardScaler or MinMaxScaler to transform the numeric data.

#### III. Select Variables
1. Select three or more variable for clustering. Try to select variables based on a theme (i.e. variables convey similar '
types of information).
    - **DO NOT USE TARGET VARIABLES TO CREATE CLUSTERS**
    - Hint: Try to avoid categorical variables or binary variables, they usually produce bad results

#### IV: Determine the number of clusters
1. Calculate Intertia, Silhouette, and/or Calinski Harabaz scores.
2. Graph the results of your scores.
3. Determine a good number of clusters to use.

#### V: Find the KMeans Clusters
1. Find the KMeans Clusters
2. Using "groupby", profile the cluster centers.
3. If possible, try to tell a story of the people who are members of each cluster. Do the clusters make sense?
4. Determine if the clusters can be used to determine the Probability of Loan Default and Loss Amount given default.
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
# -----
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# -----
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# -----
import matplotlib.pyplot as plt
import seaborn as sns
# -----
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

# Set the display settings
sns.set_theme()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

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
PHASE II: TRANSFORM THE DATA
    1. Use StandardScaler or MinMaxScaler to transform the numeric data.
"""
# Create a StandardScaler object and fit the data - we will use the StandardScaler for this transformation
theScaler = StandardScaler()
theScaler.fit(X)

# Transform the data and create a DataFrame
X_TRN = theScaler.transform(X)
X_TRN = pd.DataFrame(X_TRN)

# Rename the columns of the transformed DataFrame
varNames_trn = []
for i in varNames:
    newName = 'trn_' + i
    varNames_trn.append(newName)
print(varNames_trn)
print("\n\n")

# Give the columns of the transformed DataFrame the new names
X_TRN.columns = varNames_trn

# Print the first few rows of the transformed DataFrame and the summary statistics
print(X_TRN.head())
print("\n\n")
print(X_TRN.describe())
print("\n\n")

# %%
"""
PHASE III: SELECT VARIABLES
    1. Select three or more variable for clustering. Try to select variables based on a theme (i.e. variables convey similar 
    types of information).
    - DO NOT USE TARGET VARIABLES TO CREATE CLUSTERS
    - Hint: Try to avoid categorical variables or binary variables, they usually produce bad results
"""

"""
Variable Selection Methodology:
A am going to select my variables based on the results of the important variables included in the random forest model that I 
created in the Tree-Based Methods assignment. The reason for choosing the random forest model is because this model had the greatest
AUC and the greatest accuracy.I will select the top 5 numeric variables that were important in the model.

Selected Variables from the Important Random Forest Model Variables:
    1. DEBTINC
    2. CLAGE
    3. DELINQ
    4. LOAN
    5. VALUE
"""

# Refine the transformed data to include only the selected variables
X_TRN = X_TRN[['trn_IMP_DEBTINC', 'trn_IMP_CLAGE', 'trn_IMP_DELINQ', 'trn_IMP_LOAN', 'trn_IMP_VALUE']]

#%%
"""
PHASE IV: DETERMINE THE NUMBER OF CLUSTERS
    1. Calculate Intertia, Silhouette, and/or Calinski Harabaz scores.
    2. Graph the results of your scores.
    3. Determine a good number of clusters to use.
"""

# Create a list to store the inertia, silhouette, and calinski harabasz scores
K_LIST = []     # list to store the number of clusters
I_LIST = []     # list to store the inertia scores
S_LIST = []     # list to store the silhouette scores
C_LIST = []     # list to store the calinski harabasz scores

# Loop through the range of clusters
for K in range(3, 12):
    km = KMeans(n_clusters=K, random_state=1)
    km.fit(X_TRN)
    K_LIST.append(K)
    I_LIST.append(km.inertia_)
    S_LIST.append(silhouette_score(X_TRN, km.labels_))
    C_LIST.append(calinski_harabasz_score(X_TRN, km.labels_))

# Define the function to plot the scores
def drawElbow(K, SCORE, LABEL):
    plt.plot(K, SCORE, 'ro-', linewidth=2)
    plt.title(LABEL)
    plt.xlabel('Clusters')
    plt.ylabel('Score')
    plt.show()
    
# Draw the elbow plots
drawElbow(K_LIST, I_LIST, "Inertia")
drawElbow(K_LIST, S_LIST, "Silhouette")
drawElbow(K_LIST, C_LIST, "Calinski")

""" For the time being, lets go with 7 clusters """

# %%
"""
PHASE V: FIND THE KMEANS CLUSTERS
    1. Find the KMeans Clusters
    2. Using "groupby", profile the cluster centers.
    3. If possible, try to tell a story of the people who are members of each cluster. Do the clusters make sense?
    4. Determine if the clusters can be used to determine the Probability of Loan Default and Loss Amount given default.
"""

def clusterData( DATA, TRN_DATA, K, TARGET ) :
    # Print the number of clusters
    print("\n\n\n")
    print("K = ",K)
    print("=======")
    
    # Perform the clustering
    km = KMeans( n_clusters=K, random_state = 1 )       # Create the KMeans object
    km.fit( TRN_DATA )                                  # Fit the KMeans object to the data
    Y = km.predict( TRN_DATA )                          # Predict the clusters
    DATA["CLUSTER"] = Y                                 # Add the cluster to the dataframe
    
    # Preview the data
    print("Preview of the data:")
    print( DATA.head().T )
    print("--------------------")
    print("\n\n")

    # Print the mean for each cluster
    # Limit the `DATA` dataframe to only include the numeric variables
    DATA = DATA[ DATA.columns[ DATA.dtypes != "object" ] ]
    # Group the data by the cluster
    G = DATA.groupby("CLUSTER")
    # Print the mean for each cluster
    print("Mean for each cluster:")
    print( G.mean().T)
    print("--------------------")
    print("\n\n")
    # Print the count for each cluster
    print("Count for each cluster:")
    print( G[ TARGET ].value_counts() )

# Perform the clustering
clusterData( df, X_TRN, 7, TARGET_F)

# %%
