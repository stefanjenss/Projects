# MSDS 422 Assignments

## About the HMEQ Data Set
For the assignment in this course, we will use Python to examine a data set containing Home Equity Loans. 

This is the data that will be used throughout the class in order to develop predictive models that will be used to determine the level of risk for each loan.

As with all real world data, this data is far from perfect.
- It contains both numerical and categorical variables.
- It contains missing data.
- It contains outliers.

### Target Variables for the HMEQ Data Set
The data set contains two target variables. 

1. The first, `TARGET_BAD_FLAG`, indicates whether or not the loan defaulted. If the value is set to 1, then the loan went bad and the bank lost money. If the value is set to 0, the loan was repaid.
2. The second target, `TARGET_LOSS_AMT`, indicates the amount of money that was lost for loans that went bad.

The remaining variables contain information about the customer at the time that the loan was issued.

## HMEQ Data Set Data Dictionary

| Variable       | Definition                                                                                      | Role   | Type     | Conventional Wisdom                                                                                                   |
|----------------|-------------------------------------------------------------------------------------------------|--------|----------|-----------------------------------------------------------------------------------------------------------------------|
| TARGET_BAD_FLAG| BAD=1 (Loan was defaulted)                                                                     | TARGET | BINARY   | HMEQ = Home Equity Line of Credit Loan. BINARY TARGET                                                                 |
| TARGET_LOSS_AMT| If loan was Bad, this was the amount not repaid.                                                | TARGET | NUMBER   | HMEQ = Home Equity Line of Credit Loan. NUMERICAL                                                                    |
| LOAN           | HMEQ Credit Line                                                                               | INPUT  | NUMBER   | The bigger the loan, the more risky the person                                                                         |
| MORTDUE        | Current Outstanding Mortgage Balance                                                          | INPUT  | NUMBER   | If you owe a lot of money on your current mortgage versus the value of your house, you are more risky.               |
| VALUE          | Value of your house                                                                             | INPUT  | NUMBER   | If you owe a lot of money on your current mortgage versus the value of your house, you are more risky.               |
| REASON         | Why do you want a loan?                                                                         | INPUT  | CATEGORY | If you are consolidating debt, that might mean you are having financial trouble.                                      |
| JOB            | What do you do for a living?                                                                    | INPUT  | CATEGORY | Some jobs are unstable (and therefore are more risky)                                                                 |
| YOJ            | Years on Job                                                                                    | INPUT  | NUMBER   | If you have been at your job for a while, you are less likely to lose that job. That makes you less risky.           |
| DEROG          | Derogatory Marks on Credit Record. These are very bad things that stay on your credit report for 7 years. | INPUT  | NUMBER   | Lots of Derogatories mean that something really bad happened to you (such as a bankruptcy) in your past.            |
| DELINQ         | Delinquencies on your current credit report.                                                    | INPUT  | NUMBER   | When you have a lot of delinquencies, you might be more likely to default on a loan.                                  |
| CLAGE          | Credit Line Age (in months) is how long you have had credit.                                    | INPUT  | NUMBER   | If you have had credit for a long time, you are considered less risky than a new high school student.                 |
| NINQ           | Number of inquiries. This is the number of times within the last 3 years that you went out looking for credit. | INPUT  | NUMBER   | Conventional wisdom is that if you are looking for more credit, you might be in financial trouble. Thus you are risky. |
| CLNO           | Number of credit lines you have (credit cards, loans, etc.).                                    | INPUT  | NUMBER   | People who have a lot of credit lines tend to be safe. However, if you have too many credit lines, you might be risky. |
| DEBTINC        | Debt to Income Ratio. Take the money you spend every month and divide it by the amount of money you earn every month. | INPUT  | NUMBER   | If your debt to income ratio is high then you are risky because you might not be able to pay your bills.              |

----------

# Assignment 1: Data Preparation

In this assignment, we will explore the data and begin to prepare the data set so that it can be used in predictive models.

### Assignment Requirements
1. Download the HMEQ Data Set
2. Read the data into Python
3. Explore both the inputs and target variables using statistical techniques.
4. Explore both the inputs and target variables using graphs and other visualizations.
5. Look for relationships between the input variables and the targets.
6. Fix (imput) all missing data.
    - Note: For numerical data, create a glaf variable to indicate if the value was missing
7. Convert all categorical variables into numerical variables (using one hot encoding)

----------

# Assignment 2: Tree Based Models

In this assignment, we will continue to use Python to develop predictive models. We will use three different tree based techniques to analyze the data: DECISION TREES, RANDOM FORESTS, and GRADIENT BOOSTING.

### Assignment Requirements
#### I. Decision Trees:
1. **Develop a decision tree to predict the probability of default**
    - Calculate the accuracy of the model on both the training and test data set.
    - Create a graph that shows the ROC curves for both the training and test data set. Clearly label each curve and display the Area Under the ROC curve.
    - Display the Decision Tree using a Graphviz program
    - List the variables included in the decision tree that predict loan default.
2. **Develop a decision tree to predict the loss amount assuming that the loan defaults**
    - Calculate the RMSE for both the training data set and test data set.
    - Display the Decision Tree using a Graphviz program
    - List the variables included in the decision tree that predeict loss amount.
#### II. Random Forests:
1. **Develop a Random Forest to predict the probability of default**
    - Calculate the accuracy of the model on both the training and test data set.
    - Create a graph that shows the ROC curves for both the training and test data set. Clearly label each curve and display the Area Under the ROC curve.
    - List the variables included in the Random Forest that predict loan default.
2. **Develop a Random Forest to predict the loss amount assuming that the loan defaults**
    - Calculate the RMSE for both the training data set and the test data set
    - List the variables included in the Random Forest that predict loss amount.
#### III. Gradient Boosting:
1. **Develop a Gradient Boosting model to predict the probability of default**
    - Calculate the accuracy of the model on both the training and test data set
    - Create a graph that shows the ROC curves for both the training and test data set. Clearly label each curve and display the Area Under the ROC curve.
    - List the variables included in the Gradient Boosting that predict loan default.
2. **Develop a Gradient Boosting to predict the loss amount assuming that the loan defaults**
    - Calculate the RMSE for both the training data set and the test data set
    - List the variables included in the Gradient Boosting that predict loss amount.
#### IV. ROC Curves
- Generate a ROC curve for the Decision Tree, Random Forest, and Gradient Boosting models using the Test Data Set.
- Use different colors for each curve and clearly label them.
- Include the Area under the ROC Curve on the graph.

----------

# Assignment 3: Regression Based Models

In this assignment, we will continue to use Python to develop predictive models. We will use two different types of regression: Linear and Logistic. We will use Logistic regression to determine the probability of a crash. Linear regression will be used to calculate the damages assuming that a crash occurs.

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

----------

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
    - Create a graph that shows the ROC curves for both the training and test data set. Clearly label each curve and display the Area Under the ROC curve.
    - Display the ROC curve for the test data with all your models on the same graph (tree based, regression, and TF). Discuss which one is the most accurate. Which one would you recommend using?

#### III. Tensor Flow Model to Predict Loss Given Default
1. Develop a model using Tensor Flow that will predict Loan Default.
2. For your model, do the following:
    - Try at least three different Activation Functions
    - Try one and two hidden layers
    - Try using a Dropout Layer
3. Explore using a variable selection technique
4. For each of the models:
    - Calculate the RMSE for both the training data set and the test data set
    - List the RMSE for the test data set for all the models created (tree based, regression, and TF). Discuss which one is the most accurate. Which one would you recommend using?

----------

# Assignment 99.1: Data Transformation

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

----------

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

----------

# Assignment 99.3: KMeans Clustering

### Assignment Requirements
#### I. Assignment Set-Up
1. Download the HMEQ Data set
2. Read the data into Python
3. **Impute any missing numeric values. You may use a simple approach such as the mean or median.**
4. Create a new Data Frame that only has numeric input variables. In other words, remove the Target variables and the categorical variables.

#### II. Tranform the Data
1. Use StandardScaler or MinMaxScaler to transform the numeric data.

#### III. Select Variables
1. Select three or more variable for clustering. Try to select variables based on a theme (i.e. variables convey similar types of information).
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