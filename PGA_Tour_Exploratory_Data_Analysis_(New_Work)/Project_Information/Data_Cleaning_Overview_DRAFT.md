# Data Cleaning Process Overview for PGA Tour Dataset

## Introduction
This document outlines the data cleaning process applied to the `PGA_Tour_Raw_Data.csv` dataset, with a focus on preparing the data for exploratory analysis and modeling in a PGA Tour-focused Data Science project.

## Initial Data Loading and Inspection
1. **Libraries Import and Environment Setup**
   - Essential libraries such as Pandas, NumPy, Seaborn, Matplotlib, and Plotly were imported.
   - Interactive shell configured for enhanced output display.

2. **Data Import**
   - The raw PGA Tour data was loaded into a Pandas DataFrame.

3. **Preliminary Data Inspection**
   - Methods like `.info()`, `.head()`, `.tail()`, and `.describe()` were used for initial data inspection.

## Data Cleaning Steps
4. **Empty Column Removal**
   - Columns with no valuable data (e.g., 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4') were identified and removed.

5. **Irrelevant Variable Removal**
   - Dropped variables related to 'DKP', 'FKP', and 'SDP' as they were not required for the analysis.

6. **Player Name Column Comparison**
   - Analyzed 'Player_initial_last' and 'player' columns to check for consistency or redundancy.

7. **Strokes Gained Columns Review**
   - Inspected columns related to strokes gained metrics such as 'sg_putt', 'sg_arg', etc.

8. **Data Type Standardization**
   - Converted identifying variables (e.g., 'Player_initial_last', 'tournament id') to string data types.

9. **Value Replacement in 'pos' Column**
   - Replaced '999.0' values in 'pos' with '0' to indicate no valid finishing position.

10. **Null Value Check**
    - Performed a check for null values, particularly in the 'pos' column.

## Creation of Strokes Gained Dataset
11. **Subset Creation for Strokes Gained Data**
    - Extracted a subset of the cleaned data focusing on strokes gained metrics to create the `sg_data_clean.csv`.
    - This dataset includes only entries with relevant strokes-gained data, providing a more focused view for specific analyses.

## Creation of the `course_experience` Variable
12. **New Variable Introduction**
    - Introduced a new variable `course_experience` based on player performance and history.
    - This variable aims to quantify a player's experience and performance on specific courses, potentially serving as a significant predictor in later analysis.