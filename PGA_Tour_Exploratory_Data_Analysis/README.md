# PGA Tour EDA Description:
This exploratory data analysis (EDA) explores the raw data from all Professional Golf Association (PGA) Tour tournaments between the 2015 - 2022 seasons to discover trends and insights into the game of professional golf and which factors might influence a player's success on the tour.

**File Overview Description**
This folder within the projects repository contains the following files (not including datasets):
- *PGA_Tour_EDA_Full_Data_Cleaning_and_Analysis_Process.ipynb*: this file includes the initial inspection of the original raw dataset, exploratory of the individual variables, cleaning of the data, tailoring of the datasets for different data analysis, and the full data anlaysis that is ultimately included in the executive summary.
- *PGA_Tour_EDA_Executive_Summary_(w_Python_Code).ipynb*: This is an executive summary of the data cleaning and exploratory data analysis process that was conducted in the previously listed file. This Jupyter Notebook contains introductory information about the project, the python code used for the data analysis of the cleaned datasets created in the full data cleaning file, as well as interpreation of the EDA results.
- *PGA_Tour_EDA_Executive_Summary_Write_Up.md*: This file includes the results of the EDA, the project background information, and the results interpretation included in the executive summary, but as a Markdown file that doesn't include the Python code used to execute the EDA.
- *PGA_Tour_EDA_Executive_Summary_Write_Up.md*: This is the Write-Up file, but in .html format.

**.csv File Description:**
- *PGA_Tour_Raw_Data.csv*: this is the original raw data file that I started with.
- *pga_clean.csv*: this was the final version of all the cleaned data, even with those tournaments missing strokes gained information.
- *pga_sg.csv*: this file only contained the cleaned data from tournaments in which strokes gained data are available.
- *season_2022_exp.csv*: this file only contains tournaments from the 2022 season. The file also contains a new variable entitled "course experience," which is based on the previous seven years of experience at a given golf course.
