__author__ = 'justinkreft'


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
import pickle

#Global Definitions for variables
ID_COL = ['id']
TARGET_COL = ["over_50k"]
CATEGORICAL_COL = ['country', 'sex', 'race', 'relationship', 'occupation', 'marital_status', 'education_level', 'workclass']
NUMERIC_COL = ["age", "education_num", "capital_gain", "capital_loss", "hours_week"]
ALL_DATA_COL = NUMERIC_COL + CATEGORICAL_COL


def main():

#### Extract and data frame creation

    # Import flat table from csv into data frame
    data = pd.read_csv('Ex1_Flat_table.csv')

    # List available variables for reference
    print("\nQuick list of attributes in the DataFrame for reference?  \n", data.columns)

    # Show first 10 records of dataframe to validate import
    print("\n", data.head(10), "\n")

#### Descriptive Statistics

    # Generate count, mean, std, min, max, quartiles for all numeric variables
    for variable in NUMERIC_COL:
        print("\nGeneral Descriptive Statistics for %s (Numeric): \n" % variable, data[variable].describe())

    # Generate count, unique classes, top and freq of top for categorical values
    for variable in CATEGORICAL_COL:
        print("\nGeneral Descriptive Statistics for %s (Categorical): \n" % variable, data[variable].describe())

    # Group data by target variable (over_50k) and process descriptive statistics for each other variable
    by_target = data.groupby('over_50k')

    # Generate Descriptive Statisitics by target variable (over_50k)
    for variable in NUMERIC_COL:
        print("\nGeneral Descriptive Statistics for %s (Numeric) by target (over_50k): \n" % variable, by_target[variable].describe())
    for variable in CATEGORICAL_COL:
        print("\nGeneral Descriptive Statistics for %s (Categorical) by target (over_50k): \n" % variable, by_target[variable].describe())
    for group in CATEGORICAL_COL:
        temp_description = data.groupby(group)
        print("\nMeans of target variable for %s (Categorical) (i.e. percent of segment population at True: \n" % group, temp_description['over_50k'].mean())

#### Clean and Validate

    # Check for null values in data frames
    print("\nAre there Null values anywhere in the Imported DataFrame? \n", data.isnull().any())

    # Check for missing (?) values in data frames
    print("\nAre there Missing (?) values anywhere in the Imported DataFrame? \n", data.isin(list("?")).any())

    # Clean Null Values and Missing Values by Imputing replacements (mean for numeric, -9999 for categorical)
    # Note: No variables were null, so imputing mean was turned off,
    # Note: Special Missing value "?" transformed to -9999
    data[CATEGORICAL_COL] = data[CATEGORICAL_COL].replace("?", -9999)
    #data[CATEGORICAL_COL] = data[CATEGORICAL_COL].fillna(value = -9999)
    #data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean(), inplace=True)

    # Create numeric labels for categorical features using LabelEncoder
    # save coding for reference as dictionary in pickle
    encoding_ref = {}
    for column in CATEGORICAL_COL:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype('str'))
        encoding_ref[column] = le.classes_
    with open("encoding_ref.pkl", "wb") as fout:
        pickle.dump(encoding_ref, fout, protocol=-1)
    # Print Encoding dictionary for verification
    print(encoding_ref)


#### Create Training, Test and Validation sets

    # Split into train and test sets. Model Validation will use cross validation. Export
    train, test = train_test_split(data, test_size = 0.25)
    train.to_csv("Ex1_train.csv")
    test.to_csv("Ex1_test.csv")


main()