import numpy as np
import pandas as pd

from data_preprocessing.utils import check_column_mismatch


def missing_value_impute(df_train, df_test=None):
    check_column_mismatch(df_train, df_test)

    # Identify numerical and categorical columns in training DataFrame
    numerical_columns = df_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df_train.select_dtypes(exclude=[np.number]).columns.tolist()

    # If test DataFrame is provided, concatenate train and test DataFrames
    if df_test is not None:
        df_merged = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    else:
        df_merged = df_train.copy()

    # Calculate the mean of each numerical column in the merged DataFrame
    mean_values_train = df_merged[numerical_columns].mean()

    # Fill missing values in numerical columns with mean
    df_train[numerical_columns] = df_train[numerical_columns].fillna(mean_values_train)

    # Fill missing values in categorical columns with mode
    mode_values_train = df_merged[categorical_columns].mode().iloc[0]
    df_train[categorical_columns] = df_train[categorical_columns].fillna(mode_values_train)

    if df_test is not None:
        # Fill missing values in test DataFrame with mean of the training set for numerical columns
        df_test[numerical_columns] = df_test[numerical_columns].fillna(mean_values_train)
        df_test[categorical_columns] = df_test[categorical_columns].fillna(mode_values_train)
        return df_train, df_test
    else:
        return df_train
