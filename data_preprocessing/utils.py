import numpy as np
import pandas as pd


def check_column_mismatch(df_train, df_test):
    # Check if columns in training and test DataFrames match
    if (isinstance(df_train, pd.DataFrame) and set(df_train.columns) != set(df_test.columns)) or \
            (isinstance(df_train, np.ndarray) and df_train.shape[1] != df_test.shape[1]):
        raise Exception("Columns in training and test data do not match. Please remove labels in training data.")
