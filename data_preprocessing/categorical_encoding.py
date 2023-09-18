import inspect
from typing import Union

import pandas as pd
from category_encoders import OneHotEncoder, BinaryEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.utils import BaseEncoder
from sklearn.preprocessing import LabelEncoder

from data_preprocessing.utils import check_column_mismatch


def infer_categorical_features(X_train, X_test=None, threshold=10):
    """
    Infer categorical features in a DataFrame based on a threshold of unique values.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame.
    threshold (int): The threshold for the number of unique values to consider a feature as categorical.

    Returns:
    categorical_features (list): A list of column names identified as categorical features.
    """
    if X_test is not None:
        # Combine the training and testing data
        dataframe = pd.concat([X_train, X_test], axis=0)
    else:
        dataframe = X_train

    categorical_features = []

    for column in dataframe.columns:
        unique_values = dataframe[column].nunique()
        if unique_values <= threshold:
            categorical_features.append(column)

    return categorical_features


def categorical_encoding(df_train, df_test, cat_columns, encoding_method: Union[str, BaseEncoder] = OneHotEncoder):
    check_column_mismatch(df_train, df_test)

    if encoding_method == 'OneHotEncoder':
        # Perform label encoding on categorical columns
        label_encoder = OneHotEncoder(cols=cat_columns)
    elif encoding_method == 'BinaryEncoder':
        # Perform label encoding on categorical columns
        label_encoder = BinaryEncoder(cols=cat_columns)
    elif inspect.isclass(BaseEncoder):
        # Use a custom encoder if provided
        label_encoder = encoding_method(cols=cat_columns)
    else:
        label_encoder = OrdinalEncoder(cols=cat_columns)

    df_train = label_encoder.fit_transform(df_train)
    if df_test is not None:
        df_test = label_encoder.transform(df_test)
        return df_train, df_test
    else:
        return df_train


def ordinal_encoding(train_df, test_df, feature):
    encoder = LabelEncoder()
    encoder.fit(train_df[feature])
    train_df[feature] = encoder.transform(train_df[feature])
    test_df[feature] = encoder.transform(test_df[feature])
