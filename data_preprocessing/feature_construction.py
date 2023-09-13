import pandas as pd

from data_preprocessing.utils import check_column_mismatch


def frequency_features(train_df: pd.DataFrame, test_df: pd.DataFrame = None):
    check_column_mismatch(train_df, test_df)

    if test_df is not None:
        # Combine train and test DataFrames
        combined_df = pd.concat([train_df, test_df], axis=0)
    else:
        combined_df = train_df  # Use only train_df if test_df is not provided

    # Adding value_counts features to both train_df and test_df
    for c in train_df.columns:
        val_counts = combined_df[c].value_counts()

        # Create a new column with count-based features for train_df
        train_df[c + '_count'] = train_df[c].map(val_counts)

        if test_df is not None:
            # Create the same features for test_df
            test_df[c + '_count'] = test_df[c].map(val_counts)

    if test_df is not None:
        return train_df, test_df
    else:
        return train_df
