from typing import Union

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from data_preprocessing.utils import check_column_mismatch


def feature_selection_lgbm(X_train: Union[np.ndarray, pd.DataFrame],
                           y_train: Union[np.ndarray, pd.DataFrame],
                           X_test: Union[np.ndarray, pd.DataFrame],
                           percentage_total_weight=0.95):
    check_column_mismatch(X_train, X_test)

    best_params = {'learning_rate': 0.2, 'max_depth': 3, 'min_child_samples': 5,
                   'n_estimators': 50, 'num_leaves': 31}
    # Create a LightGBM classifier
    lgb_model = LGBMClassifier(**best_params)

    # Fit the LightGBM model to the training data
    lgb_model.fit(X_train, y_train)

    # Get feature importances
    feature_importances = lgb_model.feature_importances_

    # Sort features by importance
    sorted_idx = np.argsort(feature_importances)[::-1]

    # Calculate the total weight of all feature importances
    total_weight = np.sum(feature_importances)

    # Determine the total weight threshold based on the specified percentage
    threshold_weight = total_weight * percentage_total_weight

    # Initialize variables
    selected_feature_indices = []
    current_weight = 0

    # Select features based on the weight threshold
    for idx in sorted_idx:
        selected_feature_indices.append(idx)
        current_weight += feature_importances[idx]
        if current_weight >= threshold_weight:
            break

    # Update datasets with selected features
    if isinstance(X_train, pd.DataFrame):
        X_train_selected = X_train.iloc[:, selected_feature_indices]
        X_test_selected = X_test.iloc[:, selected_feature_indices]
    else:
        X_train_selected = X_train[:, selected_feature_indices]
        X_test_selected = X_test[:, selected_feature_indices]
    num_selected_features = len(selected_feature_indices)
    num_total_features = X_train.shape[1]
    print('num_selected_features', num_selected_features, 'num_total_features', num_total_features)
    return X_train_selected, X_test_selected
