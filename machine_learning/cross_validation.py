import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from machine_learning.auto_valid_lightgbm import AutoValidLGBMClassifier


def train_model(X, y, lgbm_best_params=None, catboost_best_params=None):
    if lgbm_best_params is None and catboost_best_params is None:
        raise Exception()
    lgbm_clf, catboost_clf = None, None

    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    ensemble_f1_scores = []

    for train_index, val_index in kf.split(X):
        if isinstance(X, pd.DataFrame):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        else:
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

        if lgbm_best_params is not None:
            # Initialize and train LightGBM classifier with best parameters
            lgbm_clf = AutoValidLGBMClassifier(**lgbm_best_params, random_state=1)
            lgbm_clf.fit(X_train, y_train)
        else:
            lgbm_clf = None

        if catboost_best_params is not None:
            # Initialize and train CatBoost classifier with best parameters
            catboost_clf = CatBoostClassifier(
                **catboost_best_params, random_state=1, silent=True
            )
            catboost_clf.fit(X_train, y_train)
        else:
            catboost_clf = None

        ensemble_pred = ensemble_prediction(catboost_clf, lgbm_clf, X_val)

        # Calculate the F1 score of the ensemble model for the current fold
        ensemble_f1 = f1_score(
            y_val, ensemble_pred, average="micro"
        )  # You can use 'micro' or 'weighted' as well
        ensemble_f1_scores.append(ensemble_f1)

        print(f"Ensemble F1 Score on validation fold: {ensemble_f1}")

    # Average F1 Score across all folds
    print(f"Average Ensemble F1 Score on validation set: {np.mean(ensemble_f1_scores)}")

    if lgbm_clf is not None:
        lgbm_clf.fit(X, y)

    if catboost_clf is not None:
        catboost_clf.fit(X, y)
    return lgbm_clf, catboost_clf


def ensemble_prediction(catboost_clf, lgbm_clf, x_val):
    if lgbm_clf != None:
        # Validate both models on the validation set
        y_val_prob_lgbm = lgbm_clf.predict_proba(x_val)
    else:
        y_val_prob_lgbm = None

    if catboost_clf != None:
        y_val_prob_catboost = catboost_clf.predict_proba(x_val)
    else:
        y_val_prob_catboost = None

    if y_val_prob_lgbm is not None and y_val_prob_catboost is not None:
        # Combine predictions using a simple voting ensemble for multiclass
        ensemble_prob = (
            y_val_prob_lgbm + y_val_prob_catboost
        ) / 2  # Take the average probability scores
    elif y_val_prob_lgbm is not None:
        ensemble_prob = y_val_prob_lgbm
    else:
        ensemble_prob = y_val_prob_catboost

    # Now, you can find the class with the highest probability for each sample
    ensemble_pred = np.argmax(
        ensemble_prob, axis=1
    )  # This gives the predicted class for each sample
    class_mapping = {index: label for index, label in enumerate(lgbm_clf.classes_)}
    # Now, you can convert the predicted class indices to string labels
    ensemble_pred = [class_mapping[class_index] for class_index in ensemble_pred]
    return ensemble_pred
