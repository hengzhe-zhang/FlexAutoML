from functools import partial

import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score


def objective_catboost(trial, X, y, scoring="f1_micro"):
    # Hyperparameter settings for CatBoost
    catboost_params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "depth": trial.suggest_int("depth", 3, 9),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
    }

    # Initialize CatBoost classifier with the suggested hyperparameters
    clf = CatBoostClassifier(**catboost_params, random_state=1, silent=True)

    # Use cross-validation to evaluate the classifier with micro F1 score
    f1_scores = cross_val_score(clf, X, y, cv=5, scoring=scoring, n_jobs=-1)

    # Calculate the mean micro F1 score across cross-validation folds
    mean_f1_micro = f1_scores.mean()
    return mean_f1_micro


def objective_lgbm(trial, X, y, scoring="f1_micro"):
    # Hyperparameter settings for LightGBM
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "num_leaves": trial.suggest_int("num_leaves", 31, 120),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
    }

    # Initialize LightGBM classifier with the suggested hyperparameters
    clf = LGBMClassifier(**params, random_state=1)

    # Use cross-validation to evaluate the classifier with micro F1 score
    f1_scores = cross_val_score(clf, X, y, cv=5, scoring=scoring, n_jobs=-1)
    # Calculate the mean micro F1 score across cross-validation folds
    mean_f1_micro = f1_scores.mean()
    return mean_f1_micro


def hyperparameter_optimization(X, y, n_trials=50, function=objective_lgbm):
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction="maximize")
    study.optimize(partial(function, X=X, y=y), n_trials=n_trials)
    # Extract best hyperparameters
    best_params = study.best_params
    return best_params
