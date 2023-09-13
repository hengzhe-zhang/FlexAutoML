import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_preprocessing.categorical_encoding import categorical_encoding, infer_categorical_features
from data_preprocessing.feature_construction import frequency_features
from data_preprocessing.feature_normalization import feature_normalization
from data_preprocessing.feature_selection import feature_selection_lgbm
from machine_learning.cross_validation import train_model, ensemble_prediction
from machine_learning.hyper_parameter_optimization import objective_catboost, objective_lgbm, \
    hyperparameter_optimization

if __name__ == '__main__':
    # Load the Breast Cancer dataset (assuming 'breast_cancer' is already imported)
    breast_cancer = load_breast_cancer()
    X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    y = breast_cancer.target

    # Encoding target column
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split the dataset into a training set and a testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_columns = infer_categorical_features(X_train)
    print('Inferred Categorical Features', cat_columns)
    X_train, X_test = frequency_features(X_train, X_test)

    # Encoding categorical features
    X_train, X_test = categorical_encoding(X_train, X_test, cat_columns=cat_columns)
    # Normalization
    X_train, X_test = feature_normalization(X_train, X_test)

    # Feature selection using LightGBM
    X_train, X_test = feature_selection_lgbm(X_train, y_train, X_test, percentage_total_weight=0.95)

    # Perform hyperparameter optimization for LightGBM
    lgbm_best_params = hyperparameter_optimization(X_train, y_train, n_trials=3, function=objective_lgbm)

    # Perform hyperparameter optimization for CatBoost
    catboost_best_params = hyperparameter_optimization(X_train, y_train, n_trials=3, function=objective_catboost)

    # Train models
    lgbm_clf, catboost_clf = train_model(X_train, y_train, lgbm_best_params, catboost_best_params)

    # Make predictions on the testing set
    ensemble_pred = ensemble_prediction(catboost_clf, lgbm_clf, X_test)
    ensemble_pred = label_encoder.inverse_transform(ensemble_pred)

    # Prepare the submission file
    submission_df = pd.DataFrame({'outcome': ensemble_pred})
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file has been generated.")
