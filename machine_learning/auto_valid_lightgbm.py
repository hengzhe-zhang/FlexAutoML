import numpy as np
from lightgbm import LGBMClassifier, early_stopping
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def lgb_f1_micro(y_true, y_pred):
    y_pred = np.argmax(y_pred.reshape(len(y_true), -1), axis=1)
    return "f1_micro", f1_score(y_true, y_pred, average="micro"), True


class AutoValidLGBMClassifier:
    def __init__(
        self, test_size=0.2, early_stopping_rounds=30, use_micro_f1=False, **kwargs
    ):
        self.test_size = test_size
        self.early_stopping_rounds = early_stopping_rounds
        self.use_micro_f1 = use_micro_f1
        self.lgbm = LGBMClassifier(**kwargs)

    def fit(self, X, y):
        # Split the data into training and validation sets
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        eval_metric = "logloss"  # default metric
        if self.use_micro_f1:
            eval_metric = lgb_f1_micro  # use micro f1

        # Fit the model with early stopping
        self.lgbm.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=eval_metric,
            callbacks=[early_stopping(stopping_rounds=self.early_stopping_rounds)],
        )
        self.classes_ = self.lgbm.classes_

    def predict(self, X):
        return self.lgbm.predict(X)

    def predict_proba(self, X):
        return self.lgbm.predict_proba(X)


# Example usage:
if __name__ == "__main__":
    # Load some sample data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Initialize the classifier
    clf = AutoValidLGBMClassifier()

    # Fit the classifier
    clf.fit(X, y)

    # Make predictions
    y_pred = clf.predict(X)

    # Output the first 10 predictions
    print("First 10 predictions:", y_pred[:10])
