import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


class NoiseDataCleaner:
    def __init__(self, X, y, num_splits=60):
        self.X = X
        self.y = y
        self.num_splits = num_splits
        self.noisy_indices = []

    def evaluate_model(self):
        kf = KFold(n_splits=self.num_splits)
        auc_list = []

        for train_index, val_index in kf.split(self.X):
            if isinstance(self.X, pd.DataFrame):
                X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
                y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]
            else:
                X_train, X_val = self.X[train_index], self.X[val_index]
                y_train, y_val = self.y[train_index], self.y[val_index]

            # Define and train the LightGBM model using Scikit-learn API
            model = LGBMClassifier(objective='binary', metric='auc', boosting_type='gbdt')
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[early_stopping(stopping_rounds=10),
                                 log_evaluation(period=0)])

            y_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            auc_list.append(auc)

        return auc_list

    def identify_noisy_data(self, auc_list):
        Q1 = np.percentile(auc_list, 25)
        Q3 = np.percentile(auc_list, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        for i, auc in enumerate(auc_list):
            if not lower_bound <= auc <= upper_bound:
                self.noisy_indices.extend(list(KFold(n_splits=self.num_splits).split(self.X))[i][1])

    def remove_noisy_data(self):
        if isinstance(self.X, pd.DataFrame):
            self.X = self.X.drop(self.noisy_indices).reset_index(drop=True)
            self.y = self.y.drop(self.noisy_indices).reset_index(drop=True)
        else:
            self.X = np.delete(self.X, self.noisy_indices, axis=0)
            self.y = np.delete(self.y, self.noisy_indices, axis=0)

    def clean(self):
        auc_list = self.evaluate_model()
        self.identify_noisy_data(auc_list)
        self.remove_noisy_data()

        return self.X, self.y
