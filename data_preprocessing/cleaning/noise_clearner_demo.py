import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from data_preprocessing.cleaning.noise_data_cleaning import NoiseDataCleaner

# Generate synthetic dataset
X, y = make_classification(n_samples=6000, n_features=20, random_state=42)

# Introducing noise in the last 10% of the data
y[-600:] = np.random.choice([0, 1], 600)

# Convert to DataFrame for the sake of demonstration
X_df = pd.DataFrame(X)
y_df = pd.Series(y)

# Initialize the cleaner
cleaner = NoiseDataCleaner(X_df, y_df)

# Get AUC scores before cleaning
auc_before = cleaner.evaluate_model()

# Clean the data
X_cleaned, y_cleaned = cleaner.clean()

# Get AUC scores after cleaning
cleaner_new = NoiseDataCleaner(X_cleaned, y_cleaned)
auc_after = cleaner_new.evaluate_model()

# Visualize the AUC scores
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(auc_before, label='Before Cleaning', marker='o')
plt.axhline(y=np.mean(auc_before), color='r', linestyle='--', label='Mean AUC')
plt.title('AUC Scores Before Cleaning')
plt.xlabel('Fold')
plt.ylabel('AUC')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(auc_after, label='After Cleaning', marker='o')
plt.axhline(y=np.mean(auc_after), color='r', linestyle='--', label='Mean AUC')
plt.title('AUC Scores After Cleaning')
plt.xlabel('Fold')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.show()
