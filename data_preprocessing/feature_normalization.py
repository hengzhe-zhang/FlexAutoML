from sklearn.preprocessing import StandardScaler


def feature_normalization(X, test_X=None):
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    if test_X is not None:
        test_X_normalized = scaler.transform(test_X)
        return X_normalized, test_X_normalized
    else:
        return X_normalized
