# custom_components.py

from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoding_maps = {}

    def fit(self, X, y):
        for col in X.columns:
            mapping = y.groupby(X[col]).mean()
            self.encoding_maps[col] = mapping
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in X.columns:
            X_transformed[col] = X[col].map(self.encoding_maps[col])
            X_transformed[col] = X_transformed[col].fillna(X_transformed[col].mean())
        return X_transformed
