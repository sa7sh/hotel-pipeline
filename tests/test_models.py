# tests/test_models.py
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def test_rf_train_predicts():
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0,1})