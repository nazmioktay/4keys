import numpy as np
import pandas as pd

from app.backtest.walk_forward import _MaskedModel
from app.ml.feature_selection import correlated_feature_clusters, redundant_features


def _frame():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    b = rng.normal(size=500)
    return pd.DataFrame(
        {
            "a": a,
            "a_copy": a * 2 + rng.normal(scale=0.01, size=500),
            "b": b,
            "a_neg": -a,
            "const": np.zeros(500),
        }
    )


def test_clusters_group_highly_correlated_features_including_negative():
    clusters = {frozenset(c) for c in correlated_feature_clusters(_frame(), threshold=0.9)}
    assert frozenset({"a", "a_copy", "a_neg"}) in clusters
    assert frozenset({"b"}) in clusters
    assert frozenset({"const"}) in clusters


def test_redundant_features_keeps_first_of_each_cluster():
    assert sorted(redundant_features(_frame(), threshold=0.9)) == ["a_copy", "a_neg"]


def test_masked_model_zeroes_dropped_columns_at_prediction():
    seen = {}

    class _Spy:
        def predict_batch(self, X):
            seen["X"] = X
            return np.zeros(len(X)), np.zeros(len(X))

    X = _frame()
    _MaskedModel(_Spy(), ["a_copy"]).predict_batch(X)
    assert (seen["X"]["a_copy"] == 0).all()
    assert (X["a_copy"] != 0).any()  # çağıranın verisi değişmez
