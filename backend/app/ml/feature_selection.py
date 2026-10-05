"""Korelasyon kümelemesiyle özellik sadeleştirme (plan 3.5).

Birbirine çok benzeyen özellikler (ör. trend bayrakları, volatilite ölçüleri)
modele yeni bilgi katmadan gürültü ve aşırı uyum riski ekler. Mutlak Pearson
korelasyonu üzerinden hiyerarşik kümeleme yapılır, her kümeden bir temsilci
tutulur; gerisi walk-forward ile A/B için `drop_features` olarak verilir
(bkz. `app.backtest.walk_forward.prepare_walk_forward`).
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def correlated_feature_clusters(X: pd.DataFrame, threshold: float = 0.9) -> list[list[str]]:
    """|korelasyon| >= `threshold` olan özellikleri aynı kümede toplar
    (tam bağlantı: kümedeki HER çift bu eşiği geçer). Sabit kolonlar kendi
    başına bir küme sayılır."""
    columns = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    if len(columns) < 2:
        return [[c] for c in X.columns]
    corr = np.array(X[columns].corr().abs().fillna(0.0), dtype=float)
    np.fill_diagonal(corr, 1.0)
    distance = np.clip(1.0 - corr, 0.0, None)
    distance = (distance + distance.T) / 2
    labels = fcluster(linkage(squareform(distance, checks=False), method="complete"), t=1.0 - threshold, criterion="distance")
    clusters: dict[int, list[str]] = {}
    for column, label in zip(columns, labels):
        clusters.setdefault(int(label), []).append(column)
    constant = [[c] for c in X.columns if c not in columns]
    return list(clusters.values()) + constant


def redundant_features(X: pd.DataFrame, threshold: float = 0.9) -> list[str]:
    """Her kümeden listede İLK geçen özellik (özellik listesi zaten
    önem/öncelik sırasıyla yazılmış) tutulur; geri kalanlar döner."""
    order = {c: i for i, c in enumerate(X.columns)}
    dropped: list[str] = []
    for cluster in correlated_feature_clusters(X, threshold):
        keep = min(cluster, key=order.__getitem__)
        dropped.extend(c for c in cluster if c != keep)
    return dropped
