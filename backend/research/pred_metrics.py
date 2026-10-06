"""Tahmin metrikleri: tarih bazlı IC / rank IC (+ t-istatistiği), QLIKE, AUC / log-loss, kalibrasyon.

IC kesitsel hesaplanır (her tarihte semboller arası korelasyon); t-istatistiği günlük IC serisi üzerinden, ufuk h için
örtüşen etiketleri hesaba katan Newey-West (gecikme = h−1) ile. NaN tahmin/etiket satırları ölçülmez; sayısı raporlanır."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import log_loss, roc_auc_score


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def daily_ic(pred: pd.Series, y: pd.Series, min_symbols: int = 5) -> tuple[pd.Series, pd.Series]:
    """(IC, rank IC) tarih serileri; `pred`/`y` (date, symbol) indeksli. En az `min_symbols` sembolü olan tarihler."""
    both = pd.concat([pred.rename("p"), y.rename("y")], axis=1).dropna()
    ic, ric = {}, {}
    for date, g in both.groupby(level="date"):
        if len(g) < min_symbols:
            continue
        p, t = g["p"].to_numpy(), g["y"].to_numpy()
        ic[date] = _corr(p, t)
        ric[date] = _corr(rankdata(p), rankdata(t))
    return pd.Series(ic, dtype=float).dropna(), pd.Series(ric, dtype=float).dropna()


def newey_west_t(series: pd.Series, lag: int) -> float:
    """Ortalama için Newey-West (Bartlett) t-istatistiği."""
    x = series.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 5:
        return float("nan")
    mu = x.mean()
    e = x - mu
    var = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        var += 2.0 * w * (e[k:] @ e[:-k]) / n
    se = math.sqrt(max(var, 1e-18) / n)
    return float(mu / se)


def ic_summary(pred: pd.Series, y: pd.Series, horizon: int = 1, min_symbols: int = 5) -> dict:
    ic, ric = daily_ic(pred, y, min_symbols)
    lag = max(horizon - 1, 0)
    return {
        "ic_mean": float(ic.mean()) if len(ic) else float("nan"),
        "ic_t": newey_west_t(ic, lag) if len(ic) else float("nan"),
        "ic_ir": float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan"),
        "rank_ic_mean": float(ric.mean()) if len(ric) else float("nan"),
        "rank_ic_t": newey_west_t(ric, lag) if len(ric) else float("nan"),
        "n_dates": int(len(ic)),
    }


def qlike(pred_vol: np.ndarray, true_vol: np.ndarray) -> float:
    """QLIKE (varyans üzerinden): ortalama( σ²/ŝ² − ln(σ²/ŝ²) − 1 ). Vol tahminleri pozitif olmalı (küçükler kırpılır)."""
    p = np.maximum(np.asarray(pred_vol, dtype=float), 1e-8) ** 2
    t = np.maximum(np.asarray(true_vol, dtype=float), 1e-8) ** 2
    ratio = t / p
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Güvenilirlik tablosu (tahmin ortalaması vs gerçekleşen sıklık) ve ECE."""
    df = pd.DataFrame({"p": p, "y": y})
    df["bin"] = np.minimum((df["p"] * bins).astype(int), bins - 1)
    table = df.groupby("bin").agg(pred=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    ece = float((table["n"] * (table["pred"] - table["observed"]).abs()).sum() / table["n"].sum())
    return table.reset_index(), ece


def evaluate(pred: pd.Series, y: pd.Series, kind: str, horizon: int, target_name: str = "") -> dict:
    """Hedef türüne göre metrik sözlüğü. `pred`/`y`: (date, symbol) indeksli; NaN satırlar atılır ve sayılır."""
    both = pd.concat([pred.rename("p"), y.rename("y")], axis=1)
    n_all = len(both)
    valid = both.dropna()
    out = {"n_rows": int(n_all), "n_scored": int(len(valid)), "n_pred_nan": int(both["p"].isna().sum())}
    if valid.empty:
        return out
    out.update(ic_summary(valid["p"], valid["y"], horizon))
    p, t = valid["p"].to_numpy(), valid["y"].to_numpy()
    if kind == "classification":
        yb = (t > 0.5).astype(int)
        if len(np.unique(yb)) == 2:
            out["auc"] = float(roc_auc_score(yb, p))
            out["log_loss"] = float(log_loss(yb, np.clip(p, 1e-6, 1 - 1e-6)))
        table, ece = calibration_table(np.clip(p, 0, 1), yb)
        out["ece"] = ece
        out["calibration"] = table.round(4).to_dict("records")
    elif kind == "regression":
        out["rmse"] = float(np.sqrt(np.mean((p - t) ** 2)))
        out["r2_oos"] = float(1 - np.sum((p - t) ** 2) / max(np.sum((t - t.mean()) ** 2), 1e-18))
        if target_name == "next_rv":
            out["qlike"] = qlike(p, t)
    return out
