"""Volatilite model kıyasları: HAR-RV (Corsi) ve GARCH(1,1).

HAR-RV: hedef(vol) ~ günlük + haftalık + aylık geçmiş gerçekleşen vol (OLS/WLS, panel havuzlu, kesme terimli).
GARCH(1,1): sembol başına `arch` paketiyle YALNIZCA MLE (eğitim getirileri); koşullu varyans, test dönemi için SABİT
parametrelerle takvim günü bazında özyineli SÜZÜLÜR (yeniden kestirim yok, ileri bilgi yok): bilinen getiri günü
`s2' = ω + α r² + β s2`, bilinmeyen gün (embargo boşluğu) `s2' = ω + (α+β) s2`. Tahmin: `h` günlük ortalama varyansın
karekökü; eğitimde hedefe (Parkinson vol) `scale` ile (sıfırdan geçen EKK) ölçeklenir."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseModel
from .registry import register_model

_HAR_FEATURES = ("ohlcv_core__rv_pk_1", "ohlcv_core__rv_pk_5", "ohlcv_core__rv_pk_22")


@register_model("har_rv")
class HarRV(BaseModel):
    task = "regression"
    handles_nan = True  # NaN satırlar eğitimden atılır, tahminde NaN döner (doldurma yok)

    def __init__(self, seed: int = 0, features: tuple[str, ...] = _HAR_FEATURES, **params):
        super().__init__(seed, features=tuple(features), **params)

    def fit(self, X, y, dates, sample_weight=None):
        cols = list(self.params["features"])
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"har_rv için gerekli özellikler yok: {missing} (kaynak 'ohlcv_core' seçili mi?)")
        A = X[cols].to_numpy(dtype=float)
        yy = y.to_numpy(dtype=float)
        ok = np.isfinite(A).all(axis=1) & np.isfinite(yy)
        w = np.ones(len(yy)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        A1 = np.column_stack([np.ones(ok.sum()), A[ok]])
        sw = np.sqrt(w[ok])
        self.coef_, *_ = np.linalg.lstsq(A1 * sw[:, None], yy[ok] * sw, rcond=None)
        self._cols = cols
        return self

    def predict(self, X):
        A = X[self._cols].to_numpy(dtype=float)
        out = self.coef_[0] + A @ self.coef_[1:]
        out[~np.isfinite(A).all(axis=1)] = np.nan
        return out


def _filter_variance(returns: pd.Series, omega: float, alpha: float, beta: float, s2_init: float, end: pd.Timestamp) -> pd.Series:
    """Takvim günü bazında koşullu varyans: dönen seri indeksi d -> d GÜNÜNÜN kapanışına kadarki bilgiyle d+1 varyansı."""
    days = pd.date_range(returns.index.min(), end, freq="D")
    r = returns.reindex(days).to_numpy()
    s2 = np.empty(len(days))
    prev = s2_init
    for i, ri in enumerate(r):
        prev = omega + alpha * ri * ri + beta * prev if np.isfinite(ri) else omega + (alpha + beta) * prev
        s2[i] = prev
    return pd.Series(s2, index=days)


@register_model("garch11")
class Garch11(BaseModel):
    task = "regression"
    handles_nan = True
    supports_sample_weight = False  # MLE tek varlık; ağırlık yok sayılır (belgelenmiş)

    def __init__(self, seed: int = 0, ret_col: str = "ohlcv_core__ret_1", horizon: int = 1, min_obs: int = 250, **params):
        super().__init__(seed, ret_col=ret_col, horizon=horizon, min_obs=min_obs, **params)

    @staticmethod
    def _returns_by_symbol(X: pd.DataFrame, col: str) -> dict[str, pd.Series]:
        if not isinstance(X.index, pd.MultiIndex) or "symbol" not in X.index.names:
            raise ValueError("garch11: X indeksi (date, symbol) MultiIndex olmalı")
        r = X[col] * 100.0  # arch % ölçeğinde daha kararlı
        out = {}
        for sym, grp in r.groupby(level="symbol"):
            s = grp.droplevel("symbol").dropna().sort_index()
            s = s[~s.index.duplicated()]
            out[sym] = s
        return out

    def fit(self, X, y, dates, sample_weight=None):
        from arch import arch_model

        col = self.params["ret_col"]
        if col not in X.columns:
            raise ValueError(f"garch11: '{col}' kolonu gerekli (kaynak 'ohlcv_core' seçili mi?)")
        self._train: dict[str, dict] = {}
        for sym, r in self._returns_by_symbol(X, col).items():
            if len(r) < self.params["min_obs"]:
                continue
            try:
                res = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1, rescale=False).fit(disp="off", show_warning=False)
                omega, alpha, beta = (float(res.params[k]) for k in ("omega", "alpha[1]", "beta[1]"))
                if not (alpha >= 0 and beta >= 0 and alpha + beta < 0.9999 and omega > 0):
                    raise ValueError("kararsız GARCH parametresi")
            except Exception:  # noqa: BLE001 - kestirim başarısızsa RiskMetrics (EWMA) yedeği
                omega, alpha, beta = 0.0, 0.06, 0.94
            self._train[sym] = {"returns": r, "omega": omega, "alpha": alpha, "beta": beta, "s2_init": float(r.var())}
        # eğitim satırları için süzülmüş tahminler -> hedefe ölçek
        raw = self._raw_forecast(X)
        ok = np.isfinite(raw) & np.isfinite(y.to_numpy(dtype=float))
        self.scale_ = float((raw[ok] * y.to_numpy(dtype=float)[ok]).sum() / max((raw[ok] ** 2).sum(), 1e-18)) if ok.any() else 1.0
        return self

    def _raw_forecast(self, X: pd.DataFrame) -> np.ndarray:
        col, h = self.params["ret_col"], self.params["horizon"]
        dates = X.index.get_level_values("date")
        symbols = X.index.get_level_values("symbol")
        out = np.full(len(X), np.nan)
        known = self._returns_by_symbol(X, col)
        for sym, st in self._train.items():
            mask = np.asarray(symbols == sym)
            if not mask.any():
                continue
            r_all = pd.concat([st["returns"], known.get(sym, pd.Series(dtype=float))]).sort_index()
            r_all = r_all[~r_all.index.duplicated(keep="first")]  # eğitim getirisi önceliklidir (aynı gün aynı değer)
            end = pd.Timestamp(dates[mask].max())
            s2 = _filter_variance(r_all, st["omega"], st["alpha"], st["beta"], st["s2_init"], end)
            persistence = st["alpha"] + st["beta"]
            long_run = st["omega"] / (1 - persistence) if persistence < 1 and st["omega"] > 0 else None
            s2_next = s2.reindex(pd.DatetimeIndex(dates[mask])).to_numpy()  # d kapanışına kadar bilgiyle d+1 varyansı
            if long_run is None:
                avg = s2_next
            else:  # k=1..h adım ortalama beklenen varyans: v̄ + p^(k-1) (s2' - v̄), ortalaması kapalı formda
                avg = long_run + ((1 - persistence**h) / (h * (1 - persistence))) * (s2_next - long_run)
            out[mask] = np.sqrt(np.maximum(avg, 0.0)) / 100.0
        return out

    def predict(self, X):
        return self._raw_forecast(X) * self.scale_
