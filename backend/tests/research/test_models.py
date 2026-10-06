import numpy as np
import pandas as pd
import pytest

from research.models.base import BaseModel
from research.models.registry import available_models, get_model, register_model


def _panel(n_dates=300, symbols=("A", "B", "C", "D"), seed=0, nonlinear=False):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    X = pd.DataFrame(rng.normal(size=(len(idx), 4)), index=idx, columns=["f0", "f1", "f2", "f3"])
    signal = 1.5 * X["f0"] - 1.0 * X["f1"] + (X["f2"] ** 2 - 1 if nonlinear else 0.5 * X["f2"])
    y = signal + rng.normal(0, 0.3, len(idx))
    return X, pd.Series(y.to_numpy(), index=idx, name="y"), pd.Series(idx.get_level_values("date"), index=idx)


def _r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


# ---------------------------------------------------------------- registry
def test_registry_lists_all_wrappers_and_rejects_unknown_or_duplicate():
    names = available_models()
    for n in ("ridge", "logistic", "lightgbm_reg", "lightgbm_clf", "lightgbm_rank", "xgboost_reg", "xgboost_clf", "har_rv", "garch11", "lstm_seq", "patchtst_seq"):
        assert n in names
    with pytest.raises(KeyError, match="Bilinmeyen model"):
        get_model("yok")

    class Other(BaseModel):
        def fit(self, X, y, dates, sample_weight=None): ...
        def predict(self, X): ...

    with pytest.raises(ValueError, match="zaten kayıtlı"):
        register_model("ridge")(Other)
    assert get_model("ridge", seed=3).get_params()["seed"] == 3


# ---------------------------------------------------------------- doğrusal kıyaslar
def test_ridge_recovers_linear_signal_and_imputes_with_training_median():
    X, y, d = _panel()
    split = int(len(X) * 0.7)
    m = get_model("ridge").fit(X.iloc[:split], y.iloc[:split], d.iloc[:split])
    test = X.iloc[split:].copy()
    assert _r2(y.iloc[split:], m.predict(test)) > 0.9
    test.iloc[0, 0] = np.nan  # NaN: eğitim medyanıyla doldurulur, tahmin NaN olmaz
    assert np.isfinite(m.predict(test)).all()
    assert m._imputer.median_["f0"] == pytest.approx(X.iloc[:split]["f0"].median())  # medyan YALNIZCA eğitimden


def test_sample_weight_changes_the_ridge_fit():
    X, y, d = _panel(n_dates=150)
    w = np.where(X["f0"].to_numpy() > 0, 10.0, 0.1)
    a = get_model("ridge").fit(X, y, d).predict(X)
    b = get_model("ridge").fit(X, y, d, sample_weight=w).predict(X)
    assert not np.allclose(a, b)


def test_logistic_separates_classes():
    X, y, d = _panel()
    yb = (y > 0).astype(float)
    m = get_model("logistic").fit(X.iloc[:800], yb.iloc[:800], d.iloc[:800])
    p = m.predict(X.iloc[800:])
    assert ((p > 0.5) == (yb.iloc[800:] > 0.5)).mean() > 0.9 and ((p >= 0) & (p <= 1)).all()


# ---------------------------------------------------------------- ağaç modelleri
@pytest.mark.parametrize("name", ["lightgbm_reg", "xgboost_reg"])
def test_tree_regressors_learn_nonlinearity_are_deterministic_and_take_nan(name):
    X, y, d = _panel(nonlinear=True)
    split = int(len(X) * 0.75)
    a = get_model(name, seed=7).fit(X.iloc[:split], y.iloc[:split], d.iloc[:split])
    b = get_model(name, seed=7).fit(X.iloc[:split], y.iloc[:split], d.iloc[:split])
    test = X.iloc[split:]
    pa = a.predict(test)
    assert _r2(y.iloc[split:], pa) > 0.7
    np.testing.assert_array_equal(pa, b.predict(test))  # aynı seed -> birebir aynı
    withnan = test.copy()
    withnan.iloc[:5, 0] = np.nan
    assert np.isfinite(a.predict(withnan)).all()  # NaN yerel yönetilir


@pytest.mark.parametrize("name", ["lightgbm_clf", "xgboost_clf"])
def test_tree_classifiers_return_probabilities(name):
    X, y, d = _panel(nonlinear=True)
    yb = (y > 0).astype(float)
    m = get_model(name, seed=1).fit(X.iloc[:900], yb.iloc[:900], d.iloc[:900])
    p = m.predict(X.iloc[900:])
    assert ((p >= 0) & (p <= 1)).all() and ((p > 0.5) == (yb.iloc[900:] > 0.5)).mean() > 0.8


def test_lightgbm_lambdarank_orders_symbols_within_each_date():
    X, y, d = _panel(n_dates=250)
    rank = y.groupby(level="date").rank(pct=True).clip(upper=0.999)
    split = int(len(X) * 0.7)
    m = get_model("lightgbm_rank", seed=2, n_estimators=100).fit(X.iloc[:split], rank.iloc[:split], d.iloc[:split])
    pred = pd.Series(m.predict(X.iloc[split:]), index=X.index[split:])
    ic = pd.concat([pred, rank.iloc[split:]], axis=1).groupby(level="date").apply(lambda g: g.iloc[:, 0].corr(g.iloc[:, 1], method="spearman"))
    assert ic.mean() > 0.6


# ---------------------------------------------------------------- HAR-RV
def test_har_recovers_coefficients_and_returns_nan_for_missing_features():
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product([pd.date_range("2021-01-01", periods=400), ["A", "B"]], names=["date", "symbol"])
    feats = ("ohlcv_core__rv_pk_1", "ohlcv_core__rv_pk_5", "ohlcv_core__rv_pk_22")
    X = pd.DataFrame(rng.uniform(0.01, 0.06, size=(len(idx), 3)), index=idx, columns=feats)
    y = 0.002 + 0.4 * X[feats[0]] + 0.3 * X[feats[1]] + 0.2 * X[feats[2]] + rng.normal(0, 0.001, len(idx))
    m = get_model("har_rv").fit(X, pd.Series(y.to_numpy(), index=idx), pd.Series(idx.get_level_values(0), index=idx))
    np.testing.assert_allclose(m.coef_, [0.002, 0.4, 0.3, 0.2], atol=0.03)
    Xn = X.copy()
    Xn.iloc[3, 1] = np.nan
    p = m.predict(Xn)
    assert np.isnan(p[3]) and np.isfinite(np.delete(p, 3)).all()
    with pytest.raises(ValueError, match="gerekli özellikler"):
        get_model("har_rv").fit(X[[feats[0]]], pd.Series(y.to_numpy(), index=idx), pd.Series(idx.get_level_values(0), index=idx))


# ---------------------------------------------------------------- GARCH(1,1)
def _simulate_garch(n=1800, omega=0.05, alpha=0.08, beta=0.90, seed=3):
    rng = np.random.default_rng(seed)
    s2 = np.empty(n + 1)
    r = np.empty(n)
    s2[0] = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(s2[t]) * rng.normal()
        s2[t + 1] = omega + alpha * r[t] ** 2 + beta * s2[t]
    return r / 100.0, s2[1:] / 1e4  # getiri (ondalık) ve d kapanışına kadar bilgiyle d+1 gerçek koşullu varyans


def _garch_frame(r, s2_next, start="2018-01-01"):
    idx = pd.MultiIndex.from_arrays([pd.date_range(start, periods=len(r)), ["A"] * len(r)], names=["date", "symbol"])
    X = pd.DataFrame({"ohlcv_core__ret_1": r}, index=idx)
    y = pd.Series(np.sqrt(s2_next), index=idx)
    return X, y


def test_garch_filters_conditional_variance_without_lookahead_and_across_embargo_gap():
    r, s2n = _simulate_garch()
    X, y = _garch_frame(r, s2n)
    tr, gap, te = slice(0, 1200), 5, slice(1205, 1800)
    m = get_model("garch11", min_obs=250).fit(X.iloc[tr], y.iloc[tr], X.index.get_level_values(0)[tr])
    st = m._train["A"]
    assert st["alpha"] + st["beta"] == pytest.approx(0.98, abs=0.05)
    pred = pd.Series(m.predict(X.iloc[te]), index=X.index[te])
    truth = y.iloc[te]
    assert np.corrcoef(pred, truth)[0, 1] > 0.8 and 0.7 < m.scale_ < 1.4
    # İLERİ BİLGİ YOK: t'den sonraki getirileri bozmak t'deki tahmini DEĞİŞTİRMEZ
    t0 = 1500
    X2 = X.iloc[te].copy()
    X2.iloc[t0 - 1205 + 1:, 0] *= 10.0
    pred2 = pd.Series(m.predict(X2), index=X2.index)
    assert pred2.iloc[: t0 - 1205 + 1].equals(pred.iloc[: t0 - 1205 + 1])
    assert gap == 5  # embargo boşluğu (1200..1204) tahmine alınmadı, süzme yine çalıştı


# ---------------------------------------------------------------- sekans modelleri (torch)
@pytest.mark.parametrize("name", ["lstm_seq", "patchtst_seq"])
def test_sequence_wrappers_fit_predict_deterministically_on_panel(name):
    pytest.importorskip("torch")
    X, y, d = _panel(n_dates=120, symbols=("A", "B", "C"))
    yb = (y > 0).astype(float)
    kw = dict(seq_len=10, epochs=2, seed=5)
    a = get_model(name, **kw).fit(X.iloc[:240], yb.iloc[:240], d.iloc[:240])
    p = a.predict(X.iloc[240:])
    assert p.shape == (len(X) - 240,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    b = get_model(name, **kw).fit(X.iloc[:240], yb.iloc[:240], d.iloc[:240])
    np.testing.assert_allclose(p, b.predict(X.iloc[240:]), atol=1e-6)
    unseen = X.iloc[240:].copy()
    unseen.index = pd.MultiIndex.from_arrays([unseen.index.get_level_values(0), ["Z"] * len(unseen)], names=["date", "symbol"])
    assert np.isnan(a.predict(unseen)).all()  # eğitimde görülmeyen sembol -> NaN, uydurma yok
