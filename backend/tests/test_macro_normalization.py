import numpy as np
import pandas as pd

from app.ml.macro_features import merge_macro_features, normalized_macro_history


def _history(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=n, freq="6h", tz="UTC"),
            "vix": 20 + rng.normal(0, 2, n).cumsum(),
            "funding_rate_btc": rng.normal(0, 0.0005, n),
        }
    )


def test_expanding_zscore_has_no_lookahead():
    full = _history(200)
    early = normalized_macro_history(full.iloc[:80])
    later = normalized_macro_history(full)
    # geleceğe ait anlık görüntüler eklenince geçmiş değerler DEĞİŞMEMELİ
    pd.testing.assert_series_equal(early["macro_vix_norm"], later["macro_vix_norm"].iloc[:80], check_names=False)


def test_extreme_future_value_does_not_shift_past_features():
    history = _history(120)
    shocked = pd.concat(
        [history, pd.DataFrame({"time": [history["time"].iloc[-1] + pd.Timedelta(hours=6)], "vix": [500.0], "funding_rate_btc": [0.0]})],
        ignore_index=True,
    )
    bars = pd.DataFrame({"timestamp": pd.date_range("2025-01-10", periods=50, freq="1h").astype("int64")})
    before = merge_macro_features(bars, history)["macro_vix_norm"]
    after = merge_macro_features(bars, shocked)["macro_vix_norm"]
    pd.testing.assert_series_equal(before, after)


def test_warmup_period_is_nan_then_finite():
    norm = normalized_macro_history(_history(60))["macro_vix_norm"]
    assert norm.iloc[:19].isna().all()
    assert np.isfinite(norm.iloc[30:]).all()
