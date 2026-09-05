import numpy as np
import pandas as pd

from app.ml.multi_timeframe_features import (
    MULTI_TIMEFRAME_FEATURE_COLUMNS,
    _resample_ohlcv,
    compute_multi_timeframe_features,
)


def _flat_ohlcv(n: int, price: float = 100.0, freq: str = "1h") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    return pd.DataFrame(
        {"timestamp": idx, "open": price, "high": price + 0.1, "low": price - 0.1, "close": price, "volume": 1000.0}
    )


def test_resample_ohlcv_labels_bucket_by_close_boundary():
    """KRİTİK: her üst-TF bar, o barın TAMAMEN KAPANDIĞI zamanla
    etiketlenmeli — [00:00,04:00) aralığı "04:00" etiketini almalı, "00:00"
    DEĞİL. Bu, `merge_asof(direction="backward")` ile geriye dönük
    eşlemenin GELECEĞE BAKMADIĞINI garanti eden temel özellik."""
    ohlcv = _flat_ohlcv(8)  # 00:00..07:00, 8 saatlik bar
    resampled = _resample_ohlcv(ohlcv, "4h")

    assert len(resampled) == 2
    assert resampled["timestamp"].iloc[0] == pd.Timestamp("2024-01-01 04:00:00")
    assert resampled["timestamp"].iloc[1] == pd.Timestamp("2024-01-01 08:00:00")


def test_compute_multi_timeframe_features_shape_and_columns():
    ohlcv = _flat_ohlcv(24 * 90)  # 90 gün, 4h/1d ısınması için yeterli
    result = compute_multi_timeframe_features(ohlcv)

    assert len(result) == len(ohlcv)
    assert list(result.columns) == MULTI_TIMEFRAME_FEATURE_COLUMNS
    # yeterli veri var -> en azından SON kısımda NaN olmamalı (ısınma sonrası)
    assert not result.iloc[-1].isna().any()


def test_compute_multi_timeframe_features_nan_when_insufficient_history():
    ohlcv = _flat_ohlcv(10)  # çok kısa, ısınma için yetersiz
    result = compute_multi_timeframe_features(ohlcv)
    assert result.isna().all().all()


def test_compute_multi_timeframe_features_no_lookahead_at_bucket_boundary():
    """Fiyat, tam bir 4h bucket sınırında (saat 200, 200 % 4 == 0) SERT bir
    sıçrama yapıyor. Bu sıçramanın etkisi, o bucket TAMAMEN KAPANMADAN
    (yani sıçramadan sonraki 4 saat boyunca) `htf_4h_ema_gap`'e YANSIMAMALI
    — aksi halde gelecek bilgisi (look-ahead bias) sızıyor demektir."""
    n = 24 * 60  # 60 gün
    jump_hour = 24 * 30  # 30. günün başı, kesinlikle bir 4h sınırı (30*24 % 4 == 0)
    assert jump_hour % 4 == 0

    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = np.full(n, 100.0)
    close[jump_hour:] = 200.0  # kalıcı, sert bir sıçrama
    ohlcv = pd.DataFrame(
        {"timestamp": idx, "open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 1000.0}
    )

    result = compute_multi_timeframe_features(ohlcv)

    # sıçramadan HEMEN ÖNCEKİ değer (referans, sıçrama öncesi rejim)
    pre_jump_value = result["htf_4h_ema_gap"].iloc[jump_hour - 1]
    # sıçramayı İÇEREN bucket [jump_hour, jump_hour+4) HENÜZ KAPANMADI ->
    # bu 4 saat boyunca hâlâ ESKİ (sıçrama öncesi) değeri görmeli
    for h in range(jump_hour, jump_hour + 4):
        assert result["htf_4h_ema_gap"].iloc[h] == pre_jump_value, f"saat {h}'de erken sızıntı (look-ahead)"

    # bucket kapandıktan (jump_hour+4) SONRA artık sıçramayı yansıtmaya BAŞLAMALI
    # (EMA gecikmeli tepki verir ama pre_jump_value'dan FARKLI olmalı)
    after_close_value = result["htf_4h_ema_gap"].iloc[jump_hour + 4]
    assert after_close_value != pre_jump_value
