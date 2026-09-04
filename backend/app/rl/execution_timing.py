"""Hurst-tabanlı işlem zamanlaması analizi.

Klasik "optimal execution" RL'i (emri parçalara bölerek piyasa etkisini
azaltma) bizim verimizle (order-book derinliği yok) ve ölçeğimizle
(çeyrek Kelly boyutları, BTC/USDT'nin likiditesine göre ihmal edilebilir)
anlamlı bir sonuç veremez — piyasa etkisi modeli olmadan "bölmek daha
iyi" sonucu yapay/yanıltıcı olurdu (bkz. sohbet).

Bunun yerine, ölçülebilir ve verimizle gerçekten test edilebilir bir
hipotez: sinyal anındaki Hurst üsteli (H) DÜŞÜKSE (< 0.5, ortalamaya-dönüş
rejimi), hemen yürütmek yerine birkaç mum GECİKTİRMEK ortalama olarak
daha iyi bir fiyat verir mi? H YÜKSEKSE (trend-devamlılığı) ise
geciktirmenin fiyatı daha da kötüleştirmesi (trend devam ettiği için)
beklenir.

Metodoloji: sabit bir gecikme (`delay_bars`, ÖNCEDEN belirlenmiş, veriye
göre optimize edilmemiş — "en iyi gecikmeyi ara" gibi bir look-ahead
yanlılığından kaçınmak için) ile karşılaştırma yapılır; H değerine göre
üç eşit gruba (düşük/orta/yüksek) ayrılıp her grupta ortalama "gecikmeli
fiyat / anlık fiyat" farkı raporlanır. Hipotez doğruysa düşük-H grubunda
BUY için bu fark negatif (daha ucuz), yüksek-H grubunda pozitif (daha
pahalı) olmalıdır.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from app.exchanges.base import Exchange

from app.ml.features import build_features


@dataclass
class HurstBucketResult:
    label: str
    hurst_range: str
    samples: int
    mean_delayed_slippage_pct: float


@dataclass
class HurstTimingReport:
    symbol: str
    delay_bars: int
    total_samples: int
    buckets: list[HurstBucketResult]


def analyze_hurst_execution_timing(
    exchange: Exchange,
    symbol: str,
    timeframe: str,
    lookback: int,
    delay_bars: int = 3,
    low_threshold: float = 0.45,
    high_threshold: float = 0.55,
) -> HurstTimingReport:
    """`side="buy"` varsayımıyla: her bar `i` için "anlık" fiyat
    `close[i+1]` (sinyal onaylandıktan bir sonraki bar — gerçekçi bir
    dolum varsayımı), "gecikmeli" fiyat `close[i+1+delay_bars]`. Fark
    yüzdesi = (gecikmeli - anlık) / anlık * 100 — BUY için NEGATİF değer
    "geciktirmek daha ucuza mal oldu" demektir (iyi), POZİTİF ise
    "geciktirmek daha pahalıya mal oldu" demektir (kötü).

    `hurst_exponent[i]` (karar anında ZATEN bilinen, geleceğe bakmayan bir
    değer) üç gruba ayrılır (`low_threshold`/`high_threshold` ile) ve her
    grup için ortalama fark raporlanır. Look-ahead yanlılığından kaçınmak
    için `delay_bars` SABİTTİR — veriye bakarak "en iyi" gecikme aranmaz.
    """
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, lookback)
    features = build_features(ohlcv)
    close = ohlcv["close"].to_numpy(dtype="float64")
    hurst = features["hurst_exponent"].to_numpy(dtype="float64")

    n = len(close)
    max_i = n - delay_bars - 2  # i+1+delay_bars sınırın içinde kalmalı
    if max_i < 1:
        return HurstTimingReport(symbol=symbol, delay_bars=delay_bars, total_samples=0, buckets=[])

    immediate = close[1 : max_i + 1]
    delayed = close[1 + delay_bars : max_i + 1 + delay_bars]
    hurst_at_decision = hurst[0:max_i]

    slippage_pct = (delayed - immediate) / immediate * 100

    valid = ~(np.isnan(hurst_at_decision) | np.isnan(slippage_pct))
    hurst_at_decision = hurst_at_decision[valid]
    slippage_pct = slippage_pct[valid]

    buckets: list[HurstBucketResult] = []
    masks = {
        "low": (hurst_at_decision < low_threshold, f"H < {low_threshold}"),
        "mid": (
            (hurst_at_decision >= low_threshold) & (hurst_at_decision <= high_threshold),
            f"{low_threshold} <= H <= {high_threshold}",
        ),
        "high": (hurst_at_decision > high_threshold, f"H > {high_threshold}"),
    }
    for label, (mask, range_label) in masks.items():
        samples = int(mask.sum())
        mean_slip = float(slippage_pct[mask].mean()) if samples > 0 else 0.0
        buckets.append(HurstBucketResult(label=label, hurst_range=range_label, samples=samples, mean_delayed_slippage_pct=mean_slip))

    return HurstTimingReport(symbol=symbol, delay_bars=delay_bars, total_samples=int(valid.sum()), buckets=buckets)
