import numpy as np
import pandas as pd

LONG = 1
SHORT = -1
NEUTRAL = 0


def label_future_direction(
    close: pd.Series, horizon: int = 5, threshold_pct: float = 1.0
) -> pd.Series:
    """Her mum için `horizon` mum sonraki getiriye göre yön etiketi üretir.

    Getiri > +threshold_pct  -> LONG (1)
    Getiri < -threshold_pct  -> SHORT (-1)
    Aksi halde                -> NEUTRAL (0)

    Serinin son `horizon` satırı, gelecek bilinmediği için NaN'dır.
    """
    future_return = (close.shift(-horizon) / close - 1) * 100

    labels = pd.Series(NEUTRAL, index=close.index, dtype="float")
    labels[future_return > threshold_pct] = LONG
    labels[future_return < -threshold_pct] = SHORT
    labels[future_return.isna()] = float("nan")
    return labels


def triple_barrier_labels(
    ohlcv: pd.DataFrame,
    take_profit_pct: float | pd.Series | np.ndarray = 2.0,
    stop_loss_pct: float | pd.Series | np.ndarray = 2.0,
    max_horizon: int = 10,
) -> pd.Series:
    """"Triple-barrier" etiketleme (Lopez de Prado): sabit bir mum sayısı sonraki
    getiriye bakmak yerine, üç bariyerden hangisi ÖNCE tetiklenirse etiketi o belirler:

    - Üst bariyer (`entry * (1 + take_profit_pct/100)`) önce dokunulursa -> LONG (1)
      (fiyat önce yukarı yönde hedefe ulaştı, bu mumda long avantajlıydı)
    - Alt bariyer (`entry * (1 - stop_loss_pct/100)`) önce dokunulursa -> SHORT (-1)
    - `max_horizon` mum içinde hiçbiri dokunulmazsa (zaman bariyeri) -> NEUTRAL (0)

    Sabit-eşikli "N mum sonra ne oldu?" etiketlemesine göre gerçek işlem
    mantığını (kâr hedefi / stop / zaman aşımı) çok daha doğru yansıtır ve
    mum içi (high/low) hareketleri kullanır, sadece kapanışı değil.

    `take_profit_pct`/`stop_loss_pct` bir SKALER (tüm barlar için sabit
    yüzde) OLABİLECEĞİ GİBİ, `ohlcv` ile AYNI uzunlukta bir dizi/Series de
    olabilir — bu, HER BAR için FARKLI (ör. o barın ATR'sine göre
    ölçeklenen, bkz. `app.ml.dataset._compute_labels`'ın
    `"atr_triple_barrier"` yolu) bariyer genişliği tanımlamayı sağlar;
    böylece model, gerçek işlemde kullanılan volatilite-duyarlı ATR
    stop/hedef mesafesiyle AYNI mantıkla etiketlenmiş olur (sabit yüzdelik
    etiketleme ile gerçek ATR tabanlı çıkış arasındaki uyumsuzluğu giderir).

    Serinin son kısmı (max_horizon mum içinde veri sonuna gelen satırlar,
    hiçbir bariyere dokunmamışsa) NaN döner — bu satırlar için zaman
    bariyerine gerçekten ulaşılıp ulaşılmadığı bilinmiyor, etiketlenemez.
    """
    close = ohlcv["close"].to_numpy(dtype=float)
    high = ohlcv["high"].to_numpy(dtype=float)
    low = ohlcv["low"].to_numpy(dtype=float)
    n = len(close)
    tp_pct = np.full(n, take_profit_pct, dtype=float) if np.isscalar(take_profit_pct) else np.asarray(take_profit_pct, dtype=float)
    sl_pct = np.full(n, stop_loss_pct, dtype=float) if np.isscalar(stop_loss_pct) else np.asarray(stop_loss_pct, dtype=float)
    labels = np.full(n, np.nan)

    for i in range(n):
        entry = close[i]
        upper = entry * (1 + tp_pct[i] / 100)
        lower = entry * (1 - sl_pct[i] / 100)
        window_end = min(i + 1 + max_horizon, n)

        label = None
        for j in range(i + 1, window_end):
            hit_up = high[j] >= upper
            hit_down = low[j] <= lower
            if hit_up and hit_down:
                # Aynı mumda ikisi de tetiklendi: hangi bariyer entry'ye daha
                # yakınsa muhafazakâr varsayımla o gerçekleşmiş kabul edilir.
                label = LONG if (upper - entry) <= (entry - lower) else SHORT
            elif hit_up:
                label = LONG
            elif hit_down:
                label = SHORT
            if label is not None:
                break

        if label is None:
            # Bariyerlerden hiçbiri dokunulmadı. Zaman bariyerine (max_horizon)
            # gerçekten ulaşıldıysa NEUTRAL; veri erken bittiyse (window_end <
            # i+1+max_horizon) bu satır hakkında karar veremeyiz, NaN kalır.
            if window_end == i + 1 + max_horizon:
                label = NEUTRAL

        if label is not None:
            labels[i] = label

    return pd.Series(labels, index=ohlcv.index)
