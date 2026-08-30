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
