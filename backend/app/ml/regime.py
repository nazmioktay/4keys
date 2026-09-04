"""Piyasa rejimi tespiti — Gaussian Mixture Model (GMM) tabanlı.

Kullanıcı Markov Rejim Değişim Modelleri'ni (Markov Regime Switching)
önerdi: piyasanın "düşük volatiliteli yükseliş" / "yüksek volatiliteli
düşüş" gibi farklı durumlar (rejimler) arasında geçiş yaptığını
varsayıp, önce rejimi tespit edip sonra rejime özel modeller eğitmek.

Tam bir Markov-Switching modeli (`statsmodels.tsa.regime_switching`)
YENİ bir bilimsel kütüphane bağımlılığı gerektirdiğinden, kullanıcının
onayıyla bunun yerine `scikit-learn`in ZATEN kurulu olan
`GaussianMixture`'ı kullanılıyor — tam bir HMM kadar zaman-serisi-
duyarlı (geçiş olasılıkları modellemez) değil, ama volatilite+trend
uzayında pratikte benzer, yorumlanabilir kümeler ("rejimler") bulur ve
ek bağımlılık/indirme gerektirmez.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from app.exchanges.base import Exchange
from sklearn.mixture import GaussianMixture

DEFAULT_REGIME_MODEL_PATH = Path(__file__).parent / "artifacts" / "regime_model.joblib"

REGIME_FEATURE_COLUMNS = ["regime_volatility", "regime_trend"]


def compute_regime_features(ohlcv: pd.DataFrame, vol_window: int = 20, trend_window: int = 20) -> pd.DataFrame:
    """Rejim tespiti için BİLİNÇLİ OLARAK yalnızca iki yorumlanabilir eksen:

    - `regime_volatility`: log-getirinin kayan std'si (piyasa ne kadar
      çalkantılı)
    - `regime_trend`: kayan ortalama log-getiri (piyasa yukarı mı aşağı
      mı yönlü)

    Yüksek boyutlu bir özellik uzayında kümeleme yapmak yerine bu ikisiyle
    sınırlı tutuluyor — amaç GMM'nin bulduğu kümelerin "düşük volatiliteli
    yükseliş" gibi İNSAN TARAFINDAN YORUMLANABİLİR rejimlere karşılık
    gelmesi; ALL_FEATURE_COLUMNS'un tamamıyla kümeleme yapmak yorumlanamaz
    ve gürültüye daha duyarlı kümeler üretirdi.
    """
    log_return = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
    return pd.DataFrame(
        {
            "regime_volatility": log_return.rolling(vol_window).std(),
            "regime_trend": log_return.rolling(trend_window).mean(),
        }
    )


class RegimeModel:
    """GMM tabanlı rejim sınıflandırıcı. Rejim etiketleri, YORUMLANABİLİRLİK
    için ortalama volatiliteye göre sıralanır (0 = en düşük volatilite
    rejimi, `n_regimes - 1` = en yüksek volatilite rejimi) — GMM'nin ham
    küme indeksleri (`predict` çağrıları arası bile) rastgele/anlamsız
    sırada olabilir.
    """

    def __init__(self, n_regimes: int = 3, seed: int = 42) -> None:
        self.n_regimes = n_regimes
        self.seed = seed
        self._gmm: GaussianMixture | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._label_order: np.ndarray | None = None

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._feature_mean) / self._feature_std

    def fit(self, regime_features: pd.DataFrame) -> None:
        clean = regime_features.dropna()
        if len(clean) < self.n_regimes * 20:
            raise ValueError(f"Rejim modeli eğitimi için yeterli veri yok ({len(clean)} satır).")

        X = clean.to_numpy(dtype="float64")
        self._feature_mean = X.mean(axis=0)
        self._feature_std = X.std(axis=0)
        self._feature_std[self._feature_std == 0] = 1.0
        X_norm = self._normalize(X)

        self._gmm = GaussianMixture(n_components=self.n_regimes, random_state=self.seed, n_init=5)
        raw_labels = self._gmm.fit_predict(X_norm)

        # Kümeleri ortalama volatiliteye (regime_volatility, 0. kolon) göre
        # sırala — 0 = en düşük volatilite.
        cluster_vol = [
            float(X[raw_labels == c, 0].mean()) if (raw_labels == c).sum() > 0 else float("inf") for c in range(self.n_regimes)
        ]
        self._label_order = np.argsort(cluster_vol)

    def predict(self, regime_features: pd.DataFrame) -> pd.Series:
        """Döner: her satır için 0..n_regimes-1 rejim etiketi (0=en düşük
        volatilite). Yeterli geçmişi olmayan (warm-up) satırlar -1 ile
        işaretlenir."""
        self._require_fitted()
        result = pd.Series(-1, index=regime_features.index, dtype="int64")
        valid = regime_features.dropna()
        if valid.empty:
            return result

        X = valid.to_numpy(dtype="float64")
        X_norm = self._normalize(X)
        raw_labels = self._gmm.predict(X_norm)
        remap = {int(old): new for new, old in enumerate(self._label_order)}
        mapped = np.array([remap[int(label)] for label in raw_labels], dtype="int64")
        result.loc[valid.index] = mapped
        return result

    def _require_fitted(self) -> None:
        if self._gmm is None:
            raise RuntimeError("Rejim modeli henüz eğitilmedi. Önce fit() veya load() çağırın.")

    def save(self, path: Path = DEFAULT_REGIME_MODEL_PATH) -> None:
        self._require_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "n_regimes": self.n_regimes,
                "seed": self.seed,
                "gmm": self._gmm,
                "feature_mean": self._feature_mean,
                "feature_std": self._feature_std,
                "label_order": self._label_order,
            },
            path,
        )

    @classmethod
    def load_from(cls, path: Path = DEFAULT_REGIME_MODEL_PATH) -> "RegimeModel":
        data = joblib.load(path)
        model = cls(n_regimes=data["n_regimes"], seed=data["seed"])
        model._gmm = data["gmm"]
        model._feature_mean = data["feature_mean"]
        model._feature_std = data["feature_std"]
        model._label_order = data["label_order"]
        return model


@dataclass
class RegimeSummary:
    regime: int
    samples: int
    mean_volatility: float
    mean_trend: float


def fit_regime_model(
    exchange: Exchange, symbols: list[str], timeframe: str, lookback: int, n_regimes: int = 3
) -> tuple[RegimeModel, list[RegimeSummary]]:
    """Tüm sembollerin (BTC-öncelikli eğitim evreni) volatilite/trend
    özelliklerini BİRLEŞTİREREK (pooled) tek, paylaşılan bir rejim modeli
    eğitir — rejimler sembole değil piyasaya özgü bir kavram olduğundan.

    Kayan pencereli özellikler (`compute_regime_features`) HER SEMBOL
    kendi kapanış serisinde AYRI AYRI hesaplanır (birleştirmeden ÖNCE) —
    aksi halde semboller arası pencere sızıntısı olurdu.
    """
    per_symbol_features: list[pd.DataFrame] = []
    for symbol in symbols:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, lookback)
        if len(ohlcv) < 60:
            continue
        per_symbol_features.append(compute_regime_features(ohlcv))

    if not per_symbol_features:
        raise ValueError("Rejim modeli eğitimi için yeterli veri yok.")

    combined = pd.concat(per_symbol_features, ignore_index=True)
    model = RegimeModel(n_regimes=n_regimes)
    model.fit(combined)

    labels = model.predict(combined)
    summaries = []
    clean = combined.dropna()
    clean_labels = labels.loc[clean.index]
    for r in range(n_regimes):
        mask = clean_labels == r
        summaries.append(
            RegimeSummary(
                regime=r,
                samples=int(mask.sum()),
                mean_volatility=float(clean.loc[mask, "regime_volatility"].mean()) if mask.sum() > 0 else 0.0,
                mean_trend=float(clean.loc[mask, "regime_trend"].mean()) if mask.sum() > 0 else 0.0,
            )
        )
    return model, summaries


def build_regime_labeled_dataset(
    exchange: Exchange,
    symbols: list[str],
    timeframe: str,
    lookback: int,
    regime_model: RegimeModel,
    horizon: int = 5,
    threshold_pct: float = 1.0,
    labeling_method: str = "threshold",
    take_profit_pct: float = 2.0,
    stop_loss_pct: float = 2.0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Hibrit rejim+ML yaklaşımı için: XGBoost'un standart eğitim setini
    (`app.ml.dataset._build_symbol_frames` ile AYNI özellik/etiketleme
    işlem hattı) her satıra bir rejim etiketi ekleyerek kurar.

    Rejim özellikleri (`compute_regime_features`), `_build_symbol_frames`
    henüz sembolleri BİRLEŞTİRMEDEN önce döndürdüğü PER-SYMBOL çerçeveler
    üzerinde hesaplanır — semboller arası pencere sızıntısını önlemek
    için (bkz. `fit_regime_model` docstring'i).

    Döner: (X, y, regime, time_frac) — warm-up nedeniyle rejim etiketi
    olmayan (-1) satırlar ELENİR.
    """
    from .dataset import _build_symbol_frames
    from .features import ALL_FEATURE_COLUMNS

    frames = _build_symbol_frames(
        exchange, symbols, timeframe, lookback, horizon, threshold_pct, labeling_method, take_profit_pct, stop_loss_pct
    )
    if not frames:
        return (
            pd.DataFrame(columns=ALL_FEATURE_COLUMNS),
            pd.Series(dtype="float64"),
            pd.Series(dtype="int64"),
            pd.Series(dtype="float64"),
        )

    labeled_frames = []
    for frame in frames:
        regime_feats = compute_regime_features(pd.DataFrame({"close": frame["close"].to_numpy()}))
        frame = frame.reset_index(drop=True).copy()
        frame["regime"] = regime_model.predict(regime_feats).to_numpy()
        labeled_frames.append(frame)

    combined = pd.concat(labeled_frames, ignore_index=True)
    combined = combined[combined["regime"] >= 0].reset_index(drop=True)

    X = combined[ALL_FEATURE_COLUMNS]
    y = combined["label"]
    regime = combined["regime"]
    time_frac = combined["time_frac"]
    return X, y, regime, time_frac
