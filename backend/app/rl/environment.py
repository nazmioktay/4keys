"""Reinforcement Learning için basit, bağımlılıksız (gymnasium GEREKTİRMEYEN
ama onunla aynı `reset()`/`step()` sözleşmesini izleyen) bir alım-satım
ortamı.

Neden gymnasium'a bağımlı değil: bu aşamada amaç "hazırlık" — ortamın
kendisini ve veri/ödül tasarımını doğru kurmak. Gerçek bir ajan (PPO/DQN
vb.) eğitmek istendiğinde `stable-baselines3` + `gymnasium` eklenip bu
sınıf ince bir `gymnasium.Env` sarmalayıcısına (adapter) dönüştürülebilir
— tasarım buna hazır (aynı `reset`/`step` imzaları).

## Ezberlemeyi önleme tasarımı (bkz. XGBoost/LSTM'de öğrenilen dersler)
- **Rastgele başlangıç noktası** (`reset(seed=...)`): her epizod, verinin
  KENDİSİ değil, İÇİNDEKİ rastgele bir pencere ile başlar — ajanın tek bir
  sabit sırayı ezberlemesini zorlaştırır (bootstrap benzeri).
- **Kronolojik out-of-sample holdout**: `holdout_frac` (varsayılan %20)
  kadar en-yeni veri `reset(..., use_holdout=True)` DIŞINDA hiçbir zaman
  eğitim epizotlarına dahil edilmez — XGBoost/LSTM'deki aynı disiplin.
- **İşlem maliyeti** (`transaction_cost_pct`): pozisyon değişiminde küçük
  bir maliyet uygulanır — maliyetsiz bir ortamda ajan anlamsızca sık işlem
  yaparak gürültüyü "kâr" sanabilir (overfitting'in RL'deki bir biçimi).
"""

from dataclasses import dataclass, field

import numpy as np

FLAT, LONG, SHORT = 0, 1, 2
ACTIONS = (FLAT, LONG, SHORT)


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict = field(default_factory=dict)


class TradingEnv:
    """Tek pozisyonlu (flat/long/short), tek sembollü basit alım-satım ortamı.

    Gözlem (observation): o barın özellik vektörü + [güncel pozisyon
    yönü (-1/0/1), pozisyonun açık olduğu bar sayısı normalize edilmiş].

    Aksiyon: `FLAT` (pozisyon yoksa hiçbir şey yapma / varsa kapat),
    `LONG`, `SHORT`.

    Ödül: bir sonraki bara geçerken pozisyonun mark-to-market getirisi
    (yüzde, ondalık) EKSİ pozisyon değiştirilmişse işlem maliyeti.
    """

    def __init__(
        self,
        features: np.ndarray,
        close: np.ndarray,
        window_len: int = 200,
        holdout_frac: float = 0.2,
        transaction_cost_pct: float = 0.05,
    ) -> None:
        if len(features) != len(close):
            raise ValueError("features ve close aynı uzunlukta olmalı")
        if len(features) < window_len + 10:
            raise ValueError(f"yeterli veri yok ({len(features)} satır, window_len={window_len})")

        self.features = features
        self.close = close
        self.window_len = window_len
        self.transaction_cost_pct = transaction_cost_pct

        cutoff = int(len(features) * (1 - holdout_frac))
        # Eğitim epizotları YALNIZCA [0, cutoff) aralığından başlangıç seçebilir
        # ve pencereleri de bu aralığı AŞAMAZ — holdout'a hiçbir sızıntı olmaz.
        self._train_start_high = max(cutoff - window_len, 1)
        self._cutoff = cutoff

        self._rng = np.random.default_rng()
        self._position = FLAT
        self._start_idx = 0
        self._cursor = 0
        self._bars_in_position = 0

    @property
    def observation_dim(self) -> int:
        return self.features.shape[1] + 2

    def _observation(self) -> np.ndarray:
        feat = self.features[self._cursor]
        pos_dir = {FLAT: 0.0, LONG: 1.0, SHORT: -1.0}[self._position]
        bars_norm = min(self._bars_in_position / self.window_len, 1.0)
        return np.concatenate([feat, [pos_dir, bars_norm]]).astype("float32")

    def reset(self, seed: int | None = None, use_holdout: bool = False) -> np.ndarray:
        """Yeni bir epizod başlatır. `use_holdout=True` verilirse (yalnızca
        DEĞERLENDİRME için — asla eğitim rollout'larında kullanılmamalı),
        epizod holdout dilimindeki TEK sabit pencerede başlar (kronolojik,
        rastgele değil — gerçek "canlıda ne olurdu" testi)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if use_holdout:
            self._start_idx = self._cutoff
        else:
            self._start_idx = int(self._rng.integers(0, self._train_start_high))

        self._cursor = self._start_idx
        self._position = FLAT
        self._bars_in_position = 0
        return self._observation()

    def _max_cursor(self, use_holdout: bool) -> int:
        return len(self.features) - 2 if use_holdout else min(self._start_idx + self.window_len, len(self.features) - 2)

    def step(self, action: int, use_holdout: bool = False) -> StepResult:
        if action not in ACTIONS:
            raise ValueError(f"geçersiz aksiyon: {action}")

        price_now = self.close[self._cursor]
        price_next = self.close[self._cursor + 1]
        raw_return = (price_next / price_now) - 1.0

        reward = 0.0
        if self._position == LONG:
            reward = raw_return
        elif self._position == SHORT:
            reward = -raw_return

        if action != self._position:
            reward -= self.transaction_cost_pct / 100.0
            self._bars_in_position = 0
        else:
            self._bars_in_position += 1

        self._position = action
        self._cursor += 1

        max_cursor = self._max_cursor(use_holdout)
        terminated = False
        truncated = self._cursor >= max_cursor
        info = {"price": float(price_next), "position": self._position}
        return StepResult(self._observation(), float(reward), terminated, truncated, info)
