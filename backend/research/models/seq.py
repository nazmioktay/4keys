"""Mevcut sekans modelleri için panel uyumlu sarmalayıcılar: `app.ml.lstm_model.LSTMSignalModel` ve
`app.ml.patchtst_model.PatchTSTSignalModel`.

Panel uyumu: X (date, symbol) MultiIndex'li; her sembolün satırları tarih sırasıyla kayan `seq_len` pencerelere çevrilir
(pencere sonundaki satırın etiketi). Tahminde, pencereyi tamamlamak için EĞİTİMDEN kalan son `seq_len-1` satır sembol
başına saklanır (geçmiş veri → ileri bilgi yok). İKİLİ sınıflandırma hedefi (y ∈ {0,1}) gerekir; P(y=1) =
tahmin sınıfı 1 ise güven, değilse 1−güven. `sample_weight` DESTEKLENMEZ (mevcut modellerde yok; yok sayılır — belgelenmiş
sınırlama). torch gerekir (içe aktarma tembeldir).

Bilinen tutarsızlık (sızıntı DEĞİL): tahminde pencere, eğitimin son `seq_len-1` satırı + test satırlarından kurulur; purge/embargo boşluğu
yüzünden pencere takvimsel olarak bitişik olmayan günleri kapsayabilir (yalnızca GEÇMİŞ veri; eğitim penceresiyle biçimsel fark)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseModel
from .registry import register_model


class _SeqPanel(BaseModel):
    task = "classification"
    supports_sample_weight = False
    _impl = ""  # "lstm" | "patchtst"

    def __init__(self, seed: int = 0, seq_len: int = 20, epochs: int = 8, **params):
        super().__init__(seed, seq_len=seq_len, epochs=epochs, **params)

    def _build(self):
        if self._impl == "lstm":
            from app.ml.lstm_model import LSTMSignalModel

            return LSTMSignalModel(seq_len=self.params["seq_len"], hidden_size=self.params.get("hidden_size", 32), num_layers=1, dropout=0.2)
        from app.ml.patchtst_model import PatchTSTSignalModel

        return PatchTSTSignalModel(seq_len=self.params["seq_len"], patch_len=self.params.get("patch_len", 5))

    @staticmethod
    def _per_symbol(frame: pd.DataFrame):
        for sym, grp in frame.groupby(level="symbol", sort=True):
            yield sym, grp.droplevel("symbol").sort_index()

    def _windows(self, values: np.ndarray) -> np.ndarray:
        s = self.params["seq_len"]
        if len(values) < s:
            return np.empty((0, s, values.shape[1]), dtype="float32")
        w = np.lib.stride_tricks.sliding_window_view(values, s, axis=0)  # (n-s+1, f, s)
        return np.ascontiguousarray(np.moveaxis(w, -1, 1), dtype="float32")

    def fit(self, X, y, dates, sample_weight=None):
        import torch

        torch.set_num_threads(1)
        s = self.params["seq_len"]
        Xp = self._prep_fit(X)
        self._cols = list(Xp.columns)
        xs, ys = [], []
        self._hist: dict[str, pd.DataFrame] = {}
        yy = y.reindex(Xp.index)
        for sym, grp in self._per_symbol(Xp):
            labels = yy.xs(sym, level="symbol").sort_index().reindex(grp.index)
            win = self._windows(grp.to_numpy(dtype="float32"))
            lab = labels.to_numpy()[s - 1:]
            ok = np.isfinite(lab)
            xs.append(win[ok])
            ys.append((lab[ok] > 0.5).astype(int))
            self._hist[sym] = grp.iloc[-(s - 1):]
        Xw, yw = np.concatenate(xs), np.concatenate(ys)
        if len(np.unique(yw)) < 2:
            raise ValueError("sekans modeli: eğitim etiketinde tek sınıf var")
        self._m = self._build()
        self._m.fit(Xw, yw, epochs=self.params["epochs"], seed=self.seed)
        return self

    def predict(self, X):
        s = self.params["seq_len"]
        Xp = self._prep(X)[self._cols]
        out = pd.Series(np.nan, index=X.index)
        for sym, grp in self._per_symbol(Xp):
            hist = self._hist.get(sym)
            if hist is None:
                continue
            both = pd.concat([hist, grp])
            both = both[~both.index.duplicated(keep="last")].sort_index()
            win = self._windows(both.to_numpy(dtype="float32"))
            idx = both.index[s - 1:]
            if len(win) == 0:
                continue
            pred, conf = self._m.predict_batch(win)
            p1 = np.where(pred == 1, conf, 1.0 - conf)
            res = pd.Series(p1, index=idx).reindex(grp.index)
            out.loc[pd.MultiIndex.from_arrays([grp.index, [sym] * len(grp)], names=X.index.names)] = res.to_numpy()
        return out.reindex(X.index).to_numpy()


@register_model("lstm_seq")
class LSTMSeq(_SeqPanel):
    _impl = "lstm"


@register_model("patchtst_seq")
class PatchTSTSeq(_SeqPanel):
    _impl = "patchtst"
