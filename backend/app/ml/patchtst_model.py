from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import Prediction

DEFAULT_PATCHTST_MODEL_PATH = Path(__file__).parent / "artifacts" / "patchtst_model.pt"

_LABEL_TO_DIRECTION = {1: "long", -1: "short", 0: "neutral"}


class _PatchTSTNet(nn.Module):
    """PatchTST'ten (Nie ve ark., 2023) ESİNLENİLMİŞ, BASİTLEŞTİRİLMİŞ bir
    patch-tabanlı Transformer sınıflandırıcı.

    Orijinal PatchTST kanal-bağımsızdır (her özellik/kanal ayrı ayrı,
    ağırlık paylaşımıyla işlenir). Burada, LSTM ile aynı çok-değişkenli
    girdi arayüzünü (seq_len, n_özellik) koruyabilmek için BİLİNÇLİ OLARAK
    basitleştirildi: her patch, o zaman aralığındaki TÜM özellikleri
    birlikte (kanal-karışık) düzleştirip tek bir token'a projekte eder.
    Bu, orijinal makaledeki "kanal-bağımsız" tasarımın tam bir uygulaması
    DEĞİLDİR — LSTM'in "sekansı sırayla, adım adım okuma" varsayımı yerine
    "sekansı örtüşmeyen zaman dilimlerine (patch) bölüp dikkat mekanizmasıyla
    birlikte değerlendirme" fikrini test etmek için yeterli, ucuz bir
    yaklaşımdır.
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        patch_len: int = 5,
        stride: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = max((seq_len - patch_len) // stride + 1, 1)

        self.patch_proj = nn.Linear(patch_len * input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_özellik) -> patch'ler: (batch, num_patches, n_özellik, patch_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        batch, num_patches, n_features, patch_len = patches.shape
        patches = patches.reshape(batch, num_patches, n_features * patch_len)
        tokens = self.patch_proj(patches) + self.pos_embedding[:, :num_patches, :]
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)  # patch'ler arası ortalama havuzlama
        return self.fc(self.dropout(pooled))


@dataclass
class PatchTSTTrainingReport:
    epochs_run: int
    final_train_loss: float
    final_train_accuracy: float
    best_val_loss: float | None = None
    stopped_early: bool = False


class PatchTSTSignalModel:
    """LSTM'e (`app.ml.lstm_model.LSTMSignalModel`) alternatif, patch-tabanlı
    Transformer mimarisiyle yön (long/short/neutral) tahmini yapan model.

    BTC-only sınamalarda LSTM'in hem lookback artırma hem de model
    kapasitesini küçültme ile ~%38-39 balanced_accuracy tavanına takılı
    kaldığı görüldü (bkz. README) — bu, LSTM'in kullanılan öznitelik
    setinden/etiketlerden daha fazlasını çıkaramadığına işaret ediyor.
    PatchTST tarzı dikkat mekanizması, LSTM'in sıralı/tekrarlayan
    varsayımından farklı bir örüntü ailesini yakalayabilir mi diye
    denemek için eklendi — otomatik olarak "daha iyi" varsayılmaz, aynı
    out-of-sample disipliniyle (holdout + erken durdurma + sınıf
    ağırlıklandırma) LSTM ile karşılaştırılmalıdır.

    Girdi şekli: (n_örnek, seq_len, n_özellik) — `LSTMSignalModel` ile
    AYNI, `app.ml.sequence_dataset` çıktısı doğrudan kullanılabilir.
    """

    def __init__(
        self,
        seq_len: int = 20,
        patch_len: int = 5,
        stride: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self._net: _PatchTSTNet | None = None
        self._is_fitted = False
        self.classes_: np.ndarray | None = None
        self._label_to_idx: dict[float, int] = {}
        self._idx_to_label: dict[int, float] = {}
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._n_features: int | None = None
        self.feature_columns: list[str] | None = None

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._feature_mean) / self._feature_std

    def _build_net(self, n_features: int, num_classes: int) -> _PatchTSTNet:
        return _PatchTSTNet(
            input_size=n_features,
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            num_classes=num_classes,
            dropout=self.dropout,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        patience: int = 5,
        max_grad_norm: float = 1.0,
    ) -> PatchTSTTrainingReport:
        """`X`: şekil (n, seq_len, n_özellik), `y`: şekil (n,) etiket dizisi.

        Erken durdurma, gradyan kırpma ve sınıf ağırlıklandırma —
        `LSTMSignalModel.fit` ile AYNI mantıkla, adil bir karşılaştırma
        için — burada da uygulanır (bkz. o dosyadaki notlar)."""
        self.classes_ = np.unique(y)
        self._label_to_idx = {label: i for i, label in enumerate(self.classes_)}
        self._idx_to_label = {i: label for label, i in self._label_to_idx.items()}
        y_idx = np.array([self._label_to_idx[v] for v in y])

        self._feature_mean = X.mean(axis=(0, 1), keepdims=True)
        self._feature_std = X.std(axis=(0, 1), keepdims=True)
        self._feature_std[self._feature_std == 0] = 1.0
        X_norm = self._normalize(X)

        n_features = X.shape[2]
        self._n_features = n_features
        self._net = self._build_net(n_features, len(self.classes_))

        X_tensor = torch.from_numpy(np.ascontiguousarray(X_norm, dtype=np.float32))
        y_tensor = torch.from_numpy(np.ascontiguousarray(y_idx, dtype=np.int64))
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        class_counts = np.bincount(y_idx, minlength=len(self.classes_)).astype("float64")
        class_counts[class_counts == 0] = 1.0
        class_weights = class_counts.sum() / (len(class_counts) * class_counts)
        weight_tensor = torch.from_numpy(class_weights.astype("float32"))
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        X_val_tensor = y_val_tensor = None
        if has_val:
            y_val_idx = np.array([self._label_to_idx.get(v, -1) for v in y_val])
            X_val_tensor = torch.from_numpy(np.ascontiguousarray(self._normalize(X_val), dtype=np.float32))
            y_val_tensor = torch.from_numpy(np.ascontiguousarray(y_val_idx, dtype=np.int64))

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        stopped_early = False

        final_loss = 0.0
        final_acc = 0.0
        epochs_run = 0
        for _epoch in range(epochs):
            self._net.train()
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = self._net(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), max_grad_norm)
                optimizer.step()

                total_loss += loss.item() * len(batch_y)
                correct += (logits.argmax(dim=1) == batch_y).sum().item()
                total += len(batch_y)
            final_loss = total_loss / max(total, 1)
            final_acc = correct / max(total, 1)
            epochs_run += 1

            if has_val:
                self._net.eval()
                with torch.no_grad():
                    val_logits = self._net(X_val_tensor)
                    val_loss = criterion(val_logits, y_val_tensor).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        stopped_early = True
                        break

        if has_val and best_state is not None:
            self._net.load_state_dict(best_state)

        self._is_fitted = True
        return PatchTSTTrainingReport(
            epochs_run=epochs_run,
            final_train_loss=final_loss,
            final_train_accuracy=final_acc,
            best_val_loss=best_val_loss if has_val else None,
            stopped_early=stopped_early,
        )

    def _require_fitted(self) -> None:
        if not self._is_fitted or self._net is None:
            raise RuntimeError("PatchTST modeli henüz eğitilmedi. Önce fit() veya load() çağırın.")

    def predict_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`X`: şekil (n, seq_len, n_özellik). Döner: (tahmin edilen etiket dizisi, güven dizisi)."""
        self._require_fitted()
        X_norm = self._normalize(X)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.tensor(X_norm, dtype=torch.float32))
            proba = torch.softmax(logits, dim=1).numpy()
        best_idx = np.argmax(proba, axis=1)
        predictions = np.array([self._idx_to_label[i] for i in best_idx])
        confidences = proba[np.arange(len(proba)), best_idx]
        return predictions, confidences

    def predict(self, sequence: np.ndarray) -> Prediction:
        """`sequence`: şekil (seq_len, n_özellik) — en son `seq_len` bar."""
        predictions, confidences = self.predict_batch(sequence[np.newaxis, :, :])
        label = int(predictions[0])
        return Prediction(direction=_LABEL_TO_DIRECTION[label], confidence=float(confidences[0]))

    def save(self, path: Path = DEFAULT_PATCHTST_MODEL_PATH) -> None:
        self._require_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "seq_len": self.seq_len,
                "patch_len": self.patch_len,
                "stride": self.stride,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "n_features": self._n_features,
                "classes": self.classes_,
                "feature_mean": self._feature_mean,
                "feature_std": self._feature_std,
                "feature_columns": self.feature_columns,
            },
            path,
        )

    def load(self, path: Path = DEFAULT_PATCHTST_MODEL_PATH) -> None:
        checkpoint = torch.load(path, weights_only=False)
        self.seq_len = checkpoint["seq_len"]
        self.patch_len = checkpoint["patch_len"]
        self.stride = checkpoint["stride"]
        self.d_model = checkpoint["d_model"]
        self.nhead = checkpoint["nhead"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]
        self.classes_ = checkpoint["classes"]
        self.feature_columns = checkpoint.get("feature_columns")
        self._label_to_idx = {label: i for i, label in enumerate(self.classes_)}
        self._idx_to_label = {i: label for label, i in self._label_to_idx.items()}
        self._feature_mean = checkpoint["feature_mean"]
        self._feature_std = checkpoint["feature_std"]
        self._n_features = checkpoint["n_features"]

        self._net = self._build_net(checkpoint["n_features"], len(self.classes_))
        self._net.load_state_dict(checkpoint["state_dict"])
        self._net.eval()
        self._is_fitted = True

    @classmethod
    def load_from(cls, path: Path = DEFAULT_PATCHTST_MODEL_PATH) -> "PatchTSTSignalModel":
        model = cls()
        model.load(path)
        return model
