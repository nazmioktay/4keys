from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import Prediction

DEFAULT_LSTM_MODEL_PATH = Path(__file__).parent / "artifacts" / "lstm_model.pt"

_LABEL_TO_DIRECTION = {1: "long", -1: "short", 0: "neutral"}


class _LSTMNet(nn.Module):
    """Çok katmanlı, dropout'lu LSTM sınıflandırıcı — rehberin "LSTM dropout
    ile aşırı uyum engellenir" (2.4 Overfitting) önerisinin karşılığı."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, num_classes: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]  # (batch, hidden_size) — en üst katmanın son zaman adımı
        return self.fc(self.dropout(last_hidden))


@dataclass
class LSTMTrainingReport:
    epochs_run: int
    final_train_loss: float
    final_train_accuracy: float
    best_val_loss: float | None = None
    stopped_early: bool = False


class LSTMSignalModel:
    """OHLCV özellik sekanslarından yön (long/short/neutral) tahmini yapan
    LSTM (Long Short-Term Memory) tabanlı model — rehberin Faz B'si.

    XGBoost (Faz A) her barı BAĞIMSIZ bir satır olarak görürken, LSTM son
    `seq_len` barın özellik vektörünü SIRAYLA okur ve önceki adımlardan
    öğrendiklerini bir gizli duruma (hidden state) taşır — sekans/zaman
    örüntülerini yakalamak için tasarlanmıştır (bkz. rehber tablosu,
    "Güçlü olduğu alan: Sekans ve zaman örüntüleri").

    Girdi şekli: (n_örnek, seq_len, n_özellik) — bkz. `app.ml.sequence_dataset`.
    """

    def __init__(self, seq_len: int = 20, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3) -> None:
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self._net: _LSTMNet | None = None
        self._is_fitted = False
        self.classes_: np.ndarray | None = None
        self._label_to_idx: dict[float, int] = {}
        self._idx_to_label: dict[int, float] = {}
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._n_features: int | None = None

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._feature_mean) / self._feature_std

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
    ) -> LSTMTrainingReport:
        """`X`: şekil (n, seq_len, n_özellik), `y`: şekil (n,) etiket dizisi.

        `weight_decay` (Adam'ın L2 regularizasyonu) ve LSTM dropout'u
        birlikte, rehberin "Regularization: LSTM dropout ile aşırı uyum
        engellenir" önerisini karşılar.

        `X_val`/`y_val` verilirse (kronolojik olarak eğitim setinin İÇİNDE,
        gerçek out-of-sample holdout'tan AYRI bir doğrulama dilimi — bkz.
        `app.ml.train.train_lstm_signal_model`), erken durdurma (early
        stopping) devreye girer: doğrulama kaybı `patience` epoch boyunca
        iyileşmezse eğitim durur ve EN İYİ doğrulama kaybına sahip ağırlıklar
        geri yüklenir. Bu, sabit 30 epoch'un veri setinin ezberlenmeye
        başladığı noktayı aşmasını (aşırı uyum) önlemeyi hedefler.
        `max_grad_norm` ile gradyan kırpma (gradient clipping) her zaman
        uygulanır — küçük, gürültülü veri setlerinde ani büyük güncellemelerin
        (ve dolayısıyla ezberlemenin) önüne geçer.
        """
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
        self._net = _LSTMNet(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=len(self.classes_),
            dropout=self.dropout,
        )

        # `torch.tensor(...)` her zaman kopyalar; `torch.from_numpy` bu
        # ölçekte (10K mum × ~20 sembol) gereksiz bir tam kopyayı (yüzlerce
        # MB) önlemek için tercih edilir — bkz. bellek notu yukarıda.
        X_tensor = torch.from_numpy(np.ascontiguousarray(X_norm, dtype=np.float32))
        y_tensor = torch.from_numpy(np.ascontiguousarray(y_idx, dtype=np.int64))
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

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
        return LSTMTrainingReport(
            epochs_run=epochs_run,
            final_train_loss=final_loss,
            final_train_accuracy=final_acc,
            best_val_loss=best_val_loss if has_val else None,
            stopped_early=stopped_early,
        )

    def _require_fitted(self) -> None:
        if not self._is_fitted or self._net is None:
            raise RuntimeError("LSTM modeli henüz eğitilmedi. Önce fit() veya load() çağırın.")

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

    def save(self, path: Path = DEFAULT_LSTM_MODEL_PATH) -> None:
        self._require_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "seq_len": self.seq_len,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "n_features": self._n_features,
                "classes": self.classes_,
                "feature_mean": self._feature_mean,
                "feature_std": self._feature_std,
            },
            path,
        )

    def load(self, path: Path = DEFAULT_LSTM_MODEL_PATH) -> None:
        checkpoint = torch.load(path, weights_only=False)
        self.seq_len = checkpoint["seq_len"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]
        self.classes_ = checkpoint["classes"]
        self._label_to_idx = {label: i for i, label in enumerate(self.classes_)}
        self._idx_to_label = {i: label for label, i in self._label_to_idx.items()}
        self._feature_mean = checkpoint["feature_mean"]
        self._feature_std = checkpoint["feature_std"]
        self._n_features = checkpoint["n_features"]

        self._net = _LSTMNet(
            input_size=checkpoint["n_features"],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=len(self.classes_),
            dropout=self.dropout,
        )
        self._net.load_state_dict(checkpoint["state_dict"])
        self._net.eval()
        self._is_fitted = True

    @classmethod
    def load_from(cls, path: Path = DEFAULT_LSTM_MODEL_PATH) -> "LSTMSignalModel":
        model = cls()
        model.load(path)
        return model
