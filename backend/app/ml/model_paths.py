"""Ağır ML kütüphaneleri (özellikle `torch`) OLMADAN model dosya yollarına
erişim.

Ölçüldü: `import torch` TEK BAŞINA process RSS'ine ~460MB ekliyor (bkz.
README "backend'in boştaki belleği neden bu kadar yüksek" araştırması) —
ama "LSTM/PatchTST modeli hiç eğitilmiş mi", "hangi dosyaya bakmalıyım"
gibi sorular için yalnızca birer `Path` sabitine ihtiyaç var, torch'un
TAMAMINI yüklemeye gerek yok. `app.ml.lstm_model`/`app.ml.patchtst_model`
(gerçek model sınıflarını, dolayısıyla torch'u içeren modüller) SADECE
gerçekten bir model yüklenecek/eğitilecek/tahmin yapılacak yerde,
kullanıldığı fonksiyonun İÇİNDE (lazy) import edilmeli — bkz. bu iki
modülün kendisi de artık bu dosyadaki sabitleri yeniden ihraç ediyor,
tek kaynak burası."""

from pathlib import Path

DEFAULT_LSTM_MODEL_PATH = Path(__file__).parent / "artifacts" / "lstm_model.pt"
DEFAULT_PATCHTST_MODEL_PATH = Path(__file__).parent / "artifacts" / "patchtst_model.pt"
