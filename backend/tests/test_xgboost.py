import numpy as np
import pandas as pd
import pytest

from app.exchanges.base import Exchange
from app.ml.advanced_indicators import average_true_range
from app.ml.dataset import build_training_dataset, build_training_dataset_with_time
from app.ml.labeling import triple_barrier_labels
from app.ml.model import SignalModel
from app.ml.train import train_signal_model_validated
from app.ml.validation import split_out_of_sample, walk_forward_splits


class TrendExchange(Exchange):
    """Testler için gerçek ağ çağrısı yapmayan, belirgin bir yön trendi
    olan sentetik borsa (XGBoost'un öğrenebileceği bir örüntü olması için)."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def list_symbols(self, quote_currency: str, market_type: str) -> list[str]:
        return ["UPUSDT", "DOWNUSDT"]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        n = max(limit, 400)
        if symbol == "UPUSDT":
            base = np.linspace(100, 400, n)
        else:
            base = np.linspace(400, 100, n)
        noise = self._rng.normal(0, 1.0, n)
        close = base + noise
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": self._rng.uniform(800, 1200, n),
            }
        )


def _synthetic_features_and_labels(n_per_class: int) -> tuple[pd.DataFrame, pd.Series]:
    """Sınıf başına TAM olarak `n_per_class` örneği olan, öğrenilebilir bir
    sentetik özellik/etiket seti kurar (kalibrasyon eşiği testleri için —
    gerçek `build_training_dataset` çıktısındaki sınıf sayılarının tam
    kontrolü zor olduğundan doğrudan kurulur)."""
    from app.ml.features import ALL_FEATURE_COLUMNS

    rng = np.random.default_rng(0)
    rows = []
    labels = []
    for label, offset in ((-1.0, -3.0), (0.0, 0.0), (1.0, 3.0)):
        for _ in range(n_per_class):
            row = {col: rng.normal(offset, 0.5) for col in ALL_FEATURE_COLUMNS}
            rows.append(row)
            labels.append(label)
    X = pd.DataFrame(rows)
    y = pd.Series(labels)
    return X, y


def test_signal_model_skips_calibration_below_per_fold_sample_threshold():
    # cv=3, eşik = 3*10 = 30/sınıf gerekiyor; 20/sınıf bunun altında kalmalı.
    X, y = _synthetic_features_and_labels(n_per_class=20)
    model = SignalModel(calibrate=True)
    model.fit(X, y)
    assert model.is_calibrated is False


def test_signal_model_calibrates_above_per_fold_sample_threshold():
    X, y = _synthetic_features_and_labels(n_per_class=40)
    model = SignalModel(calibrate=True)
    model.fit(X, y)
    assert model.is_calibrated is True


def test_signal_model_defaults_to_xgboost():
    model = SignalModel()
    assert model.algorithm == "xgboost"


def test_xgboost_model_fits_and_predicts():
    exchange = TrendExchange(seed=1)
    X, y = build_training_dataset(exchange, ["UPUSDT", "DOWNUSDT"], "4h", 400, horizon=5, threshold_pct=0.5)
    model = SignalModel(algorithm="xgboost")
    model.fit(X, y)

    predictions, confidences = model.predict_batch(X)
    assert len(predictions) == len(X)
    assert ((confidences >= 0) & (confidences <= 1)).all()


def test_mlp_algorithm_still_selectable_for_comparison():
    exchange = TrendExchange(seed=2)
    X, y = build_training_dataset(exchange, ["UPUSDT", "DOWNUSDT"], "4h", 400, horizon=5, threshold_pct=0.5)
    model = SignalModel(algorithm="mlp")
    model.fit(X, y)
    assert model.algorithm == "mlp"
    predictions, _ = model.predict_batch(X)
    assert len(predictions) == len(X)


def test_shap_values_only_supported_for_xgboost():
    exchange = TrendExchange(seed=3)
    X, y = build_training_dataset(exchange, ["UPUSDT", "DOWNUSDT"], "4h", 400, horizon=5, threshold_pct=0.5)

    xgb_model = SignalModel(algorithm="xgboost")
    xgb_model.fit(X, y)
    importance = xgb_model.shap_values(X)
    assert set(importance["feature"]) == set(X.columns) | set()
    assert (importance["mean_abs_shap"] >= 0).all()
    # en yüksek katkılı özellik en üstte olmalı (azalan sırada)
    assert (importance["mean_abs_shap"].diff().dropna() <= 1e-9).all()

    mlp_model = SignalModel(algorithm="mlp")
    mlp_model.fit(X, y)
    try:
        mlp_model.shap_values(X)
        assert False, "MLP için SHAP hata vermeliydi"
    except ValueError:
        pass


def test_split_out_of_sample_never_leaks_future_rows_into_train():
    exchange = TrendExchange(seed=4)
    X, y, time_frac, _bar_ts, _symbol = build_training_dataset_with_time(exchange, ["UPUSDT"], "4h", 400, horizon=5, threshold_pct=0.5)
    X_train, y_train, X_holdout, y_holdout = split_out_of_sample(X, y, time_frac, holdout_frac=0.2)

    assert len(X_train) + len(X_holdout) == len(X)
    assert time_frac[X_train.index].max() <= time_frac[X_holdout.index].min()
    assert len(X_holdout) > 0


def test_walk_forward_splits_are_purged_and_chronological():
    time_frac = pd.Series(np.linspace(0, 1, 500))
    splits = walk_forward_splits(time_frac, n_splits=5, embargo_frac=0.02)
    assert len(splits) > 0

    for train_idx, test_idx in splits:
        max_train_time = time_frac.iloc[train_idx].max()
        min_test_time = time_frac.iloc[test_idx].min()
        # embargo boşluğu: test penceresinin başlangıcı, eğitim setinin
        # en son gördüğü zamandan en az embargo_frac kadar sonra olmalı
        assert min_test_time - max_train_time >= 0.02 - 1e-9


def test_train_signal_model_validated_produces_walk_forward_and_oos_reports(tmp_path, monkeypatch):
    from app.ml import train as train_module

    # Gerçek paylaşılan app/ml/artifacts/ dosyasına (hem .joblib hem
    # write_model_status'un .status.json'ına) yazmamak için — diğer test
    # dosyaları (ör. test_system_backtest.py) DEFAULT_MODEL_PATH'i
    # gerçekten okur; burada bırakılacak bir holdout_start_time kaydı o
    # testleri sessizce kırar (bkz. test_backtest_* holdout testleri).
    monkeypatch.setattr(train_module, "DEFAULT_MODEL_PATH", tmp_path / "signal_model.joblib")

    exchange = TrendExchange(seed=5)
    result = train_signal_model_validated(
        exchange,
        ["UPUSDT", "DOWNUSDT"],
        horizon=5,
        threshold_pct=0.5,
        timeframe="4h",
        lookback=400,
        holdout_frac=0.2,
        walk_forward_splits=4,
    )

    assert result.rows_used > 0
    assert len(result.walk_forward.folds) > 0
    assert 0.0 <= result.walk_forward.mean_accuracy <= 1.0
    assert result.out_of_sample.holdout_rows > 0
    assert 0.0 <= result.out_of_sample.accuracy <= 1.0


def test_train_signal_model_validated_persist_false_does_not_touch_disk(tmp_path, monkeypatch):
    from app.ml.model import SignalModel

    saved_paths = []
    monkeypatch.setattr(SignalModel, "save", lambda self, path=None: saved_paths.append(path))

    exchange = TrendExchange(seed=6)
    train_signal_model_validated(
        exchange, ["UPUSDT", "DOWNUSDT"], horizon=5, threshold_pct=0.5, timeframe="4h", lookback=400, persist=False
    )
    assert saved_paths == []


def test_train_signal_model_validated_rejects_and_skips_save_below_quality_threshold(tmp_path, monkeypatch):
    """`Settings.ml_min_balanced_accuracy`'nin ALTINDA kalan bir model
    diske KAYDEDİLMEMELİ (önceki model korunmalı) ve `accepted=False`
    ile açıkça işaretlenmeli — bkz. app.ml.train kalite kapısı."""
    from app.core.config import settings
    from app.ml import train as train_module
    from app.ml.model import SignalModel

    # Eşiği kasıtlı olarak imkansız derecede yüksek (1.01) yaparak, gerçekte
    # ne kadar iyi eğitilmiş olursa olsun modelin REDDEDİLMESİNİ garantiler
    # — testin kırılganlığını (gerçek balanced_accuracy'nin rastgele altında/
    # üstünde çıkmasına bağlı olmadan) önler.
    monkeypatch.setattr(settings, "ml_min_balanced_accuracy", 1.01)
    # write_model_status'un GERÇEK paylaşılan app/ml/artifacts/ konumuna
    # yazmasını önlemek için (bkz. test_lstm.py aynı deseni kullanır).
    monkeypatch.setattr(train_module, "DEFAULT_MODEL_PATH", tmp_path / "signal_model.joblib")

    saved_paths = []
    monkeypatch.setattr(SignalModel, "save", lambda self, path=None: saved_paths.append(path))

    exchange = TrendExchange(seed=7)
    result = train_signal_model_validated(
        exchange, ["UPUSDT", "DOWNUSDT"], horizon=5, threshold_pct=0.5, timeframe="4h", lookback=400
    )

    assert result.accepted is False
    assert result.rejection_reason is not None
    assert "1.01" in result.rejection_reason
    assert saved_paths == []  # reddedilen model KAYDEDİLMEDİ


def test_train_signal_model_validated_accepts_and_saves_above_quality_threshold(tmp_path, monkeypatch):
    from app.core.config import settings
    from app.ml import train as train_module
    from app.ml.model import SignalModel

    # Eşiği 0'a çekerek (her zaman geçilir) kabul/kaydetme yolunu test eder.
    monkeypatch.setattr(settings, "ml_min_balanced_accuracy", 0.0)
    monkeypatch.setattr(train_module, "DEFAULT_MODEL_PATH", tmp_path / "signal_model.joblib")

    saved_paths = []
    monkeypatch.setattr(SignalModel, "save", lambda self, path=None: saved_paths.append(path))

    exchange = TrendExchange(seed=8)
    result = train_signal_model_validated(
        exchange, ["UPUSDT", "DOWNUSDT"], horizon=5, threshold_pct=0.5, timeframe="4h", lookback=400
    )

    assert result.accepted is True
    assert result.rejection_reason is None
    assert len(saved_paths) == 1


def test_train_signal_model_validated_records_holdout_start_time(tmp_path, monkeypatch):
    """Kabul edilen bir eğitim, holdout'un GERÇEK başlangıç zaman damgasını
    `app.ml.model_status`'a yazmalı — `app.backtest.system_runner` bunu
    okuyup backtest'i modelin hiç görmediği barlarla sınırlar (bkz.
    test_system_backtest.py::test_backtest_excludes_bars_before_model_holdout_start)."""
    from app.core.config import settings
    from app.ml import train as train_module
    from app.ml.model import SignalModel
    from app.ml.model_status import get_holdout_start_time

    monkeypatch.setattr(settings, "ml_min_balanced_accuracy", 0.0)
    monkeypatch.setattr(settings, "ml_primary_symbol", "UPUSDT")
    status_target = tmp_path / "signal_model.joblib"
    monkeypatch.setattr(train_module, "DEFAULT_MODEL_PATH", status_target)
    monkeypatch.setattr(SignalModel, "save", lambda self, path=None: None)

    exchange = TrendExchange(seed=8)
    train_signal_model_validated(exchange, ["UPUSDT", "DOWNUSDT"], horizon=5, threshold_pct=0.5, timeframe="4h", lookback=400)

    holdout_start = get_holdout_start_time(status_target)
    assert holdout_start is not None
    # kayıtlı zaman damgası ayrıştırılabilir (geçerli bir ISO tarih) olmalı
    pd.Timestamp(holdout_start)


class _TwoSymbolDifferentEraExchange(Exchange):
    """İki sembol, TAMAMEN FARKLI takvim aralıklarında: "OLD" 2020'de,
    "PRIMARY" 2024'te. Gerçek üretimdeki durumun (BTC + kendi geçmişi/DB
    önbellek kapsamı farklı, korelasyonlu bir altcoin) sadeleştirilmiş
    karşılığı."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng(3)

    def list_symbols(self, quote_currency: str, market_type: str) -> list[str]:
        return ["PRIMARY", "OLD"]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since: int | None = None) -> pd.DataFrame:
        n = max(limit, 400)
        start = "2024-01-01" if symbol == "PRIMARY" else "2020-01-01"
        close = np.linspace(100, 200, n) + self._rng.normal(0, 1.0, n)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range(start, periods=n, freq="4h"),
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": self._rng.uniform(800, 1200, n),
            }
        )


def test_holdout_start_time_uses_primary_symbol_not_earliest_across_symbols(tmp_path, monkeypatch):
    """Regresyon (gerçek üretim hatası): eğitim BTC + korelasyonlu ikinci
    bir sembolle yapıldığında, holdout başlangıcı TÜM sembollerin EN ERKEN
    tarihinden alınıyordu. İkinci sembolün geçmişi farklı bir takvim
    aralığındaysa bu, kaydedilen tarihi ÇOK ERKENE çekiyor ve backtest'in
    holdout filtresi HİÇBİR barı dışlamıyordu (0 işlem / tüm dönem).
    Doğrusu: yalnızca `settings.ml_primary_symbol`'ün kendi holdout'u."""
    from app.core.config import settings
    from app.ml import train as train_module
    from app.ml.model import SignalModel
    from app.ml.model_status import get_holdout_start_time

    monkeypatch.setattr(settings, "ml_min_balanced_accuracy", 0.0)
    monkeypatch.setattr(settings, "ml_primary_symbol", "PRIMARY")
    status_target = tmp_path / "signal_model.joblib"
    monkeypatch.setattr(train_module, "DEFAULT_MODEL_PATH", status_target)
    monkeypatch.setattr(SignalModel, "save", lambda self, path=None: None)

    exchange = _TwoSymbolDifferentEraExchange()
    train_signal_model_validated(exchange, ["PRIMARY", "OLD"], horizon=5, threshold_pct=0.5, timeframe="4h", lookback=400)

    holdout_start = pd.Timestamp(get_holdout_start_time(status_target))
    # PRIMARY 2024'te başlıyor; "OLD" (2020) sembolünün holdout'u ASLA
    # seçilmemeli — aksi halde tarih 2020'ye düşerdi.
    assert holdout_start.year == 2024
    # ve holdout, PRIMARY'nin serisinin SONLARINA doğru olmalı (ilk barı değil)
    assert holdout_start > pd.Timestamp("2024-01-01")


def test_sweep_lookback_values_returns_one_point_per_lookback():
    from app.ml.train import sweep_lookback_values

    exchange = TrendExchange(seed=7)
    points = sweep_lookback_values(exchange, ["UPUSDT", "DOWNUSDT"], [300, 400], timeframe="4h", horizon=5, threshold_pct=0.5)

    assert [p.lookback for p in points] == [300, 400]
    for point in points:
        assert point.error is None
        assert point.rows_used > 0
        assert 0.0 <= point.out_of_sample_balanced_accuracy <= 1.0


def test_sweep_lookback_values_reports_error_without_stopping_the_sweep(monkeypatch):
    import app.ml.train as train_module

    original = train_module.train_signal_model_validated

    def _fail_for_small_lookback(*args, **kwargs):
        if kwargs.get("lookback") == 5:
            raise ValueError("yeterli veri yok (test)")
        return original(*args, **kwargs)

    monkeypatch.setattr(train_module, "train_signal_model_validated", _fail_for_small_lookback)

    exchange = TrendExchange(seed=8)
    points = train_module.sweep_lookback_values(exchange, ["UPUSDT"], [5, 400], timeframe="4h", horizon=5, threshold_pct=0.5)

    assert points[0].error is not None
    assert points[1].error is None


def test_stale_model_features_raise_clear_error(tmp_path):
    """Regresyon (üretimde yaşandı): kod `ALL_FEATURE_COLUMNS`'a yeni özellik
    eklendikten sonra modeller yeniden eğitilmezse, XGBoost ham bir
    "feature_names mismatch" dökümü fırlatıyordu — kullanıcı için anlamsız.
    Artık ne yapılması gerektiğini SÖYLEYEN bir hata veriyor."""
    from app.ml.model import SignalModel, StaleModelFeaturesError

    X, y = _synthetic_features_and_labels(40)
    model = SignalModel()
    model.fit(X, y)
    path = tmp_path / "m.joblib"
    model.save(path)

    # diskteki model ESKİ bir özellik setiyle eğitilmiş gibi davran
    reloaded = SignalModel.load_from(path)
    reloaded.feature_columns = [c for c in reloaded.feature_columns if c != reloaded.feature_columns[-1]]

    with pytest.raises(StaleModelFeaturesError) as exc:
        reloaded.predict_batch(X)
    assert "train-all.sh" in str(exc.value)


def test_freshly_trained_model_records_current_feature_columns(tmp_path):
    from app.ml.features import ALL_FEATURE_COLUMNS
    from app.ml.model import SignalModel

    X, y = _synthetic_features_and_labels(40)
    model = SignalModel()
    model.fit(X, y)
    assert model.feature_columns == list(ALL_FEATURE_COLUMNS)

    path = tmp_path / "m.joblib"
    model.save(path)
    assert SignalModel.load_from(path).feature_columns == list(ALL_FEATURE_COLUMNS)


def test_all_ensemble_members_share_one_labeling_definition():
    """Regresyon (gerçek üretimde gözlendi): ensemble üyeleri FARKLI hedefler
    öğreniyordu — XGBoost "12 bar içinde ATR hedefine mi stopa mı önce
    ulaşır", LSTM "3 bar sonra %1 hareket eder mi", online "5 bar sonra %1
    hareket eder mi". Farklı soruların cevaplarını harmanlamak sağlam bir
    ensemble değil: güvenler aynı ölçekte olmaz ve `_skill_weight` farklı
    problemlerde ölçülmüş doğrulukları kıyaslar (XGBoost'un zor/dengeli
    problemdeki 0.375'i, online'ın kolay problemdeki 0.502'siyle kıyaslanıp
    haksız düşük ağırlık aldı).

    Bu test, `train_all_models`'ın TÜM üyeleri tek bir `_ENSEMBLE_LABELING`
    tanımından beslediğini garanti eder — biri elle farklı bir horizon/
    labeling ile çağrılırsa BU TEST ÇÖKER."""
    import ast
    import inspect
    import textwrap

    from app.ml import train as train_module

    # Metin araması YETMEZ (yorum satırlarındaki "horizon=3" gibi ifadeler
    # yanlış alarm verir) — gerçek çağrıların argümanları AST ile incelenir.
    tree = ast.parse(textwrap.dedent(inspect.getsource(train_module.train_all_models)))
    members = {
        "train_signal_model_validated",
        "train_meta_label_model",
        "train_lstm_signal_model",
        "train_online_signal_model",
        "train_signal_models_by_regime",
    }
    labeling_kwargs = {"horizon", "labeling_method", "threshold_pct", "take_profit_pct", "stop_loss_pct"}

    seen = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in members):
            continue
        seen.add(node.func.id)
        passed = {kw.arg for kw in node.keywords if kw.arg is not None}
        leaked = passed & labeling_kwargs
        assert not leaked, (
            f"{node.func.id} kendi etiketleme parametrelerini elle geçiriyor: {sorted(leaked)} — "
            "tüm ensemble üyeleri _ENSEMBLE_LABELING'i paylaşmalı"
        )
        assert any(kw.arg is None for kw in node.keywords), (
            f"{node.func.id} çağrısında **_ENSEMBLE_LABELING açılımı yok"
        )

    assert seen == members, f"train_all_models'ta çağrılmayan üye(ler): {sorted(members - seen)}"

    # ve paylaşılan tanım gerçekten ATR-hizalı olmalı
    assert train_module._ENSEMBLE_LABELING["labeling_method"] == "atr_triple_barrier"

    # DOĞRU kontrol "horizon >= N" gibi sabit bir eşik DEĞİL (bu, horizon'un
    # ATR çarpanıyla BİRLİKTE davrandığını gözden kaçırır — ör. horizon=3
    # 1.5xATR ile %48.9 nötr verirken 1.0xATR ile yalnızca %18.7 nötr verir,
    # bkz. sohbet). Asıl endişe: seçilen (horizon, çarpan) kombinasyonu saf
    # gürültüde AŞIRI tek-sınıf çökmesine (eski "%84.7 nötr" felaketi) yol
    # açmasın. Bunu doğrudan, sentetik saf rastgele-yürüyüş verisiyle ölçer.
    rng = np.random.default_rng(11)
    n = 20000
    close = 60000 * np.exp(np.cumsum(rng.normal(0.0, 0.004, n)))
    ohlcv = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "open": close,
            "high": close * (1 + np.abs(rng.normal(0, 0.002, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.002, n))),
            "close": close,
            "volume": rng.uniform(800, 1200, n),
        }
    )
    atr_pct = (average_true_range(ohlcv, length=14) / ohlcv["close"]) * 100
    mult = train_module._ENSEMBLE_LABELING["take_profit_pct"]
    horizon = train_module._ENSEMBLE_LABELING["horizon"]
    labels = triple_barrier_labels(ohlcv, atr_pct * mult, atr_pct * mult, max_horizon=horizon).dropna()
    class_shares = labels.value_counts(normalize=True)
    assert class_shares.max() < 0.6, (
        f"_ENSEMBLE_LABELING (horizon={horizon}, çarpan={mult}) saf gürültüde bir sınıfa "
        f"çöküyor (en yüksek pay: %{class_shares.max() * 100:.1f}) — eski '%84.7 nötr' felaketiyle AYNI desen"
    )


class _TrendingWithNoiseExchange(Exchange):
    """Guclu, ogrenilebilir bir trend + gurultu -- XGBoost'un KENDI
    egitim etiketinde (atr_triple_barrier) makul bir dogruluk yakalayabilmesi
    icin (meta-label uyusmazligi regresyonunu yeniden uretebilmek icin sart:
    primary GERCEKTEN cogunlukla dogruysa, YANLIS bir etikete gore olculunce
    "yanlis" gorunmesi ancak o zaman ACIKCA fark edilir)."""

    def __init__(self, n: int = 3000, seed: int = 21) -> None:
        rng = np.random.default_rng(seed)
        trend = np.linspace(0, 0.6, n)  # guclu, kalici yukselis
        close = 100 * np.exp(trend + rng.normal(0, 0.01, n))
        self.full_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=n, freq="1h"),
                "open": close,
                "high": close * 1.003,
                "low": close * 0.997,
                "close": close,
                "volume": rng.uniform(800, 1200, n),
            }
        )

    def list_symbols(self, quote_currency, market_type):
        return ["BTCUSDT"]

    def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
        return self.full_df.iloc[-limit:].reset_index(drop=True)


def test_meta_label_must_use_same_labeling_as_primary_model():
    """Regresyon (gerçek üretim hatası): `train_meta_label_model`, birincil
    modelin "doğru mu tahmin etti" sorusunu kendi etiketleme parametreleriyle
    (varsayılan: `threshold`/`horizon=5`) ölçüyordu — `primary_model` BAŞKA
    bir etiketle (`atr_triple_barrier`/`horizon=12`) eğitilmiş olsa bile.
    Yani meta-label, primary'nin ÖĞRENMEDİĞİ bir soruya göre "yanlış" damgası
    vuruyordu. Üretimde somut sonucu: primary kendi sorusunda makul bir
    doğrulukla (oos=0.377) çalışırken, meta-label 768 açılış girişiminin
    767'sini veto etti.

    Bu test, AYNI (`atr_triple_barrier`/`horizon=12`) etiketlemeyle eğitilmiş
    bir birincil model için meta-label'ın "doğru" oranının UYUMLU parametrelerle
    ÇAĞRILDIĞINDA UYUMSUZ parametrelerle çağrıldığından belirgin şekilde
    yüksek olduğunu doğrular."""
    from app.ml.meta_label import build_meta_dataset
    from app.ml.train import _ENSEMBLE_LABELING, train_signal_model_validated

    exchange = _TrendingWithNoiseExchange()

    primary_result = train_signal_model_validated(
        exchange, ["BTCUSDT"], timeframe="1h", lookback=3000, persist=False, **_ENSEMBLE_LABELING
    )
    primary = primary_result.model

    # AYNI (birincilin ÖĞRENDİĞİ) etiketle yeniden kurulan veri seti
    X_matched, y_matched = build_training_dataset(exchange, ["BTCUSDT"], "1h", 3000, **_ENSEMBLE_LABELING)
    _, meta_y_matched = build_meta_dataset(primary, X_matched, y_matched)

    # ESKİ (uyumsuz) varsayılanlarla kurulan veri seti -- eski buggy davranış
    X_mismatched, y_mismatched = build_training_dataset(
        exchange, ["BTCUSDT"], "1h", 3000, horizon=5, threshold_pct=1.0, labeling_method="threshold"
    )
    _, meta_y_mismatched = build_meta_dataset(primary, X_mismatched, y_mismatched)

    matched_correct_rate = meta_y_matched.mean()
    mismatched_correct_rate = meta_y_mismatched.mean()

    assert matched_correct_rate > mismatched_correct_rate + 0.1, (
        f"beklenen: AYNI etiketle 'doğru' oranı belirgin yüksek olmalı — "
        f"uyumlu={matched_correct_rate:.3f} uyumsuz={mismatched_correct_rate:.3f}"
    )
    # ve uyumlu haliyle meta-label'ın makul sayıda pozitif ("act") örneği olmalı
    # -- aksi halde ("hep veto") backtest'te gördüğümüz duruma geri döneriz.
    assert matched_correct_rate > 0.5
