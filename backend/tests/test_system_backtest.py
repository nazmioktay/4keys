import numpy as np
import pandas as pd
import pytest

from app.backtest.schemas import SystemBacktestReport, SystemBacktestRequest
from app.backtest.system_runner import (
    _compute_position_size,
    _lstm_predictions_for_series,
    run_system_backtest,
    sweep_confidence_thresholds,
)
from app.ml.features import FEATURE_COLUMNS, build_features
from app.ml.lstm_model import LSTMSignalModel
from app.ml.model import SignalModel
from app.exchanges.base import Exchange


class FakeOscillatingExchange(Exchange):
    """Osilasyonlu, göstergelerin ısınması için yeterli uzunlukta sabit bir
    geçmişi olan test borsası (bkz. test_backtest.py::FakeHistoryExchange —
    aynı desen, burada bağımsız kopyalanmış)."""

    def __init__(self, total_candles: int, timeframe: str = "1h", seed: int = 3) -> None:
        freq_minutes = 60
        idx = pd.date_range("2022-01-01", periods=total_candles, freq=f"{freq_minutes}min", tz="UTC").tz_convert(None)
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 60 * np.pi, total_candles)
        base = 20000 + 2000 * np.sin(t)
        close = base + rng.normal(0, 50.0, total_candles)
        self.full_df = pd.DataFrame(
            {
                "timestamp": idx,
                "open": close,
                "high": close + 20,
                "low": close - 20,
                "close": close,
                "volume": rng.uniform(800, 1200, total_candles),
            }
        )

    def list_symbols(self, quote_currency, market_type):
        return ["BTC/USDT:USDT"]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since: int | None = None) -> pd.DataFrame:
        df = self.full_df
        if since is None:
            return df.iloc[-limit:].reset_index(drop=True)
        since_ts = pd.Timestamp(since, unit="ms")
        mask = df["timestamp"] >= since_ts
        return df.loc[mask].iloc[:limit].reset_index(drop=True)


class FakeLongHistoryExchange(Exchange):
    """`_FAR_PAST_MS` (2017) ile "şimdi" arasında ÇOK daha uzun bir geçmişi
    olan test borsası — `run_system_backtest`'in gerçekten EN SON
    `request.candles` mumu (2017'den itibaren en ESKİ değil) çektiğini
    doğrulamak için (bkz. gerçek üretim regresyonu: backtest raporu
    2019-2020 gibi alakasız bir dönem gösteriyordu, çünkü `fetch_full_history`
    bilerek en eski geçmişten başlıyor)."""

    def __init__(self, total_candles: int) -> None:
        idx = pd.date_range("2017-01-01", periods=total_candles, freq="1h", tz="UTC").tz_convert(None)
        rng = np.random.default_rng(11)
        close = 20000 + np.cumsum(rng.normal(0, 5, total_candles))
        self.full_df = pd.DataFrame(
            {
                "timestamp": idx,
                "open": close,
                "high": close + 20,
                "low": close - 20,
                "close": close,
                "volume": rng.uniform(800, 1200, total_candles),
            }
        )

    def list_symbols(self, quote_currency, market_type):
        return ["BTC/USDT:USDT"]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since: int | None = None) -> pd.DataFrame:
        df = self.full_df
        if since is None:
            return df.iloc[-limit:].reset_index(drop=True)
        since_ts = pd.Timestamp(since, unit="ms")
        mask = df["timestamp"] >= since_ts
        return df.loc[mask].iloc[:limit].reset_index(drop=True)


def test_run_system_backtest_uses_most_recent_candles_not_earliest(monkeypatch):
    from app.db.session import reset_for_tests
    from app.core.config import settings

    monkeypatch.setattr(settings, "database_url", "")
    reset_for_tests()

    # 5 yıldan fazla (2017'den itibaren) veri var, ama yalnızca son 600
    # mum istiyoruz — rapor 2017'ye değil, verinin EN SONUNA yakın olmalı.
    exchange = FakeLongHistoryExchange(total_candles=50000)
    train_ohlcv = exchange.full_df.iloc[-1000:-600].reset_index(drop=True)
    model = _trained_model(train_ohlcv)

    # restrict_to_holdout=False: bu test tarih ARALIĞINI doğruluyor, holdout
    # filtresini değil — gerçek DEFAULT_MODEL_PATH'te başka bir testten
    # kalan bir holdout kaydı varsa buradaki senkronize sentetik veriyle
    # ilgisiz şekilde çakışıp testi kırabilir (bkz. holdout-özel testler).
    request = SystemBacktestRequest(
        symbol="BTC/USDT:USDT", timeframe="1h", candles=600, initial_balance=1000.0, restrict_to_holdout=False
    )
    report = run_system_backtest(exchange, model, None, request)

    expected_period_end = exchange.full_df["timestamp"].iloc[-1]
    actual_period_end = pd.Timestamp(report.period_end)
    assert actual_period_end.year == expected_period_end.year
    assert (expected_period_end - actual_period_end).days < 30  # ısınma barları yüzünden küçük bir kayma olabilir


def _trained_model(ohlcv: pd.DataFrame) -> SignalModel:
    features = build_features(ohlcv).dropna().reset_index(drop=True)
    # Basit, gerçekçi olmayan ama deterministik bir etiket: getiri işaretine göre.
    returns = features["close"].pct_change().shift(-1).fillna(0.0)
    y = pd.Series(np.where(returns > 0.001, 1, np.where(returns < -0.001, -1, 0)), index=features.index)
    model = SignalModel()
    model.fit(features, y)
    return model


def test_run_system_backtest_produces_report_and_persists_ohlcv(tmp_path, monkeypatch):
    from app.db import session as db_session

    monkeypatch.setattr(db_session.settings, "database_url", f"sqlite:///{tmp_path}/test.db")
    db_session.reset_for_tests()
    db_session.init_db()

    exchange = FakeOscillatingExchange(total_candles=600)
    train_ohlcv = exchange.full_df.iloc[:400].reset_index(drop=True)
    model = _trained_model(train_ohlcv)

    request = SystemBacktestRequest(
        symbol="BTC/USDT:USDT", timeframe="1h", candles=600, initial_balance=1000.0, restrict_to_holdout=False
    )
    report = run_system_backtest(exchange, model, None, request)

    assert report.candles_used == 600
    assert report.initial_balance == 1000.0
    assert report.trades_closed >= 0
    assert report.win_rate_pct >= 0.0
    # equity eğrisi trade sayısına tutarlı: final_equity, initial_balance +
    # tüm işlemlerin pnl_quote toplamı olmalı
    if report.trades:
        expected_final = 1000.0 + sum(t.pnl_quote for t in report.trades)
        assert report.final_equity == pytest.approx(expected_final, abs=0.05)

    # DB'ye backfill edilen OHLCV geri okunabilmeli
    from app.db import repository as db

    latest = db.get_latest_backtest_run(symbol="BTC/USDT:USDT")
    assert latest is not None
    assert latest["trades_closed"] == report.trades_closed

    db_session.reset_for_tests()


def test_run_system_backtest_raises_on_insufficient_history():
    # Borsada 100 mum var ama en az 300 istenir -> exchange bunun altında
    # döner, göstergelerin ısınması (250 bar) için yetersiz kalır.
    exchange = FakeOscillatingExchange(total_candles=100)
    model = _trained_model(exchange.full_df)
    request = SystemBacktestRequest(symbol="BTC/USDT:USDT", timeframe="1h", candles=300)

    with pytest.raises(ValueError):
        run_system_backtest(exchange, model, None, request)


def test_stop_loss_closes_long_position_on_crash():
    exchange = FakeOscillatingExchange(total_candles=600)
    train_ohlcv = exchange.full_df.iloc[:400].reset_index(drop=True)
    model = _trained_model(train_ohlcv)

    request = SystemBacktestRequest(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        candles=600,
        initial_balance=1000.0,
        atr_stop_loss_mult=1.0,
        atr_take_profit_mult=None,
        atr_trailing_mult=None,
        open_confidence=0.5,  # izin verilen minimum -> daha çok işlem tetiklensin
        close_confidence=0.9,  # kapanış yalnızca stop-loss'tan gelsin, sinyalden değil
        restrict_to_holdout=False,
    )
    report = run_system_backtest(exchange, model, None, request)

    stop_loss_trades = [t for t in report.trades if t.exit_reason == "stop_loss"]
    for t in stop_loss_trades:
        if t.direction == "long":
            assert t.exit_price < t.entry_price
        else:
            assert t.exit_price > t.entry_price


def test_take_profit_closes_position_at_target():
    exchange = FakeOscillatingExchange(total_candles=600)
    train_ohlcv = exchange.full_df.iloc[:400].reset_index(drop=True)
    model = _trained_model(train_ohlcv)

    request = SystemBacktestRequest(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        candles=600,
        initial_balance=1000.0,
        atr_stop_loss_mult=None,
        atr_take_profit_mult=0.5,
        atr_trailing_mult=None,
        open_confidence=0.5,
        close_confidence=0.9,
        restrict_to_holdout=False,
    )
    report = run_system_backtest(exchange, model, None, request)

    take_profit_trades = [t for t in report.trades if t.exit_reason == "take_profit"]
    for t in take_profit_trades:
        if t.direction == "long":
            assert t.exit_price > t.entry_price
        else:
            assert t.exit_price < t.entry_price


def test_trades_record_position_size_and_decision_breakdown():
    exchange = FakeOscillatingExchange(total_candles=600)
    train_ohlcv = exchange.full_df.iloc[:400].reset_index(drop=True)
    model = _trained_model(train_ohlcv)

    request = SystemBacktestRequest(
        symbol="BTC/USDT:USDT", timeframe="1h", candles=600, initial_balance=1000.0, restrict_to_holdout=False
    )
    report = run_system_backtest(exchange, model, None, request)

    assert report.trades, "test verisiyle en az bir işlem beklenir"
    for t in report.trades:
        assert t.size_quote > 0
        assert t.size_explanation
        assert t.xgboost_direction in ("long", "short", "neutral")
        assert t.decision_reason and "XGBoost=" in t.decision_reason
        # bu testte LSTM/online model geçirilmedi -> ensemble alanları boş kalmalı
        assert t.lstm_direction is None
        assert t.online_direction is None


def test_backtest_excludes_bars_before_model_holdout_start(monkeypatch):
    """Regresyon: backtest varsayılan olarak `request.candles` (en son N mum)
    ister, ama bu neredeyse her zaman modelin KENDİ eğitim penceresiyle
    çakışır. Modelin holdout'unun (hiç görmediği veri) 300. bardan
    başladığını simüle edip, raporun yalnızca O NOKTADAN SONRAKİ barları
    içerdiğini ve ilgili uyarının göründüğünü doğrular."""
    exchange = FakeOscillatingExchange(total_candles=600)
    train_ohlcv = exchange.full_df.iloc[:400].reset_index(drop=True)
    model = _trained_model(train_ohlcv)

    features_full = build_features(exchange.full_df).dropna().reset_index(drop=True)
    holdout_cutoff = pd.Timestamp(int(features_full.iloc[300]["timestamp"]))
    monkeypatch.setattr(
        "app.backtest.system_runner.get_holdout_start_time",
        lambda path: holdout_cutoff.isoformat(),
    )

    request = SystemBacktestRequest(symbol="BTC/USDT:USDT", timeframe="1h", candles=600, initial_balance=1000.0)
    report = run_system_backtest(exchange, model, None, request)

    assert pd.Timestamp(report.period_start) >= holdout_cutoff
    assert any("EĞİTİM verisiyle çakışmaması" in w for w in report.warnings)


def test_backtest_warns_when_no_holdout_recorded(monkeypatch):
    """Eski (bu güvenlik özelliği eklenmeden önce eğitilmiş) bir model için
    kayıtlı holdout yoksa backtest çökmemeli, ama kullanıcıyı UYARMALI."""
    exchange = FakeOscillatingExchange(total_candles=600)
    train_ohlcv = exchange.full_df.iloc[:400].reset_index(drop=True)
    model = _trained_model(train_ohlcv)

    monkeypatch.setattr("app.backtest.system_runner.get_holdout_start_time", lambda path: None)

    request = SystemBacktestRequest(symbol="BTC/USDT:USDT", timeframe="1h", candles=600, initial_balance=1000.0)
    report = run_system_backtest(exchange, model, None, request)

    assert any("holdout başlangıç tarihi bulunamadı" in w for w in report.warnings)


def test_backtest_restrict_to_holdout_false_bypasses_filter(monkeypatch):
    exchange = FakeOscillatingExchange(total_candles=600)
    train_ohlcv = exchange.full_df.iloc[:400].reset_index(drop=True)
    model = _trained_model(train_ohlcv)

    features_full = build_features(exchange.full_df).dropna().reset_index(drop=True)
    holdout_cutoff = pd.Timestamp(int(features_full.iloc[300]["timestamp"]))
    monkeypatch.setattr(
        "app.backtest.system_runner.get_holdout_start_time",
        lambda path: holdout_cutoff.isoformat(),
    )

    request = SystemBacktestRequest(
        symbol="BTC/USDT:USDT", timeframe="1h", candles=600, initial_balance=1000.0, restrict_to_holdout=False
    )
    report = run_system_backtest(exchange, model, None, request)

    assert pd.Timestamp(report.period_start) < holdout_cutoff
    assert not any("EĞİTİM verisiyle çakışmaması" in w for w in report.warnings)


class FakeNoisyBtcExchange(Exchange):
    """GERÇEKÇİ (gürültülü, rastgele yürüyüş) bir BTC serisi — `FakeOscillatingExchange`'in
    temiz sinüsünden FARKLI olarak kolayca öğrenilemez, bu yüzden modelin
    kalibre edilmiş güveni gerçek üretimdeki gibi ~0.4-0.6 bandında kalır.
    Güven eşiği regresyonunu yeniden üretebilmek için bu şart."""

    def __init__(self, total_candles: int, seed: int = 5) -> None:
        rng = np.random.default_rng(seed)
        close = 60000 * np.exp(np.cumsum(rng.normal(0.0001, 0.004, total_candles)))
        self.full_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=total_candles, freq="1h"),
                "open": close,
                "high": close * (1 + np.abs(rng.normal(0, 0.002, total_candles))),
                "low": close * (1 - np.abs(rng.normal(0, 0.002, total_candles))),
                "close": close,
                "volume": rng.uniform(800, 1200, total_candles),
            }
        )

    def list_symbols(self, quote_currency, market_type):
        return ["BTC/USDT:USDT"]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since: int | None = None) -> pd.DataFrame:
        return self.full_df.iloc[-limit:].reset_index(drop=True)


def test_default_confidence_thresholds_are_reachable_on_calibrated_scale(monkeypatch):
    """Regresyon (üretimde "0 işlem / %0 PnL"): `open_confidence` varsayılanı
    0.6'ydı, ama `SignalModel` güveni KALİBRE EDİLMİŞ 3 sınıflı olasılıktan
    okuyor — kalibrasyon olasılıkları taban orana (0.333) sıkıştırdığı için
    yönlü güven gerçek veride 0.56-0.61'i aşmıyor; 0.6 eşiği pratikte
    ULAŞILAMAZDI ve ne backtest ne de CANLI motor pozisyon açabiliyordu.

    Bu test, gerçekçi GÜRÜLTÜLÜ bir seri + gerçek `atr_triple_barrier`
    etiketlemesiyle (üretimdeki AYNI yol) VARSAYILAN eşiklerin işlem
    açabildiğini garanti eden bir DUMAN TESTİDİR.

    NOT (dürüstlük): bu senaryo tek başına 0.6 varsayılanını GÜVENİLİR
    şekilde yakalayamaz — ölçülen maksimum yönlü güven (~0.60) tam eşiğin
    sınırında olduğu ve XGBoost eğitimi tam deterministik olmadığı için
    sonuç koşudan koşuya değişebiliyor. Eşiğin ulaşılabilir kalmasını
    DETERMİNİSTİK olarak garanti eden asıl koruma:
    `test_default_confidence_thresholds_stay_on_calibrated_scale`."""
    from app.ml.dataset import build_training_dataset

    exchange = FakeNoisyBtcExchange(total_candles=2500)
    X, y = build_training_dataset(
        exchange, ["BTC/USDT:USDT"], "1h", 2500, horizon=12,
        labeling_method="atr_triple_barrier", take_profit_pct=1.5, stop_loss_pct=1.5,
    )
    model = SignalModel()
    model.fit(X, y)
    monkeypatch.setattr("app.backtest.system_runner.get_holdout_start_time", lambda path: None)

    # eşikler AÇIKÇA verilmiyor -> şemadaki VARSAYILANLAR kullanılır
    request = SystemBacktestRequest(symbol="BTC/USDT:USDT", timeframe="1h", candles=2500, initial_balance=1000.0)
    report = run_system_backtest(exchange, model, None, request)

    assert report.trades_closed > 0, (
        f"varsayılan eşiklerle (open={request.open_confidence}) hiç işlem açılmadı — uyarılar: {report.warnings}"
    )


def test_default_confidence_thresholds_stay_on_calibrated_scale():
    """DETERMİNİSTİK koruma (bkz. yukarıdaki duman testinin notu): güven
    eşikleri KALİBRE EDİLMİŞ 3 sınıflı olasılık ölçeğinde yaşar — rastgele
    seviye 0.333, gerçek ölçümde yönlü güvenin tavanı ~0.56-0.61. Bu yüzden
    0.55 ÜZERİ bir varsayılan, kapıyı pratikte ULAŞILAMAZ yapar ve sistem
    (hem backtest hem CANLI motor) hiç pozisyon açamaz — üretimde tam olarak
    bu yaşandı ("0 işlem / %0 PnL" backtest'i).

    Ayrıca `ge` alt sınırı da 0.5'in ÜZERİNDE olmamalı: eski `ge=0.5`,
    doğru değerin API'den verilmesini bile engelliyordu."""
    from app.engine.decision import DecisionEngine

    request = SystemBacktestRequest()
    assert request.open_confidence <= 0.55, "backtest open_confidence varsayılanı kalibre ölçekte ulaşılamaz"
    assert request.close_confidence <= request.open_confidence, "çıkış eşiği girişten yüksek olmamalı"

    field = SystemBacktestRequest.model_fields["open_confidence"]
    lower_bounds = [m.ge for m in field.metadata if hasattr(m, "ge")]
    assert lower_bounds and lower_bounds[0] <= 0.5, "ge alt sınırı doğru eşiğin verilmesini engelliyor"

    # CANLI karar motorunun kendi varsayılanları da AYNI ölçekte olmalı —
    # `app.engine.service.run_cycle_once` bu eşikleri açıkça geçmiyor.
    import inspect

    live_defaults = inspect.signature(DecisionEngine.__init__).parameters
    assert live_defaults["open_confidence"].default <= 0.55
    assert live_defaults["close_confidence"].default <= live_defaults["open_confidence"].default


def test_zero_trade_backtest_explains_why(monkeypatch):
    """Regresyon: üretimde backtest "0 işlem / %0 PnL" döndü ve NEDENİ hiçbir
    yerde görünmüyordu. Artık sıfır işlem durumunda rapor, sinyalin hiç yönlü
    çıkmadığını mı yoksa güven eşiğinin mi aşılamadığını (ve görülen en yüksek
    yönlü güveni) AÇIKÇA yazmalı."""
    exchange = FakeOscillatingExchange(total_candles=600)
    train_ohlcv = exchange.full_df.iloc[:400].reset_index(drop=True)
    model = _trained_model(train_ohlcv)
    monkeypatch.setattr("app.backtest.system_runner.get_holdout_start_time", lambda path: None)

    # Eşiği ulaşılamaz yaparak "hiç işlem açılmadı" durumunu zorla
    request = SystemBacktestRequest(
        symbol="BTC/USDT:USDT", timeframe="1h", candles=600, initial_balance=1000.0, open_confidence=1.0
    )
    report = run_system_backtest(exchange, model, None, request)

    assert report.trades_closed == 0
    assert any("HİÇ işlem açılmadı" in w for w in report.warnings)
    # bu senaryoda yönlü karar ÜRETİLDİ ama eşik aşılamadı -> en yüksek güven raporlanmalı
    assert any("EN YÜKSEK yönlü güven" in w for w in report.warnings)


def test_lstm_predictions_skip_windows_that_contain_a_mid_series_nan():
    """Regresyon: `raw_features.dropna()` sonrası `valid_positions` ISINMA
    SONRASI olsa da ardışık olmak ZORUNDA değildir — bir gösterge (ör.
    `linreg_zscore`, `di_diff_norm`, bir doji mumda mum fitil oranları) bir
    rolling payda tam SIFIR olduğunda ısınma bittikten ÇOK SONRA bile ARA
    SIRA NaN üretebilir. Böyle bir satır bir LSTM penceresinin İÇİNDE
    (bitişinde değil) kalsa bile rekürrent ileri geçişi NaN'a bulaştırır —
    üretimde gözlenen `LSTM=short(nan)` hatasının kök nedeni buydu. Bu
    testte seq_len=5 pencereli sentetik bir seride 50. satıra bilerek bir
    NaN enjekte edilip, o satırı İÇEREN pencerelerin (bitişleri 50-54
    arası) `None` (LSTM'siz) dönmesi, UZAK pencerelerin ise normal tahmin
    üretmesi doğrulanır."""
    seq_len = 5
    n_rows = 200
    rng = np.random.default_rng(7)
    columns = FEATURE_COLUMNS[:6]
    data = rng.normal(0, 1, (n_rows, len(columns))).astype("float64")
    raw_features = pd.DataFrame(data, columns=columns)

    contaminated_row = 50
    raw_features.loc[contaminated_row, columns[2]] = float("nan")

    model = LSTMSignalModel(seq_len=seq_len, hidden_size=8, num_layers=1)
    X_train = rng.normal(0, 1, (60, seq_len, len(columns))).astype("float32")
    y_train = rng.choice([-1, 0, 1], size=60)
    model.fit(X_train, y_train, epochs=1, batch_size=16)
    model.feature_columns = columns

    valid_positions = np.arange(seq_len - 1, n_rows)
    directions, confidences = _lstm_predictions_for_series(model, raw_features, valid_positions)

    pos_to_row = {int(p): i for i, p in enumerate(valid_positions)}
    # bitişi 50..54 olan pencereler NaN'lı satırı içerir -> LSTM'siz kalmalı
    for end_pos in range(contaminated_row, contaminated_row + seq_len):
        row_idx = pos_to_row[end_pos]
        assert directions[row_idx] is None
        assert confidences[row_idx] is None

    # kontamine bölgeden uzak bir pencere (ör. bitişi 100) normal tahmin üretmeli
    far_row_idx = pos_to_row[100]
    assert directions[far_row_idx] is not None
    assert confidences[far_row_idx] is not None
    assert not np.isnan(confidences[far_row_idx])


# --- Pozisyon boyutlandırma (_compute_position_size) ---
# Doğrudan birim test: tam bir backtest koşusundan bağımsız, deterministik
# senaryolarla Kelly/fixed_risk dallanmasını, güven ölçeğini ve
# max_position_exposure_pct güvenlik ağını doğrular.


def test_fixed_risk_sizing_matches_stop_distance_formula():
    """fixed_risk boyutu, `calculate_position_size` ile AYNI formülü
    (stop mesafesine göre geriye hesaplanan risk) üretmeli: equity=1000,
    risk=%1 -> risk_amount=10; stop mesafesi %2 -> size=10/0.02=500."""
    request = SystemBacktestRequest(
        position_sizing_method="fixed_risk",
        risk_per_trade_pct=1.0,
        confidence_scaling_enabled=False,
        max_position_exposure_pct=100.0,  # bu testte üst sınır devre dışı
    )
    size_quote, explanation = _compute_position_size(
        equity=1000.0, entry_price=100.0, stop_loss_price=98.0, atr_now=2.0,
        direction="long", confidence=0.9, closed_trade_pnls=[], request=request,
    )
    assert size_quote == pytest.approx(500.0)
    assert "fixed_risk" in explanation


def test_kelly_falls_back_to_fixed_risk_before_min_trades():
    """`position_sizing_method="kelly"` seçili olsa bile, kapanmış işlem
    sayısı `kelly_min_trades`'in ALTINDAYSA Kelly formülü hiç çalıştırılmamalı
    — fixed_risk'e (AYNI stop-mesafesi formülüne) düşmeli."""
    request = SystemBacktestRequest(
        position_sizing_method="kelly",
        kelly_min_trades=20,
        risk_per_trade_pct=1.0,
        confidence_scaling_enabled=False,
        max_position_exposure_pct=100.0,
    )
    size_quote, explanation = _compute_position_size(
        equity=1000.0, entry_price=100.0, stop_loss_price=98.0, atr_now=2.0,
        direction="long", confidence=0.9, closed_trade_pnls=[2.0, -1.0, 1.5, -0.5, 3.0],
        request=request,
    )
    assert size_quote == pytest.approx(500.0), "yetersiz geçmişte fixed_risk ile AYNI boyut beklenir"
    assert "Kelly için yeterli işlem geçmişi yok" in explanation


def test_kelly_activates_after_min_trades_and_uses_kelly_formula():
    """Yeterli (>= kelly_min_trades) kapanmış işlem VE en az bir kayıp
    varsa, boyut Kelly formülünden gelmeli — fixed_risk'ten FARKLI bir
    sonuç (bu senaryoda 250, fixed_risk'in vereceği 500'den farklı)."""
    request = SystemBacktestRequest(
        position_sizing_method="kelly",
        kelly_min_trades=5,
        kelly_multiplier=0.5,
        max_kelly_fraction_pct=25.0,
        risk_per_trade_pct=1.0,
        confidence_scaling_enabled=False,
        max_position_exposure_pct=100.0,
    )
    # 5 işlem: 4 kazanç %2, 1 kayıp %-1 -> kazanma=%80, b=2.0, full Kelly=%70,
    # yarım Kelly=%35, max_kelly_fraction_pct=%25 ile kırpılır -> equity*0.25=250.
    closed_trade_pnls = [2.0, 2.0, 2.0, 2.0, -1.0]
    size_quote, explanation = _compute_position_size(
        equity=1000.0, entry_price=100.0, stop_loss_price=98.0, atr_now=2.0,
        direction="long", confidence=0.9, closed_trade_pnls=closed_trade_pnls, request=request,
    )
    assert size_quote == pytest.approx(250.0)
    assert "Kelly (" in explanation
    assert "tam Kelly=%70.0" in explanation
    assert "uygulanan 0.5x=%25.0" in explanation


def test_kelly_falls_back_safely_when_no_losing_trades_yet():
    """Kenar durum: yeterli işlem sayısı var ama HİÇ kayıp yok (avg_loss=0) —
    Kelly'nin b=kazanç/kayıp oranı sıfıra bölünmeden çöker. fixed_risk'e
    güvenli şekilde düşmeli ve nedeni AÇIKÇA ('yeterli geçmiş yok' değil,
    'henüz kayıp yok') söylemeli — aksi halde işlem geçmişi zaten yeterliyken
    yanlış bir sebep raporlanmış olurdu."""
    request = SystemBacktestRequest(
        position_sizing_method="kelly",
        kelly_min_trades=20,
        risk_per_trade_pct=1.0,
        confidence_scaling_enabled=False,
        max_position_exposure_pct=100.0,
    )
    size_quote, explanation = _compute_position_size(
        equity=1000.0, entry_price=100.0, stop_loss_price=98.0, atr_now=2.0,
        direction="long", confidence=0.9, closed_trade_pnls=[1.0] * 20, request=request,
    )
    assert size_quote == pytest.approx(500.0), "kayıpsız kenar durumda fixed_risk'e düşmeli"
    assert "henüz kayıp işlem yok" in explanation
    assert "yeterli işlem geçmişi yok" not in explanation, "yanıltıcı olurdu: işlem geçmişi zaten yeterli"


@pytest.mark.parametrize(
    "confidence,expected_scale",
    [
        (0.5, 0.5),   # == open_confidence (eşik) -> min_scale
        (0.75, 0.75), # tam ortada -> min_scale ile 1.0 arası ortada
        (1.0, 1.0),   # maksimum güven -> tam boyut
        (0.3, 0.5),   # eşiğin ALTINDA bile olsa min_scale'de kırpılır (negatif ölçek yok)
    ],
)
def test_confidence_scaling_interpolates_between_min_scale_and_full_size(confidence, expected_scale):
    request = SystemBacktestRequest(
        position_sizing_method="fixed_risk",
        risk_per_trade_pct=1.0,
        confidence_scaling_enabled=True,
        open_confidence=0.5,
        confidence_scaling_min_scale=0.5,
        max_position_exposure_pct=100.0,
    )
    size_quote, explanation = _compute_position_size(
        equity=1000.0, entry_price=100.0, stop_loss_price=98.0, atr_now=2.0,
        direction="long", confidence=confidence, closed_trade_pnls=[], request=request,
    )
    base_size = 500.0  # bkz. test_fixed_risk_sizing_matches_stop_distance_formula
    assert size_quote == pytest.approx(base_size * expected_scale)
    assert f"x{expected_scale:.2f}" in explanation


def test_max_position_exposure_pct_caps_oversized_position():
    """Çok dar bir stop mesafesinde (fixed_risk formülü kaldıraç gibi
    davranıp equity'nin KAT KAT üstünde bir boyut önerebilir) güvenlik ağı
    devreye girip `max_position_exposure_pct` sınırına küçültmeli."""
    request = SystemBacktestRequest(
        position_sizing_method="fixed_risk",
        risk_per_trade_pct=1.0,
        confidence_scaling_enabled=False,
        max_position_exposure_pct=15.0,
    )
    # stop mesafesi yalnızca %0.1 -> ham formül equity'nin 10 katını (10000) önerir
    size_quote, explanation = _compute_position_size(
        equity=1000.0, entry_price=100.0, stop_loss_price=99.9, atr_now=2.0,
        direction="long", confidence=0.9, closed_trade_pnls=[], request=request,
    )
    assert size_quote == pytest.approx(150.0), "equity'nin %15'ine (max_position_exposure_pct) kırpılmalı"
    assert "sınırına küçültüldü" in explanation


def test_position_size_falls_back_to_one_atr_when_stop_loss_disabled():
    """`atr_stop_loss_mult=None` (stop-loss kapalı) olsa bile boyutlandırma
    BİR mesafe varsayımına ihtiyaç duyar — kod 1×ATR kullanır. `stop_loss_price=None`
    geçilerek bu yol tetiklenir ve sonucun 1×ATR mesafesiyle tutarlı olduğu
    doğrulanır (entry=100, atr=5 -> mesafe=%5 -> size=10/0.05=200)."""
    request = SystemBacktestRequest(
        position_sizing_method="fixed_risk",
        risk_per_trade_pct=1.0,
        confidence_scaling_enabled=False,
        max_position_exposure_pct=100.0,
    )
    size_quote, _explanation = _compute_position_size(
        equity=1000.0, entry_price=100.0, stop_loss_price=None, atr_now=5.0,
        direction="long", confidence=0.9, closed_trade_pnls=[], request=request,
    )
    assert size_quote == pytest.approx(200.0)


def test_position_size_wired_into_backtest_is_not_flat_full_equity():
    """Regresyon: ESKİ davranışta HER işlem `size_quote = equity` (o anki
    TÜM sermaye) idi — Kelly/fixed-risk yalnızca `warnings`'te bir YORUMDU,
    gerçekte UYGULANMIYORDU. ATR bar bar değiştiği için fixed_risk formülü
    hemen hemen HİÇBİR ZAMAN flat-equity ile aynı sonucu vermez; bu test
    gerçek bir backtest koşusunda en az bir işlemin boyutunun o anki
    equity'den FARKLI olduğunu doğrular."""
    exchange = FakeOscillatingExchange(total_candles=600)
    train_ohlcv = exchange.full_df.iloc[:400].reset_index(drop=True)
    model = _trained_model(train_ohlcv)

    request = SystemBacktestRequest(
        symbol="BTC/USDT:USDT", timeframe="1h", candles=600, initial_balance=1000.0,
        position_sizing_method="fixed_risk", risk_per_trade_pct=1.0,
        atr_stop_loss_mult=1.5, confidence_scaling_enabled=False,
        restrict_to_holdout=False,
    )
    report = run_system_backtest(exchange, model, None, request)

    assert report.trades, "test verisiyle en az bir işlem beklenir"
    equity_before = request.initial_balance
    saw_non_flat_size = False
    for t in report.trades:
        if t.size_quote != pytest.approx(equity_before, rel=1e-6):
            saw_non_flat_size = True
        equity_before = t.equity_after
    assert saw_non_flat_size, "boyutlandırma hâlâ eski 'her zaman tüm equity' davranışına eşit görünüyor"


# --- Güven eşiği taraması (sweep_confidence_thresholds) ---


def test_persist_false_does_not_write_to_backtest_run_table(tmp_path, monkeypatch):
    """`persist=False`, `sweep_confidence_thresholds` gibi arka arkaya çok
    sayıda deneme yapan çağrılarda `backtest_runs` tablosunu/Grafana
    panellerini kirletmemek için vardır — rapor dönmeli ama DB'ye YAZMAMALI."""
    from app.db import repository as db
    from app.db import session as db_session

    monkeypatch.setattr(db_session.settings, "database_url", f"sqlite:///{tmp_path}/test.db")
    db_session.reset_for_tests()
    db_session.init_db()

    exchange = FakeOscillatingExchange(total_candles=600)
    train_ohlcv = exchange.full_df.iloc[:400].reset_index(drop=True)
    model = _trained_model(train_ohlcv)

    request = SystemBacktestRequest(
        symbol="BTC/USDT:USDT", timeframe="1h", candles=600, initial_balance=1000.0, restrict_to_holdout=False
    )
    report = run_system_backtest(exchange, model, None, request, persist=False)

    assert report.id is None
    assert db.get_latest_backtest_run(symbol="BTC/USDT:USDT") is None, "persist=False olsa bile DB'ye yazılmış"

    db_session.reset_for_tests()


def test_sweep_confidence_thresholds_covers_all_values_without_persisting(tmp_path, monkeypatch):
    """Her `open_confidence` adayı için bir nokta dönmeli, `close_confidence`
    doğru şekilde türetilmeli (open - gap) ve ara denemeler `backtest_runs`
    tablosunu KİRLETMEMELİ (bkz. `test_persist_false_...`)."""
    from app.db import repository as db
    from app.db import session as db_session

    monkeypatch.setattr(db_session.settings, "database_url", f"sqlite:///{tmp_path}/test.db")
    db_session.reset_for_tests()
    db_session.init_db()

    exchange = FakeOscillatingExchange(total_candles=600)
    train_ohlcv = exchange.full_df.iloc[:400].reset_index(drop=True)
    model = _trained_model(train_ohlcv)

    base_request = SystemBacktestRequest(
        symbol="BTC/USDT:USDT", timeframe="1h", candles=600, initial_balance=1000.0, restrict_to_holdout=False
    )
    values = [0.5, 0.6, 0.7]
    points = sweep_confidence_thresholds(exchange, model, None, base_request, values, close_confidence_gap=0.05)

    assert [p.open_confidence for p in points] == values
    for p in points:
        assert p.close_confidence == pytest.approx(p.open_confidence - 0.05)
        assert p.error is None

    assert db.get_latest_backtest_run(symbol="BTC/USDT:USDT") is None, "tarama backtest_runs tablosunu kirletmemeli"

    db_session.reset_for_tests()


def test_sweep_confidence_thresholds_records_error_without_stopping(monkeypatch):
    """Bir eşikte backtest çökerse (ör. beklenmedik bir ValueError) taramanın
    GERİ KALANI durmamalı — hatalı nokta `error` alanıyla işaretlenip diğer
    eşikler normal şekilde denenmeye devam etmeli (bkz.
    `app.ml.train.sweep_lookback_values` ile AYNI dayanıklılık deseni)."""
    from app.backtest import system_runner

    def fake_run_system_backtest(exchange, model, meta_model, request, lstm_model=None, online_model=None, persist=True):
        if request.open_confidence == 0.6:
            raise ValueError("simüle edilmiş hata")
        return SystemBacktestReport(
            symbol=request.symbol,
            timeframe="1h",
            candles_used=100,
            period_start="2024-01-01T00:00:00",
            period_end="2024-01-05T00:00:00",
            initial_balance=1000.0,
            final_equity=1000.0,
            trades_closed=1,
            win_rate_pct=100.0,
            total_pnl_quote=0.0,
            total_pnl_pct=0.0,
            daily_pnl_quote=0.0,
            daily_pnl_pct=0.0,
            monthly_pnl_quote=0.0,
            monthly_pnl_pct=0.0,
            max_drawdown_pct=0.0,
            trades=[],
        )

    monkeypatch.setattr(system_runner, "run_system_backtest", fake_run_system_backtest)

    points = system_runner.sweep_confidence_thresholds(
        exchange=None,
        model=None,
        meta_model=None,
        base_request=SystemBacktestRequest(),
        open_confidence_values=[0.5, 0.6, 0.7],
    )

    assert [p.open_confidence for p in points] == [0.5, 0.6, 0.7]
    assert points[1].error is not None and "simüle edilmiş hata" in points[1].error
    assert points[0].error is None and points[0].trades_closed == 1
    assert points[2].error is None and points[2].trades_closed == 1


# --- Pozisyon boyutlandırma taraması (sweep_position_sizing) ---


def test_sweep_position_sizing_covers_full_grid_without_persisting(tmp_path, monkeypatch):
    """`kelly_min_trades` x `kelly_multiplier` ızgarasının HER kombinasyonu
    için bir nokta dönmeli, ara denemeler `backtest_runs` tablosunu
    KİRLETMEMELİ (bkz. `test_persist_false_...`)."""
    from app.backtest.system_runner import sweep_position_sizing
    from app.db import repository as db
    from app.db import session as db_session

    monkeypatch.setattr(db_session.settings, "database_url", f"sqlite:///{tmp_path}/test.db")
    db_session.reset_for_tests()
    db_session.init_db()

    exchange = FakeOscillatingExchange(total_candles=600)
    train_ohlcv = exchange.full_df.iloc[:400].reset_index(drop=True)
    model = _trained_model(train_ohlcv)

    base_request = SystemBacktestRequest(
        symbol="BTC/USDT:USDT", timeframe="1h", candles=600, initial_balance=1000.0, restrict_to_holdout=False
    )
    min_trades_values = [10, 20]
    multiplier_values = [0.5, 1.0]
    points = sweep_position_sizing(exchange, model, None, base_request, min_trades_values, multiplier_values)

    assert len(points) == len(min_trades_values) * len(multiplier_values)
    assert {(p.kelly_min_trades, p.kelly_multiplier) for p in points} == {
        (mt, mu) for mt in min_trades_values for mu in multiplier_values
    }
    for p in points:
        assert p.error is None

    assert db.get_latest_backtest_run(symbol="BTC/USDT:USDT") is None, "tarama backtest_runs tablosunu kirletmemeli"

    db_session.reset_for_tests()


def test_sweep_position_sizing_records_error_without_stopping(monkeypatch):
    """Bir kombinasyonda backtest çökerse taramanın GERİ KALANI durmamalı —
    hatalı nokta `error` alanıyla işaretlenip diğer kombinasyonlar normal
    şekilde denenmeye devam etmeli (bkz. `sweep_confidence_thresholds` ile
    AYNI dayanıklılık deseni)."""
    from app.backtest import system_runner

    def fake_run_system_backtest(exchange, model, meta_model, request, lstm_model=None, online_model=None, persist=True):
        if request.kelly_min_trades == 20:
            raise ValueError("simüle edilmiş hata")
        return SystemBacktestReport(
            symbol=request.symbol,
            timeframe="1h",
            candles_used=100,
            period_start="2024-01-01T00:00:00",
            period_end="2024-01-05T00:00:00",
            initial_balance=1000.0,
            final_equity=1000.0,
            trades_closed=1,
            win_rate_pct=100.0,
            total_pnl_quote=0.0,
            total_pnl_pct=0.0,
            daily_pnl_quote=0.0,
            daily_pnl_pct=0.0,
            monthly_pnl_quote=0.0,
            monthly_pnl_pct=0.0,
            max_drawdown_pct=0.0,
            trades=[],
        )

    monkeypatch.setattr(system_runner, "run_system_backtest", fake_run_system_backtest)

    points = system_runner.sweep_position_sizing(
        exchange=None,
        model=None,
        meta_model=None,
        base_request=SystemBacktestRequest(),
        kelly_min_trades_values=[10, 20],
        kelly_multiplier_values=[0.5],
    )

    assert len(points) == 2
    ok_point = next(p for p in points if p.kelly_min_trades == 10)
    err_point = next(p for p in points if p.kelly_min_trades == 20)
    assert ok_point.error is None and ok_point.trades_closed == 1
    assert err_point.error is not None and "simüle edilmiş hata" in err_point.error


# --- Periyodik optimizasyon (run_periodic_optimization) ---


def test_run_periodic_optimization_skips_unreliable_points_and_never_persists(monkeypatch):
    """`run_periodic_optimization` (1) az işlemli/güvenilmez noktaları
    (`MIN_RELIABLE_TRADES_FOR_OPTIMIZATION`'ın altında) ASLA seçmemeli —
    ne kadar cazip bir PnL gösterirse göstersin, (2) her çağrıyı
    `persist=False` ile yapmalı (bkz. README 'karlılık': canlı ayarları
    OTOMATİK DEĞİŞTİRMEZ, `backtest_runs` tablosunu kirletmemeli)."""
    from app.backtest import system_runner

    def fake_run_system_backtest(exchange, model, meta_model, request, lstm_model=None, online_model=None, persist=True):
        assert persist is False, "sweep/optimizasyon çağrıları backtest_runs'ı kirletmemeli"
        if request.open_confidence == 0.7:
            trades, base_pnl = 5, 999.0  # cazip ama GÜVENİLMEZ (az işlem) — seçilmemeli
        elif request.open_confidence == 0.6:
            trades, base_pnl = 50, 3.0
        else:
            trades, base_pnl = 50, 1.0
        kelly_bonus = 2.0 if request.kelly_multiplier == 1.0 else 0.0
        min_trades_bonus = 0.5 if request.kelly_min_trades == 20 else 0.0
        return SystemBacktestReport(
            symbol=request.symbol,
            timeframe="1h",
            candles_used=100,
            period_start="2024-01-01T00:00:00",
            period_end="2024-01-05T00:00:00",
            initial_balance=1000.0,
            final_equity=1000.0,
            trades_closed=trades,
            win_rate_pct=60.0,
            total_pnl_quote=0.0,
            total_pnl_pct=base_pnl + kelly_bonus + min_trades_bonus,
            daily_pnl_quote=0.0,
            daily_pnl_pct=0.0,
            monthly_pnl_quote=0.0,
            monthly_pnl_pct=0.0,
            max_drawdown_pct=1.0,
            trades=[],
        )

    monkeypatch.setattr(system_runner, "run_system_backtest", fake_run_system_backtest)

    base_request = SystemBacktestRequest()  # varsayılan: open=0.5, kelly_multiplier=0.75, kelly_min_trades=40
    result = system_runner.run_periodic_optimization(
        exchange=None,
        model=None,
        meta_model=None,
        base_request=base_request,
        open_confidence_values=[0.5, 0.6, 0.7],
        kelly_min_trades_values=[10, 20],
        kelly_multiplier_values=[0.5, 1.0],
    )

    # 0.7 cazip (999) ama trades=5 < MIN_RELIABLE_TRADES_FOR_OPTIMIZATION -> ASLA seçilmemeli
    assert result.recommended_open_confidence == 0.6
    assert result.recommended_close_confidence == pytest.approx(0.55)
    assert result.recommended_kelly_multiplier == 1.0
    assert result.recommended_kelly_min_trades == 20
    assert result.recommended_total_pnl_pct == pytest.approx(3.0 + 2.0 + 0.5)

    # mevcut (base_request'in KENDİ değerleri) ayrı raporlanmalı, önerilenle KARIŞTIRILMAMALI
    assert result.current_open_confidence == base_request.open_confidence
    assert result.current_kelly_multiplier == base_request.kelly_multiplier
    assert result.current_kelly_min_trades == base_request.kelly_min_trades
    assert result.current_total_pnl_pct == pytest.approx(1.0)
