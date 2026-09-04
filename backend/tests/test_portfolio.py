import numpy as np
import pandas as pd
import pytest

from app.engine.decision import DecisionEngine
from app.engine.positions import PaperPositionStore
from app.exchanges.base import Exchange
from app.portfolio.manager import PortfolioManager
from app.portfolio.risk_manager import calculate_position_size, evaluate_risk
from app.portfolio.schemas import PositionExposure, RiskRules


def test_calculate_position_size_matches_risk_amount():
    size_quote, risk_amount, stop_distance_pct = calculate_position_size(
        equity=1000, entry_price=100, stop_loss_price=95, risk_per_trade_pct=1.0, direction="long"
    )
    assert risk_amount == pytest.approx(10.0)
    assert stop_distance_pct == pytest.approx(5.0)
    # SL'e çarpılırsa kaybedilen tutar tam olarak risk_amount olmalı
    loss_if_sl_hit = size_quote * (stop_distance_pct / 100)
    assert loss_if_sl_hit == pytest.approx(risk_amount)


def test_calculate_position_size_rejects_invalid_stop_loss():
    with pytest.raises(ValueError):
        calculate_position_size(1000, 100, 105, 1.0, direction="long")


def test_evaluate_risk_blocks_after_daily_loss_limit():
    rules = RiskRules(daily_loss_limit_pct=5.0)
    decision = evaluate_risk(
        equity=1000,
        open_positions=[],
        realized_pnl_session=-60.0,  # -%6, limit %5
        proposed_symbol="BTC/USDT",
        proposed_size_quote=100,
        rules=rules,
    )
    assert decision.allowed is False


def test_evaluate_risk_caps_symbol_exposure():
    rules = RiskRules(max_symbol_exposure_pct=10.0, max_total_exposure_pct=100.0)
    decision = evaluate_risk(
        equity=1000,
        open_positions=[PositionExposure(symbol="BTC/USDT", size_quote=80)],
        realized_pnl_session=0.0,
        proposed_symbol="BTC/USDT",
        proposed_size_quote=100,
        rules=rules,
    )
    assert decision.allowed is True
    assert decision.size_quote == pytest.approx(20.0)  # 1000*%10 - 80 = 20


def test_evaluate_risk_blocks_new_symbol_over_concurrent_limit():
    rules = RiskRules(max_concurrent_positions=1)
    decision = evaluate_risk(
        equity=1000,
        open_positions=[PositionExposure(symbol="BTC/USDT", size_quote=50)],
        realized_pnl_session=0.0,
        proposed_symbol="ETH/USDT",
        proposed_size_quote=50,
        rules=rules,
    )
    assert decision.allowed is False


def test_portfolio_manager_open_close_updates_equity():
    # commission_pct/slippage_pct=0: bu test saf fiyat farkı PnL'ini
    # ölçüyor, işlem maliyetlerini değil (bkz. test_close_applies_transaction_costs).
    portfolio = PortfolioManager(starting_equity=1000, rules=RiskRules(commission_pct=0, slippage_pct=0))
    decision = portfolio.propose_open("BTC/USDT", "long", entry_price=100, stop_loss_price=97)
    assert decision.allowed is True
    portfolio.open("BTC/USDT", "long", 100, decision.size_quote)
    assert portfolio.get("BTC/USDT") is not None

    record = portfolio.close("BTC/USDT", exit_price=106)  # +%6 kâr, avg_price bazlı değil doğrudan pnl
    assert record is not None
    assert record["pnl_pct"] == pytest.approx(6.0)
    assert portfolio.equity > portfolio.starting_equity


def test_open_only_fills_first_entry_tranche():
    rules = RiskRules(entry_tranche_weights=[0.4, 0.6])
    portfolio = PortfolioManager(starting_equity=1000, rules=rules)
    portfolio.open("BTC/USDT", "long", entry_price=100, size_quote=1000)

    position = portfolio.get("BTC/USDT")
    assert position.size_quote == pytest.approx(400.0)
    assert position.target_size_quote == pytest.approx(1000.0)
    assert position.entry_fill_index == 1
    assert not position.entry_fully_filled()


def test_add_entry_tranche_fills_remaining_and_recomputes_avg_price():
    rules = RiskRules(entry_tranche_weights=[0.5, 0.5])
    portfolio = PortfolioManager(starting_equity=1000, rules=rules)
    portfolio.open("BTC/USDT", "long", entry_price=100, size_quote=1000)  # 500 @ 100

    portfolio.add_entry_tranche("BTC/USDT", price=110)  # +500 @ 110

    position = portfolio.get("BTC/USDT")
    assert position.size_quote == pytest.approx(1000.0)
    assert position.entry_price == pytest.approx(105.0)  # ağırlıklı ortalama (500*100 + 500*110)/1000
    assert position.entry_fully_filled()


def test_add_entry_tranche_is_noop_when_already_fully_filled():
    rules = RiskRules(entry_tranche_weights=[1.0])
    portfolio = PortfolioManager(starting_equity=1000, rules=rules)
    portfolio.open("BTC/USDT", "long", entry_price=100, size_quote=1000)

    result = portfolio.add_entry_tranche("BTC/USDT", price=110)
    assert result is None
    assert portfolio.get("BTC/USDT").size_quote == pytest.approx(1000.0)


def test_close_tranche_keeps_position_open_until_final_tranche():
    rules = RiskRules(entry_tranche_weights=[1.0], exit_tranche_weights=[0.5, 0.5])
    portfolio = PortfolioManager(starting_equity=1000, rules=rules)
    portfolio.open("BTC/USDT", "long", entry_price=100, size_quote=1000)

    first = portfolio.close_tranche("BTC/USDT", exit_price=110)
    assert first["partial"] is True
    assert first["size_quote"] == pytest.approx(500.0)
    assert portfolio.get("BTC/USDT") is not None
    assert portfolio.get("BTC/USDT").size_quote == pytest.approx(500.0)

    second = portfolio.close_tranche("BTC/USDT", exit_price=115)
    assert second["partial"] is False
    assert second["size_quote"] == pytest.approx(500.0)
    assert portfolio.get("BTC/USDT") is None  # tamamen kapandı


def test_close_tranche_last_tranche_closes_full_remaining_size_avoiding_dust():
    # 3 eşit olmayan dilim: yuvarlama artığı son dilimde TAMAMEN kapatılarak giderilmeli
    rules = RiskRules(entry_tranche_weights=[1.0], exit_tranche_weights=[0.33, 0.33, 0.34])
    portfolio = PortfolioManager(starting_equity=1000, rules=rules)
    portfolio.open("BTC/USDT", "long", entry_price=100, size_quote=999)

    portfolio.close_tranche("BTC/USDT", exit_price=100)
    portfolio.close_tranche("BTC/USDT", exit_price=100)
    last = portfolio.close_tranche("BTC/USDT", exit_price=100)

    assert last["partial"] is False
    assert portfolio.get("BTC/USDT") is None


def test_close_forces_full_immediate_close_ignoring_exit_tranches():
    rules = RiskRules(entry_tranche_weights=[1.0], exit_tranche_weights=[0.5, 0.5])
    portfolio = PortfolioManager(starting_equity=1000, rules=rules)
    portfolio.open("BTC/USDT", "long", entry_price=100, size_quote=1000)

    record = portfolio.close("BTC/USDT", exit_price=110)
    assert record["partial"] is False
    assert record["size_quote"] == pytest.approx(1000.0)
    assert portfolio.get("BTC/USDT") is None


def test_propose_open_scales_size_by_confidence():
    rules = RiskRules(max_symbol_exposure_pct=100, max_total_exposure_pct=100, confidence_scaling_min_confidence=0.6, confidence_scaling_min_scale=0.5)
    portfolio = PortfolioManager(starting_equity=1000, rules=rules)

    low = portfolio.propose_open("BTC/USDT", "long", 100, 97, confidence=0.6)
    high = portfolio.propose_open("BTC/USDT", "long", 100, 97, confidence=1.0)
    mid = portfolio.propose_open("BTC/USDT", "long", 100, 97, confidence=0.8)

    assert low.size_quote == pytest.approx(high.size_quote * 0.5)
    assert low.size_quote < mid.size_quote < high.size_quote


def test_propose_open_confidence_scaling_disabled_ignores_confidence():
    rules = RiskRules(max_symbol_exposure_pct=100, max_total_exposure_pct=100, confidence_scaling_enabled=False)
    portfolio = PortfolioManager(starting_equity=1000, rules=rules)

    low = portfolio.propose_open("BTC/USDT", "long", 100, 97, confidence=0.6)
    high = portfolio.propose_open("BTC/USDT", "long", 100, 97, confidence=1.0)
    assert low.size_quote == pytest.approx(high.size_quote)


def test_propose_open_vix_regime_filter_blocks_above_threshold():
    rules = RiskRules(
        max_symbol_exposure_pct=100, max_total_exposure_pct=100, vix_regime_filter_enabled=True, vix_zscore_block_threshold=2.5
    )
    portfolio = PortfolioManager(starting_equity=1000, rules=rules)

    decision = portfolio.propose_open("BTC/USDT", "long", 100, 97, vix_zscore=3.0)
    assert decision.allowed is False
    assert decision.size_quote == 0


def test_propose_open_vix_regime_filter_halves_size_above_reduce_threshold():
    rules = RiskRules(
        max_symbol_exposure_pct=100,
        max_total_exposure_pct=100,
        vix_regime_filter_enabled=True,
        vix_zscore_reduce_threshold=1.5,
        vix_zscore_block_threshold=2.5,
        confidence_scaling_enabled=False,
    )
    portfolio = PortfolioManager(starting_equity=1000, rules=rules)

    normal = portfolio.propose_open("BTC/USDT", "long", 100, 97, vix_zscore=0.0)
    stressed = portfolio.propose_open("BTC/USDT", "long", 100, 97, vix_zscore=2.0)
    assert stressed.size_quote == pytest.approx(normal.size_quote * 0.5)


def test_propose_open_vix_regime_filter_disabled_by_default():
    rules = RiskRules(max_symbol_exposure_pct=100, max_total_exposure_pct=100, confidence_scaling_enabled=False)
    portfolio = PortfolioManager(starting_equity=1000, rules=rules)

    decision = portfolio.propose_open("BTC/USDT", "long", 100, 97, vix_zscore=10.0)  # aşırı yüksek olsa da filtre kapalı
    assert decision.allowed is True
    assert decision.size_quote > 0


def test_pnl_summary_totals_match_closed_history():
    # commission_pct/slippage_pct=0: bu test saf fiyat farkı PnL'ini ölçüyor.
    portfolio = PortfolioManager(starting_equity=1000, rules=RiskRules(entry_tranche_weights=[1.0], commission_pct=0, slippage_pct=0))
    portfolio.open("BTC/USDT", "long", entry_price=100, size_quote=1000)
    portfolio.close("BTC/USDT", exit_price=110)  # +%10 -> +100 quote

    summary = portfolio.pnl_summary()
    assert summary.total.pnl_quote == pytest.approx(100.0)
    assert summary.total.trade_count == 1
    assert summary.daily.pnl_quote == pytest.approx(100.0)  # az önce kapandı -> son 24s içinde
    assert summary.weekly.pnl_quote == pytest.approx(100.0)
    assert summary.monthly.pnl_quote == pytest.approx(100.0)


class _StaticOhlcvExchange(Exchange):
    """`DecisionEngine._predict`'in ihtiyaç duyduğu kadar OHLCV üreten,
    ama tahmini SABİT bir sahte modelden alan testler için minimal borsa."""

    def list_symbols(self, quote_currency: str, market_type: str) -> list[str]:
        return ["BTC/USDT"]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, since=None) -> pd.DataFrame:
        n = max(limit, 220)
        close = np.linspace(100, 110, n)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.full(n, 1000.0),
            }
        )


class _FixedModel:
    """Her zaman aynı tahmini döndüren sahte model — kademeli alım/satım
    akışını gerçek bir XGBoost eğitimi olmadan, deterministik test etmek için."""

    def __init__(self, direction: str, confidence: float) -> None:
        from app.ml.model import Prediction

        self._prediction = Prediction(direction=direction, confidence=confidence)

    def predict(self, feature_row):
        return self._prediction


def test_decision_engine_fills_second_entry_tranche_on_next_cycle_when_signal_persists():
    portfolio = PortfolioManager(
        starting_equity=1000,
        rules=RiskRules(entry_tranche_weights=[0.5, 0.5], max_symbol_exposure_pct=100, max_total_exposure_pct=100),
    )
    engine = DecisionEngine(
        exchange=_StaticOhlcvExchange(),
        model=_FixedModel("long", 0.9),
        positions=PaperPositionStore(),
        timeframe="4h",
        lookback=220,
        open_confidence=0.6,
        close_confidence=0.55,
        portfolio=portfolio,
    )

    first_actions = engine.run_cycle(["BTC/USDT"])
    assert first_actions[0].type == "open_long"
    position = portfolio.get("BTC/USDT")
    assert position.entry_fill_index == 1
    first_size = position.size_quote

    second_actions = engine.run_cycle(["BTC/USDT"])  # sinyal hâlâ aynı yönde -> 2. dilim eklenmeli
    assert second_actions[0].type == "add_entry_tranche"
    position = portfolio.get("BTC/USDT")
    assert position.entry_fully_filled()
    assert position.size_quote > first_size

    third_actions = engine.run_cycle(["BTC/USDT"])  # tüm dilimler dolu -> artık "hold"
    assert third_actions[0].type == "hold"


def test_decision_engine_closes_position_in_stages_across_cycles():
    portfolio = PortfolioManager(
        starting_equity=1000,
        rules=RiskRules(
            entry_tranche_weights=[1.0], exit_tranche_weights=[0.5, 0.5], max_symbol_exposure_pct=100, max_total_exposure_pct=100
        ),
    )
    engine = DecisionEngine(
        exchange=_StaticOhlcvExchange(),
        model=_FixedModel("long", 0.9),
        positions=PaperPositionStore(),
        timeframe="4h",
        lookback=220,
        open_confidence=0.6,
        close_confidence=0.55,
        portfolio=portfolio,
    )
    engine.run_cycle(["BTC/USDT"])
    assert portfolio.get("BTC/USDT").entry_fully_filled()

    engine.model = _FixedModel("short", 0.9)  # ters sinyal -> kapanış tetiklenir
    first_close = engine.run_cycle(["BTC/USDT"])
    assert first_close[0].type == "close_partial"
    assert portfolio.get("BTC/USDT") is not None

    second_close = engine.run_cycle(["BTC/USDT"])
    assert second_close[0].type == "close"
    assert portfolio.get("BTC/USDT") is None


class _TrendExchange(Exchange):
    def __init__(self) -> None:
        self._rng = np.random.default_rng(11)

    def list_symbols(self, quote_currency: str, market_type: str) -> list[str]:
        return ["UPUSDT"]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        n = max(limit, 220)
        base = np.linspace(100, 260, n)
        close = base + self._rng.normal(0, 1.0, n)
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


def test_decision_engine_uses_portfolio_manager_for_sizing():
    from app.ml.dataset import build_training_dataset
    from app.ml.model import SignalModel

    exchange = _TrendExchange()
    X, y = build_training_dataset(exchange, ["UPUSDT"], "4h", 220, horizon=5, threshold_pct=1.0)
    model = SignalModel()
    model.fit(X, y)

    portfolio = PortfolioManager(starting_equity=1000, rules=RiskRules(max_risk_per_trade_pct=1.0))
    engine = DecisionEngine(
        exchange=exchange,
        model=model,
        positions=PaperPositionStore(),
        timeframe="4h",
        lookback=220,
        open_confidence=0.0,
        close_confidence=0.0,
        portfolio=portfolio,
    )

    actions = engine.run_cycle(["UPUSDT"])
    assert len(actions) == 1
    action = actions[0]
    if action.type == "open_long":
        position = portfolio.get("UPUSDT")
        assert position is not None
        assert position.size_quote > 0
        # %1 risk, %3 varsayılan SL mesafesi -> boyut equity'nin küçük bir kısmı olmalı
        assert position.size_quote < portfolio.equity


def test_close_applies_transaction_costs():
    # commission_pct=0.05, slippage_pct=0.03 -> round-trip maliyet = (0.05+0.03)*2 = %0.16
    portfolio = PortfolioManager(
        starting_equity=1000,
        rules=RiskRules(entry_tranche_weights=[1.0], commission_pct=0.05, slippage_pct=0.03),
    )
    portfolio.open("BTC/USDT", "long", entry_price=100, size_quote=1000)
    record = portfolio.close("BTC/USDT", exit_price=110)  # brüt +%10
    assert record is not None
    assert record["pnl_pct"] == pytest.approx(10.0 - 0.16)


def test_close_zero_cost_matches_gross_pnl():
    portfolio = PortfolioManager(
        starting_equity=1000, rules=RiskRules(entry_tranche_weights=[1.0], commission_pct=0, slippage_pct=0)
    )
    portfolio.open("BTC/USDT", "long", entry_price=100, size_quote=1000)
    record = portfolio.close("BTC/USDT", exit_price=110)
    assert record["pnl_pct"] == pytest.approx(10.0)


def test_portfolio_position_stop_loss_breached_long():
    position = PortfolioManager(starting_equity=1000).open("BTC/USDT", "long", entry_price=100, size_quote=100, stop_loss_price=95)
    assert position.stop_loss_breached(96) is False
    assert position.stop_loss_breached(95) is True
    assert position.stop_loss_breached(94) is True


def test_portfolio_position_stop_loss_breached_short():
    position = PortfolioManager(starting_equity=1000).open("BTC/USDT", "short", entry_price=100, size_quote=100, stop_loss_price=105)
    assert position.stop_loss_breached(104) is False
    assert position.stop_loss_breached(105) is True
    assert position.stop_loss_breached(106) is True


def test_portfolio_position_no_stop_loss_never_breached():
    position = PortfolioManager(starting_equity=1000).open("BTC/USDT", "long", entry_price=100, size_quote=100)
    assert position.stop_loss_price is None
    assert position.stop_loss_breached(0.01) is False


def test_decision_engine_force_closes_on_stop_loss_breach():
    """Model hâlâ "tut" (nötr güven eşiğinin altında) dese bile, fiyat
    stop-loss seviyesini geçtiyse pozisyon zorla kapatılmalı."""

    class _StubModel:
        def predict(self, feature_row):
            from app.ml.model import Prediction

            return Prediction(direction="neutral", confidence=0.0)

    class _CrashExchange(Exchange):
        def list_symbols(self, quote_currency, market_type):
            return ["UPUSDT"]

        def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
            n = max(limit, 220)
            rng = np.random.default_rng(0)
            close = 100 + rng.normal(0, 0.1, n)
            # Son ~30 bar düşük (90) — Ichimoku bulutu gibi ileri kaydırmalı
            # göstergeler `latest_feature_vector`in EN SON bar yerine dropna
            # sonrası kalan son geçerli bardan (biraz daha geriden) okumasına
            # yol açabiliyor; tek bir bar yerine bir blok düşürmek testi
            # bu ayrıntıya duyarsız kılıyor.
            close[-30:] = 90.0  # ani düşüş -> stop-loss'u tetiklemeli
            return pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=n, freq="4h"),
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": rng.uniform(800, 1200, n),
                }
            )

    exchange = _CrashExchange()
    portfolio = PortfolioManager(starting_equity=1000, rules=RiskRules(entry_tranche_weights=[1.0]))
    portfolio.open("UPUSDT", "long", entry_price=100, size_quote=100, stop_loss_price=95)

    engine = DecisionEngine(
        exchange=exchange,
        model=_StubModel(),
        positions=PaperPositionStore(),
        timeframe="4h",
        lookback=220,
        open_confidence=0.9,
        close_confidence=0.9,
        portfolio=portfolio,
    )

    action = engine.evaluate("UPUSDT")
    assert action is not None
    assert action.type == "close"
    assert "stop-loss" in action.reason
