from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.dca.schemas import DCAParams
from app.strategy.schemas import StrategyDefinition


class PerformanceMetrics(BaseModel):
    num_trades: int
    win_rate_pct: float
    total_return_pct: float
    cagr_pct: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    max_drawdown_pct: float
    calmar_ratio: float | None
    profit_factor: float | None
    avg_win_pct: float
    avg_loss_pct: float
    expectancy_pct: float


class MonteCarloReport(BaseModel):
    """İşlem sırasının/örneklemenin "şans" bileşenini ölçmek için, GERÇEKLEŞEN
    işlem getirilerinin YERİNE KOYARAK (with replacement) tekrar tekrar
    örneklenmesiyle üretilen dağılım — `vectorbt` gibi ağır bağımlılıklar
    olmadan, mevcut backtest motorunun üstüne eklenen bir katman.

    Yorumlama: `total_return_pct_p5`, "simülasyonların %95'i bundan daha
    iyi sonuç verdi" demektir (kötümser uç); `max_drawdown_pct_p95`,
    "simülasyonların yalnızca %5'i bundan daha kötü bir drawdown gördü"
    demektir (karşılaşabileceğiniz kötü senaryolara yakın bir üst sınır)."""

    num_simulations: int
    num_trades_per_simulation: int
    total_return_pct_p5: float
    total_return_pct_p50: float
    total_return_pct_p95: float
    max_drawdown_pct_p50: float
    max_drawdown_pct_p95: float
    probability_of_loss_pct: float


class DataSufficiency(BaseModel):
    candles_used: int
    candles_requested_final: int
    trades_found: int
    min_trades_target: int
    sufficient: bool
    reason: str


class BacktestReport(BaseModel):
    symbol: str
    timeframe: str
    data_sufficiency: DataSufficiency
    train_metrics: PerformanceMetrics | None
    test_metrics: PerformanceMetrics | None
    full_period_metrics: PerformanceMetrics | None
    monte_carlo: MonteCarloReport | None = None
    warnings: list[str] = Field(default_factory=list)


class BacktestRequest(BaseModel):
    symbol: str
    timeframe: str | None = None
    dca_params: DCAParams | None = None
    strategy: StrategyDefinition | None = None
    min_trades: int = Field(30, ge=5, le=500, description="İstatistiksel olarak yeterli sayılacak minimum kapanan işlem sayısı")
    max_candles: int = Field(5000, ge=100, le=20000, description="Geriye doğru çekilecek maksimum mum sayısı")
    initial_candles: int = Field(500, ge=50, description="Veri yeterliliği taramasının başlangıç adımı")
    train_ratio: float = Field(0.7, gt=0.3, lt=0.95, description="Kronolojik eğitim (in-sample) bölümünün oranı")
    monte_carlo_simulations: int = Field(
        1000, ge=0, le=20000,
        description="Test (out-of-sample) dönemi işlem getirilerinin yeniden örneklenme (bootstrap) sayısı. 0 = Monte Carlo atlanır.",
    )

    @model_validator(mode="after")
    def _exactly_one_target(self) -> "BacktestRequest":
        if (self.dca_params is None) == (self.strategy is None):
            raise ValueError("Tam olarak biri verilmeli: dca_params VEYA strategy.")
        return self

    @property
    def kind(self) -> Literal["dca", "strategy"]:
        return "dca" if self.dca_params is not None else "strategy"


class SystemBacktestRequest(BaseModel):
    """`app.backtest.system_runner`: DCA/JSON-strateji motorlarından FARKLI
    olarak burada canlıdaki AYNI ML modeli (XGBoost + varsa meta-label
    filtresi) gerçek geçmiş mumlar üzerinde bar-bar tekrar oynatılır —
    "sistemin kendisinin" geçmişte nasıl performans gösterdiğini ölçer."""

    symbol: str = Field(default="BTC/USDT:USDT", description="Varsayılan: futures perpetual BTC/USDT (ml_primary_symbol)")
    timeframe: str | None = Field(default=None, description="Boş bırakılırsa ml_train_timeframe (1h) kullanılır")
    candles: int = Field(default=10000, ge=300, le=20000)
    initial_balance: float = Field(default=1000.0, gt=0)
    open_confidence: float = Field(default=0.6, ge=0.5, le=1.0)
    close_confidence: float = Field(default=0.55, ge=0.5, le=1.0)
    commission_pct: float = Field(default=0.04, ge=0)
    slippage_pct: float = Field(default=0.02, ge=0)
    use_meta_label: bool = Field(default=True, description="Eğitilmiş bir meta-label modeli varsa sinyal filtresi olarak kullanılır")
    use_ensemble: bool = Field(
        default=True, description="Kullanılabilirse (bkz. app.ml.model_status) LSTM/online modeli de canlıdaki gibi ensemble'a katar"
    )

    # --- ATR tabanlı risk yönetimi ---
    # Sabit yüzdelik stop-loss YERİNE: volatiliteye göre ölçeklenen ATR
    # (Average True Range) tabanlı stop-loss (opsiyonel olarak kâr-alma +
    # trailing stop da desteklenir, ama VARSAYILAN OLARAK KAPALI).
    #
    # Neden: ilk denemede (kullanıcı isteğiyle) kâr-alma=1.5xATR +
    # trailing=0.5xATR VARSAYILAN AÇIKTI — gerçek üretim testinde toplam
    # PnL %88'den %32'ye, kazanma oranı %90'dan %57'ye düştü. Stop-loss
    # yalnızca KAYBEDEN işlemleri sınırlar (kazananları asla kesmez), ama
    # kâr-alma ve (özellikle 0.5xATR gibi dar bir) trailing MEKANİK olarak
    # kazanan bir işlemi SABİT bir mesafede keser — modelin kendi (canlı
    # ensemble) sinyali hâlâ o yönde güçlüyken bile. Eski (ATR öncesi)
    # "dinamik" yöntemde çıkış kararını her barda YENİDEN üretilen model
    # sinyali veriyordu (bkz. `close_confidence`), sabit bir mesafe değil
    # — bu, kazananların "koşmasına" izin veriyordu. Bu yüzden varsayılan
    # olarak kâr-alma/trailing KAPALI: çıkış yine birincil olarak dinamik
    # sinyale dayanıyor, stop-loss ise SADECE ATR ile ölçeklenen (eski
    # sabit yüzdelik `stop_loss_pct=3.0`'tan daha isabetli) bir güvenlik
    # ağı olarak kalıyor. İsteyen kullanıcı bu alanları elle açabilir.
    atr_period: int = Field(default=14, ge=2, le=100)
    atr_stop_loss_mult: float | None = Field(default=1.5, gt=0, description="null ise stop-loss uygulanmaz")
    atr_take_profit_mult: float | None = Field(
        default=None, gt=0, description="null (varsayılan) ise kâr-alma uygulanmaz, çıkış dinamik sinyale bağlı kalır"
    )
    atr_trailing_mult: float | None = Field(
        default=None,
        gt=0,
        description="En iyi fiyattan bu kadar ATR geride trailing stop; null (varsayılan) ise trailing yok",
    )


class SystemTradeRecord(BaseModel):
    direction: str  # "long" | "short"
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_quote: float
    equity_after: float
    exit_reason: str  # "signal" | "stop_loss" | "take_profit" | "trailing_stop"
    duration_candles: int
    size_quote: float
    size_explanation: str
    xgboost_direction: str
    xgboost_confidence: float
    lstm_direction: str | None = None
    lstm_confidence: float | None = None
    online_direction: str | None = None
    online_confidence: float | None = None
    decision_reason: str


class SystemBacktestReport(BaseModel):
    id: int | None = None
    created_at: str | None = None
    symbol: str
    timeframe: str
    candles_used: int
    period_start: str
    period_end: str
    initial_balance: float
    final_equity: float
    trades_closed: int
    win_rate_pct: float
    total_pnl_quote: float
    total_pnl_pct: float
    daily_pnl_quote: float
    daily_pnl_pct: float
    monthly_pnl_quote: float
    monthly_pnl_pct: float
    max_drawdown_pct: float
    trades: list[SystemTradeRecord]
    warnings: list[str] = Field(default_factory=list)
