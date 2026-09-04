from typing import Literal

from pydantic import BaseModel, Field, field_validator


KELLY_VARIANTS: dict[str, float] = {"quarter": 0.25, "half": 0.5, "full": 1.0}


class RiskRules(BaseModel):
    """Ana para yönetimi kuralları — tüm botlar/stratejiler bu kurallar
    üzerinden pozisyon açar; kurallar aşılıyorsa işlem küçültülür veya reddedilir.
    """

    max_risk_per_trade_pct: float = Field(1.0, gt=0, description="Bir işlemde riske edilecek sermaye yüzdesi (SL mesafesine göre boyutlandırma için) — position_sizing_method='fixed_risk' iken kullanılır")
    max_total_exposure_pct: float = Field(50.0, gt=0, description="Tüm açık pozisyonların toplamının sermayeye oranı üst sınırı")
    max_symbol_exposure_pct: float = Field(15.0, gt=0, description="Tek bir sembole ayrılabilecek maksimum sermaye yüzdesi")
    max_concurrent_positions: int = Field(5, ge=1, description="Aynı anda açık olabilecek maksimum farklı sembol sayısı")
    daily_loss_limit_pct: float = Field(5.0, gt=0, description="Bu yüzdeye ulaşan günlük/oturum zararında yeni işlem açılmaz")

    # --- Kelly kriteri tabanlı pozisyon boyutlandırma ---
    position_sizing_method: Literal["fixed_risk", "kelly"] = Field(
        "fixed_risk", description="'fixed_risk': SL mesafesine göre sabit risk yüzdesi. 'kelly': Kelly kriteri."
    )
    kelly_multiplier: float = Field(
        0.5, gt=0, le=1.5,
        description="Full Kelly'nin uygulanacak kesri. Çeyrek Kelly=0.25, yarım Kelly=0.5 (önerilen/varsayılan), tam Kelly=1.0",
    )
    kelly_min_trades: int = Field(
        20, ge=5,
        description="Kelly istatistiklerinin (kazanma oranı, ort. kazanç/kayıp) güvenilir sayılması için gereken minimum kapanmış işlem sayısı. Yeterli geçmiş yoksa fixed_risk'e düşülür.",
    )
    max_kelly_fraction_pct: float = Field(
        25.0, gt=0,
        description="Kelly formülü ne derse desin, bir işleme ayrılacak sermayenin üst güvenlik sınırı (%)",
    )

    # --- Kademeli (aşamalı) alım/satım ---
    # Bir pozisyon TEK seferde değil, birden çok "tranche" (dilim) halinde
    # açılır/kapatılır — piyasayı tek büyük emirle hareket ettirmemek ve
    # sinyalin bir sonraki döngüde de kalıcı olduğunu teyit etmek için.
    entry_tranche_weights: list[float] = Field(
        default_factory=lambda: [0.5, 0.5],
        description="Hesaplanan tam pozisyon boyutunun her alım diliminde ne kadarının kullanılacağı (toplamı ~1.0 olmalı). Örn. [0.5, 0.5] = çeyrek Kelly ile hesaplanan tutarın yarısı ilk döngüde, yarısı sinyal bir sonraki döngüde de kalıcıysa açılır.",
    )
    exit_tranche_weights: list[float] = Field(
        default_factory=lambda: [0.5, 0.5],
        description="Kapanış sinyali geldiğinde pozisyonun ne kadarının her dilimde satılacağı (toplamı ~1.0 olmalı). Son dilim, yuvarlama artığı kalmaması için pozisyonun TAMAMINI kapatır.",
    )

    # --- Confidence-weighted boyutlandırma ---
    # Kelly/fixed_risk'in önerdiği boyut, modelin O ANKİ tahmininin
    # güvenine göre ek olarak ölçeklenir — yalnızca "kazanma oranı"
    # geçmişine değil, "bu spesifik sinyal ne kadar güçlü" bilgisine de
    # duyarlı olmak için (rehberin "5+6: karar doğruluğunu artırma"
    # önerilerinden biri).
    confidence_scaling_enabled: bool = Field(
        True, description="Açık ise pozisyon boyutu, tahminin confidence'ına göre (confidence_scaling_min_scale..1.0 arası) ek olarak ölçeklenir."
    )
    confidence_scaling_min_confidence: float = Field(
        0.6, ge=0, le=1, description="Bu confidence'ta (ve altında) ölçek confidence_scaling_min_scale'e sabitlenir; genelde open_confidence eşiğiyle aynı tutulmalı."
    )
    confidence_scaling_min_scale: float = Field(
        0.5, gt=0, le=1, description="confidence_scaling_min_confidence'taki (veya altındaki) ölçek — 1.0 confidence'ta ölçek her zaman 1.0'dır."
    )

    # --- Piyasa rejimi filtresi (VIX bazlı, opsiyonel) ---
    # Ekstrem piyasa stresi anlarında (VIX kendi geçmişine göre çok
    # yüksekse) yeni pozisyon açmayı kısıtlar/engeller — modelin normal
    # piyasa koşullarında öğrendiği örüntülerin kriz anlarında güvenilmez
    # olabileceği varsayımıyla.
    vix_regime_filter_enabled: bool = Field(False, description="Açık ise VIX z-skoru eşiği aşıldığında yeni pozisyon açma kısıtlanır/engellenir.")
    vix_zscore_block_threshold: float = Field(2.5, gt=0, description="VIX z-skoru (macro_vix_norm) bu değeri aşarsa yeni pozisyon TAMAMEN engellenir.")
    vix_zscore_reduce_threshold: float = Field(1.5, gt=0, description="VIX z-skoru bu değeri aşarsa (block eşiğine kadar) pozisyon boyutu yarıya indirilir.")

    @field_validator("entry_tranche_weights", "exit_tranche_weights")
    @classmethod
    def _validate_tranche_weights(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("en az bir tranche ağırlığı gerekli")
        if any(w <= 0 for w in value):
            raise ValueError("tranche ağırlıkları pozitif olmalı")
        total = sum(value)
        if not (0.98 <= total <= 1.02):
            raise ValueError(f"tranche ağırlıklarının toplamı ~1.0 olmalı (şu an {total:.3f})")
        return value


class PositionSizeRequest(BaseModel):
    equity: float = Field(..., gt=0)
    entry_price: float = Field(..., gt=0)
    stop_loss_price: float = Field(..., gt=0)
    direction: Literal["long", "short"] = "long"
    risk_per_trade_pct: float = Field(1.0, gt=0)


class PositionSizeResponse(BaseModel):
    size_quote: float
    risk_amount_quote: float
    stop_distance_pct: float


class PositionExposure(BaseModel):
    symbol: str
    size_quote: float


class RiskCheckRequest(BaseModel):
    equity: float = Field(..., gt=0)
    open_positions: list[PositionExposure] = Field(default_factory=list)
    realized_pnl_session: float = 0.0
    proposed_symbol: str
    proposed_size_quote: float = Field(..., gt=0)
    rules: RiskRules = Field(default_factory=RiskRules)


class RiskDecision(BaseModel):
    allowed: bool
    size_quote: float
    reasons: list[str] = Field(default_factory=list)


class TradeStats(BaseModel):
    num_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float


class PortfolioStatus(BaseModel):
    equity: float
    starting_equity: float
    realized_pnl_session: float
    open_positions: list[dict]
    closed_history: list[dict]
    rules: RiskRules
    trade_stats: TradeStats


class KellySizeRequest(BaseModel):
    equity: float = Field(..., gt=0)
    win_rate_pct: float = Field(..., ge=0, le=100)
    avg_win_pct: float = Field(..., gt=0, description="Ortalama kazanan işlem getirisi (pozitif yüzde)")
    avg_loss_pct: float = Field(..., lt=0, description="Ortalama kaybeden işlem getirisi (negatif yüzde, örn. -2.5)")
    variant: Literal["quarter", "half", "full", "custom"] = "half"
    custom_multiplier: float | None = Field(default=None, gt=0, le=1.5, description="variant='custom' iken kullanılır")
    max_kelly_fraction_pct: float = Field(25.0, gt=0)


class PnlWindow(BaseModel):
    pnl_quote: float
    trade_count: int
    win_rate_pct: float


class PnlSummary(BaseModel):
    """Kayan pencereli (rolling) PNL özeti — takvim günü/haftası/ayı
    sınırlarına göre DEĞİL, "son 24 saat / son 7 gün / son 30 gün"
    şeklinde hesaplanır (basitlik için; zaman dilimi belirsizliğinden
    kaynaklanan sınır hatalarını önler)."""

    total: PnlWindow
    daily: PnlWindow
    weekly: PnlWindow
    monthly: PnlWindow


class KellySizeResponse(BaseModel):
    full_kelly_pct: float
    applied_kelly_pct: float
    kelly_multiplier_used: float
    size_quote: float
