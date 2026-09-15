from typing import Literal

from pydantic import BaseModel, Field, field_validator


KELLY_VARIANTS: dict[str, float] = {"quarter": 0.25, "half": 0.5, "full": 1.0}


class RiskRules(BaseModel):
    """Ana para yönetimi kuralları — tüm botlar/stratejiler bu kurallar
    üzerinden pozisyon açar; kurallar aşılıyorsa işlem küçültülür veya reddedilir.
    """

    max_risk_per_trade_pct: float = Field(1.0, gt=0, description="Bir işlemde riske edilecek sermaye yüzdesi (SL mesafesine göre boyutlandırma için) — position_sizing_method='fixed_risk' iken kullanılır")
    # GÜNCELLEME (bkz. README "Karlılık", kullanıcı isteği: "portföyün en
    # yüksek kullanımı olacak şekilde"): sistem TEK sembolle (BTC-only)
    # çalıştığı için `max_symbol_exposure_pct` fiilen TÜM portföyün tavanı
    # oluyordu — eski %15 sermayenin %85'ini sürekli atıl bırakıyordu (gerçek
    # backtest'te ölçüldü: ort. pozisyon equity'nin yalnızca %13,49'u).
    # Gerçek üretim modeliyle taranıp ölçüldü: %15/%25 -> +%30,50 PnL; %40/%40
    # -> +%49,16; %60/%60 -> +%50,29 (TEPE); %100/%100 -> +%48,65 (DÜŞÜYOR,
    # drawdown ise artmaya devam ediyor: %0,978 -> %1,625 -> %2,155 -> %2,785)
    # — %60 sonrası saf kaldıraç riski, karşılığında getiri YOK (aşırı
    # kaldıraçın klasik imzası). %60 seçildi: PnL'i FEDA ETMEDEN maksimum
    # sermaye kullanımı. NOT: bu ölçüm backtest'te yapıldı, backtest kademeli
    # (tranche) girişi SİMÜLE ETMEZ — canlı/paper motor `entry_tranche_weights`
    # ile 2 dilimde açtığından gerçek risk muhtemelen biraz daha yumuşaktır.
    max_total_exposure_pct: float = Field(60.0, gt=0, description="Tüm açık pozisyonların toplamının sermayeye oranı üst sınırı")
    max_symbol_exposure_pct: float = Field(60.0, gt=0, description="Tek bir sembole ayrılabilecek maksimum sermaye yüzdesi")
    max_concurrent_positions: int = Field(5, ge=1, description="Aynı anda açık olabilecek maksimum farklı sembol sayısı")
    daily_loss_limit_pct: float = Field(5.0, gt=0, description="Bu yüzdeye ulaşan günlük/oturum zararında yeni işlem açılmaz")

    # --- Kelly kriteri tabanlı pozisyon boyutlandırma ---
    # GÜNCELLEME (bkz. README "karlılık", kullanıcı isteği: "paper trading'e
    # de uygula"): önceden canlı/paper motor VARSAYILAN OLARAK "fixed_risk"
    # kullanıyordu, Kelly yalnızca backtest'in kendi varsayılanıydı — yani
    # `POST /backtest/system/sweep-position-sizing` ile ölçülen iyileşme
    # (bkz. `app.backtest.schemas.SystemBacktestRequest.kelly_multiplier`
    # docstring'i — aynı gerekçe/ölçüm) gerçek paper-trading sermayesine
    # HİÇ yansımıyordu. Artık ikisi SENKRON: `kelly`/`1.0`/`40` (bkz.
    # `app.backtest.schemas.SystemBacktestRequest.kelly_multiplier`
    # "GÜNCELLEME 2" notu — büyümüş örneklemle (523 işlem) tam Kelly'nin
    # düşük risk/yüksek getiri sunduğu doğrulandı, kullanıcı onayıyla
    # 0.75'ten 1.0'a geçildi).
    position_sizing_method: Literal["fixed_risk", "kelly"] = Field(
        "kelly", description="'fixed_risk': SL mesafesine göre sabit risk yüzdesi. 'kelly' (varsayılan): Kelly kriteri."
    )
    kelly_multiplier: float = Field(
        1.0, gt=0, le=1.5,
        description="Full Kelly'nin uygulanacak kesri. Çeyrek Kelly=0.25, yarım Kelly=0.5, 0.75, tam Kelly=1.0 (varsayılan, bkz. yukarıdaki ölçüm)",
    )
    kelly_min_trades: int = Field(
        40, ge=5,
        description="Kelly istatistiklerinin (kazanma oranı, ort. kazanç/kayıp) güvenilir sayılması için gereken minimum kapanmış işlem sayısı. Yeterli geçmiş yoksa fixed_risk'e düşülür.",
    )
    max_kelly_fraction_pct: float = Field(
        60.0, gt=0,
        description="Kelly formülü ne derse desin, bir işleme ayrılacak sermayenin üst güvenlik sınırı (%) — bkz. yukarıdaki max_symbol_exposure_pct notu, ikisi SENKRON tutulmalı",
    )
    # Bkz. `app.backtest.schemas.SystemBacktestRequest.leverage` (AYNI
    # semantik/gerekçe, gerçek üretim modeliyle ölçüldü: 1x -> +%63,50 PnL/
    # %1,83 drawdown, 3x -> +%331,59 PnL/%5,43 drawdown). Pozisyon
    # boyutlandırma (Kelly/fixed_risk) DEĞİŞMEZ, hep TEMİNAT (equity yüzdesi)
    # anlamına gelir — kaldıraç yalnızca gerçekleşen PnL'i (kâr VE zarar) bu
    # teminat üzerinden büyütür. Üst sınır `MAX_LEVERAGE` (bkz.
    # `app.security.safety`) ile AYNI (3) — GERÇEK borsaya asla bundan fazla
    # kaldıraç gönderilemeyeceği için (`enforce_leverage_cap`), paper'da da
    # test edilemez bir senaryonun bir anlamı yok. VARSAYILAN 3 (kod içi
    # tavanın tamamı, kullanıcı isteğiyle) — likidasyon riski ayrıca
    # değerlendirildi: gerçek stop mesafeleri (ort. %0,77, en dar %0,33)
    # 3x'teki ~%33 likidasyon eşiğinin çok altında.
    leverage: int = Field(
        3, ge=1, le=3,
        description="Teminatın kontrol ettiği nominal pozisyonun çarpanı — sizing'i DEĞİL, gerçekleşen PnL'in büyüklüğünü etkiler.",
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

    # --- İşlem maliyetleri (komisyon + kayma/slippage) ---
    # Önceden PnL yalnızca fiyat farkından hesaplanıyordu — gerçek bir
    # işlemde her BACAK (açılış VE kapanış) komisyon ve kayma (piyasa
    # emrinin gösterilen fiyattan biraz sapmayla dolması) maliyeti taşır.
    # Varsayılanlar Binance Futures taker ücretine (~%0.04) ve mütevazı
    # bir kayma tahminine (~%0.02) dayanır — kesin değerler değildir,
    # ayarlanabilir.
    commission_pct: float = Field(0.04, ge=0, description="Her işlem bacağı (açılış veya kapanış) için komisyon yüzdesi.")
    slippage_pct: float = Field(0.02, ge=0, description="Her işlem bacağı için varsayılan kayma (slippage) yüzdesi.")

    # --- Stop-loss uygulaması (canlı/paper trading) ---
    # Önceden `assumed_stop_loss_pct` (bkz. DecisionEngine) yalnızca Kelly
    # boyutlandırma HESABI için kullanılıyordu — pozisyona gerçekten
    # KAYDEDİLMİYOR ve fiyat o seviyeyi geçse bile hiçbir zaman
    # KONTROL EDİLMİYORDU. Artık her döngüde kontrol edilip aşılırsa
    # pozisyon zorla kapatılır (bkz. DecisionEngine.evaluate).
    stop_loss_enabled: bool = Field(True, description="Açık ise açılıştaki stop-loss seviyesi her döngüde kontrol edilir; aşılırsa pozisyon (modelin sinyalinden BAĞIMSIZ) zorla kapatılır.")

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
