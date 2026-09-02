from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .session import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OHLCVRaw(Base):
    """Ham mum verisi — Kripto Bot Rehberi Bölüm 3.2'deki `ohlcv_raw` tablosu.

    TimescaleDB varsa `time` üzerinde hypertable'a çevrilir (bkz.
    `session.init_db`); düz PostgreSQL/SQLite'ta normal bir tablo olarak kalır.
    """

    __tablename__ = "ohlcv_raw"
    __table_args__ = (UniqueConstraint("time", "symbol", "timeframe", name="uq_ohlcv_raw_time_symbol_tf"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    timeframe: Mapped[str] = mapped_column(String)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)


class SignalRecord(Base):
    """Üretilen her tahmin — Bölüm 3.4-3.6'daki `signals` tablosu.

    `source`: "screener" (teknik skor), "ml" (birincil model) veya "meta"
    (meta-label kararı) — hangi bileşenin ürettiğini ayırt eder.
    """

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    source: Mapped[str] = mapped_column(String, index=True)
    direction: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)


class FeatureSnapshot(Base):
    """ML özellik vektörünün (bkz. `app.ml.features.FEATURE_COLUMNS`) her
    tarama döngüsünde zaman damgasıyla kaydı.

    Amaç: XGBoost şu an eğitim için her seferinde Binance'ten anlık ham veri
    çekip özellikleri yeniden hesaplıyor — bu tablo bunun YERİNE geçmiyor,
    zamanla ayrı bir "canlı piyasadan toplanmış gerçek veri seti" biriktiriyor.
    Bu birikim, ileride LSTM (sekans/zaman serisi modeli) ve Reinforcement
    Learning ajanının eğitimi için gereken uzun, kesintisiz zaman serisini
    sağlayacak — o modeller borsadan anlık çekilen sınırlı geçmişle değil,
    burada biriken gerçek geçmişle eğitilecek.
    """

    __tablename__ = "feature_snapshots"
    __table_args__ = (UniqueConstraint("time", "symbol", "timeframe", name="uq_feature_snapshot_time_symbol_tf"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    timeframe: Mapped[str] = mapped_column(String)
    close: Mapped[float] = mapped_column(Float)
    rsi_norm: Mapped[float] = mapped_column(Float)
    macd_hist_norm: Mapped[float] = mapped_column(Float)
    ema_gap: Mapped[float] = mapped_column(Float)
    momentum: Mapped[float] = mapped_column(Float)
    volume_ratio: Mapped[float] = mapped_column(Float)
    price_position: Mapped[float] = mapped_column(Float)
    return_1: Mapped[float] = mapped_column(Float)
    return_3: Mapped[float] = mapped_column(Float)
    return_5: Mapped[float] = mapped_column(Float)

    # --- Kullanıcının manuel işlemde kullandığı ek göstergeler ---
    # nullable=True: bu kolonlar tabloya SONRADAN eklendi (bkz.
    # app.db.session._add_missing_columns); önceden kaydedilmiş satırlar
    # bu kolonlar için NULL içerir.
    ha_trend: Mapped[float | None] = mapped_column(Float, nullable=True)
    ha_body_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    stoch_rsi_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    stoch_rsi_d: Mapped[float | None] = mapped_column(Float, nullable=True)
    mavilim_gap: Mapped[float | None] = mapped_column(Float, nullable=True)
    pmax_trend: Mapped[float | None] = mapped_column(Float, nullable=True)
    pmax_dist_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    linreg_zscore: Mapped[float | None] = mapped_column(Float, nullable=True)
    linreg_slope_norm: Mapped[float | None] = mapped_column(Float, nullable=True)
    wt_diff_norm: Mapped[float | None] = mapped_column(Float, nullable=True)
    wt_cross: Mapped[float | None] = mapped_column(Float, nullable=True)
    nwe_position: Mapped[float | None] = mapped_column(Float, nullable=True)
    sr_dist_support_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    sr_dist_resistance_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    sr_level_count_norm: Mapped[float | None] = mapped_column(Float, nullable=True)


class MacroSnapshot(Base):
    """Ücretsiz makro/piyasa bağlamı verilerinin periyodik anlık görüntüsü
    (bkz. `app.macro.data`). Kripto fiyatı yalnızca kendi grafiğinde değil,
    daha geniş piyasa bağlamında hareket eder — TOTAL, BTC dominansı,
    funding rate, VIX, altın, dünya borsa endeksleri, Fed/ECB faiz
    oranları. LSTM/RL eğitiminde OHLCV tabanlı özelliklerin yanına ek
    bağlam olarak kullanılabilir.
    """

    __tablename__ = "macro_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    total_market_cap: Mapped[float | None] = mapped_column(Float, nullable=True)
    btc_dominance: Mapped[float | None] = mapped_column(Float, nullable=True)
    funding_rate_btc: Mapped[float | None] = mapped_column(Float, nullable=True)
    vix: Mapped[float | None] = mapped_column(Float, nullable=True)
    gold_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    sp500: Mapped[float | None] = mapped_column(Float, nullable=True)
    nasdaq: Mapped[float | None] = mapped_column(Float, nullable=True)
    nikkei: Mapped[float | None] = mapped_column(Float, nullable=True)
    dax: Mapped[float | None] = mapped_column(Float, nullable=True)
    fed_funds_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    ecb_deposit_rate: Mapped[float | None] = mapped_column(Float, nullable=True)


class TradeRecord(Base):
    """Gerçekleşen (kapanan) her işlem — Bölüm 3.4-3.6'daki `trades` tablosu.

    Sharpe/win-rate/drawdown gibi tüm performans ölçümlerinin temelidir;
    `app.portfolio.manager.PortfolioManager.close()` her pozisyon
    kapanışında buraya bir kayıt yazar (DB etkinse).
    """

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    direction: Mapped[str] = mapped_column(String)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float] = mapped_column(Float)
    size_quote: Mapped[float] = mapped_column(Float)
    pnl_pct: Mapped[float] = mapped_column(Float)
    pnl_quote: Mapped[float] = mapped_column(Float)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source: Mapped[str] = mapped_column(String, default="engine")
