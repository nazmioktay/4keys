import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.types import Float, String
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


_engine = None
_session_factory: sessionmaker | None = None


def is_enabled() -> bool:
    """Kalıcı veritabanı yapılandırılmış mı (FOURKEYS_DATABASE_URL boş değil)."""
    return bool(settings.database_url)


def get_engine():
    global _engine
    if _engine is None and is_enabled():
        kwargs: dict = {"pool_pre_ping": True}
        if settings.database_url.startswith("sqlite") and ":memory:" in settings.database_url:
            # Testlerde paylaşılan bellek içi SQLite: tek bağlantı havuzunda
            # tutulmazsa her session farklı, boş bir veritabanı görür.
            kwargs = {"connect_args": {"check_same_thread": False}, "poolclass": StaticPool}
        _engine = create_engine(settings.database_url, **kwargs)
    return _engine


def _get_session_factory() -> sessionmaker:
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        if engine is None:
            raise RuntimeError("Veritabanı yapılandırılmamış (FOURKEYS_DATABASE_URL boş).")
        _session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    session = _get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> bool:
    """Tabloları oluşturur ve mümkünse `ohlcv_raw`'ı TimescaleDB hypertable'ına
    çevirir. Veritabanı yapılandırılmamışsa hiçbir şey yapmadan False döner —
    bu katman tamamen opsiyoneldir, sistemin geri kalanı DB olmadan da çalışır.
    """
    if not is_enabled():
        logger.info("database disabled (FOURKEYS_DATABASE_URL not set)")
        return False

    from . import models  # noqa: F401 - Base.metadata'ya tabloları kaydettirir

    engine = get_engine()
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)

    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
            conn.execute(text("SELECT create_hypertable('ohlcv_raw', 'time', if_not_exists => TRUE)"))
        logger.info("TimescaleDB hypertable hazır")
    except Exception as exc:  # noqa: BLE001 - düz PostgreSQL/SQLite'ta bu adım opsiyoneldir
        logger.info("TimescaleDB hypertable kurulamadı (düz PostgreSQL/SQLite olabilir): %s", exc)

    return True


def _add_missing_columns(engine) -> None:
    """Hafif şema göçü: `Base.metadata.create_all` yalnızca EKSİK tabloları
    oluşturur, VAR OLAN bir tabloya sonradan eklenen kolonları eklemez.
    Modele yeni bir Float veya String kolonu eklendiğinde (ör. yeni bir ML
    özelliği veya `BacktestTradeRow`'a eklenen karar-açıklama kolonları),
    var olan tablo üzerinde bu fonksiyon eksik kolonları `ALTER TABLE ADD
    COLUMN` ile (NULL edilebilir olarak, eski satırlar bozulmadan) tamamlar.
    Yalnızca yeni Float/String kolonlar için güvenlidir; kolon tipi/isim
    değişikliği veya silme desteklenmez.
    """
    inspector = inspect(engine)
    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        existing_columns = {col["name"] for col in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing_columns:
                continue
            if isinstance(column.type, Float):
                sql_type = "FLOAT"
            elif isinstance(column.type, String):
                sql_type = "TEXT"
            else:
                continue  # yalnızca yeni Float/String kolonlar için otomatik ekleme yapılır
            try:
                with engine.begin() as conn:
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{column.name}" {sql_type}'))
                logger.info("added missing column %s.%s", table.name, column.name)
            except Exception as exc:  # noqa: BLE001 - en kötü ihtimalle kolon eklenmez, sistem çalışmaya devam eder
                logger.warning("could not add column %s.%s: %s", table.name, column.name, exc)


def check_connection() -> bool:
    if not is_enabled():
        return False
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


def reset_for_tests() -> None:
    """Testler arasında motoru/oturum fabrikasını sıfırlar (farklı
    `database_url` değerleriyle temiz başlamak için)."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
