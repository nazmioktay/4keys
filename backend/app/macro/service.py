from app.core.config import settings
from app.db import repository as db
from app.exchanges.base import Exchange

from . import data


def fetch_macro_snapshot(exchange: Exchange) -> dict:
    """Tüm ücretsiz makro veri kaynaklarını toplar. Her kaynak izole
    başarısız olabilir (bkz. `app.macro.data`) — bu fonksiyon kendisi
    hiçbir exception fırlatmaz, eksik kaynaklar None olarak kalır."""
    market = data.get_total_market_cap_and_btc_dominance()
    indices = data.get_world_indices()
    return {
        "total_market_cap": market["total_market_cap"],
        "btc_dominance": market["btc_dominance"],
        "funding_rate_btc": data.get_binance_btc_funding_rate(exchange),
        "vix": data.get_vix(),
        "gold_price": data.get_gold_price(),
        "sp500": indices["sp500"],
        "nasdaq": indices["nasdaq"],
        "nikkei": indices["nikkei"],
        "dax": indices["dax"],
        "fed_funds_rate": data.get_fed_funds_rate(settings.fred_api_key or None),
        "ecb_deposit_rate": data.get_ecb_deposit_rate(),
    }


def refresh_and_record_macro_snapshot(exchange: Exchange) -> dict:
    """Makro anlık görüntüyü toplayıp (DB açıksa) kalıcı hale getirir."""
    snapshot = fetch_macro_snapshot(exchange)
    db.record_macro_snapshot(snapshot)
    return snapshot
