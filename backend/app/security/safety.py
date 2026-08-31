import logging

logger = logging.getLogger(__name__)

# Kripto Bot Rehberi Bölüm 9.3: "Maksimum kaldıraç (kod içi sabit limit) 3x".
# Bilinçli olarak Settings/.env üzerinden AYARLANAMAZ — amacı tam olarak bunun
# çalışma zamanında (yanlışlıkla ya da baskıyla) yükseltilememesidir; değiştirmek
# isteyen biri bu satırı elle düzenleyip yeniden deploy etmek zorunda kalmalı.
MAX_LEVERAGE = 3


def enforce_leverage_cap(requested_leverage: int) -> int:
    """İstenen kaldıracı kod içi sabit tavana karşı denetler.

    Aşıyorsa reddeder (izin verilen en yükseğe sessizce indirmez) — kullanıcı
    farkında olmadan daha yüksek kaldıraçla işlem yapmasın diye."""
    if requested_leverage > MAX_LEVERAGE:
        raise ValueError(
            f"İstenen kaldıraç {requested_leverage}x, kod içi güvenlik tavanını "
            f"(MAX_LEVERAGE={MAX_LEVERAGE}x) aşıyor. Bu sınır ortam değişkeniyle "
            "değiştirilemez; bilinçli bir tasarım kararıdır (bkz. Güvenlik Protokolü Bölüm 9.3)."
        )
    return requested_leverage


def check_withdrawals_disabled(exchange) -> tuple[bool, str]:
    """Binance'ten API anahtarının izinlerini sorup çekim (withdrawal) izninin
    kapalı olduğunu doğrular (Güvenlik Protokolü Bölüm 9.1, birinci madde).

    Döner: (verified_safe, message)
    - (True, ...)  -> çekim izni kapalı, doğrulandı.
    - (False, ...) -> ya çekim izni AÇIK ya da doğrulama başarısız oldu; her
      iki durumda da çağıran taraf temkinli davranmalı (varsayılan: reddet).
    """
    try:
        permissions = exchange.get_api_key_permissions()
    except Exception as exc:  # noqa: BLE001 - borsa/izin sorunları burada yakalanır
        logger.warning("API key permission check failed: %s", exc)
        return False, f"API anahtarı izinleri doğrulanamadı ({exc}); güvenlik gereği canlı işlem engellendi."

    withdrawals_enabled = bool(permissions.get("enableWithdrawals", False))
    if withdrawals_enabled:
        return False, "API anahtarında ÇEKİM (withdrawal) izni AÇIK — Güvenlik Protokolü Bölüm 9.1'e aykırı, canlı işlem engellendi."
    return True, "Çekim izni kapalı, doğrulandı."
