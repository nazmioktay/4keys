import base64
import hashlib
import hmac
import json
import time

from app.core.config import settings


def _secret() -> bytes:
    return settings.auth_secret_key.get_secret_value().encode()


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64decode(data: str) -> bytes:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded)


def create_token(username: str) -> str:
    """İmzalı, süresi dolan basit bir oturum token'ı üretir.

    JWT kütüphanesi eklemek yerine (tek yöneticili bu uygulama için
    gereksiz bir bağımlılık) yalnızca stdlib (`hmac`/`hashlib`) kullanır:
    `base64(payload).imza` biçiminde, HMAC-SHA256 ile imzalanmış."""
    payload = {"sub": username, "exp": int(time.time()) + settings.auth_token_ttl_hours * 3600}
    payload_b64 = _b64encode(json.dumps(payload).encode())
    signature = hmac.new(_secret(), payload_b64.encode(), hashlib.sha256).digest()
    return f"{payload_b64}.{_b64encode(signature)}"


def verify_token(token: str) -> str | None:
    """Token geçerliyse kullanıcı adını, değilse (imza uyuşmuyor/süresi
    dolmuş/bozuk) `None` döner."""
    try:
        payload_b64, signature_b64 = token.split(".", 1)
    except ValueError:
        return None

    expected_signature = hmac.new(_secret(), payload_b64.encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(_b64encode(expected_signature), signature_b64):
        return None

    try:
        payload = json.loads(_b64decode(payload_b64))
    except (ValueError, UnicodeDecodeError):
        return None

    if payload.get("exp", 0) < time.time():
        return None
    return payload.get("sub")
