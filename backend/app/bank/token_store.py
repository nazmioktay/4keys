from .schemas import TokenResponse

# Süreç ömrü boyunca bellek içi, tek kullanıcılı token saklama.
# NOT: Üretimde bu, şifrelenmiş bir veritabanında (ör. KVKK'ya uygun,
# erişimi kısıtlı bir secrets store'da) saklanmalıdır — süreç yeniden
# başladığında burada tutulan token kaybolur, kullanıcı onay akışını
# tekrar etmelidir.
_token: TokenResponse | None = None


def save_token(token: TokenResponse) -> None:
    global _token
    _token = token


def get_token() -> TokenResponse | None:
    return _token


def clear_token() -> None:
    global _token
    _token = None
