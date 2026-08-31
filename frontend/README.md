# 4keys Frontend

Backend'e (FastAPI, `../backend`) bağlanan basit bir React panel. Görsel
olarak koyu temalı, alt sekmeli bir mobil uygulama düzenini izler; her sekme
backend'in gerçek bir modülüne bağlıdır:

| Sekme | Bağlı olduğu backend modülü |
|---|---|
| Portföy | `/portfolio/status`, `/security/status` |
| Al-Sat | `/dca/optimize`, `/strategy/examples` + `/strategy/backtest`, `/engine/run-cycle` |
| AI Asistan | *(henüz backend özelliği yok — dürüstçe belirtilir)* |
| Araştırıcı | `/screener/top` |
| Ayarlar | `/portfolio/rules`, `/security/*`, `/scheduler/status`, `/db/status` |

## Çalıştırma

```bash
npm install
npm run dev
```

Backend'in `http://localhost:8000`'de çalıştığını varsayar (bkz. `../backend`).
Farklı bir adres için `.env.example`'ı `.env` olarak kopyalayıp
`VITE_API_BASE_URL`'i düzenleyin.
