# Portföy paper modu — bekleyen tasarım (kullanıcı talimatı, 2026-10-08)

**Durum:** UYGULANMADI. Ön koşullar: `research/configs/aday_v1.yaml` var VE `deneyler.md`'de aday_v1'in nihai testi GEÇTİ
(`FINAL-TEST-ACILDI aday_v1` kaydı ile). 2026-10-08 itibarıyla ikisi de yok (kabul edilmiş kol yok; nihai pencere kilitli). Dal:
`feature/portfoy-paper`. Paper modda `enable_live_trading` kapalı kalır; gerçek emir gönderen hiçbir kod yolu çağrılmaz.

1. **Hedef ağırlık motoru:** aday_v1, `research/` kodu DOĞRUDAN çağrılarak hesaplanır (araştırma–üretim arasında kod kopyası yok).
   Her gün 00:05 UTC'de, mevcut `app/scheduler` deseni ve catch-up mekanizmasıyla.
2. **Emir planlayıcı:** hedef − mevcut → no-trade bandı → min notional/adım yuvarlaması → emir listesi. Paper yürütme: önce post-only
   limit; 30 dakikada dolmazsa piyasa emri; dolum gerçek 1m/1h veriden simüle edilir. Her emir için karar fiyatı, dolum fiyatı, kayma (bps).
3. **Carry kolu varsa:** iki bacaklı paper pozisyon (spot + perp); her funding anında gerçek funding geçmişinden tahakkuk; marj oranı
   izleme, eşiğin altında uyarı ve pozisyon azaltma.
4. **Risk:** mevcut PortfolioManager kuralları (günlük zarar limiti, kill switch, allow_new_entries, kalıcı durum). Otomatik durdurma:
   portföy DD'si aday_v1 walk-forward DD dağılımının p95'ini aşarsa yeni risk alınmaz.
5. **Gölge backtest:** her gün paper başlangıcından bugüne aday_v1 backtest'i yeniden; paper NAV ile fark (tracking error), kayma ve
   maliyet farkı `GET /portfolio/paper-vs-backtest` ucunda ve Prometheus/Grafana metriklerinde.
6. **Mevcut saatlik ML motoru silinmez;** ayarla seçilebilir, varsayılan yeni motor; README'de açıkça yazılır.
7. **KURALLAR.md'ye canlıya geçiş kriterleri:** en az 8 hafta ve 30 yeniden dengeleme; yıllık tracking error < %5; ortalama kayma ≤
   varsayılan kaymanın 1,5 katı; hiçbir otomatik durdurma tetiklenmemiş. Canlıya geçiş ayrı ve kullanıcı onaylı adımdır: küçük sermaye,
   kaldıraç ≤ 1x.

Testler: yuvarlama, limit/piyasa dolum modeli, funding tahakkuku, gölge backtest tutarlılığı. Tam paket yeşil; PR.
