# Deney günlüğü

Kurallar için bkz. `KURALLAR.md`. Her çalıştırma `research.registry.register_experiment` ile buraya tek satır ekler.
Toplam deneme sayacının tek doğruluk kaynağı `research/results/_registry.json`'dır (Deflated Sharpe bunu kullanır).

**Toplam deneme: 0**

| Tarih (UTC) | Deney | Hipotez | Varyant | Net Sharpe | Maks. DD | Karar | Veri hash | Commit |
|---|---|---|---|---|---|---|---|---|

<!-- Nihai pencere açma kayıtları (yalnızca Aşama 5): "FINAL-TEST-ACILDI <deney_id>" -->
<!-- Forward test açma kayıtları (en erken 2028-05-01; KURALLAR.md §10): "FORWARD-TEST-ACILDI <deney_id>" -->

## Kural kayıtları
- 2026-10-07 — KURALLAR.md §10 (ileriye dönük doğrulama bölgesi) önceden kayda geçirildi: nihai pencere [2025-10-01, 2026-11-01);
  forward keşif [2026-11-01, 2027-11-01); forward test 2027-11-01+, en erken 2028-05-01'de açılır. Bu tarihte hiçbir forward_only
  veri toplanmamıştı ve hiçbir deney nihai pencereye bakmamıştı.
