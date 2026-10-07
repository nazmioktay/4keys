# Portföy ve nihai test — bekleyen tasarım (kullanıcı talimatı, 2026-10-08)

**Durum:** KOŞTURULMADI. Ön koşul: `deneyler.md`'de en az bir KABUL EDİLMİŞ kol. Bu belge ön kayıt DEĞİLDİR. Nihai test penceresi
(KURALLAR §1) bu talimat uygulanana kadar KİLİTLİ kalır. Dal: `research/portfoy`.

1. **Kabul edilen kolların günlük net getirilerini birleştir.**
   - Ağırlık yöntemleri (önceden sabit; yalnızca geçmiş veriyle, aylık güncelleme): ters vol; eşit risk katkısı (ERC; 120 günlük,
     shrinkage'lı kovaryans); sabit oranlar (ör. trend %50, carry %30, kesitsel %20).
   - Portföy vol hedefi yıllık %15 ve %20; kol başına risk tavanı %50; brüt kaldıraç tavanı.
   - Drawdown kontrol katmanı (varsayılan kapalı, test edilir): DD > %15 iken risk yarıya, yeni zirvede geri; yalnızca net Calmar
     artarsa kabul.
   - Netleştirme: aynı sembolde çakışan pozisyonlar borsa düzeyinde netleştirilir; turnover ve maliyet tasarrufu ölçülür.
2. **Portföy düzeyinde ölç:** tüm KURALLAR metrikleri, stres dönemleri, ücret/kayma/gecikme duyarlılığı; NAV 1.000 / 10.000 / 50.000
   USDT senaryoları; asgari emir kısıtlarının küçük hesapta çeşitlendirmeyi nasıl bozduğu ve makul en küçük hesap önerisi.
3. **KİLİTLE:** son tanım (kollar, parametreler, ağırlık yöntemi) `research/configs/aday_v1.yaml` olarak commit'lenir; sonra değişiklik yok.
4. **Nihai test:** aday_v1, `allow_final_test=True` ile 2025-10-01 sonrası pencerede YALNIZCA BİR KEZ çalıştırılır ve `deneyler.md`'ye
   yazılır (`FINAL-TEST-ACILDI aday_v1`). Ölçütler: gerçekleşen Sharpe ve DD walk-forward döneminin blok bootstrap %5–95 aralığında mı;
   al-tut'u Calmar'da yeniyor mu.
5. **Sonuç:** GEÇTİ → Aşama 6. KALDI → hangi kol ve dönem yüzünden olduğu analiz edilir; parametre değiştirip tekrar test EDİLMEZ;
   nihai pencere KURALLAR.md'de "kullanıldı" olarak işaretlenir.

Özet PR: araştırma dallarını main'e birleştirir; açıklamada aday_v1 tanımı ve nihai test sonucu.
