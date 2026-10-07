# trend_001 — düzeltmeler

Bu klasördeki `sonuc.md` koşunun özgün çıktısıdır ve değiştirilmez. Düzeltmeler ve yorum notları: `docs/research/deneyler.md` → "Not ve düzeltmeler (trend_001)".

**Not ve düzeltmeler (trend_001)** — kayıtlar değiştirilmez; düzeltmeler bu notla yapılır. Karar DEĞİŞMEZ.

**Kod sürümü:**
- Kayıttaki `git_commit` "unknown" görünüyor, çünkü araştırma imajında `git` yok.
- Sıra: ön kayıt `5de2cc8` → uygulama `dd11280` → gerçek veride tek koşu. Koşu sonrası `backend/` temizdi (HEAD = `dd11280`).
- Koşunun son düzenlemeden sonra başladığı zaman damgalarıyla kesin olarak doğrulanamaz. Ana aday serisi `dd11280` koduyla yeniden kuruldu ve kayıttaki seriyle birebir aynı çıktı.

**Raporlama düzeltmeleri** (denetimde bulundu; kod düzeltildi; değerler kayıttaki walk-forward seçimleriyle aynı koşu yeniden kurularak hesaplandı):
- *Short bacak katkısı (main LS):* İlk hesap çıkış maliyetini hiçbir bacağa yazmıyordu. Doğru değerler: long %37,7, short **%0,8**, toplam %38,5 (raporlanan: short %2,2). Short bacağın katkısı pratikte sıfır.
- *NAV 1.000'de atlanan emir oranı:* İlk hesabın paydası, gerçek işlem sayısı yerine pozisyon-gün sayısıydı. Doğru değerler: emir denemelerinin LF'de **%31,6**'sı, LS'de **%27,5**'i min notional yüzünden atlanıyor (raporlanan %10,5 ve %8,7). NAV 10.000'de bu oran %1,0 ve %0,01. Küçük hesapta uygulanabilirlik ciddi biçimde bozuluyor.
- *+1 gün gecikme senaryosu:* Katman geçişlerinde bant bir gün erken hizalanıyordu. main'de tüm katmanlar b0,25 olduğu için etkisi yok; kod düzeltildi.

**Ön kayıtta açık bırakılıp kodda yorumlanan noktalar** (sapma değil, yorum):
- (a) Plato oranının paydası, walk-forward Sharpe değil, son seçimdeki sabit N30/b0,25 varyantının Sharpe'ı (0,52 ve 0,38); komşular da aynı sabit kombinasyonla koşturuldu. Walk-forward Sharpe ile oran LF 0,92, LS 0,55 olurdu; karar aynı.
- (b) Pencerenin son getiri günü 2025-09-29. 09-30 getirisi 10-01 açılışını, yani nihai pencereyi gerektirir.
- (c) Stres DD'si, serinin o güne kadarki kümülatif tepesinden ölçülüyor; yani dönemin kendi kaybını değil, o gün yaşanan toplam düşüşü gösteriyor. Örneğin FTX satırı 2021-11'den beri biriken düşüşü içerir.
- (d) Donchian kanalları kapanışla değil, high/low ile kuruluyor.
- (e) Walk-forward ısınma günleri (2020-10-01 → 2021-03-30) ilk katmanın seçimini kullanıyor; bu günler değerlendirme penceresinin dışında.

**Ek kontroller:**
- 216 varyant matrisinde boş ya da birebir aynı sütun yok. PBO tekrar hesabı 0,497; rastgele gürültü matrisinde 0,68.
- EW ilk-10 kıyası bağımsız, maliyetsiz hesapla yıllık -%21 çıktı (raporlanan -%19,9).

**Bütçe notu:** KURALLAR §9'daki "soru başına 40 config" bütçesi kodda kayıtlı deney başına sayılıyor. trend_001 bütçeden 1/40 düşüyor; deneme sayacına ise 248 varyant olarak işlendi.
