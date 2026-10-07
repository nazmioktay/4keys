# Durum raporu — 2026-10-08

Tek sayfada: neyi denedik, ne öğrendik, elimizde ne var, sırada ne var. Ayrıntılar `deneyler.md`'de; kurallar `KURALLAR.md`'de.

## Kısa cevap
Kâr getiren bir sistem bulunmadı. Ama kaybettiren bir sistemi gerçek parayla çalıştırmaktan kurtulundu ve neyin işlemediği artık
önceden kayıtlı, belgeli deneylerle biliniyor.

## Öğrenilenler

**Saatlik ML yön tahmini (canlı botun motoru)**
- İlk iyimser backtest sonuçları ölçüm hatalarından geliyordu: yarım/donuk mum verisi, iyimser yürütme (kapanıştan giriş, yalnızca
  kapanışla stop, funding yok), modellerin test dönemini görmesi ve backtest ile paper arasında farklı kaldıraç. Bu hatalar backtest'i
  olduğundan İYİ gösterdi; paper'daki zarar modelin bozulması değil, baştan avantajı olmamasıydı.
- Dürüst ölçümde (sızıntılar kapalı, sonraki mumun açılışında giriş, high/low stop, funding dahil) ~8 ayda 24–47 işlem, PnL ≈ 0.

**Önceden kayıtlı üç hipotez (2021-04 → 2025-09; toplam 321 deneme)**

| Deney | Sonuç | Özü |
|---|---|---|
| `trend_001` — zaman serisi momentumu | KALDI | Net Sharpe 0,43 (eşik 0,8; BTC al-tut 0,54). Maks. düşüş −%48 (BTC al-tut −%77): riski azaltıyor ama yeterli getiri üretmiyor. Short bacak katkısı ≈ 0. |
| `kesitsel_001` — kesitsel momentum | KALDI | Net Sharpe 0,14. En iyi − en kötü dilim farkı 1–60 günün hiçbirinde pozitif değil: bu evrende/dönemde kesitsel momentum yok. |
| `carry_001` — funding/basis carry | KALDI | Gerçek ve çok düşük riskli pozitif getiri (en iyi A\|BTC\|PM: yıllık %7,2, maks. düşüş −%0,3), ama risksiz faizin (%4) 2 katı koşulunu geçmiyor; funding geliri eriyor (BTC carry 2021 %18,3 → 2025 %2,0). |

**Piyasa ve hesap**
- Altcoin evreni bu dönemde çok kötü: eşit ağırlıklı ilk-10 al-tut yıllık ≈ −%20, maks. düşüş −%92,6.
- 250 USDT'lik hesapta asgari emir tutarları çeşitlendirmeyi kısıtlıyor (ör. BTC'de 1 lot ≈ 114 USDT); en iyi senaryoda bile yıllık
  kazanç ~20 USDT civarı. 1x ile BTC'yi rahat taşımak için önerilen asgari hesap ~570 USDT.

## Elimizde olanlar
- **Düzeltilmiş canlı/paper motor** (main'de): yarım mumu atan önbellek, kapanmış mum başına tek karar, canlı fiyatla stop kontrolü,
  gerçekçi backtest yürütmesi, sızıntı düzeltmeleri, kalıcı portföy durumu. Portföy kaldıracı varsayılanı 1x (kod içi tavan 3x);
  `binance_testnet=True` ve `enable_live_trading=False` varsayılan.
- **Araştırma altyapısı:** noktasal-zamanlı evren, vektörel motor ve iki bacaklı carry simülatörü, Deflated Sharpe, PBO, ön kayıt,
  deney günlüğü ve deneme bütçesi, 250 USDT hesap ve Binance lot kuralları.
- **Kullanılmamış nihai test penceresi** (2025-10-01 → 2026-11-01): hiçbir deney görmedi. İleride iyi bir aday bulunursa onu dürüstçe
  sınamanın tek yolu; tek kullanımlık.
- **İleriye dönük veri toplayıcılar** (VPS, opt-in): tasfiye akışı, emir defteri derinliği, ayrıntılı açık pozisyon. Hiçbir backtest'te
  kullanılmadı; kurallar gereği 12 ay birikmeden kullanılamaz (forward keşif dönemi 2026-11-01'de başlar, forward test en erken 2028-05-01).
- **Bekleyen tasarımlar** (`sablonlar/`): ML turnuvası, portföy + nihai test, portföy paper modu. Hepsi kabul edilmiş en az bir kol
  gerektiriyor.
- **Çalışma düzeni:** `CLAUDE.md`, reviewer denetimi, ruff; 702 test (2026-10-08, `pytest --collect-only`).
- **Veri dışa aktarımı:** BTC/ETH araştırma verisi Excel olarak (nihai pencere hariç).

## Dürüst sınırlar
- Test edilen alan dar: 2021–2025, Binance USDⓈ-M perpetual + spot + quarterly, günlük/saatlik, kamuya açık fiyat verisi. "Hiçbir yerde
  avantaj yok" denemez; yalnızca "bu fikirlerde, bu maliyetlerle, bu hesapla yok" denebilir.
- Deneme sayısının bedeli var: her yeni deneme gelecekteki adayların Deflated Sharpe çıtasını yükseltir.
- carry_001'de denetçi ajan kayıttan önce bazı sonuçları gördü; karar değişmedi ve ön kayıt ekinde belgelendi.
- Tasfiye kontrolü son işlem fiyatının gün içi en yükseğiyle yapıldı (Binance mark fiyatı kullanır): kısa iğnelerde kötümser.
- Canlı ML motorunun ilk iyimser backtest rakamları önceki oturumların özetinden; bu rapor için yeniden doğrulanmadı. Araştırma
  rakamları deney çıktılarından (`research/results/`) kontrol edildi.

## Öneriler
1. **Gerçek parayla otomatik işlem yapma.** Saatlik ML motorunun avantajı yok; testnet açık, `enable_live_trading` kapalı kalsın.
2. **Hesap büyüklüğünü gerçekçi değerlendir.** 250 USDT'de beklenen kazanç harcanan zamanın karşılığı değil: proje ya öğrenme projesi
   olarak sürer ya da sermaye belirgin artarsa yeniden değerlendirilir (kullanıcı kararı).
3. **Carry bir strateji değil, nakit yönetimi aracı.** BTC'de Portföy Marjıyla delta-nötr carry geçmişte yılda ~%7 verdi, 2025'te ~%2'ye
   düştü (risksiz faize yakın). Funding yeniden yükselirse anlamlı olur; izleme konusu.
4. **Araştırmayı acele ettirme.** En mantıklı sonraki deney, ileriye dönük veriyle (tasfiye, emir defteri) kurulacak bir hipotez. O
   zamana kadar yeni deneme yalnızca gerçekten farklı ve gerekçeli bir fikir varsa yapılmalı. Nihai pencere kilitli kalsın.
