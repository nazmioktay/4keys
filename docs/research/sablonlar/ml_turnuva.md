# ML turnuvası — bekleyen tasarım (kullanıcı talimatı, 2026-10-08)

**Durum:** KOŞTURULMADI. Ön koşul: `deneyler.md`'de en az bir KABUL EDİLMİŞ kol. Bu belge ön kayıt DEĞİLDİR; bir kol kabul edildiğinde
her soru için config listesi deneyden önce `deneyler.md`'ye ön kayıt olarak yazılır.

İlke: ML yalnızca kabul edilmiş bir kolun net performansını, deneme sayısı düzeltmesinden sonra da anlamlı biçimde iyileştirirse
portföye girer. Dal: `research/ml-turnuva`. Soru başına bütçe ≤ 40 config.

**Veri kademeleri** (her biri bir öncekine göre ablasyonla): (a) ohlcv_core; (b) + funding + taker_flow; (c) + derivatives_metrics;
(d) + macro + sentiment_fng.

**Soru 1 — Volatilite tahmini: vol hedeflemeyi iyileştiriyor mu?**
- Hedef: next_rv(1), next_rv(7), panel. Modeller: EWMA (kıyas), HAR-RV, GARCH(1,1), ridge, LightGBM, en iyi ikisinin yığını.
- Başarı: QLIKE'ta HAR'a karşı Diebold-Mariano p < 0,05 VE kabul edilmiş kolun vol hedeflemesine takıldığında net Sharpe artışı blok
  bootstrap aralığının dışında.

**Soru 2 — Kesitsel sıralama (yalnızca kesitsel kol kabul edildiyse)**
- Hedef: cross_sectional_rank(7) veya vol_adj_return(7). Modeller: ridge (kıyas), LightGBM regresyon, LightGBM LambdaRank, küçük MLP, yığın.
- Başarı: ortalama günlük rank IC > 0,02 ve t > 2 VE kesitsel kolun net Sharpe'ı kural tabanlı skora göre bootstrap aralığı dışında
  daha iyi VE turnover artışı maliyet sonrası telafi ediliyor.

**Soru 3 — Meta-label filtresi (trend kolu kabul edildiyse)**
- Hedef: strategy_outcome(trend_kolu, H); H = kolun ortalama tutma süresi. Modeller: lojistik regresyon (kıyas), sığ LightGBM, sığ XGBoost;
  LSTM/PatchTST yalnızca ağaç modelleri anlamlı beceri gösterirse ek deneme olarak.
- Başarı: OOF AUC > 0,55 ve anlamlı VE filtre/boyutlandırmayla trend kolunun net Calmar'ı bootstrap aralığı dışında iyileşiyor.

**Her soru için zorunlu:** önce en basit doğrusal model; karmaşık model ancak onu anlamlı farkla yenerse kazanan olur. Ablasyon tablosu:
katkısı sıfırdan istatistiksel olarak ayrılmayan kaynak üretim tasarımına girmez. Tüm sızıntı testleri geçmeli. Güncel global deneme
sayısıyla Deflated Sharpe ≥ 0,95.

**Çıktı:** her soru için aday modeller tablosu, ablasyon tablosu ve kazanan (ya da gerekçesiyle "ML katkısı yok");
kazanan config'ler `research/configs/` altına.

**Rehberden gelenler** (bkz. `REHBER_DEGERLENDIRME.md`): ilk proje rehberinin Faz A'sı (XGBoost) buradaki ağaç modelleri (LightGBM/XGBoost)
olarak, doğrusal kıyastan SONRA koşulur; Faz B (LSTM/PatchTST) yalnızca ağaç modelleri anlamlı beceri gösterirse ek deneme olarak.
Faz C (RL) bu turnuvanın DIŞINDADIR: yalnızca kabul edilmiş bir kolun yürütme/boyutlandırma katmanı olarak, ayrı ön kayıtla.
