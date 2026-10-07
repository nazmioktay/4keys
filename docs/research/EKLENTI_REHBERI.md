# Eklenti rehberi: yeni model, kaynak, hedef veya sinyal adaptörü eklemek

Önce `docs/research/KURALLAR.md` ve `deneyler.md`'yi okuyun. Bu rehber, çerçevenin (`backend/research/`) genişletilmesini anlatır.
Strateji sonucu üretmeden önce **eklentiniz kendi testlerinden ve çerçevenin sızıntı testlerinden geçmelidir**.

## Ortam
```bash
docker build -t 4keys-backend-test -f backend/Dockerfile backend            # temel imaj (bir kez)
docker build -t 4keys-research-test -f backend/research/Dockerfile backend  # + pyarrow, matplotlib, lightgbm, arch, libgomp1
docker run --rm -v "$PWD:/repo" -w /repo/backend 4keys-research-test python -m pytest -q tests/research
```
Veri önbelleği: `python -m research.data.download --all --hourly-top 60` (günlük/funding/spot/vadeli/1h) ve
`python -m research.data.download_metrics --symbols BTCUSDT ETHUSDT` (futures `metrics`; `--coverage` yalnızca listeler).

## Deney nasıl koşar
```bash
cd backend && python -m research.runner research/configs/ornek_vol.yaml           # config'te smoke: true ise sayaç artmaz
python -m research.runner research/configs/ornek_vol.yaml --smoke                 # her config'i smoke olarak koştur
python -m research.ablation research/configs/ornek_vol.yaml [--smoke]              # her kaynak tek tek çıkarılır (varsayılan: her biri KAYITLI deneme)
```
Tek YAML: `id, question, zone (main|forward; varsayılan main, bkz. KURALLAR §10), sources, universe{n}, period{start,end}, target{name,h}, models[], cv{n_splits,embargo_days,min_train_days},
signal{adapter,params}, arm, seed, checks{...}` (bkz. `research/configs/ornek_vol.yaml`). Akış: noktasal-zamanlı evren → kaynak
özellikleri (PIT birleştirme) → hedef → purge+embargo'lu walk-forward → OOF → tahmin metrikleri → sinyal adaptörü → Aşama 0
motoru → sızıntı testleri → kayıt (sayaç, Deflated Sharpe, deneyler.md) → rapor.

## 1) Yeni veri kaynağı
1. `docs/research/sablonlar/kaynak_sablonu.py` → `backend/research/sources/<ad>.py`; sınıfı `Source`'tan türetin, `@register_source`.
2. `backend/research/sources/registry.py::_MODULES` listesine modül adını ekleyin.
3. **Zaman sözleşmesi** (en kritik kısım): satır `date` = gözlem günü, karar o günün kapanışında (date+1 gün 00:00 UTC). Her satır
   `available_at` taşır; `available_at <= date+1 gün` ise kullanılabilir. Yayın gecikmesini `publication_lag` ve `available_at`'e yansıtın;
   şüphede MUHAFAZAKÂR (daha geç) olun. Özellikler yalnızca geçmişe bakan pencerelerle hesaplanır.
4. **Eksik değer**: 0'la doldurmayın — NaN bırakın. Birleştirme `<kaynak>__missing` bayrağı ekler (önceki sürümde makroyu 0'la
   doldurmak eğitim/canlı uyumsuzluğu yaratmıştı).
5. **forward_only** (geçmişi olmayan): `forward_only = True`, `accumulated_days()` ve `feature_names` uygulayın. Koşucu 12 aydan az
   birikimle bu kaynağı kullanan deneyi REDDEDER (`ForwardOnlyTooShortError`).
6. **Kaynak kontrol listesi (testler)**: kayıt+bilinmeyen ad hatası; `validate_panel` sözleşmesi; bilinen değerle özellik hesabı;
   `available_at` doğru; **gelecek bozulunca geçmiş özellik değişmiyor**; boş/eksik veri davranışı; önbellek anahtarı (version/params).
   Örnekler: `tests/research/test_sources.py`, `test_forward_sources.py`.
7. Kapsam raporu: yeni kaynak otomatik dahil olur; `cd backend && python -m research.coverage_report` ile `docs/research/kaynak_kapsami.md`'yi yeniden üretin.

## 2) Yeni model
1. `docs/research/sablonlar/model_sablonu.py` → `backend/research/models/<ad>.py`; `BaseModel`'den türetin, `@register_model("ad")`.
2. `backend/research/models/registry.py::_MODULES` listesine ekleyin. (Yeni = **tek dosya + bu satır**.)
3. Arayüz: `fit(X, y, dates, sample_weight=None)`, `predict(X)`, `get_params()`, sabit `seed`; `task` ∈ regression / classification (ikili) / rank.
   NaN: `handles_nan=False` ise eğitim medyanıyla doldurulur (medyan YALNIZCA eğitim katmanından); aksi halde modeliniz NaN'ı yönetir.
4. **Determinizm zorunlu**: tohumsuz/global RNG kullanmayın, çok iş parçacığı sıra farkı yaratıyorsa tek iş parçacığı (`n_jobs=1`).
   Tekrarlanabilirlik kontrolü (aynı config iki kez → aynı OOF) bozuk modeli GEÇERSİZ sayar.
5. **Model kontrol listesi (testler)**: kayıt; öğrenme (sentetik veri); aynı seed → birebir aynı; NaN girdi; `sample_weight`; tahmin edilemeyen
   satırda NaN. Örnek: `tests/research/test_models.py`. Hedef türü uyumu: `regression` model `rank` hedefte de kullanılabilir.
6. Bağımlılık eklerseniz `backend/requirements-research.txt` ve `backend/research/Dockerfile`'a ekleyin (sistem kütüphanesi gerekebilir:
   LightGBM → libgomp1).

## 3) Yeni hedef / sinyal adaptörü
- **Hedef** (`targets.py`): `Target(name, horizon, kind, ...)` döndüren fabrika + `TARGET_FACTORIES`. Etiket hizası: karar t kapanışı → ileri getiri
  `open[t+1] → open[t+1+h]`; etiket `date + (h+1)` günde tamamlanır. `horizon` bildirin (purge = h, embargo ≥ h zorunlu). Test: etiket yalnızca
  t+1.. verisinden gelir (t ve öncesini bozmak etiketi değiştirmemeli).
- **Sinyal adaptörü** (`signals.py`): `(pred_wide, membership, **params) -> ağırlık` fonksiyonu + `ADAPTERS`. Evren dışı → 0 ağırlık; NaN tahmin → pozisyon yok.

## Otomatik sızıntı / sağlamlık testleri (biri başarısızsa deney GEÇERSİZ)
| Test | Ne yakalar | Not |
|---|---|---|
| karıştırılmış hedef | eğitimin test etiketini görmesi (CV sızıntısı) | etiketler tarih içinde karıştırılır; IC ≈ 0 (\|t\| < 3) olmalı |
| available_at denetimi | karar zamanından sonra yayımlanan değer; test dönemine taşan eğitim etiketi | ihlal sayısı 0 olmalı |
| ek gecikme (+1 gün) | zamanlama sızıntısı şüphesi (UYARI) | **sınır:** kalıcı hedeflerde (volatilite) güçsüz — orada diğer ikisine güvenin |
| tekrarlanabilirlik | tohumsuz/nondeterministik model | aynı config iki kez → aynı OOF |

Geçersiz deney yine KAYDEDİLİR (karar "GEÇERSİZ: ..."; sayaç artar — gizlenmez). Her test kendi negatif kontrolüyle doğrulanır (`tests/research/test_checks.py`).

## Deneme bütçesi ve sayaç
Soru adı normalize edilir (`strip`+küçük harf); `smoke` koşuları sayaca girmez ama soru başına en fazla 10'dur ve sonuçları karar için kullanılamaz. Soru (`question`) başına en fazla **40 varyant** (deneme sayacına giren her varyant; ablasyonlar dahil; tek deney de aşamaz); dolunca
`BudgetExceededError`, yeni deneme yapılmaz — mevcut sonuçlardan karar verilir. Hesap varsayılanı 250 USDT NAV, 3x kaldıraç, en yakın lot + Binance min notional/adım
(`account: {nav, leverage, lot_rounding, limits}`; KURALLAR §3); rapor 1x için önerilen asgari hesabı da yazar.
Her varyant (model) bir deneme sayılır (Deflated Sharpe için global sayaç). `smoke: true` / `--smoke` hiçbir şey kaydetmez.

## İleriye dönük toplayıcılar (canlı koda dokunan tek istisna)
Tasfiye akışı, emir defteri derinlik bantları ve ayrıntılı açık pozisyon `backend/app/forwardcollect/` ile toplanır; **varsayılan KAPALI**
(`FOURKEYS_FORWARD_COLLECTORS_ENABLED=true` ile açılır; semboller `FOURKEYS_FORWARD_COLLECTOR_SYMBOLS`). Yeni tablolar: `liquidation_events`,
`depth_band_snapshots`, `oi_detail_snapshots` (mevcut tabloları değiştirmez). 12 ay birikmeden bu kaynaklarla deney yapılamaz; yapılınca da yalnızca `zone: forward` deneyde ve forward keşif aralığında
[2026-11-01, 2027-11-01) (KURALLAR.md §10). Evren ilk ~90 gün boş kalır: etkin keşif ~9 ay, `cv.min_train_days`'i buna göre seçin. WebSocket
bağlantısı birim testlerinde sahte akışla sınanır; **gerçek bağlantı elle doğrulanmalıdır** (açıp `docker logs` ve `liquidation_events` satırlarına bakın).

## Yeniden üretilebilirlik notu (önbellek veri imzası)
DB/ağ kaynaklarının (forward_only, `sentiment_fng`, `macro`) önbellek anahtarı veri imzasını içerir: `sentiment_fng` için 12 saatlik dilim,
`macro` için UTC günü. Bu yüzden bir deney bu sınırları aşan bir sürede yeniden koşulursa önbellek yeniden inşa edilir (beklenen). Tam
yeniden üretilebilirlik için veri anlık görüntüsünü (parquet/DB) sabitleyin; deney kaydındaki `data_snapshot_hash` yalnızca dosya tabanlı veriyi kapsar.

