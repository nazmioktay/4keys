# İlk proje rehberinin değerlendirmesi — "Bölüm 2: ML Metodolojisi" (2026-10-08)

Projenin başındaki rehber (XGBoost → LSTM → RL fazları, overfitting önlemleri, ensemble) canlı botun saatlik ML motorunun yol haritasıydı
ve büyük ölçüde UYGULANDI. Bu belge, rehberin hangi kısmının bugünkü bulgularla (bkz. `DURUM_RAPORU.md`, `deneyler.md`) hâlâ geçerli
olduğunu, hangisinin düzeltilmesi gerektiğini ve yeniden kullanılırken hangi kurallara uyulacağını kayda geçirir. Kural değişikliği
değildir; KURALLAR.md geçerlidir.

## Rehber ne kadar uygulandı?
| Faz | Durum (kodda) | Sonuç |
|---|---|---|
| A — XGBoost | Uygulandı (`backend/app/ml/model.py`, `train.py`, meta-label `meta_label.py`) | Dürüst ölçümde avantaj yok: ~8 ayda 24–47 işlem, PnL ≈ 0. |
| B — LSTM | Uygulandı (`backend/app/ml/lstm_model.py`); PatchTST alternatifi de var (`patchtst_model.py`) | Balanced accuracy ~%38'de takıldı; rafta. Ensemble'a katılımı elle açılmaz: her eğitimin sonunda kalite eşiğini (`ml_min_balanced_accuracy`, `app.ml.model_status`) geçerse otomatik devreye girer. |
| C — RL | Yalnızca hazırlık: ortam (`backend/app/rl/environment.py`), epizod verisi (`rl/dataset.py`), rastgele politika referansı (`rl/train.py`), Hurst tabanlı yürütme zamanlaması testi (`rl/execution_timing.py`) | Ajan (PPO/DQN) EĞİTİLMEDİ; `stable-baselines3` bilinçli olarak eklenmedi. |
| Ensemble | Rehberdeki sabit %60/%30 değil: kural ve beceri ağırlıklı birleştirme (`DecisionEngine._combine_predictions`, `_skill_weight`, `backend/app/engine/decision.py`) | Üyelerin hiçbirinin avantajı olmadığı için ensemble de avantaj üretmedi. |

## Bölüm bölüm değerlendirme
| Rehber bölümü | Karar | Gerekçe |
|---|---|---|
| 2.1 "Model kuralı kendisi bulur; insan gözünün göremeyeceği ilişkileri görür" | DÜZELT | Tanım doğru, vaat yanıltıcı. Gürültülü fiyat verisinde çok değişken, ezber riskini büyütür. ML avantaj YARATMAZ, var olanı büyütür: saatlik yön tahmini bunun kanıtı. |
| 2.2 Önce XGBoost | KORU | Tablo verisinde en sağlam aile (XGBoost/LightGBM). Ek kural: ondan önce doğrusal kıyas (lojistik/ridge); ağaç modeli ancak onu anlamlı farkla yenerse kazanan olur. |
| 2.2 "Yorumlanabilir (SHAP), overfitting'i fark etmek kolay" | DÜZELT | SHAP eğitim verisindeki katkıyı gösterir, aşırı uyumu ÖNLEMEZ ve ÖLÇMEZ. Önem test (OOF) verisinde ölçülmeli: permütasyon önemi (`backend/research/importance.py`) ve kaynak ablasyonu (`backend/research/ablation.py`). Kanıt: özellik ekleme izole doğruluğu artırırken sistem performansını düşürdü (README "madde 18"). |
| 2.3 LSTM (zamanı öğrenen model) | KOŞULLU | Rafta kalır. Ancak ağaç modelleri aynı soruda anlamlı beceri gösterirse, ek ve sayılan deneme olarak (bkz. `sablonlar/ml_turnuva.md`). Az ve gürültülü veride derin modeller nadiren fayda sağlar. |
| 2.4 Reinforcement Learning | ERTELE | Rehberin kendi uyarısı doğru: simülatör gerçeği yansıtmazsa ajan yanlış şey öğrenir. Artık gerçekçi simülatörler var (`backend/research/engine.py`, `backend/research/carry.py`) ama RL, en çok deneme tüketen ve en kolay aşırı uyan yöntem. Yalnızca kabul edilmiş bir kolun yürütme/boyutlandırma katmanı olarak, ayrı ön kayıt ve bütçeyle. Doğrudan al/sat sinyali için kullanılmaz. |
| 2.5 Overfitting önlemleri | KORU + GÜÇLENDİRİLDİ | Hepsi KURALLAR.md'de daha sıkı: purged walk-forward + embargo; kilitli nihai pencere 13 ay (rehber: 6 ay) ve tek kullanımlık; paper için 8 hafta / 30 dengeleme önerisi (`sablonlar/portfoy_paper.md`). Rehberde OLMAYAN en önemli önlem: **deneme sayımı** (Deflated Sharpe, PBO, soru başına 40 varyant bütçesi, ön kayıt). |
| 2.5 "Geçmişte %85, gerçekte %48" örneği | KORU, EKLE | İlk +%38'lik backtest bunun canlı örneği; ama asıl nedeni ezber değil **ölçüm hatası**: yarım mum, kapanıştan giriş, yalnızca kapanışla stop, funding yok. Rehbere eklenmesi gereken ikinci tehlike: gerçekçi olmayan backtest. |
| 2.6 Ensemble (%60 XGBoost, %30 LSTM, RL karar katmanı) | DÜZELT | "Tek modelden daha dayanıklı" ancak her üyenin kendi avantajı varsa ve hataları düşük korelasyonluysa doğru. Avantajı olmayan modellerin birleşimi avantaj üretmez. Ağırlıklar elle sabitlenmez; yalnızca geçmiş/OOF veriyle öğrenilir (`backend/research/stacking.py`). |

## Rehberde olmayan ama gereken
- **Ekonomik ölçüt:** doğruluk/AUC değil; ücret, kayma, funding, gecikme ve asgari emir sonrası net getiri, düşüş ve turnover.
- **Noktasal-zamanlı veri:** her satır, bilginin gerçekte ne zaman bilindiğini (`available_at`) taşır; delist olan semboller evrende kalır.
- **Etiketlerin örtüşmesi:** ardışık örnekler aynı gelecek getiriyi paylaşınca istatistikler aşırı güvenli olur (purge/embargo, benzersizlik ağırlığı).
- **Hesap gerçekliği:** 250 USDT'de asgari emir tutarları çeşitlendirmeyi ve modelin önerdiği boyutları sınırlar.

## Yeniden kullanım kuralları (rehberden bir şey alınırken)
1. ML yalnızca **kabul edilmiş bir kolu** büyütmek için kullanılır (KURALLAR.md, `sablonlar/ml_turnuva.md`).
2. Her soruda önce en basit doğrusal model; karmaşık model ancak anlamlı farkla kazanırsa.
3. Özellik/kaynak önemi yalnızca test verisinde ve ablasyonla; SHAP yalnızca açıklama aracı.
4. LSTM/PatchTST yalnızca ağaç modelleri beceri gösterirse; RL yalnızca yürütme/boyutlandırma katmanı ve ayrı ön kayıtla.
5. Ensemble ağırlıkları öğrenilir, elle sabitlenmez; üyelerin her biri tek başına beceri göstermelidir.
6. Her deneme sayılır; sonuç görülmeden önce ön kayda yazılır.
