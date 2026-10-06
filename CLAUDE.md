# Proje Kuralları

Python backend (FastAPI, APScheduler, ML) + React frontend + Docker Compose (TimescaleDB, Prometheus, Grafana).
Sistem gerçek borsa hesaplarına bağlanabilir; güvenlik her şeyden önce gelir.

## Çalışma düzeni
- Geliştirme ve testlerde HER ZAMAN dry-run / testnet / paper-trading kullan. Canlı emir gönderen kodu çalıştırma.
- `.env` ve `secrets/` dosyalarını okuma, yazdırma, loglama.
- Yeni davranış için test yaz. Backend testi: `cd backend && pytest -x -q`.
- Python lint: `ruff check .` — frontend lint: `cd frontend && npm run lint`.
- Yeni bağımlılık eklersen `requirements` dosyasına da yaz.
- Zaman serisi verisinde look-ahead bias yapma; doğrulama walk-forward / purged olmalı. Tüm zamanlar UTC.

## İş bitirme protokolü (zorunlu)
Kod değişikliği içeren her anlamlı iş bittiğinde, kullanıcıya nihai cevabı yazmadan ÖNCE:

1. Testleri ve lint'i çalıştır, hataları düzelt.
2. `reviewer` subagent'ını çağır (değişen dosyaları ve amacı kısaca ver).
3. Reviewer kararı:
   - **ONAY** → özeti yaz.
   - **DÜZELTME GEREKLİ** → bulguları düzelt, testleri tekrar çalıştır, reviewer'ı bir kez daha çağır.
   - **ENGEL** → durup kullanıcıya bildir, kendi kararınla ilerleme.
4. Düzeltme-denetim döngüsü en fazla 3 tur sürsün; hâlâ çözülmediyse kalan sorunları açıkça kullanıcıya yaz.
5. Nihai cevapta: ne yapıldı, reviewer'ın kararı, çözülen bulgular ve çözülmeyen riskler yer alsın. Reviewer'ın doğrulayamadığı şeyleri kesinmiş gibi sunma.

Küçük işlerde (yorum, yazım düzeltmesi, tek satırlık config) reviewer atlanabilir.

## Onay gerektirenler
`git push`, `deploy/*.sh`, `docker compose up/down/restart` ve veritabanı işlemleri için kullanıcıya sor.

## Araştırma işleri
Araştırma işlerinde önce docs/research/KURALLAR.md ve deneyler.md'yi oku. Araştırma kodu (`backend/research/`) canlı/paper koduna (`backend/app/**`) dokunmaz (istisna: opt-in, varsayılan KAPALI `app/forwardcollect`). Araştırma testleri için `backend/research/Dockerfile` imajı gerekir (bkz. docs/research/EKLENTI_REHBERI.md).
