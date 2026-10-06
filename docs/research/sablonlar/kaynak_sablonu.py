"""KAYNAK ŞABLONU — yeni bir veri kaynağı eklemek için kopyalayın:

    cp docs/research/sablonlar/kaynak_sablonu.py backend/research/sources/benim_kaynagim.py

Sonra: (1) sınıfı doldurun, (2) `backend/research/sources/registry.py::_MODULES` listesine "benim_kaynagim" ekleyin,
(3) `backend/tests/research/` altına test yazın (ZORUNLU testler rehberin "Kaynak kontrol listesi"nde).
Kurallar: docs/research/EKLENTI_REHBERI.md. Bu dosya paketin parçası DEĞİLDİR (içe aktarılmaz)."""

from __future__ import annotations

import pandas as pd

from research.data import store
from research.sources.base import MARKET, Source  # MARKET: piyasa-geneli kaynaklar için sembol değeri
from research.sources.registry import register_source


@register_source
class BenimKaynagim(Source):
    name = "benim_kaynagim"            # config'te `sources: [benim_kaynagim]`; özellik kolonları `benim_kaynagim__...` önekli olur
    version = "1"                      # veri/özellik tanımı DEĞİŞİNCE artırın (önbellek anahtarına girer)
    scope = "symbol"                   # "symbol" (sembol bazlı) | "market" (piyasa-geneli; symbol=MARKET)
    forward_only = False               # True: geçmişi YOK, yalnızca bugünden itibaren toplanır -> `accumulated_days()` zorunlu (12 ay kuralı)
    publication_lag = pd.Timedelta(0)  # verinin gerçekte ne kadar gecikmeyle yayımlandığı (belgeleme + available_at'e yansıtın)
    max_staleness = pd.Timedelta(days=3)  # birleştirmede en çok bu kadar eski değer kullanılır, ötesi NaN
    feature_names = ()                 # BOŞ olabilen kaynaklarda (ör. forward_only) önekli özellik adları (boş panelde kolonlar var olsun)

    @property
    def history_start(self):
        """Kaynağın en erken tarihi (bilinmiyorsa None)."""
        return None

    def fetch(self, start, end, symbols=None):
        """Ham veriyi döner (dosya/ağ/DB). Nihai test penceresi (>= 2025-10-01) için `store.load_*` zaten KESER;
        kendi kaynağınız harici veriyse `research.guard.cut_final_test(...)` ile siz kesin."""
        raise NotImplementedError

    def to_panel(self, universe, dates):
        """[date, symbol, <özellikler>, available_at] döndürün.

        - `date`: GÖZLEM günü. Karar o günün kapanışında (date + 1 gün 00:00 UTC) verilir.
        - `available_at`: bu satırın değerinin GERÇEKTEN bilinebildiği an. `available_at <= date + 1 gün` ise o gün kullanılabilir.
          Emin değilseniz MUHAFAZAKÂR olun (daha geç).
        - Özellikler yalnızca `date` ve ÖNCESİ veriyle hesaplanmalı (geriye bakan pencereler; gelecek YOK).
        - Eksik değer 0'la doldurmayın: NaN bırakın (birleştirme `<kaynak>__missing` bayrağı ekler).
        - Özellik kolonlarını `self.prefix(frame)` ile önekleyin.
        """
        frame = pd.DataFrame(columns=["date", "symbol", "x", "available_at"])  # örnek iskelet
        return self.prefix(frame)
