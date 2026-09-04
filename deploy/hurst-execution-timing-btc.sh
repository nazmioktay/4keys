#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Hurst-tabanli islem zamanlamasi hipotezini BTC/USDT'nin gercek
# gecmisinde test eder: sinyal aninda H<0.5 (ortalamaya-donus) iken
# birkac mum GECIKTIRMEK, H>0.5 (trend-devamliligi) durumuna gore daha
# ucuz bir fiyat veriyor mu? (bkz. app/rl/execution_timing.py docstring'i
# — "optimal execution"un piyasa etkisi modeli olmadan neden anlamli
# olmadigi ve bunun yerine neden bu hipotezin test edildigi.)
#
# UYARI: sonuc, canli bir ajan/strateji DEGILDIR — yalnizca tarihsel bir
# gozlem/hipotez testidir. Delay_bars SABITTIR (veriye bakarak "en iyi"
# gecikme aranmaz, look-ahead yanliligindan kacinmak icin).
# ============================================================

curl -sS "http://127.0.0.1:8000/rl/hurst-execution-timing?symbol=BTC%2FUSDT%3AUSDT&delay_bars=3"
echo
echo "Tamamlandi. 'low' (H<0.45) grubundaki mean_delayed_slippage_pct negatifse ve 'high' (H>0.55) grubundakinden daha dusukse (daha ucuzsa), hipotez destekleniyor demektir."
