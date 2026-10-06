"""Araştırma sabitleri — docs/research/KURALLAR.md ile BİREBİR aynı olmalıdır.

Kabul eşikleri ve nihai pencere tarihi önceden kayıtlıdır; sonuç görüldükten sonra DEĞİŞTİRİLEMEZ."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# backend/research/config.py -> repo kökü
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
DOCS_DIR = REPO_ROOT / "docs" / "research"
EXPERIMENT_LOG = DOCS_DIR / "deneyler.md"
RESULTS_DIR = REPO_ROOT / "research" / "results"
REGISTRY_FILE = RESULTS_DIR / "_registry.json"
CACHE_DIR = BACKEND_ROOT / "research" / "_cache"

# --- Nihai test penceresi (KURALLAR.md §1) ---
FINAL_TEST_START = pd.Timestamp("2025-10-01")
FINAL_TEST_MARKER = "FINAL-TEST-ACILDI"

# --- Maliyet varsayılanları (KURALLAR.md §3), oran olarak (0,0005 = %0,05) ---
FUTURES_TAKER_FEE = 0.0005
FUTURES_MAKER_FEE = 0.0002
SPOT_FEE = 0.0010
SLIPPAGE_BPS_BTC_ETH = 2.0
SLIPPAGE_BPS_TOP20 = 5.0
SLIPPAGE_BPS_OTHER = 15.0
TOP_TIER_SYMBOLS = ("BTCUSDT", "ETHUSDT")
TOP_TIER_COUNT = 20  # "ilk 20 sembol" = noktasal-zamanlı hacim sıralaması

ANNUALIZATION_DAYS = 365

# --- Kabul eşikleri (KURALLAR.md §5) ---
ACCEPTANCE = {
    "sharpe_arm": 0.8,
    "sharpe_portfolio": 1.0,
    "deflated_sharpe": 0.95,
    "pbo_max": 0.25,
    "max_drawdown_arm": 0.35,
    "max_drawdown_portfolio": 0.30,
    "calmar_floor": 0.7,  # Calmar ≥ max(BTC al-tut Calmar'ı, bu değer)
    "positive_year_fraction": 0.60,
    "stress_sharpe": 0.5,  # ücret ×2, kayma ×3
    "stress_fee_mult": 2.0,
    "stress_slippage_mult": 3.0,
    "plateau_ratio": 0.70,
}

# Stablecoin / kaldıraçlı token / endeks sembolleri (evren dışı) — USDT-M perpetual için
STABLE_BASES = {"USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI", "USDD", "EUR", "AEUR", "USDE", "RLUSD"}
INDEX_SYMBOLS = {"BTCDOMUSDT", "DEFIUSDT", "FOOTBALLUSDT", "BLUEBIRDUSDT"}  # Binance endeks perpetual'ları
