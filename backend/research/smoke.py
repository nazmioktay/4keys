"""Altyapı kontrolü: BTC al-tut ve EMA200 long/flat (günlük) — motor + veri + istatistik + rapor.

BU BİR DENEY DEĞİLDİR: deneme sayacı artmaz, deneyler.md/results kaydı yazılmaz.
Çıktı `research/results/_smoke/` altına gider (git'e girmez). Nihai pencere kesilir.
Kullanım: cd backend && python -m research.smoke   (önce: python -m research.data.download --symbols BTCUSDT)
"""

from __future__ import annotations

import sys

import pandas as pd

from . import config, engine
from .data import store
from .registry import current_trial_count
from .report import render_report

SYMBOL = "BTCUSDT"


def ema200_long_flat(close: pd.Series, span: int = 200) -> pd.Series:
    """close > EMA(span) ise 1 (long), değilse 0 (nakit). Isınma süresince 0."""
    ema = close.ewm(span=span, adjust=False).mean()
    signal = (close > ema).astype(float)
    signal.iloc[: span - 1] = 0.0
    return signal.rename("w")


def main() -> int:
    trials_before = current_trial_count()
    daily = store.load_klines(SYMBOL, "um_1d")
    if daily is None or daily.empty:
        print(f"Veri yok: önce `python -m research.data.download --symbols {SYMBOL}` çalıştırın.")
        return 2
    open_px = daily[["open"]].rename(columns={"open": SYMBOL})
    close = daily["close"]
    funding = store.load_funding(SYMBOL)
    events = (
        pd.DataFrame({"symbol": SYMBOL, "time": funding.index, "rate": funding["rate"].to_numpy()})
        if funding is not None
        else None
    )
    fund = engine.daily_funding(events, open_px.index, [SYMBOL]) if events is not None else None

    strategies = {
        "btc_al_tut": pd.DataFrame({SYMBOL: 1.0}, index=open_px.index),
        "ema200_long_flat": ema200_long_flat(close).to_frame(SYMBOL),
    }
    results = {}
    out_root = config.RESULTS_DIR / "_smoke"
    for name, weights in strategies.items():
        res = engine.run(
            weights, open_px,
            fee_rate=config.FUTURES_TAKER_FEE, slippage_bps=config.SLIPPAGE_BPS_BTC_ETH, funding=fund,
        )
        results[name] = res
    benchmark = results["btc_al_tut"].returns
    print(f"Veri: {SYMBOL} günlük {open_px.index.min():%Y-%m-%d} → {open_px.index.max():%Y-%m-%d} (nihai pencere kesildi: "
          f"< {config.FINAL_TEST_START:%Y-%m-%d}); veri hash: {store.snapshot_hash()}")
    for name, res in results.items():
        costs = engine.cost_summary(res)
        rep = render_report(
            res.returns, benchmark, out_root / name, name,
            extra={"toplam maliyet (NAV oranı)": f"{costs['total_cost']:.4f}", "komisyon": f"{costs['commission']:.4f}",
                   "kayma": f"{costs['slippage']:.4f}", "funding": f"{costs['funding']:.4f}",
                   "devir (toplam)": f"{float(res.turnover.sum()):.1f}"},
        )
        m = rep["metrics"]
        print(f"{name:18s} yıllık={m['annual_return']*100:6.1f}%  Sharpe={m['sharpe']:.2f}  Calmar={m['calmar']:.2f}  "
              f"maksDD={m['max_drawdown']*100:6.1f}%  -> {rep['md']}")
    trials_after = current_trial_count()
    assert trials_after == trials_before, "smoke deneme sayacını artırmamalı"
    print(f"Deneme sayacı: {trials_after} (smoke deney sayılmaz).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
