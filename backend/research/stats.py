"""İstatistik araç kutusu — günlük net getiri serisi üzerinde (KURALLAR.md §2).

Not: `app.backtest.metrics.compute_metrics` İŞLEM bazlıdır (işlem pnl'i listesi); araştırma
birimi portföyün GÜNLÜK getiri serisi olduğundan metrikler burada yeniden yazıldı.
Bölücü `purged_walk_forward_splits`, `app.ml.validation.walk_forward_splits`'i çağırır.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import ACCEPTANCE, ANNUALIZATION_DAYS

EULER_GAMMA = 0.5772156649015329


def _clean(returns) -> pd.Series:
    r = pd.Series(returns, dtype=float).dropna()
    return r


# ---------------------------------------------------------------- temel metrikler
def annual_return(returns, days: int = ANNUALIZATION_DAYS) -> float:
    """Bileşik (geometrik) yıllık getiri; kesir (0,10 = %10)."""
    r = _clean(returns)
    if len(r) == 0:
        return float("nan")
    growth = float(np.prod(1.0 + r.to_numpy()))
    if growth <= 0:
        return -1.0
    return growth ** (days / len(r)) - 1.0


def annual_vol(returns, days: int = ANNUALIZATION_DAYS) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * math.sqrt(days))


def sharpe(returns, days: int = ANNUALIZATION_DAYS) -> float:
    """Yıllıklaştırılmış Sharpe (risksiz oran 0; günlük net getiri)."""
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(r.mean() / sd * math.sqrt(days))


def sortino(returns, days: int = ANNUALIZATION_DAYS) -> float:
    """Ortalama / aşağı sapma (hedef 0, tüm gözlemler üzerinden kök-ortalama-kare)."""
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    downside = np.minimum(r.to_numpy(), 0.0)
    dd = math.sqrt(float(np.mean(downside**2)))
    if dd == 0:
        return float("nan")
    return float(r.mean() / dd * math.sqrt(days))


def drawdown_series(returns) -> pd.Series:
    r = _clean(returns)
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(returns) -> tuple[float, int]:
    """(maks. drawdown (negatif kesir), en uzun ssular altında kalma süresi (gün))."""
    r = _clean(returns)
    if len(r) == 0:
        return 0.0, 0
    dd = drawdown_series(r)
    longest = current = 0
    for under in (dd < 0).to_numpy():
        current = current + 1 if under else 0
        longest = max(longest, current)
    return float(dd.min()), int(longest)


def calmar(returns, days: int = ANNUALIZATION_DAYS) -> float:
    mdd, _ = max_drawdown(returns)
    if mdd == 0:
        return float("nan")
    return annual_return(returns, days) / abs(mdd)


def yearly_returns(returns) -> pd.Series:
    """Takvim yılı başına bileşik getiri."""
    r = _clean(returns)
    if len(r) == 0:
        return pd.Series(dtype=float)
    return (1.0 + r).groupby(r.index.year).prod() - 1.0


def beta_correlation(returns, benchmark) -> tuple[float, float]:
    """(beta, korelasyon) — ortak tarihler üzerinde."""
    both = pd.concat([_clean(returns).rename("r"), _clean(benchmark).rename("b")], axis=1, join="inner").dropna()
    if len(both) < 3 or both["b"].var() == 0:
        return float("nan"), float("nan")
    beta = float(both["r"].cov(both["b"]) / both["b"].var())
    corr = float(both["r"].corr(both["b"]))
    return beta, corr


def summary(returns, benchmark=None) -> dict:
    r = _clean(returns)
    mdd, mdd_days = max_drawdown(r)
    out = {
        "n_days": int(len(r)),
        "annual_return": annual_return(r),
        "annual_vol": annual_vol(r),
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "calmar": calmar(r),
        "max_drawdown": mdd,
        "max_drawdown_days": mdd_days,
        "yearly_returns": {int(k): float(v) for k, v in yearly_returns(r).items()},
    }
    if benchmark is not None:
        out["beta_vs_btc"], out["corr_vs_btc"] = beta_correlation(r, benchmark)
    return out


# ---------------------------------------------------------------- durağan blok bootstrap
def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano (1994): her adımda p=1/mean_block olasılıkla yeni rastgele başlangıç,
    aksi halde bir sonraki gözlem (dairesel). Blok uzunluğu geometrik, ortalaması `mean_block`."""
    p = 1.0 / mean_block
    new_block = rng.random(n) < p
    new_block[0] = True
    starts = rng.integers(0, n, size=n)
    idx = np.empty(n, dtype=np.int64)
    idx[0] = starts[0]
    for i in range(1, n):
        idx[i] = starts[i] if new_block[i] else (idx[i - 1] + 1) % n
    return idx


def bootstrap_ci(
    returns,
    stat: Callable[[pd.Series], float] = sharpe,
    n_boot: int = 1000,
    mean_block: float = 15.0,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """`stat` (varsayılan Sharpe) için durağan blok bootstrap yüzdelik güven aralığı."""
    r = _clean(returns).reset_index(drop=True)
    n = len(r)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = r.to_numpy()
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, mean_block, rng)
        stats[b] = stat(pd.Series(values[idx]))
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def max_drawdown_value(returns) -> float:
    """Bootstrap için tek sayı: maks. drawdown büyüklüğü (pozitif kesir)."""
    return abs(max_drawdown(returns)[0])


# ---------------------------------------------------------------- Deflated Sharpe
def _skew_kurt(r: np.ndarray) -> tuple[float, float]:
    m = r.mean()
    s = r.std(ddof=0)
    if s == 0:
        return 0.0, 3.0
    z = (r - m) / s
    return float(np.mean(z**3)), float(np.mean(z**4))  # ham basıklık (normal = 3)


def probabilistic_sharpe(sr_hat: float, sr_benchmark: float, n: int, skew: float, kurt: float) -> float:
    """PSR (Bailey & López de Prado 2012): gözlenen (yıllıklaştırılMAMIŞ) SR'ın benchmark'ı aşma olasılığı."""
    denom = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2
    if denom <= 0 or n < 2:
        return float("nan")
    return float(norm.cdf((sr_hat - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """SR0 (BLdP 2014): N bağımsız denemede, gerçek beceri sıfırken beklenen en iyi (yıllıklaştırılmamış) SR."""
    if n_trials <= 1 or sharpe_variance <= 0:
        return 0.0
    return math.sqrt(sharpe_variance) * (
        (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials) + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe(returns, n_trials: int, trial_sharpes=None) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014) — [0,1] olasılık.

    `n_trials`: o ana kadarki TOPLAM deneme sayısı (global sayaç). `trial_sharpes`: denemelerin
    (yıllıklaştırılmamış, günlük) Sharpe'ları; verilmezse denemeler arası varyans, tek bir SR
    kestiriminin asimptotik varyansıyla yaklaşıklanır (belgelenmiş yaklaşım)."""
    r = _clean(returns).to_numpy()
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr_hat = float(r.mean() / r.std(ddof=1))
    skew, kurt = _skew_kurt(r)
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        variance = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1))
    else:
        variance = max(1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2, 1e-12) / (n - 1)
    sr0 = expected_max_sharpe(max(int(n_trials), 1), variance)
    return probabilistic_sharpe(sr_hat, sr0, n, skew, kurt)


# ---------------------------------------------------------------- PBO (CSCV)
def pbo_cscv(returns_matrix, n_blocks: int = 16) -> float:
    """Probability of Backtest Overfitting (Bailey, Borwein, López de Prado, Zhu).

    `returns_matrix`: T × N (satır = gün, sütun = bir deneyin varyantı) günlük getiriler.
    CSCV: T, `n_blocks` bitişik bloğa bölünür; her C(S, S/2) bölünmede IS'te en iyi (Sharpe)
    varyantın OOS göreli sırası ω = rank/(N+1); λ = ln(ω/(1-ω)); PBO = P(λ ≤ 0)."""
    m = np.asarray(returns_matrix, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("PBO için en az 2 varyant (sütun) gerekli")
    if n_blocks % 2 != 0 or n_blocks < 2:
        raise ValueError("n_blocks çift olmalı")
    t, n_var = m.shape
    usable = (t // n_blocks) * n_blocks
    if usable < n_blocks * 2:
        raise ValueError("PBO için yeterli gözlem yok")
    m = m[t - usable :]  # en eski kalıntıyı at
    blocks = np.array_split(m, n_blocks)
    cnt = np.array([len(b) for b in blocks], dtype=float)
    s1 = np.array([b.sum(axis=0) for b in blocks])  # S × N
    s2 = np.array([(b**2).sum(axis=0) for b in blocks])

    def sharpe_of(sel: tuple[int, ...]) -> np.ndarray:
        c = cnt[list(sel)].sum()
        mean = s1[list(sel)].sum(axis=0) / c
        var = s2[list(sel)].sum(axis=0) / c - mean**2
        with np.errstate(divide="ignore", invalid="ignore"):
            sr = np.where(var > 1e-18, mean / np.sqrt(np.maximum(var, 1e-18)), 0.0)
        return sr

    all_blocks = tuple(range(n_blocks))
    logits = []
    for is_blocks in itertools.combinations(all_blocks, n_blocks // 2):
        oos_blocks = tuple(b for b in all_blocks if b not in is_blocks)
        sr_is = sharpe_of(is_blocks)
        sr_oos = sharpe_of(oos_blocks)
        best = int(np.argmax(sr_is))
        rank = int(np.sum(sr_oos <= sr_oos[best]))  # 1..N
        omega = rank / (n_var + 1.0)
        logits.append(math.log(omega / (1.0 - omega)))
    logits_arr = np.asarray(logits)
    return float(np.mean(logits_arr <= 0.0))


# ---------------------------------------------------------------- purge + embargo bölücü
def purged_walk_forward_splits(
    dates, n_splits: int = 5, embargo_days: float = 0.0, purge_days: float = 0.0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tarih bazlı walk-forward bölmeleri. `purge_days` (etiket ufku) + `embargo_days`, eğitim
    ucuyla test başı arasında bırakılan boşluktur. `app.ml.validation.walk_forward_splits`'i
    (`time_frac` ekseni) çağırır."""
    from app.ml.validation import walk_forward_splits

    idx = pd.DatetimeIndex(dates)
    span_days = (idx.max() - idx.min()).total_seconds() / 86400.0
    if span_days <= 0:
        raise ValueError("tarih aralığı sıfır")
    time_frac = pd.Series(((idx - idx.min()).total_seconds() / 86400.0 / span_days).to_numpy())
    embargo_frac = (embargo_days + purge_days) / span_days
    return walk_forward_splits(time_frac, n_splits=n_splits, embargo_frac=embargo_frac)


# ---------------------------------------------------------------- kabul eşikleri
def check_acceptance(
    *,
    sharpe_net: float,
    deflated: float,
    pbo: float,
    max_dd: float,
    calmar_value: float,
    btc_calmar: float,
    positive_year_fraction: float,
    stress_sharpe: float,
    plateau_ratio: float,
    is_portfolio: bool,
) -> dict[str, bool]:
    """KURALLAR.md §5 eşiklerine karşı geçti/kaldı (eşikler `research.config.ACCEPTANCE`)."""
    a = ACCEPTANCE
    ok = lambda cond: bool(cond) if cond == cond else False  # NaN -> False
    return {
        "net_sharpe": ok(sharpe_net >= (a["sharpe_portfolio"] if is_portfolio else a["sharpe_arm"])),
        "deflated_sharpe": ok(deflated >= a["deflated_sharpe"]),
        "pbo": ok(pbo <= a["pbo_max"]),
        "max_drawdown": ok(abs(max_dd) <= (a["max_drawdown_portfolio"] if is_portfolio else a["max_drawdown_arm"])),
        "calmar": ok(calmar_value >= max(btc_calmar if btc_calmar == btc_calmar else 0.0, a["calmar_floor"])),
        "positive_years": ok(positive_year_fraction >= a["positive_year_fraction"]),
        "stress": ok(stress_sharpe >= a["stress_sharpe"]),
        "plateau": ok(plateau_ratio >= a["plateau_ratio"]),
    }
