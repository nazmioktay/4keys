import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from research import stats


def _daily(values, start="2021-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"), dtype=float)


# --- temel metrikler -------------------------------------------------------
def test_annual_return_constant_daily_return_is_compounded_geometrically():
    r = _daily([0.001] * 365)
    assert stats.annual_return(r) == pytest.approx(1.001**365 - 1)
    # 730 günlük aynı getiri de yıllıklaştırılınca aynı değeri verir
    assert stats.annual_return(_daily([0.001] * 730)) == pytest.approx(1.001**365 - 1)


def test_annual_vol_and_sharpe_match_hand_computation():
    r = _daily([0.01, -0.01, 0.02, 0.0, 0.01, -0.02])
    sd = np.std(r.to_numpy(), ddof=1)
    assert stats.annual_vol(r) == pytest.approx(sd * math.sqrt(365))
    assert stats.sharpe(r) == pytest.approx(r.mean() / sd * math.sqrt(365))


def test_sharpe_is_zero_for_symmetric_alternation_and_nan_for_constant():
    assert stats.sharpe(_daily([0.01, -0.01] * 50)) == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(stats.sharpe(_daily([0.0] * 10)))


def test_sortino_uses_root_mean_square_of_negative_returns():
    r = _daily([0.01, -0.02, 0.03, -0.01])
    downside = math.sqrt((0.02**2 + 0.01**2) / 4)
    assert stats.sortino(r) == pytest.approx(r.mean() / downside * math.sqrt(365))


def test_max_drawdown_and_duration_known_path():
    # eşitlik: 1.1, 0.55, 0.66, 0.726 -> tepe 1.1, dip 0.55 => -%50; 3 gün su altında
    r = _daily([0.1, -0.5, 0.2, 0.1])
    mdd, days = stats.max_drawdown(r)
    assert mdd == pytest.approx(-0.5)
    assert days == 3
    assert stats.max_drawdown(_daily([0.01, 0.01, 0.01])) == (0.0, 0)


def test_calmar_is_annual_return_over_abs_max_drawdown():
    r = _daily([0.1, -0.5, 0.2, 0.1])
    assert stats.calmar(r) == pytest.approx(stats.annual_return(r) / 0.5)


def test_yearly_returns_compound_per_calendar_year():
    idx = pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-01", "2021-01-02"])
    r = pd.Series([0.1, 0.1, -0.1, 0.0], index=idx)
    yr = stats.yearly_returns(r)
    assert yr[2020] == pytest.approx(1.1 * 1.1 - 1)
    assert yr[2021] == pytest.approx(0.9 - 1)


def test_beta_and_correlation_of_scaled_benchmark():
    rng = np.random.default_rng(0)
    b = _daily(rng.normal(0, 0.02, 500))
    beta, corr = stats.beta_correlation(2.0 * b, b)
    assert beta == pytest.approx(2.0)
    assert corr == pytest.approx(1.0)
    beta0, corr0 = stats.beta_correlation(_daily(rng.normal(0, 0.02, 500)), b)
    assert abs(corr0) < 0.15


# --- durağan blok bootstrap -------------------------------------------------
def test_stationary_bootstrap_mean_block_length_is_about_requested():
    rng = np.random.default_rng(1)
    n = 20000
    idx = stats.stationary_bootstrap_indices(n, 15.0, rng)
    breaks = np.sum(idx[1:] != (idx[:-1] + 1) % n)  # yeni blok başlangıçları
    mean_block = n / (breaks + 1)
    assert mean_block == pytest.approx(15.0, rel=0.1)


def test_bootstrap_ci_is_reproducible_and_brackets_point_estimate():
    rng = np.random.default_rng(2)
    r = _daily(rng.normal(0.001, 0.01, 800))
    lo1, hi1 = stats.bootstrap_ci(r, stats.sharpe, n_boot=300, seed=7)
    lo2, hi2 = stats.bootstrap_ci(r, stats.sharpe, n_boot=300, seed=7)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 < stats.sharpe(r) < hi1
    dlo, dhi = stats.bootstrap_ci(r, stats.max_drawdown_value, n_boot=300, seed=7)
    assert 0 < dlo < dhi


# --- Deflated Sharpe ---------------------------------------------------------
def test_expected_max_sharpe_known_value():
    # N=10, V=0.01: 0.1 * ((1-γ)Φ⁻¹(0,9) + γΦ⁻¹(1-1/(10e))) ≈ 0.1577 (elle hesap)
    assert stats.expected_max_sharpe(10, 0.01) == pytest.approx(0.1577, abs=2e-3)
    assert stats.expected_max_sharpe(1, 0.01) == 0.0


def test_probabilistic_sharpe_matches_closed_form_for_normal_returns():
    sr, n = 0.1, 250
    expected = norm.cdf(sr * math.sqrt(n - 1) / math.sqrt(1 + 0.5 * sr**2))
    assert stats.probabilistic_sharpe(sr, 0.0, n, 0.0, 3.0) == pytest.approx(expected)


def test_deflated_sharpe_with_single_trial_equals_psr_against_zero():
    rng = np.random.default_rng(3)
    r = _daily(rng.normal(0.0008, 0.01, 600))
    arr = r.to_numpy()
    sr = arr.mean() / arr.std(ddof=1)
    z = (arr - arr.mean()) / arr.std()
    psr = stats.probabilistic_sharpe(sr, 0.0, len(arr), float(np.mean(z**3)), float(np.mean(z**4)))
    assert stats.deflated_sharpe(r, n_trials=1) == pytest.approx(psr)


def test_deflated_sharpe_decreases_as_trial_count_grows():
    rng = np.random.default_rng(4)
    r = _daily(rng.normal(0.0008, 0.01, 600))
    values = [stats.deflated_sharpe(r, n_trials=n, trial_sharpes=rng.normal(0, 0.05, 50)) for n in (1, 10, 100, 1000)]
    assert values == sorted(values, reverse=True)
    assert values[0] > values[-1]


# --- PBO (CSCV) ---------------------------------------------------------------
def _pbo_bruteforce(m: np.ndarray, s: int) -> float:
    t, n = m.shape
    blocks = np.array_split(m[t - (t // s) * s :], s)
    logits = []
    for is_b in itertools.combinations(range(s), s // 2):
        oos_b = [b for b in range(s) if b not in is_b]
        is_data = np.vstack([blocks[b] for b in is_b])
        oos_data = np.vstack([blocks[b] for b in oos_b])

        def sr(d):
            return d.mean(axis=0) / d.std(axis=0)

        best = int(np.argmax(sr(is_data)))
        so = sr(oos_data)
        rank = int((so <= so[best]).sum())
        omega = rank / (n + 1)
        logits.append(math.log(omega / (1 - omega)))
    return float(np.mean(np.array(logits) <= 0))


def test_pbo_matches_bruteforce_on_small_example():
    rng = np.random.default_rng(5)
    m = rng.normal(0, 0.01, size=(400, 4))
    assert stats.pbo_cscv(m, n_blocks=8) == pytest.approx(_pbo_bruteforce(m, 8))


def test_pbo_is_zero_when_one_variant_dominates_everywhere():
    rng = np.random.default_rng(6)
    m = rng.normal(0, 0.01, size=(800, 6))
    m[:, 2] += 0.02  # her blokta açık ara en iyi
    assert stats.pbo_cscv(m, n_blocks=8) == 0.0


def test_pbo_is_near_half_for_pure_noise_variants():
    # CSCV birleşimleri birbirine bağlı: tek örneklemde varyans yüksek, birkaç tohumun ortalaması ~0,5 olmalı.
    values = [stats.pbo_cscv(np.random.default_rng(seed).normal(0, 0.01, size=(800, 20)), n_blocks=16) for seed in range(8)]
    assert 0.38 <= float(np.mean(values)) <= 0.62


# --- purge + embargo bölücü -------------------------------------------------------
def test_purged_splits_leave_embargo_gap_and_keep_chronology():
    dates = pd.date_range("2020-01-01", periods=1500, freq="D")
    splits = stats.purged_walk_forward_splits(dates, n_splits=5, embargo_days=10, purge_days=5)
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert dates[train_idx].max() < dates[test_idx].min()
        gap_days = (dates[test_idx].min() - dates[train_idx].max()).days
        assert gap_days >= 15  # purge + embargo
        assert len(set(train_idx) & set(test_idx)) == 0
    # test pencereleri çakışmaz ve ileri kayar
    starts = [dates[t].min() for _, t in splits]
    assert starts == sorted(starts)


# --- kabul eşikleri ----------------------------------------------------------------
def test_check_acceptance_applies_preregistered_thresholds():
    good = dict(
        sharpe_net=1.1, deflated=0.97, pbo=0.2, max_dd=-0.25, calmar_value=0.9, btc_calmar=0.6,
        positive_year_fraction=0.7, stress_sharpe=0.6, plateau_ratio=0.8, is_portfolio=True,
    )
    assert all(stats.check_acceptance(**good).values())
    assert not stats.check_acceptance(**{**good, "sharpe_net": 0.9})["net_sharpe"]  # portföy eşiği 1,0
    assert stats.check_acceptance(**{**good, "sharpe_net": 0.9, "is_portfolio": False})["net_sharpe"]  # kol eşiği 0,8
    assert not stats.check_acceptance(**{**good, "calmar_value": 0.8, "btc_calmar": 0.85})["calmar"]
    assert not stats.check_acceptance(**{**good, "max_dd": -0.31})["max_drawdown"]
    assert not stats.check_acceptance(**{**good, "deflated": float("nan")})["deflated_sharpe"]
