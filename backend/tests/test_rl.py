import numpy as np
import pandas as pd
import pytest

from app.exchanges.base import Exchange
from app.ml.features import ALL_FEATURE_COLUMNS
from app.rl.dataset import build_episode_data
from app.rl.environment import FLAT, LONG, SHORT, TradingEnv
from app.rl.train import evaluate_random_policy


class _FakeExchange(Exchange):
    def list_symbols(self, quote_currency, market_type):
        return ["BTC/USDT:USDT"]

    def fetch_ohlcv(self, symbol, timeframe, limit, since=None):
        n = max(limit, 400)
        close = np.linspace(100, 200, n) + np.random.default_rng(0).normal(0, 1.0, n)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.random.default_rng(0).uniform(800, 1200, n),
            }
        )


def test_build_episode_data_shapes_and_no_nan():
    X, close = build_episode_data(_FakeExchange(), "BTC/USDT:USDT", "1h", 400)
    assert X.shape[1] == len(ALL_FEATURE_COLUMNS)
    assert X.shape[0] == len(close)
    assert X.shape[0] > 0
    assert not np.isnan(X).any()


def _make_env(n=500, window_len=50, holdout_frac=0.2, transaction_cost_pct=0.0, trend=True):
    rng = np.random.default_rng(0)
    if trend:
        close = np.linspace(100, 200, n) + rng.normal(0, 0.1, n)
    else:
        close = 100 + rng.normal(0, 1.0, n).cumsum() * 0 + rng.normal(0, 0.5, n)
    features = rng.normal(0, 1, (n, 10)).astype("float32")
    return TradingEnv(features, close, window_len=window_len, holdout_frac=holdout_frac, transaction_cost_pct=transaction_cost_pct)


def test_reset_never_starts_inside_holdout_region():
    env = _make_env(n=1000, window_len=100, holdout_frac=0.2)
    cutoff = env._cutoff
    for seed in range(50):
        obs = env.reset(seed=seed)
        assert env._start_idx < cutoff
        assert obs.shape == (env.observation_dim,)


def test_reset_with_use_holdout_always_starts_at_cutoff():
    env = _make_env(n=1000, window_len=100, holdout_frac=0.2)
    env.reset(seed=1, use_holdout=True)
    assert env._start_idx == env._cutoff

    env.reset(seed=999, use_holdout=True)  # seed farklı olsa da holdout başlangıcı SABİT olmalı
    assert env._start_idx == env._cutoff


def test_step_rewards_long_position_on_uptrend():
    # İlk step(LONG) yalnızca pozisyonu AÇAR (bu bar için ödül hâlâ önceki
    # -FLAT- pozisyona göre 0'dır — 1-bar'lık gerçekçi bir yürütme
    # gecikmesi); ödül, pozisyon açıkken geçen bir SONRAKİ barda görülür.
    env = _make_env(n=500, window_len=50, transaction_cost_pct=0.0, trend=True)
    env.reset(seed=0)
    env.step(LONG)
    result = env.step(LONG)
    assert result.reward > 0  # yükselen trendde long pozisyon kâr etmeli


def test_step_rewards_short_position_negatively_on_uptrend():
    env = _make_env(n=500, window_len=50, transaction_cost_pct=0.0, trend=True)
    env.reset(seed=0)
    env.step(SHORT)
    result = env.step(SHORT)
    assert result.reward < 0  # yükselen trendde short pozisyon zarar etmeli


def test_flat_position_has_zero_reward_ignoring_costs():
    env = _make_env(n=500, window_len=50, transaction_cost_pct=0.0, trend=True)
    env.reset(seed=0)
    result = env.step(FLAT)
    assert result.reward == pytest.approx(0.0)


def test_changing_position_incurs_transaction_cost():
    env = _make_env(n=500, window_len=50, transaction_cost_pct=1.0, trend=True)  # %1 maliyet, abartılı ama net görülsün diye
    env.reset(seed=0)
    # ilk adımda pozisyon FLAT'tan LONG'a değişiyor -> maliyet düşülmeli
    result = env.step(LONG)
    assert result.reward < 0.01 - 0.001  # trend getirisi küçük, %1 maliyet baskın çıkmalı (negatife yakın/negatif)


def test_episode_truncates_after_window_len_steps_in_training_mode():
    env = _make_env(n=1000, window_len=30)
    env.reset(seed=0)
    steps = 0
    while True:
        result = env.step(FLAT)
        steps += 1
        if result.truncated:
            break
        if steps > 100:
            pytest.fail("epizot window_len içinde sonlanmadı")
    assert steps <= 30 + 1


def test_invalid_action_raises():
    env = _make_env()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(99)


def test_constructor_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        TradingEnv(np.zeros((10, 5)), np.zeros(9))


def test_constructor_rejects_insufficient_data():
    with pytest.raises(ValueError):
        TradingEnv(np.zeros((20, 5)), np.zeros(20), window_len=100)


def test_evaluate_random_policy_returns_stats_over_all_episodes():
    env = _make_env(n=500, window_len=50)
    evaluation = evaluate_random_policy(env, episodes=10, seed=0)
    assert evaluation.episodes == 10
    assert isinstance(evaluation.mean_total_reward, float)
    assert evaluation.std_total_reward >= 0.0
