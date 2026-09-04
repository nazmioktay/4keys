from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core.config import settings
from app.exchanges import get_exchange
from app.rl.dataset import build_episode_data
from app.rl.environment import TradingEnv
from app.rl.execution_timing import analyze_hurst_execution_timing
from app.rl.train import evaluate_random_policy

router = APIRouter(prefix="/rl", tags=["rl"])


class RandomBaselineResponse(BaseModel):
    symbol: str
    rows_used: int
    observation_dim: int
    episodes: int
    mean_total_reward: float
    median_total_reward: float
    std_total_reward: float


@router.get("/random-baseline", response_model=RandomBaselineResponse)
def random_baseline(
    symbol: str = Query(..., description="Örn: BTC/USDT:USDT"),
    episodes: int = Query(20, ge=1, le=200),
    use_holdout: bool = Query(False, description="True ise holdout diliminde çalıştırır (yalnızca referans, ajan değil)"),
) -> RandomBaselineResponse:
    """RL hazırlığı (Faz C): ortamın/veri pipeline'ının doğru çalıştığını
    doğrular ve rastgele bir politikanın (gerçek ajan değil) "şans"
    seviyesini ölçer — gelecekte eğitilecek gerçek bir ajanın (PPO/DQN,
    `stable-baselines3` gerektirir, henüz eklenmedi) aşması gereken
    referans noktasıdır."""
    exchange = get_exchange(settings.exchange_id)
    try:
        features, close = build_episode_data(exchange, symbol, settings.ml_train_timeframe, settings.ml_train_lookback)
        env = TradingEnv(features, close)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    evaluation = evaluate_random_policy(env, episodes=episodes, use_holdout=use_holdout)
    return RandomBaselineResponse(
        symbol=symbol,
        rows_used=len(features),
        observation_dim=env.observation_dim,
        episodes=evaluation.episodes,
        mean_total_reward=evaluation.mean_total_reward,
        median_total_reward=evaluation.median_total_reward,
        std_total_reward=evaluation.std_total_reward,
    )


class HurstBucketPoint(BaseModel):
    label: str
    hurst_range: str
    samples: int
    mean_delayed_slippage_pct: float


class HurstExecutionTimingResponse(BaseModel):
    symbol: str
    delay_bars: int
    total_samples: int
    buckets: list[HurstBucketPoint]


@router.get("/hurst-execution-timing", response_model=HurstExecutionTimingResponse)
def hurst_execution_timing(
    symbol: str = Query(..., description="Örn: BTC/USDT:USDT"),
    delay_bars: int = Query(3, ge=1, le=20, description="Sabit gecikme (mum sayısı) — look-ahead yanlılığından kaçınmak için veriye göre optimize edilmez"),
) -> HurstExecutionTimingResponse:
    """"Optimal execution" (emri parçalara bölerek piyasa etkisini azaltma)
    bizim verimizle (order-book derinliği yok) ve ölçeğimizle (çeyrek
    Kelly boyutları, BTC/USDT likiditesine göre ihmal edilebilir) anlamlı
    bir sonuç veremez — piyasa etkisi modeli olmadan "bölmek daha iyi"
    sonucu yapay olurdu.

    Bunun yerine test edilen hipotez: sinyal anındaki Hurst üsteli (H)
    DÜŞÜKSE (ortalamaya-dönüş rejimi), BUY için hemen yürütmek yerine
    `delay_bars` mum GECİKTİRMEK ortalama olarak daha iyi (daha ucuz) bir
    fiyat verir mi? Bkz. `app.rl.execution_timing` docstring'i — metodoloji
    ve look-ahead yanlılığından nasıl kaçınıldığı için."""
    exchange = get_exchange(settings.exchange_id)
    report = analyze_hurst_execution_timing(
        exchange, symbol, settings.ml_train_timeframe, settings.ml_train_lookback, delay_bars=delay_bars
    )
    if report.total_samples == 0:
        raise HTTPException(status_code=422, detail="Yeterli veri yok.")
    return HurstExecutionTimingResponse(
        symbol=report.symbol,
        delay_bars=report.delay_bars,
        total_samples=report.total_samples,
        buckets=[HurstBucketPoint(**b.__dict__) for b in report.buckets],
    )
