from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core.config import settings
from app.exchanges import get_exchange
from app.rl.dataset import build_episode_data
from app.rl.environment import TradingEnv
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
