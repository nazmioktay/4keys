"""RL eğitimi için hazırlık: gerçek bir ajan (PPO/DQN vb., `stable-baselines3`
gerektirir — bu bağımlılık BİLİNÇLİ OLARAK henüz eklenmedi, bkz. README
"Faz C — RL" notu) eğitilmeden ÖNCE, ortamın/veri pipeline'ının doğru
çalıştığını doğrulamak ve gelecekteki gerçek ajan için bir "şans" referans
noktası (rastgele politika) oluşturmak için kullanılır.
"""

from dataclasses import dataclass

import numpy as np

from .environment import ACTIONS, TradingEnv


@dataclass
class PolicyEvaluation:
    episodes: int
    mean_total_reward: float
    median_total_reward: float
    std_total_reward: float


def evaluate_random_policy(env: TradingEnv, episodes: int = 20, use_holdout: bool = False, seed: int = 0) -> PolicyEvaluation:
    """Tamamen rastgele aksiyonlarla `episodes` epizod çalıştırır.

    Bu bir "ajan" değildir — gelecekte eğitilecek gerçek bir RL ajanının
    (PPO/DQN) aşması gereken ŞANS SEVİYESİ referans noktasıdır (tıpkı
    XGBoost/LSTM'de "%33 rastgele tahmin" referansı gibi). `use_holdout=True`
    ile holdout diliminde de çalıştırılabilir — ama holdout'ta rastgele
    politikanın "iyi" çıkması hiçbir şey ifade etmez (referans amaçlıdır).
    """
    rng = np.random.default_rng(seed)
    totals: list[float] = []
    for _ in range(episodes):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)), use_holdout=use_holdout)
        total_reward = 0.0
        while True:
            action = int(rng.choice(ACTIONS))
            result = env.step(action, use_holdout=use_holdout)
            total_reward += result.reward
            if result.terminated or result.truncated:
                break
        totals.append(total_reward)

    arr = np.array(totals)
    return PolicyEvaluation(
        episodes=episodes,
        mean_total_reward=float(arr.mean()),
        median_total_reward=float(np.median(arr)),
        std_total_reward=float(arr.std()),
    )
