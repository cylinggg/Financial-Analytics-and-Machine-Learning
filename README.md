# Deep Reinforcement Learning for Automated Stock Trading: Replication

Replication of Yang et al. (2020) — an ensemble RL strategy for automated stock trading using PPO, A2C, and DDPG.

## Overview

The paper casts portfolio management as a Markov Decision Process (MDP) and trains three actor-critic agents concurrently. Every quarter, the agent with the highest Sharpe ratio on the previous quarter is deployed. This replication reproduces the environment, data pipeline, and ensemble logic with a key simplification: agents are trained once (not retrained each quarter) due to computational constraints.

## Data

- 29 DJIA constituent stocks (UTX excluded — delisted), 2009-01-01 to 2020-05-08
- Downloaded via `yfinance`; technical indicators (MACD, RSI, CCI, ADX) computed using `ta`
- Turbulence index: rolling 252-day Mahalanobis distance (threshold = 140)

## Implementation

- **Environment:** custom `gymnasium` env, 175-dim state space, action space [−1,1]^29 (scaled by H_max = 100), 0.1% transaction cost
- **Training:** PPO & A2C — 150k steps; DDPG — 80k steps; MLP 256×256, lr = 1e-4
- **Ensemble:** quarterly selection based on previous-quarter Sharpe ratio

## Results

| Strategy        | Cum. Return | Sharpe | Max Drawdown |
|-----------------|-------------|--------|--------------|
| Ensemble        | 16.5%       | 0.38   | −21.7%       |
| PPO             | 24.5%       | 0.55   | −18.3%       |
| A2C             | 23.7%       | 0.47   | −21.9%       |
| DDPG            | 29.5%       | 0.57   | −20.1%       |
| DJIA            | 39.2%       | 0.51   | −37.1%       |
| Min-Variance    | 69.4%       | 0.90   | −25.3%       |

The DJIA benchmark matches the paper almost exactly (Sharpe 0.51 vs. 0.47), confirming correct data and metric implementation. Performance gaps in RL strategies reflect the absence of quarterly retraining — a methodological necessity the original paper does not sufficiently emphasise.

## Files

- `trading_replication.py` — full implementation (environment, training, ensemble, evaluation)
- `outputs/` — cumulative return plots, performance tables, trained model files
- `rl.pdf` — original paper (Yang et al., 2020)

## Reference

Yang, H., Liu, X.-Y., Zhong, S., & Walid, A. (2020). Deep reinforcement learning for automated stock trading: An ensemble strategy. *ICAIF '20*. https://doi.org/10.1145/3383455.3422540
