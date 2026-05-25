#!/usr/bin/env python3
"""
Empirical Replication:
  Deep Reinforcement Learning for Automated Stock Trading – An Ensemble Strategy
  Yang et al., ICAIF 2020

Simplifications vs. the paper:
  - Agents trained ONCE on 2009-2015; the paper retrains every quarter.
  - 100-150 k timesteps per agent (paper uses longer training).
  - Technical indicators use the `ta` library (identical formulae).
  - Turbulence uses a fixed 252-day rolling window Mahalanobis distance.
"""

import os, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.noise import NormalActionNoise

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── 1. CONFIGURATION ────────────────────────────────────────────────────────
TICKERS = [
    "MMM", "AXP", "AAPL", "BA",  "CAT",  "CVX",  "CSCO", "KO",  "XOM",
    "GE",  "GS",  "HD",   "IBM", "INTC", "JNJ",  "JPM",  "MCD", "MRK",
    "MSFT","NKE", "PFE",  "PG",  "TRV",  "UNH",  "VZ",   "V",   "WMT",
    "DIS", "DD",
]

TRAIN_START  = "2009-01-01";  TRAIN_END   = "2015-09-30"
VAL_START    = "2015-10-01";  VAL_END     = "2015-12-31"
TRADE_START  = "2016-01-01";  TRADE_END   = "2020-05-08"

INITIAL_BALANCE      = 1_000_000
TRANSACTION_COST_PCT = 0.001        # 0.1 %
HMAX                 = 100          # max shares per single stock action
TURBULENCE_THRESHOLD = 140

TRAIN_STEPS = {"PPO": 500_000, "A2C": 500_000, "DDPG": 200_000}
OUTPUT_DIR  = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── 2. DATA DOWNLOAD & PREPROCESSING ───────────────────────────────────────
def download_data():
    print("Downloading DJIA stock data from Yahoo Finance …")
    raw = yf.download(
        TICKERS, start="2008-01-01", end="2026-05-21",
        auto_adjust=True, progress=False, group_by="column",
        threads=True,
    )
    close = raw["Close"].ffill().dropna(axis=1, thresh=500)
    high  = raw["High"].reindex(columns=close.columns).ffill()
    low   = raw["Low"].reindex(columns=close.columns).ffill()
    tickers = list(close.columns)
    print(f"  → {len(tickers)} stocks retained: {tickers}")
    return close, high, low, tickers


def compute_indicators(close, high, low):
    """MACD, RSI, CCI, ADX as defined in Yang et al. (2020)."""
    from ta.trend    import MACD as MACDInd, CCIIndicator, ADXIndicator
    from ta.momentum import RSIIndicator

    macd_df = pd.DataFrame(index=close.index)
    rsi_df  = pd.DataFrame(index=close.index)
    cci_df  = pd.DataFrame(index=close.index)
    adx_df  = pd.DataFrame(index=close.index)

    for col in close.columns:
        c, h, l = close[col], high[col], low[col]
        macd_df[col] = MACDInd(c, window_slow=26, window_fast=12, window_sign=9).macd()
        rsi_df[col]  = RSIIndicator(c, window=14).rsi()
        cci_df[col]  = CCIIndicator(h, l, c, window=20).cci()
        adx_df[col]  = ADXIndicator(h, l, c, window=14).adx()

    macd_df.fillna(0,  inplace=True)
    rsi_df.fillna(50,  inplace=True)
    cci_df.fillna(0,   inplace=True)
    adx_df.fillna(25,  inplace=True)
    return macd_df, rsi_df, cci_df, adx_df


def compute_turbulence(close, window=252):
    """Mahalanobis-distance turbulence index (Kritzman & Li 2010)."""
    returns = close.pct_change().fillna(0)
    n = len(returns)
    turb = np.zeros(n)
    for i in range(window, n):
        hist = returns.iloc[i - window : i]
        curr = returns.iloc[i].values
        mu   = hist.mean().values
        try:
            cov_inv = np.linalg.pinv(hist.cov().values)
            diff    = curr - mu
            turb[i] = float(diff @ cov_inv @ diff)
        except Exception:
            turb[i] = 0.0
    return pd.Series(np.clip(turb, 0, None), index=returns.index)


def slice_data(df, start, end):
    return df.loc[(df.index >= start) & (df.index <= end)]


# ─── 3. TRADING ENVIRONMENT ──────────────────────────────────────────────────
class StockTradingEnv(gym.Env):
    """
    Multi-stock trading environment (Yang et al. 2020).
    State  : [balance, prices×D, holdings×D, MACD×D, RSI×D, CCI×D, ADX×D]
    Action : continuous vector ∈ [−1, 1]^D  (scaled to share counts)
    Reward : daily change in portfolio value net of transaction costs
    """
    metadata = {"render_modes": []}

    def __init__(self, close, macd, rsi, cci, adx, turbulence,
                 initial_balance=INITIAL_BALANCE,
                 hmax=HMAX,
                 transaction_cost=TRANSACTION_COST_PCT,
                 turbulence_threshold=TURBULENCE_THRESHOLD):
        super().__init__()
        self.close     = close.values.astype(np.float32)
        self.macd      = macd.values.astype(np.float32)
        self.rsi       = rsi.values.astype(np.float32)
        self.cci       = cci.values.astype(np.float32)
        self.adx       = adx.values.astype(np.float32)
        self.turb      = turbulence.values.astype(np.float32)
        self.dates     = close.index

        self.D         = close.shape[1]
        self.init_bal  = initial_balance
        self.hmax      = hmax
        self.tc        = transaction_cost
        self.turb_thr  = turbulence_threshold

        obs_dim = 1 + 6 * self.D          # balance + 6 feature vectors
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.D,), dtype=np.float32)

        self.reset()

    # ── internal helpers ──────────────────────────────────────────────────
    def _get_obs(self):
        p = self.close[self.day]
        obs = np.concatenate([
            [self.balance / self.init_bal],
            p / 1_000.0,
            self.shares / self.hmax,
            self.macd[self.day] / 100.0,
            self.rsi[self.day]  / 100.0,
            self.cci[self.day]  / 100.0,
            self.adx[self.day]  / 100.0,
        ]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _portfolio_value(self, day=None):
        d = self.day if day is None else day
        return float(self.balance + np.dot(self.shares, self.close[d]))

    # ── gym API ───────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day     = 0
        self.balance = float(self.init_bal)
        self.shares  = np.zeros(self.D, dtype=np.float32)
        self._pv_hist = [self._portfolio_value()]
        return self._get_obs(), {}

    def step(self, action):
        prices = self.close[self.day]
        turb   = self.turb[self.day]

        acts = (np.array(action, dtype=np.float32) * self.hmax).astype(int)

        # Turbulence guard: sell everything
        if turb > self.turb_thr:
            acts = -self.shares.astype(int)

        # Execute sells first
        for i in np.where(acts < 0)[0]:
            sell_sh = min(-acts[i], int(self.shares[i]))
            if sell_sh > 0:
                self.balance   += sell_sh * prices[i] * (1 - self.tc)
                self.shares[i] -= sell_sh

        # Then buys
        for i in np.where(acts > 0)[0]:
            cost_per = prices[i] * (1 + self.tc)
            max_buy  = int(self.balance / (cost_per + 1e-8))
            buy_sh   = min(acts[i], max_buy)
            if buy_sh > 0:
                self.balance   -= buy_sh * cost_per
                self.shares[i] += buy_sh

        self.day += 1
        pv_new   = self._portfolio_value()
        pv_prev  = self._pv_hist[-1]
        reward   = (pv_new - pv_prev) / self.init_bal

        self._pv_hist.append(pv_new)
        terminated = (self.day == len(self.close) - 1)
        return self._get_obs(), reward, terminated, False, {}

    def portfolio_history(self):
        return np.array(self._pv_hist)


# ─── 4. TRAINING ─────────────────────────────────────────────────────────────
def make_env(close, macd, rsi, cci, adx, turb, **kwargs):
    return StockTradingEnv(close, macd, rsi, cci, adx, turb, **kwargs)


def train_agent(name, env, steps, seed=42):
    print(f"  Training {name} for {steps:,} timesteps …")
    if name == "PPO":
        m = PPO("MlpPolicy", env, seed=seed, verbose=0,
                learning_rate=1e-4, n_steps=2048, batch_size=64,
                policy_kwargs=dict(net_arch=[256, 256]))
    elif name == "A2C":
        m = A2C("MlpPolicy", env, seed=seed, verbose=0,
                learning_rate=1e-4, n_steps=5,
                policy_kwargs=dict(net_arch=[256, 256]))
    elif name == "DDPG":
        n_act = env.action_space.shape[-1]
        noise = NormalActionNoise(np.zeros(n_act), 0.1 * np.ones(n_act))
        m = DDPG("MlpPolicy", env, seed=seed, verbose=0,
                 learning_rate=1e-4, batch_size=64,
                 action_noise=noise, learning_starts=1_000,
                 policy_kwargs=dict(net_arch=[256, 256]))
    m.learn(total_timesteps=steps)
    return m


# ─── 5. BACKTESTING ──────────────────────────────────────────────────────────
def backtest(model, env):
    """Run one full episode; return portfolio value time-series."""
    obs, _ = env.reset()
    done   = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(act)
        done = terminated or truncated
    return env.portfolio_history()          # shape: (T,)


# ─── 6. PERFORMANCE METRICS ──────────────────────────────────────────────────
def _daily_returns(vals):
    v = np.array(vals, dtype=float)
    r = np.zeros(len(v))
    r[1:] = np.diff(v) / np.maximum(v[:-1], 1e-8)
    return r


def sharpe(vals, rf=0.0, ann=252):
    r = _daily_returns(vals)[1:]
    if len(r) < 2 or r.std() < 1e-10:
        return 0.0
    ann_ret = (1 + r.mean()) ** ann - 1
    ann_vol = r.std() * ann ** 0.5
    return (ann_ret - rf) / ann_vol


def performance_table(results_dict):
    rows = []
    for name, vals in results_dict.items():
        v = np.array(vals, dtype=float)
        r = _daily_returns(v)[1:]
        cum_ret  = (v[-1] - v[0]) / v[0]
        ann_ret  = (1 + r.mean()) ** 252 - 1
        ann_vol  = r.std() * 252 ** 0.5
        sr       = sharpe(v)
        peak     = np.maximum.accumulate(v)
        max_dd   = ((v - peak) / np.maximum(peak, 1e-8)).min()
        rows.append({
            "Strategy"        : name,
            "Cumulative Return": f"{cum_ret:.1%}",
            "Annual Return"   : f"{ann_ret:.1%}",
            "Annual Volatility": f"{ann_vol:.1%}",
            "Sharpe Ratio"    : f"{sr:.2f}",
            "Max Drawdown"    : f"{max_dd:.1%}",
        })
    df = pd.DataFrame(rows).set_index("Strategy")
    print("\n" + "=" * 72)
    print("Performance Evaluation (2016-01-04 to 2020-05-08)")
    print("=" * 72)
    print(df.to_string())
    print("=" * 72)
    return df


# ─── 7. ENSEMBLE STRATEGY ────────────────────────────────────────────────────
def build_ensemble(individual_results, trade_dates):
    """
    Quarterly rolling model selection (Yang et al. §5.4):
      - Evaluate each agent's Sharpe on the *previous* quarter.
      - Deploy the best agent for the *current* quarter.
    The daily percentage returns from each agent's full backtest are
    re-applied to the running ensemble portfolio value, preserving
    realistic compounding.
    """
    names  = list(individual_results.keys())
    n      = len(trade_dates)

    # Pre-compute daily returns for each agent
    agent_rets = {}
    for nm in names:
        v = np.array(individual_results[nm], dtype=float)[:n]
        r = _daily_returns(v)
        agent_rets[nm] = r

    # Quarter label for each trading day
    q_labels  = pd.PeriodIndex(trade_dates, freq="Q")
    unique_qs = q_labels.unique().sort_values()

    portfolio  = np.zeros(n, dtype=float)
    portfolio[0] = INITIAL_BALANCE
    sel_model  = names[0]           # default first quarter
    sel_log    = []

    for qi, q in enumerate(unique_qs):
        q_mask = (q_labels == q)
        q_idx  = np.where(q_mask)[0]

        if qi > 0:
            prev_q    = unique_qs[qi - 1]
            prev_mask = (q_labels == prev_q)
            prev_idx  = np.where(prev_mask)[0]

            best_sr = -np.inf
            for nm in names:
                prev_r = agent_rets[nm][prev_idx]
                sr_val = sharpe(
                    INITIAL_BALANCE * np.cumprod(1 + prev_r)
                )
                if sr_val > best_sr:
                    best_sr   = sr_val
                    sel_model = nm

        sel_log.append({"Quarter": str(q), "Selected Model": sel_model})

        # Apply selected model's returns to ensemble portfolio
        for i in q_idx:
            if i == 0:
                continue
            portfolio[i] = portfolio[i - 1] * (1 + agent_rets[sel_model][i])

    sel_df = pd.DataFrame(sel_log)
    print("\nQuarterly model selection:")
    print(sel_df.to_string(index=False))
    return portfolio, sel_df


# ─── 8. BASELINE STRATEGIES ──────────────────────────────────────────────────
def djia_index(trade_dates):
    """Download actual DJIA index (^DJI) and scale to initial portfolio value."""
    start = str(trade_dates[0].date())
    end   = str(trade_dates[-1].date())
    raw   = yf.download("^DJI", start=start, end=end,
                         auto_adjust=True, progress=False)
    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    prices = raw["Close"].copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    trade_idx = pd.to_datetime(trade_dates).tz_localize(None)
    prices = prices.reindex(trade_idx).ffill().bfill()
    return (prices / prices.iloc[0] * INITIAL_BALANCE).values


def min_variance(close_train, close_trade):
    """Analytical minimum-variance long-only portfolio."""
    ret_tr = close_train.pct_change().dropna()
    cov    = ret_tr.cov().values
    n      = cov.shape[0]
    try:
        cov_inv = np.linalg.pinv(cov)
        ones    = np.ones(n)
        w       = cov_inv @ ones / (ones @ cov_inv @ ones)
        w       = np.clip(w, 0, None)
        w      /= w.sum()
    except Exception:
        w = np.ones(n) / n

    prices = close_trade.values.astype(float)
    shares = INITIAL_BALANCE * w / prices[0]
    return (prices * shares).sum(axis=1)


# ─── 9. VISUALISATION ────────────────────────────────────────────────────────
STYLE = {
    "Ensemble": ("red",    "-",  2.5),
    "PPO":      ("blue",   "--", 1.5),
    "A2C":      ("green",  "--", 1.5),
    "DDPG":     ("purple", "--", 1.5),
    "DJIA":     ("orange", "-",  1.5),
    "Min-Var":  ("cyan",   "-",  1.5),
}


def plot_cumulative_returns(trade_dates, results, fname):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, vals in results.items():
        c, ls, lw = STYLE.get(name, ("gray", ":", 1.0))
        cum = (np.array(vals) - vals[0]) / vals[0]
        ax.plot(trade_dates[: len(cum)], cum,
                label=name, color=c, linestyle=ls, linewidth=lw)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return", fontsize=12)
    ax.set_title(
        "Cumulative Return with Transaction Costs\n"
        "(Initial Portfolio: $1,000,000 | 2016-01-04 to 2020-05-08)",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_performance_table(df_perf, fname):
    """Save performance table as a figure for easy inclusion in report."""
    fig, ax = plt.subplots(figsize=(11, 2.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=df_perf.values,
        rowLabels=df_perf.index,
        colLabels=df_perf.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    ax.set_title(
        "Table: Performance Comparison (2016/01/04 – 2020/05/08)",
        fontsize=11, pad=12,
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_selection_table(sel_df, fname):
    fig, ax = plt.subplots(figsize=(6, max(3, len(sel_df) * 0.35)))
    ax.axis("off")
    tbl = ax.table(
        cellText=sel_df.values,
        colLabels=sel_df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)
    ax.set_title("Quarterly Model Selection (Ensemble)", fontsize=11, pad=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── 10. ROLLING RETRAIN ENSEMBLE ───────────────────────────────────────────
RETRAIN_STEPS = {"PPO": 100_000, "A2C": 100_000, "DDPG": 50_000}


def rolling_retrain_ensemble(close, macd, rsi, cci, adx, turb,
                              val_start, val_end, trade_start, trade_end):
    """
    Proper quarterly rolling-retrain ensemble (Yang et al. §5.4):
      For each trading quarter Q:
        1. Train on growing window  2009-01-01 → Q_start
        2. Validate on previous quarter to pick best Sharpe agent
        3. Deploy best agent for Q
    Returns portfolio value array aligned to trade_dates.
    """
    # All trading dates — pre-filter all data to trade period so q_mask aligns
    all_trade_mask = (close.index >= trade_start) & (close.index <= trade_end)
    trade_dates    = pd.DatetimeIndex(close.index[all_trade_mask])
    n              = len(trade_dates)

    close_td = close[all_trade_mask]
    macd_td  = macd[all_trade_mask]
    rsi_td   = rsi[all_trade_mask]
    cci_td   = cci[all_trade_mask]
    adx_td   = adx[all_trade_mask]
    turb_td  = turb[all_trade_mask]

    q_labels  = pd.PeriodIndex(trade_dates, freq="Q")
    unique_qs = q_labels.unique().sort_values()

    portfolio  = np.zeros(n, dtype=float)
    portfolio[0] = INITIAL_BALANCE
    sel_log    = []

    # Initial validation period (paper: 2015-10-01 to 2015-12-31)
    cur_val_start = val_start
    cur_val_end   = val_end

    print(f"\nRolling retraining: {len(unique_qs)} quarters "
          f"({unique_qs[0]} to {unique_qs[-1]})")

    for qi, q in enumerate(unique_qs):
        q_mask = (q_labels == q)
        q_idx  = np.where(q_mask)[0]

        q_start_date = trade_dates[q_idx[0]]
        q_end_date   = trade_dates[q_idx[-1]]

        # ── Train on growing window up to start of validation ────────────
        train_end_dt = cur_val_start
        c_tr = slice_data(close, TRAIN_START, str(train_end_dt.date()))
        m_tr = slice_data(macd,  TRAIN_START, str(train_end_dt.date()))
        r_tr = slice_data(rsi,   TRAIN_START, str(train_end_dt.date()))
        cc_tr= slice_data(cci,   TRAIN_START, str(train_end_dt.date()))
        a_tr = slice_data(adx,   TRAIN_START, str(train_end_dt.date()))
        t_tr = slice_data(turb.to_frame("t"),
                          TRAIN_START, str(train_end_dt.date()))["t"]

        print(f"  Q{qi+1} ({q}): train 2009→{str(train_end_dt.date())[:7]} "
              f"({len(c_tr)} days) | val {str(cur_val_start.date())[:7]}"
              f"→{str(cur_val_end.date())[:7]}")

        train_env = make_env(c_tr, m_tr, r_tr, cc_tr, a_tr, t_tr)
        models_q  = {}
        for algo, steps in RETRAIN_STEPS.items():
            if algo == "DDPG":
                n_act = train_env.action_space.shape[-1]
                noise = NormalActionNoise(np.zeros(n_act), 0.1*np.ones(n_act))
                m = DDPG("MlpPolicy", train_env, verbose=0, seed=42,
                         learning_rate=1e-4, batch_size=64,
                         action_noise=noise, learning_starts=500,
                         policy_kwargs=dict(net_arch=[256, 256]))
            elif algo == "PPO":
                m = PPO("MlpPolicy", train_env, verbose=0, seed=42,
                        learning_rate=1e-4, n_steps=2048, batch_size=64,
                        policy_kwargs=dict(net_arch=[256, 256]))
            else:
                m = A2C("MlpPolicy", train_env, verbose=0, seed=42,
                        learning_rate=1e-4, n_steps=5,
                        policy_kwargs=dict(net_arch=[256, 256]))
            m.learn(total_timesteps=steps)
            models_q[algo] = m
            train_env.reset()

        # ── Validate on previous quarter ─────────────────────────────────
        c_v  = slice_data(close, str(cur_val_start.date()),
                          str(cur_val_end.date()))
        m_v  = slice_data(macd,  str(cur_val_start.date()),
                          str(cur_val_end.date()))
        r_v  = slice_data(rsi,   str(cur_val_start.date()),
                          str(cur_val_end.date()))
        cc_v = slice_data(cci,   str(cur_val_start.date()),
                          str(cur_val_end.date()))
        a_v  = slice_data(adx,   str(cur_val_start.date()),
                          str(cur_val_end.date()))
        t_v  = slice_data(turb.to_frame("t"),
                          str(cur_val_start.date()),
                          str(cur_val_end.date()))["t"]

        best_model, best_sr = "PPO", -np.inf
        if len(c_v) > 5:
            for algo, m in models_q.items():
                env_v = make_env(c_v, m_v, r_v, cc_v, a_v, t_v)
                v_vals = backtest(m, env_v)
                sr_v   = sharpe(v_vals)
                if sr_v > best_sr:
                    best_sr, best_model = sr_v, algo

        sel_log.append({"Quarter": str(q), "Selected": best_model,
                        "Val Sharpe": f"{best_sr:.3f}"})
        print(f"    → Selected {best_model} (val Sharpe={best_sr:.3f})")

        # ── Trade this quarter with best model ────────────────────────────
        c_q  = close_td[q_mask].ffill().fillna(0)
        m_q  = macd_td[q_mask].ffill().fillna(0)
        r_q  = rsi_td[q_mask].ffill().fillna(0)
        cc_q = cci_td[q_mask].ffill().fillna(0)
        a_q  = adx_td[q_mask].ffill().fillna(0)
        t_q  = turb_td[q_mask].ffill().fillna(0)

        # Carry forward the last known portfolio value as the starting balance
        init_bal = INITIAL_BALANCE if qi == 0 else float(portfolio[q_idx[0] - 1])
        env_q  = make_env(c_q, m_q, r_q, cc_q, a_q, t_q,
                          initial_balance=init_bal)
        q_vals = backtest(models_q[best_model], env_q)

        for step, i in enumerate(q_idx):
            portfolio[i] = q_vals[step]

        # ── Advance validation window for next quarter ────────────────────
        cur_val_start = q_start_date
        cur_val_end   = q_end_date

    sel_df = pd.DataFrame(sel_log)
    return portfolio, sel_df


# ─── 11. MAIN ────────────────────────────────────────────────────────────────
def main():
    # ── 10.1 Data ────────────────────────────────────────────────────────
    close, high, low, tickers = download_data()
    n_stocks = len(tickers)

    print("Computing technical indicators …")
    macd, rsi, cci, adx = compute_indicators(close, high, low)

    print("Computing turbulence index …")
    turb = compute_turbulence(close)

    # Align all series
    for df in (macd, rsi, cci, adx, turb):
        df.reindex(close.index)
    turb = turb.reindex(close.index).fillna(0)

    # ── 10.2 Split ───────────────────────────────────────────────────────
    def S(df, s, e):
        return slice_data(df if isinstance(df, pd.DataFrame) else df.to_frame(), s, e) \
               if isinstance(df, pd.Series) else slice_data(df, s, e)

    c_tr = slice_data(close, TRAIN_START, TRAIN_END)
    h_tr = slice_data(high,  TRAIN_START, TRAIN_END)
    l_tr = slice_data(low,   TRAIN_START, TRAIN_END)
    m_tr = slice_data(macd,  TRAIN_START, TRAIN_END)
    r_tr = slice_data(rsi,   TRAIN_START, TRAIN_END)
    cc_tr= slice_data(cci,   TRAIN_START, TRAIN_END)
    a_tr = slice_data(adx,   TRAIN_START, TRAIN_END)
    t_tr = slice_data(turb.to_frame("t"), TRAIN_START, TRAIN_END)["t"]

    c_td = slice_data(close, TRADE_START, TRADE_END)
    m_td = slice_data(macd,  TRADE_START, TRADE_END)
    r_td = slice_data(rsi,   TRADE_START, TRADE_END)
    cc_td= slice_data(cci,   TRADE_START, TRADE_END)
    a_td = slice_data(adx,   TRADE_START, TRADE_END)
    t_td = slice_data(turb.to_frame("t"), TRADE_START, TRADE_END)["t"]
    trade_dates = pd.DatetimeIndex(c_td.index)

    print(f"Train: {len(c_tr)} days | Trade: {len(c_td)} days | "
          f"Stocks: {n_stocks}")

    # ── 10.3 Train agents (or load if already saved) ─────────────────────
    train_env = make_env(c_tr, m_tr, r_tr, cc_tr, a_tr, t_tr)
    models    = {}
    algo_cls  = {"PPO": PPO, "A2C": A2C, "DDPG": DDPG}

    for algo, steps in TRAIN_STEPS.items():
        path = os.path.join(OUTPUT_DIR, f"model_{algo}.zip")
        if os.path.exists(path):
            print(f"  Loading saved {algo} from {path} …")
            models[algo] = algo_cls[algo].load(path, env=train_env)
        else:
            print(f"\nTraining agents on 2009-01-01 to 2015-09-30 …")
            models[algo] = train_agent(algo, train_env, steps)
            models[algo].save(os.path.join(OUTPUT_DIR, f"model_{algo}"))
            print(f"  {algo} saved.")
        train_env.reset()

    # ── 10.4 Individual backtests (full trading period) ──────────────────
    print("\nRunning individual backtests …")
    individual = {}
    for name, model in models.items():
        env_t = make_env(c_td, m_td, r_td, cc_td, a_td, t_td)
        vals  = backtest(model, env_t)
        individual[name] = vals[: len(c_td)]      # align length
        sr = sharpe(individual[name])
        print(f"  {name}: final=${individual[name][-1]:,.0f}  Sharpe={sr:.3f}")

    # ── 10.5 Ensemble ────────────────────────────────────────────────────
    print("\nBuilding ensemble portfolio …")
    ensemble_vals, sel_df = build_ensemble(individual, trade_dates)

    # ── 10.6 Baselines ───────────────────────────────────────────────────
    djia_vals   = djia_index(trade_dates)
    minvar_vals = min_variance(c_tr, c_td)

    # ── 10.7 Results ─────────────────────────────────────────────────────
    results = {
        "Ensemble": ensemble_vals,
        "PPO":      individual["PPO"],
        "A2C":      individual["A2C"],
        "DDPG":     individual["DDPG"],
        "DJIA":     djia_vals,
        "Min-Var":  minvar_vals,
    }

    df_perf = performance_table(results)
    df_perf.to_csv(os.path.join(OUTPUT_DIR, "performance_table.csv"))
    sel_df.to_csv(os.path.join(OUTPUT_DIR, "model_selection.csv"), index=False)

    # ── 10.8 Plots (primary: 2016-2020) ──────────────────────────────────
    print("\nGenerating plots ...")
    plot_cumulative_returns(trade_dates, results, "cumulative_returns.png")
    plot_performance_table(df_perf, "performance_table.png")
    plot_selection_table(sel_df, "model_selection.png")

    # ── 10.8b Rolling quarterly retrain (paper methodology) ─────────────
    print("\nRunning rolling quarterly retrain ensemble (~1-2 hours) ...")
    VAL_START = pd.Timestamp("2015-10-01")
    VAL_END   = pd.Timestamp("2015-12-31")
    rr_vals, rr_sel = rolling_retrain_ensemble(
        close, macd, rsi, cci, adx, turb,
        val_start=VAL_START, val_end=VAL_END,
        trade_start=TRADE_START, trade_end=TRADE_END,
    )
    rr_sel.to_csv(os.path.join(OUTPUT_DIR, "model_selection_retrain.csv"), index=False)

    results_retrain = {
        "Rolling Ensemble": rr_vals,
        "PPO":              individual["PPO"],
        "A2C":              individual["A2C"],
        "DDPG":             individual["DDPG"],
        "DJIA":             djia_vals,
        "Min-Var":          minvar_vals,
    }
    df_retrain = performance_table(results_retrain)
    df_retrain.to_csv(os.path.join(OUTPUT_DIR, "performance_table_retrain.csv"))
    plot_cumulative_returns(trade_dates, results_retrain, "cumulative_returns_retrain.png")
    print("\nRolling Retrain Results (2016-2020):")
    print(df_retrain.to_string())

    # ── 10.9 Extended backtest: 2016-2026 ────────────────────────────────
    EXTENDED_END = "2026-05-20"
    print(f"\nRunning extended backtest (2016-{EXTENDED_END}) ...")

    ext_mask  = (close.index >= TRADE_START) & (close.index <= EXTENDED_END)
    c_ext     = close[ext_mask]
    m_ext     = slice_data(macd, TRADE_START, EXTENDED_END)
    r_ext     = slice_data(rsi,  TRADE_START, EXTENDED_END)
    cc_ext    = slice_data(cci,  TRADE_START, EXTENDED_END)
    a_ext     = slice_data(adx,  TRADE_START, EXTENDED_END)
    t_ext     = slice_data(turb.to_frame("t"), TRADE_START, EXTENDED_END)["t"]
    ext_dates = pd.DatetimeIndex(c_ext.index)

    ind_ext = {}
    for name, model in models.items():
        env_e         = make_env(c_ext, m_ext, r_ext, cc_ext, a_ext, t_ext)
        vals_e        = backtest(model, env_e)
        ind_ext[name] = vals_e[: len(c_ext)]
        print(f"  {name} extended: Sharpe={sharpe(ind_ext[name]):.3f}")

    ens_ext, _ = build_ensemble(ind_ext, ext_dates)
    djia_ext   = djia_index(ext_dates)
    minvar_ext = min_variance(c_tr, c_ext)

    results_ext = {
        "Ensemble": ens_ext,
        "PPO":      ind_ext["PPO"],
        "A2C":      ind_ext["A2C"],
        "DDPG":     ind_ext["DDPG"],
        "DJIA":     djia_ext,
        "Min-Var":  minvar_ext,
    }

    # Extended performance table
    rows_ext = []
    for name, vals in results_ext.items():
        v = np.array(vals, dtype=float)
        r = _daily_returns(v)[1:]
        peak = np.maximum.accumulate(v)
        rows_ext.append({
            "Strategy"         : name,
            "Cumulative Return" : f"{(v[-1]-v[0])/v[0]:.1%}",
            "Annual Return"     : f"{(1+r.mean())**252-1:.1%}",
            "Annual Volatility" : f"{r.std()*252**0.5:.1%}",
            "Sharpe Ratio"      : f"{sharpe(v):.2f}",
            "Max Drawdown"      : f"{((v-peak)/np.maximum(peak,1e-8)).min():.1%}",
        })
    df_ext = pd.DataFrame(rows_ext).set_index("Strategy")
    print("\n" + "=" * 72)
    print(f"Extended Performance (2016-01-04 to {EXTENDED_END})")
    print("=" * 72)
    print(df_ext.to_string())
    print("=" * 72)
    df_ext.to_csv(os.path.join(OUTPUT_DIR, "performance_table_extended.csv"))

    # Extended cumulative returns plot with market event annotations
    _, ax = plt.subplots(figsize=(13, 6))
    for name, vals in results_ext.items():
        c, ls, lw = STYLE.get(name, ("gray", ":", 1.0))
        cum = (np.array(vals) - vals[0]) / vals[0]
        ax.plot(ext_dates[: len(cum)], cum,
                label=name, color=c, linestyle=ls, linewidth=lw)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)

    for s, e, col, lbl in [
        ("2020-02-19", "2020-03-23", "red",   "COVID crash"),
        ("2022-01-03", "2022-12-30", "orange","Rate-hike bear"),
    ]:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   alpha=0.10, color=col)
        ax.text(pd.Timestamp(s), ax.get_ylim()[1] * 0.95,
                lbl, fontsize=7, color=col, rotation=90, va="top")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return", fontsize=12)
    ax.set_title(
        "Extended Cumulative Return with Transaction Costs (2016-2026)\n"
        "Initial Portfolio: $1,000,000",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(alpha=0.3)
    plt.tight_layout()
    ext_path = os.path.join(OUTPUT_DIR, "cumulative_returns_extended.png")
    plt.savefig(ext_path, dpi=150)
    plt.close()
    print(f"  Saved: {ext_path}")

    print(f"\nAll outputs written to ./{OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
