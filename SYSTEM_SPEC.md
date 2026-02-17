# Quant System Spec v1
Version: 1.0
Status: Frozen (Do not change without version bump)

============================================================
1. OBJECTIVE
============================================================

Goal:
- Build a robust, rule-based tactical allocation system
- Long-term target CAGR: 9–11% (aspirational)
- Design Max Drawdown target: <= 25%
- No leverage in v1
- Manual execution
- Weekly monitoring
- Monthly rebalance decision

Primary metric:
- Calmar ratio (CAGR / MaxDD)

============================================================
2. DATA & CONVENTIONS
============================================================

Data source:
- Yahoo Finance via yfinance

Price series:
- Adjusted Close ONLY

Return frequency:
- Daily returns

Calendar:
- Align all assets to common trading-day index
- Forward-fill only after first valid observation

Look-ahead rule:
- At rebalance date R:
  signals must use data up to R-1 only

============================================================
3. ASSET UNIVERSE
============================================================

Tickers:

- SPY
- EFA
- EEM
- TLT
- GLD
- DBC
- BTC-USD
- ETH-USD
- BIL

Universe must be defined in one config location.

============================================================
4. STRATEGY LOGIC
============================================================

4.1 Trend Filter (MA200)
------------------------

- Compute 200 trading-day moving average
- Eligible if price > MA200
- Ineligible if price <= MA200

4.2 Momentum (12-Month)
------------------------

- 252-day return:
  momentum = price(t) / price(t-252) - 1

- Rank only eligible assets
- Select Top 4 assets

If fewer than 4 eligible:
- Use all eligible
- Allocate remainder to BIL

============================================================
5. PORTFOLIO CONSTRUCTION
============================================================

5.1 Base Allocation
--------------------

- Equal weight selected Top 4
- Remaining allocation to BIL if fewer than 4

5.2 Crypto Cap
---------------

Constraint:
- BTC-USD + ETH-USD <= 25%

If exceeded:
- Scale BTC & ETH proportionally
- Redistribute remainder to non-crypto selected assets
- If none available → allocate to BIL

5.3 Long-Only
--------------

- No shorting
- All weights >= 0
- Sum(weights) = 1

============================================================
6. RISK CONTROLS
============================================================

6.1 Volatility Targeting
-------------------------

Lookback:
- 63 trading days

Annualized vol:
- std(daily_returns) * sqrt(252)

Target vol:
- 12%

Scale:
- scale = min(1.0, target_vol / realized_vol)
- No leverage allowed

Remaining allocation → BIL

6.2 Drawdown Governor
----------------------

Compute:
- equity / rolling peak - 1

Rules:

If DD <= -15%:
    Reduce risky exposure by 30%

If DD <= -22%:
    Ensure BIL weight >= 50%

If DD <= -30%:
    100% BIL

Recovery:
- If DD > -10%, resume normal sizing next rebalance

============================================================
7. REBALANCE LOGIC
============================================================

Decision timing:
- Last trading Friday of each month

Execution assumption:
- Executed next trading day

Monitoring:
- Weekly run (Monday 08:00)

Band rebalance:
- Only trade if weight diff > 5%

Transaction cost:
- 0.15% per trade notional

============================================================
8. OUTPUT REQUIREMENTS
============================================================

Must output:

- Current weights
- Selected assets
- Eligibility status
- Momentum values
- Realized vol
- Current drawdown
- Trade instructions (weight delta)

Backtest metrics:

- CAGR
- MaxDD
- Calmar
- Sharpe
- Worst 12M
- Rolling 3Y CAGR
- Rolling 3Y MaxDD

============================================================
9. ARCHITECTURE REQUIREMENTS
============================================================

Required modules:

- data.py
- signals.py
- portfolio.py
- risk.py
- backtest.py
- report.py
- main.py

All parameters configurable centrally.

============================================================
10. NON-GOALS
============================================================

- No leverage
- No shorting
- No intraday trading
- No brokerage API
- No hyperparameter optimization in v1
