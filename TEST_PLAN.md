# Quant System Test Plan v1

============================================================
A. CORE CORRECTNESS TESTS (MUST PASS)
============================================================

A1. No Look-Ahead Bias
-----------------------

For rebalance date R:
- Signals must use data up to R-1 only.

Test:
- Shift signals by +1 day and verify different results.

A2. Weight Validity
--------------------

At each rebalance:

- Sum(weights) == 1 (tolerance 1e-6)
- All weights >= 0

Fail if violated.

A3. Cash Fallback
------------------

If no assets eligible:
- Portfolio = 100% BIL

A4. Crypto Cap
---------------

BTC-USD + ETH-USD <= 25% always.

Test scenario:
- Force top selection heavy in crypto.
- Confirm cap enforcement.

A5. Vol Target Constraint
---------------------------

After vol targeting:

- Total risky allocation <= 100%
- If realized vol > target → exposure reduced

A6. Drawdown Governor
----------------------

Simulate artificial DD:

- DD <= -15% → risky exposure reduced
- DD <= -22% → BIL >= 50%
- DD <= -30% → BIL = 100%

Recovery:
- DD > -10% → resume normal sizing

============================================================
B. BACKTEST INTEGRITY TESTS
============================================================

B1. Monthly Rebalance Timing
------------------------------

Rebalance only on last trading Friday of month.

B2. Transaction Costs Applied
------------------------------

Backtest must include 0.15% cost per trade.

Run comparison:
- cost=0
- cost=0.0015

Results must differ.

B3. Band Rebalance
-------------------

Small deviations (<5%) should not trigger trade.

============================================================
C. ROBUSTNESS TESTS
============================================================

Run backtest variations:

MA Window:
- 180
- 200
- 220

Vol Target:
- 10%
- 12%
- 15%

Top-N:
- 3
- 4
- 5

Pass Criteria:
- No collapse of performance.
- No explosive DD from small changes.

============================================================
D. STRESS WINDOW ANALYSIS
============================================================

Evaluate separately:

- 2008 crisis
- 2020 crash
- 2022 inflation regime

Report:
- MaxDD in window
- Recovery duration

============================================================
E. MONITORING CHECKS
============================================================

- Data freshness within last 5 trading days
- No NaN in selected assets
- No abnormal turnover spike
