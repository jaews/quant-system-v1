# Test Plan

- `tests/test_sanity.py`: basic import and end-to-end sanity check using sample data.

Suggested further tests:
- Unit tests for `signals.generate_signals` correctness
- Risk metric unit tests in `risk.py`
- Backtest edge cases (no signals, constant price, etc.)

Run tests:

```bash
pytest -q
```
