def run_backtest(df, signals_df, initial_cash=100000):
    cash = initial_cash
    position = 0.0
    trades = 0
    for idx, row in signals_df.iterrows():
        pos = row.get('positions', 0)
        price = row['close']
        if pos == 1 and position == 0:
            # enter long
            position = cash / price
            cash = 0
            trades += 1
        elif pos == -1 and position > 0:
            # exit long
            cash = position * price
            position = 0
            trades += 1
    final_value = cash + (position * signals_df['close'].iloc[-1])
    pnl = final_value - initial_cash
    return {'pnl': float(pnl), 'trades': trades}
