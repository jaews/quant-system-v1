def generate_signals(df, short=5, long=20):
    df = df.copy()
    df['short_ma'] = df['close'].rolling(short).mean()
    df['long_ma'] = df['close'].rolling(long).mean()
    df['signal'] = 0
    df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1
    df.loc[df['short_ma'] <= df['long_ma'], 'signal'] = 0
    df['positions'] = df['signal'].diff().fillna(0)
    return df
