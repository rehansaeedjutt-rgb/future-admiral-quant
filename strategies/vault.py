import pandas as pd
import numpy as np

def add_indicators(df):
    df = df.copy()
    df["ema5"]   = df["close"].ewm(span=5, adjust=False).mean()
    df["ema21"]  = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI 14, RSI 3, and Connors RSI 2
    delta = df["close"].diff()
    for p in [2, 3, 14]:
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = (-delta.clip(upper=0)).rolling(p).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f"rsi{p}"] = 100 - (100 / (1 + rs))

    # Bollinger & Z-Score
    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_low"] = df["bb_mid"] - (2 * df["bb_std"])
    df["zscore"] = (df["close"] - df["bb_mid"]) / df["bb_std"].replace(0, np.nan)

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()

    return df.dropna().reset_index(drop=True)

def strat_connors_rsi2(df):
    # Institutional High-Win-Rate Mean Reversion (75-85%+ historical win rate in uptrends)
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > df["ema200"]) & (df["rsi2"] < 10)] = 1
    sig[(df["close"] > df["ema5"]) | (df["rsi2"] > 70)] = -1
    return sig

def strat_zscore_reversion(df):
    # Statistical Z-Score Snapback
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > df["ema200"]) & (df["zscore"] < -2.0)] = 1
    sig[(df["zscore"] >= 0.0)] = -1
    return sig

def strat_double_extreme_bb(df):
    # Bollinger Lower Band + RSI(3) Extreme Snapback
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] < df["bb_low"]) & (df["rsi3"] < 15)] = 1
    sig[(df["close"] > df["bb_mid"]) | (df["rsi3"] > 65)] = -1
    return sig

def strat_macd_trend(df):
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > df["ema200"]) & (df["macd"] > df["macd_sig"]) & (df["rsi14"] > 48)] = 1
    sig[(df["macd"] < df["macd_sig"]) | (df["close"] < df["ema50"])] = -1
    return sig

STRATEGY_VAULT = {
    "Connors RSI(2) High-Win Snapback": strat_connors_rsi2,
    "Z-Score Statistical Reversion": strat_zscore_reversion,
    "Double-Extreme BB + RSI(3)": strat_double_extreme_bb,
    "MACD + EMA200 Institutional Trend": strat_macd_trend,
}