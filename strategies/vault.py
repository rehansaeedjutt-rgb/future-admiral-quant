import pandas as pd
import numpy as np

def add_indicators(df):
    df = df.copy()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_low"] = df["bb_mid"] - (2 * df["bb_std"])

    df["dc_high"] = df["high"].rolling(20).max().shift(1)
    df["dc_low"]  = df["low"].rolling(20).min().shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)
    return df.dropna().reset_index(drop=True)

def strat_ema_rsi(df):
    sig = pd.Series(0, index=df.index)
    sig[(df["ema21"] > df["ema50"]) & (df["rsi"] > 50) & (df["rsi"] < 75)] = 1
    sig[(df["ema21"] < df["ema50"]) | (df["rsi"] > 80)] = -1
    return sig

def strat_bollinger_reversion(df):
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] < df["bb_low"]) & (df["rsi"] < 35)] = 1
    sig[(df["close"] > df["bb_mid"]) | (df["rsi"] > 65)] = -1
    return sig

def strat_donchian_breakout(df):
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > df["dc_high"]) & (df["vol_ratio"] > 1.1)] = 1
    sig[(df["close"] < df["dc_low"])] = -1
    return sig

STRATEGY_VAULT = {
    "EMA 21/50 + RSI Momentum": strat_ema_rsi,
    "Bollinger Mean Reversion": strat_bollinger_reversion,
    "Donchian Volume Breakout": strat_donchian_breakout,
}