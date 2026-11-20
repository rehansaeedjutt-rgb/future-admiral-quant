import pandas as pd
import numpy as np

def add_indicators(df):
    df = df.copy()
    df["ema21"]  = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

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

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()

    # SuperTrend Bands (ATR 14, Mult 2.5)
    hl2 = (df["high"] + df["low"]) / 2
    df["st_up"] = hl2 + (2.5 * df["atr"])
    df["st_dn"] = hl2 - (2.5 * df["atr"])
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

def strat_macd_trend(df):
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > df["ema200"]) & (df["macd"] > df["macd_sig"]) & (df["rsi"] > 48)] = 1
    sig[(df["macd"] < df["macd_sig"]) | (df["close"] < df["ema50"])] = -1
    return sig

def strat_supertrend_momentum(df):
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > df["st_dn"].shift(1)) & (df["ema21"] > df["ema50"]) & (df["vol_ratio"] > 1.0)] = 1
    sig[(df["close"] < df["st_dn"]) | (df["ema21"] < df["ema50"])] = -1
    return sig

STRATEGY_VAULT = {
    "EMA 21/50 + RSI Momentum": strat_ema_rsi,
    "Bollinger Mean Reversion": strat_bollinger_reversion,
    "Donchian Volume Breakout": strat_donchian_breakout,
    "MACD + EMA200 Institutional Trend": strat_macd_trend,
    "SuperTrend ATR Momentum": strat_supertrend_momentum,
}