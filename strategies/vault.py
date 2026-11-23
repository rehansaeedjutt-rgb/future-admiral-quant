import pandas as pd
import numpy as np

def add_indicators(df):
    df = df.copy()
    df["ema5"]   = df["close"].ewm(span=5, adjust=False).mean()
    df["ema21"]  = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # Internal Bar Strength (IBS): 0 = closed at low, 1 = closed at high
    bar_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["ibs"] = (df["close"] - df["low"]) / bar_range

    # RSI 2, 3, 14 (+ alias 'rsi' for ML filter compatibility)
    delta = df["close"].diff()
    for p in [2, 3, 14]:
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = (-delta.clip(upper=0)).rolling(p).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f"rsi{p}"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi14"]
    df["cum2_rsi2"] = df["rsi2"] + df["rsi2"].shift(1)

    # ATR & Volume Ratio (required by ML filter)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)

    # Consecutive Down Bars
    down = (df["close"] < df["close"].shift(1)).astype(int)
    df["down3"] = (down + down.shift(1) + down.shift(2)) == 3

    # Bollinger Bands & Z-Score
    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_lower"] = df["bb_mid"] - (2.0 * df["bb_std"])
    df["zscore"] = (df["close"] - df["bb_mid"]) / df["bb_std"].replace(0, np.nan)

    return df.dropna().reset_index(drop=True)

def strat_ibs_rsi2_snapback(df):
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > df["ema200"]) & (df["ibs"] < 0.20) & (df["rsi2"] < 10)] = 1
    sig[(df["close"] > df["high"].shift(1)) | (df["ibs"] > 0.80)] = -1
    return sig

def strat_3bar_pullback(df):
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > df["ema200"]) & (df["down3"]) & (df["rsi3"] < 20)] = 1
    sig[(df["close"] > df["ema5"]) | (df["close"] > df["high"].shift(1))] = -1
    return sig

def strat_connors_rsi2(df):
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > df["ema200"]) & (df["rsi2"] < 5)] = 1
    sig[(df["close"] > df["ema5"])] = -1
    return sig

def strat_cum2_rsi2(df):
    # Connors 2-Day Cumulative RSI(2) < 10 Snapback
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > df["ema200"]) & (df["cum2_rsi2"] < 10)] = 1
    sig[(df["close"] > df["ema5"])] = -1
    return sig

def strat_bb_rsi2(df):
    # Lower Bollinger Band (2.0 SD) + RSI(2) Capitulation
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > df["ema200"]) & (df["close"] < df["bb_lower"]) & (df["rsi2"] < 15)] = 1
    sig[(df["close"] >= df["bb_mid"]) | (df["close"] > df["ema5"])] = -1
    return sig

def strat_zscore_reversion(df):
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > df["ema200"]) & (df["zscore"] < -2.0)] = 1
    sig[(df["zscore"] >= 0.0)] = -1
    return sig

STRATEGY_VAULT = {
    "IBS + RSI(2) Institutional Snapback": strat_ibs_rsi2_snapback,
    "3-Bar Panic Pullback (Quick Exit)": strat_3bar_pullback,
    "Connors RSI(2) Extreme (<5)": strat_connors_rsi2,
    "2-Day Cumulative RSI(2) Snapback": strat_cum2_rsi2,
    "Bollinger Band + RSI(2) Capitulation": strat_bb_rsi2,
    "Z-Score Statistical Reversion": strat_zscore_reversion,
}
