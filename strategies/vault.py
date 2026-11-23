import pandas as pd
import numpy as np

def add_indicators(df):
    df = df.copy()
    df["ema5"]   = df["close"].ewm(span=5, adjust=False).mean()
    df["ema21"]  = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=50, adjust=False).mean()
    df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()
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

    # Bollinger Bands, Squeeze Width & Z-Score
    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + (2.0 * df["bb_std"])
    df["bb_lower"] = df["bb_mid"] - (2.0 * df["bb_std"])
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
    df["bb_width_q25"] = df["bb_width"].rolling(50).quantile(0.25)
    df["zscore"] = (df["close"] - df["bb_mid"]) / df["bb_std"].replace(0, np.nan)

    # Crypto Microstructure Indicators (Liquidation Wicks, Keltner Bands, Donchian Highs)
    df["body"] = (df["close"] - df["open"]).abs().clip(lower=1e-9)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["keltner_lower"] = df["ema21"] - (1.5 * df["atr"])
    df["high_5"] = df["high"].rolling(5).max().shift(1)
    df["high_10"] = df["high"].rolling(10).max().shift(1)
    _hh14 = df["high"].rolling(14).max()
    _ll14 = df["low"].rolling(14).min()
    df["willr"] = -100.0 * (_hh14 - df["close"]) / (_hh14 - _ll14).replace(0, np.nan)
    df["low_10"] = df["low"].rolling(10).min().shift(1)
    df["bar_range"] = df["high"] - df["low"]

    return df.dropna().reset_index(drop=True)

# --- EXISTING 6 PROVEN MEAN-REVERSION STRATEGIES (UNTOUCHED) ---
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
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > df["ema200"]) & (df["cum2_rsi2"] < 10)] = 1
    sig[(df["close"] > df["ema5"])] = -1
    return sig

def strat_bb_rsi2(df):
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > df["ema200"]) & (df["close"] < df["bb_lower"]) & (df["rsi2"] < 15)] = 1
    sig[(df["close"] >= df["bb_mid"]) | (df["close"] > df["ema5"])] = -1
    return sig

def strat_zscore_reversion(df):
    sig = pd.Series(0, index=df.index)
    sig[(df["close"] > df["ema200"]) & (df["zscore"] < -2.0)] = 1
    sig[(df["zscore"] >= 0.0)] = -1
    return sig

# --- NEW 4 OUT-OF-THE-BOX CRYPTO-NATIVE STRATEGIES ---
def strat_liquidation_wick_reclaim(df):
    # Buys aggressive absorption of intraday liquidation cascades
    sig = pd.Series(0, index=df.index)
    entry = (
        (df["close"] > df["ema200"]) &
        (df["low"] < df["ema21"]) &
        (df["lower_wick"] > (1.5 * df["body"])) &
        (df["ibs"] > 0.55) &
        (df["vol_ratio"] > 1.2)
    )
    exit_cond = (df["close"] > df["high"].shift(1)) | (df["rsi2"] > 85)
    sig[entry] = 1
    sig[exit_cond & ~entry] = -1
    return sig

def strat_vol_squeeze_breakout(df):
    # Volatility compression (bottom 25% BB width) -> volume-backed upside expansion
    sig = pd.Series(0, index=df.index)
    squeeze_recent = (df["bb_width"].shift(1) <= df["bb_width_q25"].shift(1)) | (df["bb_width"].shift(2) <= df["bb_width_q25"].shift(2))
    entry = (
        (df["close"] > df["ema50"]) &
        squeeze_recent &
        (df["close"] > df["high_10"]) &
        (df["vol_ratio"] > 1.3)
    )
    exit_cond = (df["close"] < df["ema21"]) | (df["rsi2"] > 92)
    sig[entry] = 1
    sig[exit_cond & ~entry] = -1
    return sig

def strat_trend_pullback_reclaim(df):
    # Stacked uptrend pullback into EMA21 value zone + green confirmation reclaim
    sig = pd.Series(0, index=df.index)
    entry = (
        (df["ema21"] > df["ema50"]) &
        (df["ema50"] > df["ema200"]) &
        (df["low"] <= df["ema21"]) &
        (df["rsi3"] < 35) &
        (df["close"] > df["open"]) &
        (df["close"] > df["ema21"])
    )
    exit_cond = (df["close"] > df["high_5"]) | (df["rsi2"] > 85)
    sig[entry] = 1
    sig[exit_cond & ~entry] = -1
    return sig

def strat_keltner_reversal(df):
    # ATR-adaptive Keltner lower band overshoot + seller exhaustion close
    sig = pd.Series(0, index=df.index)
    entry = (
        (df["close"] > df["ema200"]) &
        (df["ema50"] > df["ema200"]) &
        (df["low"] < df["keltner_lower"]) &
        (df["ibs"] > 0.45)
    )
    exit_cond = (df["close"] >= df["ema21"]) | (df["close"] > df["high"].shift(1))
    sig[entry] = 1
    sig[exit_cond & ~entry] = -1
    return sig


def strat_atr_panic_flush(df):
    # Volatility-normalized panic flush below EMA(5) in macro uptrend, fast 1-3 bar exit
    sig = pd.Series(0, index=df.index)
    entry = (
        (df["close"] > df["ema200"]) &
        ((df["ema5"] - df["close"]) > (1.2 * df["atr"])) &
        (df["ibs"] < 0.35)
    )
    exit_cond = (df["close"] > df["ema5"]) | (df["close"] > df["high"].shift(1))
    sig[entry] = 1
    sig[exit_cond & ~entry] = -1
    return sig

def strat_volume_climax_absorption(df):
    # Forced liquidation volume spike (>1.5x avg) + deep panic close -> fast snapback exit
    sig = pd.Series(0, index=df.index)
    entry = (
        (df["close"] > df["ema200"]) &
        (df["vol_ratio"] > 1.5) &
        (df["ibs"] < 0.30) &
        (df["rsi3"] < 20)
    )
    exit_cond = (df["close"] > df["ema5"]) | (df["ibs"] > 0.70)
    sig[entry] = 1
    sig[exit_cond & ~entry] = -1
    return sig


def strat_liq_range_fade(df):
    # Forced-liquidation single-bar panic expansion (>1.8x ATR red bar) in macro uptrend
    sig = pd.Series(0, index=df.index)
    entry = (
        (df["close"] > df["ema200"]) &
        (df["close"] < df["open"]) &
        (df["bar_range"] > (1.8 * df["atr"])) &
        (df["rsi2"] < 15)
    )
    exit_cond = (df["close"] > df["ema5"]) | (df["close"] > df["high"].shift(1))
    sig[entry] = 1
    sig[exit_cond & ~entry] = -1
    return sig

def strat_stophunt_low10_sweep(df):
    # Turtle-Soup liquidity sweep below prior 10-bar low inside macro uptrend
    sig = pd.Series(0, index=df.index)
    entry = (
        (df["close"] > df["ema200"]) &
        (df["low"] < df["low_10"]) &
        (df["rsi3"] < 25)
    )
    exit_cond = (df["close"] > df["ema5"]) | (df["close"] > df["high"].shift(1))
    sig[entry] = 1
    sig[exit_cond & ~entry] = -1
    return sig

def strat_regime_rsi2_10(df):
    # Higher-frequency RSI(2) < 10 dip-buyer locked to strong uptrend regime (EMA50 > EMA200)
    sig = pd.Series(0, index=df.index)
    entry = (
        (df["close"] > df["ema200"]) &
        (df["ema50"] > df["ema200"]) &
        (df["close"] < df["ema5"]) &
        (df["rsi2"] < 10)
    )
    exit_cond = (df["close"] > df["ema5"])
    sig[entry] = 1
    sig[exit_cond & ~entry] = -1
    return sig


def strat_capitulation_rubberband(df):
    # All-regime forced-liquidation rubber-band (no EMA200 lock -> works in bull & bear markets)
    sig = pd.Series(0, index=df.index)
    entry = (
        (df["rsi2"] < 3) &
        (df["close"] < df["bb_lower"]) &
        (df["ibs"] < 0.25)
    )
    exit_cond = (df["close"] > df["high"].shift(1)) | (df["rsi2"] > 65)
    sig[entry] = 1
    sig[exit_cond & ~entry] = -1
    return sig

def strat_ema50_altseason_snapback(df):
    # Fast-cycle EMA50 regime dip-buyer -> captures every 2-4 month crypto mini-season
    sig = pd.Series(0, index=df.index)
    entry = (
        (df["close"] > df["ema50"]) &
        (df["rsi2"] < 5) &
        (df["ibs"] < 0.30)
    )
    exit_cond = (df["close"] > df["ema5"]) | (df["close"] > df["high"].shift(1))
    sig[entry] = 1
    sig[exit_cond & ~entry] = -1
    return sig

def strat_willr_extreme_flush(df):
    # Deep exhaustion flush: Williams %R(14) < -95 + RSI(2) < 5 above EMA100
    sig = pd.Series(0, index=df.index)
    entry = (
        (df["close"] > df["ema100"]) &
        (df["willr"] < -95) &
        (df["rsi2"] < 5)
    )
    exit_cond = (df["close"] > df["ema5"]) | (df["ibs"] > 0.75)
    sig[entry] = 1
    sig[exit_cond & ~entry] = -1
    return sig

STRATEGY_VAULT = {
    "IBS + RSI(2) Institutional Snapback": strat_ibs_rsi2_snapback,
    "3-Bar Panic Pullback (Quick Exit)": strat_3bar_pullback,
    "Connors RSI(2) Extreme (<5)": strat_connors_rsi2,
    "2-Day Cumulative RSI(2) Snapback": strat_cum2_rsi2,
    "Bollinger Band + RSI(2) Capitulation": strat_bb_rsi2,
    "Z-Score Statistical Reversion": strat_zscore_reversion,
    "Liquidation Wick Absorption Reclaim": strat_liquidation_wick_reclaim,
    "Volatility Squeeze Volume Breakout": strat_vol_squeeze_breakout,
    "Trend Pullback EMA21 Reclaim": strat_trend_pullback_reclaim,
    "Keltner ATR Volatility Reversal": strat_keltner_reversal,
    "ATR Panic Flush Snapback": strat_atr_panic_flush,
    "Volume Climax Whale Absorption": strat_volume_climax_absorption,
    "Liquidation Range-Expansion Fade": strat_liq_range_fade,
    "Stop-Hunt 10-Bar Low Sweep": strat_stophunt_low10_sweep,
    "Regime-Aligned RSI(2) Pullback (<10)": strat_regime_rsi2_10,
    "All-Regime Capitulation Rubber-Band": strat_capitulation_rubberband,
    "Fast-Cycle EMA50 Alt-Season Snapback": strat_ema50_altseason_snapback,
    "Williams %R + RSI(2) Exhaustion Flush": strat_willr_extreme_flush,
}
