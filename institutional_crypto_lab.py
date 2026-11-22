import numpy as np
import pandas as pd
import requests, os

INSTITUTIONAL_COINS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
    "LINKUSDT","AVAXUSDT","ADAUSDT","SUIUSDT","AAVEUSDT",
    "UNIUSDT","INJUSDT","NEARUSDT","DOTUSDT","LTCUSDT",
    "BCHUSDT","ATOMUSDT","STXUSDT","RUNEUSDT","GRTUSDT"
]

def fetch_binance(sym, total=2200):
    url = "https://data-api.binance.vision/api/v3/klines"
    rows, end_t = [], None
    while len(rows) < total:
        params = {"symbol": sym, "interval": "1d", "limit": 1000}
        if end_t: params["endTime"] = end_t
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200 or not r.json(): break
            data = r.json()
            rows = data + rows
            end_t = data[0][0] - 1
            if len(data) < 1000: break
        except Exception:
            break
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["t","open","high","low","close","volume","ct","qv","n","tb","tq","i"])
    for c in ["open","high","low","close","volume"]: df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
    return df[["timestamp","open","high","low","close","volume"]].drop_duplicates("timestamp").reset_index(drop=True)

def calc_rsi(s, period=2):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + g / (l + 1e-9)))

def get_strategies(df):
    c, h, l = df["close"], df["high"], df["low"]
    ema200 = c.ewm(span=200, adjust=False).mean()
    ema50  = c.ewm(span=50, adjust=False).mean()
    sma5   = c.rolling(5).mean()
    sma20  = c.rolling(20).mean()
    std20  = c.rolling(20).std()
    rsi2   = calc_rsi(c, 2)
    rsi14  = calc_rsi(c, 14)

    # Strategy 1: Extreme RSI(2) Mean Reversion
    s1 = pd.Series(0, index=df.index)
    s1[(c > ema200) & (rsi2 < 5)] = 1
    s1[c > sma5] = -1

    # Strategy 2: Institutional Trend Pullback (Uptrend + Pullback to EMA50/RSI2<15)
    s2 = pd.Series(0, index=df.index)
    s2[(c > ema200) & (ema50 > ema200) & (rsi2 < 15) & (c < sma20)] = 1
    s2[(c > sma20) | (c < ema200)] = -1

    # Strategy 3: Bollinger Volatility Snapback
    bb_low = sma20 - 2.0 * std20
    s3 = pd.Series(0, index=df.index)
    s3[(c > ema200) & (c < bb_low) & (rsi14 < 45)] = 1
    s3[c > sma20] = -1

    return {
        "RSI2_Extreme_MR": s1,
        "EMA50_Trend_Pullback": s2,
        "BB_Volatility_Snapback": s3
    }

def run_bt(df, sig):
    c, o = df["close"], df["open"]
    cap, pos, prev_cap, trades, eq = 10000.0, 0.0, 10000.0, [], []
    fee, slip = 0.0010, 0.0005
    for i in range(201, len(df)):
        ps = int(sig.iloc[i - 1])  # Strict shift(1)
        op = float(o.iloc[i])
        dt = df["timestamp"].iloc[i]
        if ps == 1 and pos == 0:
            prev_cap = cap
            fill = op * (1 + slip)
            pos = (cap * (1 - fee)) / fill
            cap = 0.0
        elif ps == -1 and pos > 0:
            fill = op * (1 - slip)
            cap = (pos * fill) * (1 - fee)
            trades.append({"year": dt.year, "pnl": ((cap - prev_cap) / prev_cap) * 100})
            pos = 0.0
        eq.append(cap if pos == 0 else pos * float(c.iloc[i]))
    if len(trades) < 8: return None
    tdf = pd.DataFrame(trades)
    eq_s = pd.Series(eq)
    dd = abs(((eq_s - eq_s.cummax()) / eq_s.cummax() * 100).min())
    wr = (tdf["pnl"] > 0).mean() * 100
    ret = (eq[-1] - 10000.0) / 100.0
    wins, losses = tdf[tdf["pnl"] > 0]["pnl"], tdf[tdf["pnl"] <= 0]["pnl"]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) else 9.99
    oos = tdf[tdf["year"] >= 2025]
    oos_wr = round((oos["pnl"] > 0).mean() * 100, 1) if len(oos) else np.nan
    return {"trades": len(tdf), "wr": round(wr, 1), "ret": round(ret, 1), "pf": round(pf, 2), "dd": round(dd, 1), "oos_n": len(oos), "oos_wr": oos_wr}

best_rows = []
for sym in INSTITUTIONAL_COINS:
    df = fetch_binance(sym)
    if len(df) < 250: continue
    strats = get_strategies(df)
    best = None
    for sname, sig in strats.items():
        r = run_bt(df, sig)
        if not r: continue
        # Score prioritizes high WR, positive return, and low drawdown
        score = (r["wr"] * 1.5) + min(r["ret"], 150) * 0.3 - (r["dd"] * 1.2)
        if best is None or score > best["score"]:
            best = {"coin": sym.replace("USDT", "/USDT"), "best_strategy": sname, "score": score, **r}
    if best:
        best.pop("score")
        best_rows.append(best)

res = pd.DataFrame(best_rows).sort_values(["wr", "ret"], ascending=[False, False])
os.makedirs("reports", exist_ok=True)
res.to_csv("reports/institutional_crypto_matrix.csv", index=False)

print("================================================================================")
print("      INSTITUTIONAL CRYPTO CUSTOM STRATEGY MATRIX (ZERO MEME COINS)             ")
print("================================================================================")
print(res.to_string(index=False))
