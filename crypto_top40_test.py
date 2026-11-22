import numpy as np
import pandas as pd
import requests, os

COINS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","DOTUSDT","TRXUSDT",
    "SUIUSDT","LTCUSDT","BCHUSDT","NEARUSDT","UNIUSDT","ETCUSDT","HBARUSDT","FILUSDT","ATOMUSDT","ARBUSDT",
    "OPUSDT","INJUSDT","STXUSDT","FETUSDT","AAVEUSDT","GRTUSDT","VETUSDT","ALGOUSDT","RUNEUSDT","AXSUSDT",
    "DOGEUSDT","SHIBUSDT","APTUSDT","ICPUSDT","IMXUSDT","RENDERUSDT","THETAUSDT","SANDUSDT","MANAUSDT","CRVUSDT"
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

def calc_rsi(series, period=2):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + (gain / (loss + 1e-9))))

def backtest_extreme_rsi2(df):
    if len(df) < 250: return None
    c, o = df["close"], df["open"]
    ema200 = c.ewm(span=200, adjust=False).mean()
    sma5 = c.rolling(5).mean()
    rsi2 = calc_rsi(c, 2)

    sig = pd.Series(0, index=df.index)
    sig[(c > ema200) & (rsi2 < 5)] = 1
    sig[c > sma5] = -1

    cap, pos, prev_cap, trades, eq = 10000.0, 0.0, 10000.0, [], []
    fee, slip = 0.0010, 0.0005

    for i in range(201, len(df)):
        ps = int(sig.iloc[i - 1])  # Strict shift(1) next-bar open
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

    if not trades: return None
    tdf = pd.DataFrame(trades)
    eq_s = pd.Series(eq)
    dd = abs(((eq_s - eq_s.cummax()) / eq_s.cummax() * 100).min())
    wr = (tdf["pnl"] > 0).mean() * 100
    ret = (eq[-1] - 10000.0) / 100.0
    wins, losses = tdf[tdf["pnl"] > 0]["pnl"], tdf[tdf["pnl"] <= 0]["pnl"]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) else 9.99
    oos = tdf[tdf["year"] >= 2025]
    return {
        "trades": len(tdf), "wr_pct": round(wr, 1), "ret_pct": round(ret, 1),
        "pf": round(pf, 2), "dd_pct": round(dd, 1),
        "oos_trades": len(oos), "oos_wr_pct": round((oos["pnl"] > 0).mean() * 100, 1) if len(oos) else None
    }

results = []
for sym in COINS:
    d = fetch_binance(sym)
    if d.empty: continue
    r = backtest_extreme_rsi2(d)
    if r: results.append({"coin": sym.replace("USDT", "/USDT"), **r})

res_df = pd.DataFrame(results).sort_values(["wr_pct", "ret_pct"], ascending=[False, False])
os.makedirs("reports", exist_ok=True)
res_df.to_csv("reports/crypto_top40_results.csv", index=False)

passed = res_df[(res_df["trades"] >= 10) & (res_df["wr_pct"] >= 80.0) & (res_df["ret_pct"] > 0)]
print("=========================================================================")
print("   CRYPTO TOP 40 — EXTREME RSI(2) < 5 -> SMA(5) EXIT (6-YEAR BINANCE)    ")
print("=========================================================================")
print(f"Total Tested: {len(res_df)} | Passed (>=10 trades, >=80% WR, Profitable): {len(passed)}")
print(f"All-Coin Avg WR: {res_df['wr_pct'].mean():.1f}% | All-Coin Avg Return: {res_df['ret_pct'].mean():+.1f}%")
print("\n80%+ WIN RATE VERIFIED COINS:")
print(passed.to_string(index=False) if not passed.empty else "None")
print("\nALL 40 COINS SUMMARY:")
print(res_df.to_string(index=False))
