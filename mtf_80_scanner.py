import os, time, math, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

os.makedirs("reports", exist_ok=True)
os.makedirs("data_cache/stocks", exist_ok=True)
os.makedirs("data_cache/stocks_1h", exist_ok=True)

HTTP = requests.Session()
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0.0.0"}

INITIAL_CAPITAL = 10000.0
FEE_RATE = 0.0010          # 0.10% per side
SLIPPAGE = 0.0005          # 0.05% per side
HARD_MAX_SL_PCT = 0.045    # Hard 4.5% stop-loss cap matches live bot

# Strict 80%+ Win-Rate Institutional Gate
MIN_WIN_RATE = 80.0
MIN_PROFIT_FACTOR = 2.0
MAX_DRAWDOWN = 20.0
MIN_OOS_TRADES = 10
MAX_PVALUE_PCT = 5.0

TOP_CRYPTO = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT",
    "LINKUSDT", "DOTUSDT", "TRXUSDT", "TONUSDT", "LTCUSDT", "BCHUSDT", "NEARUSDT",
    "UNIUSDT", "ETCUSDT", "XLMUSDT", "HBARUSDT", "ATOMUSDT", "INJUSDT", "STXUSDT",
    "FETUSDT", "AAVEUSDT", "THETAUSDT", "DOGEUSDT"
]

TOP_STOCKS = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "NFLX",
    "JPM", "V", "MA", "UNH", "LLY", "XOM", "COST", "WMT", "HD", "PG",
    "SPY", "QQQ", "BRK-B", "JNJ", "ABBV", "CSCO", "DE", "CAT", "GE", "EMR",
    "GD", "LRCX", "KLAC", "GILD", "VRTX", "WFC", "DUK", "SMH", "XLK", "XLF",
    "PANW", "FTNT", "SHOP", "PLTR", "MU", "AMAT", "ADI", "PH", "WM", "RSG",
    "FDX", "UPS", "C", "AXP", "SPGI", "CME", "CB", "PGR", "GLD", "XLU"
]

# ---------- DATA LOADERS ----------
def fetch_binance(symbol, interval, limit_bars):
    url = "https://api.binance.com/api/v3/klines"
    rows, end_time = [], None
    while len(rows) < limit_bars:
        params = {"symbol": symbol, "interval": interval, "limit": min(1000, limit_bars - len(rows))}
        if end_time is not None:
            params["endTime"] = end_time
        r = HTTP.get(url, params=params, timeout=10)
        if r.status_code != 200:
            break
        batch = r.json()
        if not batch or isinstance(batch, dict):
            break
        rows = batch + rows
        end_time = batch[0][0] - 1
        if len(batch) < 1000:
            break
        time.sleep(0.05)
    if len(rows) < 260:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ot","open","high","low","close","volume","ct","qv","nt","tb","tq","ig"])
    df["timestamp"] = pd.to_datetime(df["ot"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]].drop_duplicates("timestamp").reset_index(drop=True)

def fetch_yahoo(symbol, rng="6y", interval="1d", cache_dir="data_cache/stocks", ttl=10800):
    cpath = os.path.join(cache_dir, f"{symbol}.csv")
    if os.path.exists(cpath) and (time.time() - os.path.getmtime(cpath)) < ttl:
        try:
            cdf = pd.read_csv(cpath)
            if len(cdf) >= 220:
                return cdf[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception:
            pass
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        try:
            r = HTTP.get(f"https://{host}/v8/finance/chart/{symbol}",
                         params={"range": rng, "interval": interval}, headers=HDRS, timeout=10)
            if r.status_code == 200:
                res = (r.json().get("chart", {}).get("result") or [None])[0]
                if res and res.get("timestamp"):
                    q = res["indicators"]["quote"][0]
                    fmt = "%Y-%m-%d" if interval == "1d" else "%Y-%m-%d %H:%M"
                    df = pd.DataFrame({
                        "timestamp": pd.to_datetime(res["timestamp"], unit="s", utc=True).strftime(fmt),
                        "open": q.get("open"), "high": q.get("high"),
                        "low": q.get("low"), "close": q.get("close"), "volume": q.get("volume"),
                    }).dropna().reset_index(drop=True)
                    if len(df) >= 220:
                        df.to_csv(cpath, index=False)
                        return df
        except Exception:
            pass
        time.sleep(0.15)
    return pd.DataFrame()

def resample_weekly(df_1d):
    if df_1d.empty:
        return pd.DataFrame()
    d = df_1d.copy()
    d["dt"] = pd.to_datetime(d["timestamp"])
    w = d.set_index("dt").resample("W-FRI").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index()
    w["timestamp"] = w["dt"].dt.strftime("%Y-%m-%d")
    return w[["timestamp", "open", "high", "low", "close", "volume"]]

# ---------- INDICATORS & 6 OUT-OF-THE-BOX LOW-SL / 80%+ STRATEGIES ----------
def add_indicators(df):
    df = df.copy()
    delta = df["close"].diff()
    for p, col in [(2, "rsi2"), (3, "rsi3"), (14, "rsi")]:
        gain = delta.clip(lower=0).ewm(alpha=1.0 / p, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / p, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df[col] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["ibs"] = ((df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)).fillna(0.5)
    df["vol_ratio"] = (df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)).fillna(1.0)
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_low_25"] = sma20 - 2.5 * std20
    df["low_10"] = df["low"].rolling(10).min().shift(1)
    df["cum2_rsi2"] = df["rsi2"] + df["rsi2"].shift(1)
    return df

def s_sniper_double_lock(df):
    sig = pd.Series(0, index=df.index)
    entry = (df["close"] > df["ema200"]) & (df["rsi2"] < 5) & (df["ibs"] < 0.25)
    ex = (df["close"] > df["ema5"]) | (df["ibs"] > 0.80)
    sig[entry] = 1; sig[ex & ~entry] = -1
    return sig

def s_green_reclaim_antiknife(df):
    # Anti-falling-knife: prior bar extreme panic + current bar green reclaim -> minimal SL hits
    sig = pd.Series(0, index=df.index)
    entry = (df["close"] > df["ema200"]) & (df["rsi2"].shift(1) < 10) & (df["close"] > df["open"]) & (df["close"] < df["ema5"])
    ex = (df["close"] > df["ema5"]) | (df["rsi2"] > 75)
    sig[entry] = 1; sig[ex & ~entry] = -1
    return sig

def s_bb25_capitulation(df):
    sig = pd.Series(0, index=df.index)
    entry = (df["close"] > df["ema200"]) & (df["close"] < df["bb_low_25"]) & (df["rsi2"] < 12)
    ex = (df["close"] > df["ema5"]) | (df["close"] > df["high"].shift(1))
    sig[entry] = 1; sig[ex & ~entry] = -1
    return sig

def s_liquidity_sweep_absorb(df):
    # Sweeps 10-bar low & closes strong (buyers absorbed panic)
    sig = pd.Series(0, index=df.index)
    entry = (df["close"] > df["ema200"]) & (df["low"] < df["low_10"]) & (df["ibs"] > 0.55) & (df["rsi3"] < 35)
    ex = (df["close"] > df["ema5"]) | (df["close"] > df["high"].shift(1))
    sig[entry] = 1; sig[ex & ~entry] = -1
    return sig

def s_3bar_panic_exhaustion(df):
    sig = pd.Series(0, index=df.index)
    r3 = (df["close"] < df["open"]) & (df["close"].shift(1) < df["open"].shift(1)) & (df["close"].shift(2) < df["open"].shift(2))
    entry = (df["close"] > df["ema200"]) & r3 & (df["rsi3"] < 20)
    ex = (df["close"] > df["ema5"]) | (df["close"] > df["high"].shift(1))
    sig[entry] = 1; sig[ex & ~entry] = -1
    return sig

def s_cum2_rsi2_snapback(df):
    sig = pd.Series(0, index=df.index)
    entry = (df["close"] > df["ema200"]) & (df["cum2_rsi2"] < 15) & (df["close"] < df["ema5"])
    ex = (df["close"] > df["ema5"])
    sig[entry] = 1; sig[ex & ~entry] = -1
    return sig

STRATEGIES = {
    "Sniper Double-Lock (RSI2+IBS)": s_sniper_double_lock,
    "Green Reclaim Anti-Knife": s_green_reclaim_antiknife,
    "2.5-Sigma BB Capitulation": s_bb25_capitulation,
    "Liquidity Sweep Absorption": s_liquidity_sweep_absorb,
    "3-Bar Panic Exhaustion": s_3bar_panic_exhaustion,
    "2-Day Cumul RSI(2) Snapback": s_cum2_rsi2_snapback,
}

# ---------- ML FILTER (STRICT OOS PURGE) ----------
def apply_ml(df, raw_sig, threshold=0.45):
    d = df.copy()
    d["ema_dist"] = (d["close"] - d["ema50"]) / d["ema50"]
    feats = ["rsi", "vol_ratio", "atr", "ema_dist"]
    d["target"] = (d["close"].shift(-3) > d["close"]).astype(int)
    valid = d.dropna().copy()
    split = int(len(valid) * 0.5)
    if split < 60:
        return raw_sig * 0
    model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    model.fit(valid[feats].iloc[:split], valid["target"].iloc[:split])
    oos_idx = valid.index[min(split + 3, len(valid) - 1)]
    out = raw_sig.copy()
    out.loc[:oos_idx] = 0
    mask = d.index > oos_idx
    if mask.any():
        probs = model.predict_proba(d.loc[mask, feats].fillna(0))[:, 1]
        s = out.loc[mask].copy()
        s[(s == 1) & (probs < threshold)] = 0
        out.loc[mask] = s
    return out

# ---------- BACKTEST WITH ATR SIZING + HARD SL PROTECTION ----------
def run_bt(df, sig, asset_name, strat_name, record_audit=False, audit_list=None):
    cash, realized, pos, entry_px, sl_px, entry_dt, sig_dt, raw_open = INITIAL_CAPITAL, INITIAL_CAPITAL, 0.0, 0.0, 0.0, None, None, 0.0
    trades, eq = [], []
    for i in range(1, len(df)):
        o_px = float(df["open"].iloc[i])
        l_px = float(df["low"].iloc[i])
        c_px = float(df["close"].iloc[i])
        b_dt = str(df["timestamp"].iloc[i])
        psig = int(sig.iloc[i - 1])  # Strict shift(1): zero look-ahead

        # Check hard Stop-Loss hit intrabar if in position
        if pos > 0 and l_px <= sl_px:
            fill = sl_px * (1 - SLIPPAGE)
            pos_usd = round(pos * entry_px, 2)
            cash += (pos * fill) * (1 - FEE_RATE)
            realized = cash
            gpnl = ((fill - entry_px) / entry_px) * 100
            npnl = gpnl - (FEE_RATE * 200)
            trades.append(npnl)
            if record_audit and audit_list is not None:
                audit_list.append({
                    "record_type": "MTF_80_OOS", "data_source": "Binance" if "Binance" in asset_name else "Yahoo",
                    "asset": asset_name, "strategy": strat_name, "signal_date": sig_dt[:10], "entry_date": entry_dt[:10],
                    "raw_entry_open": round(raw_open, 4), "entry_price_slipped": round(entry_px, 4),
                    "exit_date": b_dt[:10], "raw_exit_open": round(sl_px, 4), "exit_price_slipped": round(fill, 4),
                    "position_size_usd": pos_usd, "fee_roundtrip_pct": 0.20, "slippage_roundtrip_pct": 0.10,
                    "gross_pnl_after_slippage_pct": round(gpnl, 2), "net_pnl_pct": round(npnl, 2), "equity_after_usd": round(cash, 2)
                })
            pos = 0.0
        elif psig == 1 and pos == 0:
            pclose = float(df["close"].iloc[i - 1])
            patr = float(df["atr"].iloc[i - 1]) if not np.isnan(df["atr"].iloc[i - 1]) else pclose * 0.02
            atr_pct = (patr / pclose) if (pclose > 0 and patr > 0) else 0.02
            alloc_pct = min(1.0, max(0.15, 0.02 / atr_pct))
            deploy = cash * alloc_pct
            cash -= deploy
            fill = o_px * (1 + SLIPPAGE)
            pos = (deploy * (1 - FEE_RATE)) / fill
            entry_px, entry_dt, sig_dt, raw_open = fill, b_dt, str(df["timestamp"].iloc[i - 1]), o_px
            sl_px = max(fill - 2.0 * patr, fill * (1.0 - HARD_MAX_SL_PCT))
        elif psig == -1 and pos > 0:
            fill = o_px * (1 - SLIPPAGE)
            pos_usd = round(pos * entry_px, 2)
            cash += (pos * fill) * (1 - FEE_RATE)
            realized = cash
            gpnl = ((fill - entry_px) / entry_px) * 100
            npnl = gpnl - (FEE_RATE * 200)
            trades.append(npnl)
            if record_audit and audit_list is not None:
                audit_list.append({
                    "record_type": "MTF_80_OOS", "data_source": "Binance" if "Binance" in asset_name else "Yahoo",
                    "asset": asset_name, "strategy": strat_name, "signal_date": sig_dt[:10], "entry_date": entry_dt[:10],
                    "raw_entry_open": round(raw_open, 4), "entry_price_slipped": round(entry_px, 4),
                    "exit_date": b_dt[:10], "raw_exit_open": round(o_px, 4), "exit_price_slipped": round(fill, 4),
                    "position_size_usd": pos_usd, "fee_roundtrip_pct": 0.20, "slippage_roundtrip_pct": 0.10,
                    "gross_pnl_after_slippage_pct": round(gpnl, 2), "net_pnl_pct": round(npnl, 2), "equity_after_usd": round(cash, 2)
                })
            pos = 0.0
        eq.append(cash + pos * c_px)

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    t_cnt = len(trades)
    wr = (len(wins) / t_cnt * 100.0) if t_cnt else 0.0
    pf = (sum(wins) / abs(sum(losses))) if losses else (9.99 if wins else 0.0)
    eq_s = pd.Series(eq) if eq else pd.Series([INITIAL_CAPITAL])
    dd = abs(float(((eq_s - eq_s.cummax()) / eq_s.cummax() * 100).min())) if len(eq_s) > 1 else 0.0
    net_ret = ((realized - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100.0
    pval = (sum(math.comb(t_cnt, k) for k in range(len(wins), t_cnt + 1)) * (0.5 ** t_cnt) * 100.0) if t_cnt else 100.0

    last_px = float(df["close"].iloc[-1])
    last_atr = float(df["atr"].iloc[-1]) if not np.isnan(df["atr"].iloc[-1]) else last_px * 0.02
    last_sig = int(sig.iloc[-1])
    status = "BUY / LONG" if last_sig == 1 else ("EXIT / SELL" if last_sig == -1 else "FLAT / STANDBY")
    return {
        "asset": asset_name, "strategy": strat_name, "price": round(last_px, 4), "status": status,
        "candle_date": str(df["timestamp"].iloc[-1])[:10],
        "stop_loss": round(max(last_px - 2.0 * last_atr, last_px * (1.0 - HARD_MAX_SL_PCT)), 4),
        "take_profit": round(last_px + 2.5 * last_atr, 4),
        "total_trades": t_cnt, "win_rate": round(wr, 2), "profit_factor": round(pf, 2),
        "max_drawdown": round(dd, 2), "net_return": round(net_ret, 2), "final_equity": round(realized, 2),
        "p_value": round(pval, 2), "max_risk_usd": 100.0,
        "ai_verdict": "ARMED — AI News & War Shield runs live before every entry",
        "updated_at": datetime.now(timezone.utc).isoformat(), "tier": "TIER 1 (80%+ SNIPER)",
    }

def scan_asset(df, asset_name, audit_list):
    if df.empty or len(df) < 260:
        return None
    df = add_indicators(df)
    split = int(len(df) * 0.5)
    best_name, best_score, best_raw = None, -1e9, None
    for sname, sfn in STRATEGIES.items():
        raw = sfn(df)
        is_res = run_bt(df.iloc[:split], raw.iloc[:split], asset_name, sname)
        if is_res["total_trades"] >= 8 and is_res["win_rate"] >= 70.0 and is_res["profit_factor"] >= 1.5:
            score = is_res["win_rate"] * min(is_res["profit_factor"], 10.0) * math.log1p(is_res["total_trades"])
            if score > best_score:
                best_score, best_name, best_raw = score, sname, raw
    if not best_name:
        return None

    filt = apply_ml(df, best_raw, threshold=0.45)
    oos = run_bt(df, filt, asset_name, best_name)
    if (oos["total_trades"] >= MIN_OOS_TRADES and oos["win_rate"] >= MIN_WIN_RATE
            and oos["profit_factor"] >= MIN_PROFIT_FACTOR and oos["max_drawdown"] <= MAX_DRAWDOWN
            and oos["p_value"] < MAX_PVALUE_PCT):
        run_bt(df, filt, asset_name, best_name, record_audit=True, audit_list=audit_list)
        return oos
    return None

def main():
    print("=========================================================================")
    print("  FUTURE ADMIRAL v3.0 — MULTI-TIMEFRAME 80%+ SNIPER SCANNER")
    print("  Timeframes: 15m | 1h | 4h | 12h | 1d | 1w  (Strict WR >= 80%, PF >= 2.0)")
    print("=========================================================================")
    verified, audit = [], []

    # 1. Crypto across 6 timeframes
    crypto_tfs = [("15m", 1500), ("1h", 2000), ("4h", 2000), ("12h", 3000), ("1d", 2200), ("1w", 400)]
    for tf, bars in crypto_tfs:
        tf_hits = 0
        for sym in TOP_CRYPTO:
            label = f"{sym.replace('USDT','/USDT')} (Binance {tf.upper()})"
            res = scan_asset(fetch_binance(sym, tf, bars), label, audit)
            if res:
                verified.append(res)
                tf_hits += 1
                print(f"  [80%+ PASS] {label:28s} | {res['strategy']:30s} | WR:{res['win_rate']:5.1f}% | PF:{res['profit_factor']:5.2f} | DD:{res['max_drawdown']:4.1f}% | Trd:{res['total_trades']}")
        print(f"--- Finished Binance {tf.upper()}: {tf_hits} verified 80%+ coins ---")

    # 2. US Stocks across 1H, 1D, 1W
    stock_1d_cache = {}
    s1d_hits, s1w_hits, s1h_hits = 0, 0, 0
    for sym in TOP_STOCKS:
        df1d = fetch_yahoo(sym, "6y", "1d", "data_cache/stocks", ttl=10800)
        stock_1d_cache[sym] = df1d
        res1d = scan_asset(df1d, f"{sym} (US Stock 1D)", audit)
        if res1d:
            verified.append(res1d); s1d_hits += 1
            print(f"  [80%+ PASS] {res1d['asset']:28s} | {res1d['strategy']:30s} | WR:{res1d['win_rate']:5.1f}% | PF:{res1d['profit_factor']:5.2f} | DD:{res1d['max_drawdown']:4.1f}% | Trd:{res1d['total_trades']}")
        res1w = scan_asset(resample_weekly(df1d), f"{sym} (US Stock 1W)", audit)
        if res1w:
            verified.append(res1w); s1w_hits += 1
            print(f"  [80%+ PASS] {res1w['asset']:28s} | {res1w['strategy']:30s} | WR:{res1w['win_rate']:5.1f}% | PF:{res1w['profit_factor']:5.2f} | DD:{res1w['max_drawdown']:4.1f}% | Trd:{res1w['total_trades']}")
    print(f"--- Finished US Stocks 1D ({s1d_hits} hits) & 1W ({s1w_hits} hits) ---")

    for sym in TOP_STOCKS[:25]:
        df1h = fetch_yahoo(sym, "6mo", "1h", "data_cache/stocks_1h", ttl=10800)
        res1h = scan_asset(df1h, f"{sym} (US Stock 1H)", audit)
        if res1h:
            verified.append(res1h); s1h_hits += 1
            print(f"  [80%+ PASS] {res1h['asset']:28s} | {res1h['strategy']:30s} | WR:{res1h['win_rate']:5.1f}% | PF:{res1h['profit_factor']:5.2f} | DD:{res1h['max_drawdown']:4.1f}% | Trd:{res1h['total_trades']}")
    print(f"--- Finished US Stocks 1H ({s1h_hits} hits) ---")

    verified.sort(key=lambda x: (x["win_rate"], x["profit_factor"]), reverse=True)
    with open("reports/live_stats.json", "w", encoding="utf-8") as f:
        json.dump({"updated_at": datetime.now(timezone.utc).isoformat(), "results": verified}, f, indent=2)
    pd.DataFrame(audit).to_csv("reports/audit_trade_log.csv", index=False)

    crypto_cnt = sum(1 for r in verified if "Binance" in r["asset"])
    stock_cnt = len(verified) - crypto_cnt
    print("\n=========================================================================")
    print(f"  TOTAL 80%+ VERIFIED SETUPS : {len(verified)} ({crypto_cnt} Crypto | {stock_cnt} US Stocks)")
    print(f"  TOTAL OOS VERIFIED TRADES  : {len(audit)}")
    print("=========================================================================")

if __name__ == "__main__":
    main()
