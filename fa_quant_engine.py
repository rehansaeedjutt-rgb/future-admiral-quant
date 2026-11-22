from ai_committee import run_dual_ai_committee
import math
import os, json, requests, yfinance as yf, pandas as pd, numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

from strategies.vault import add_indicators, STRATEGY_VAULT
from ai_models.ml_filter import apply_ml_filter
from ai_models.llm_analyst import get_ai_verdict

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
AUDIT_TRADES = []
INITIAL_CAPITAL     = float(os.getenv("ACCOUNT_EQUITY", 10000.0))
MAX_RISK_PCT        = float(os.getenv("MAX_RISK_PER_TRADE", 1.0))
FEE_RATE            = 0.001

_HTTP_SESSION = requests.Session()

def fetch_binance_data(symbol, interval="1d", total_candles=2200):
    print(f"Fetching Binance Data: {symbol} ({interval}) [6Y]...")
    url = "https://data-api.binance.vision/api/v3/klines"
    all_rows = []
    end_time = None
    while len(all_rows) < total_candles:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time is not None:
            params["endTime"] = end_time
        data = None
        for attempt in range(3):
            try:
                resp = _HTTP_SESSION.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    break
            except Exception:
                pass
        if not isinstance(data, list) or len(data) == 0:
            break
        all_rows = data + all_rows
        end_time = data[0][0] - 1
        if len(data) < 1000:
            break
    if not all_rows:
        raise ValueError(f"No Binance data returned for {symbol}")
    all_rows = all_rows[-total_candles:]
    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "taker_base", "taker_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_stock_data(ticker="NVDA", period="6mo", interval="1h"):
    print(f"Fetching US Stock Data: {ticker} ({interval})...")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    if "datetime" in df.columns:
        df.rename(columns={"datetime": "timestamp"}, inplace=True)
    elif "date" in df.columns:
        df.rename(columns={"date": "timestamp"}, inplace=True)
    return df[["timestamp","open","high","low","close","volume"]].dropna()

def run_backtest(df, signals, asset_name, strategy_name):
    if "4H" in asset_name and len(df) >= 1200:
        signals = signals.copy()
        ema_macro = df["close"].ewm(span=1200, adjust=False).mean()
        signals[(signals == 1) & (df["close"] <= ema_macro)] = 0
    capital, position, entry_price, entry_date, signal_date, raw_entry_open, trades, eq = INITIAL_CAPITAL, 0.0, 0.0, None, None, 0.0, [], []
    slippage = 0.0005  # 0.05% per-side slippage on top of FEE_RATE (0.10%)
    total_cost = FEE_RATE + slippage

    for i in range(1, len(df)):
        exec_open = float(df["open"].iloc[i])
        close_px  = float(df["close"].iloc[i])
        bar_date  = str(df["timestamp"].iloc[i])[:10]
        prev_sig  = int(signals.iloc[i - 1])  # Strict shift(1): zero look-ahead bias

        if prev_sig == 1 and position == 0:
            fill_px = exec_open * (1 + slippage)
            position = (capital * (1 - FEE_RATE)) / fill_px
            signal_date = str(df["timestamp"].iloc[i - 1])[:10]
            raw_entry_open = exec_open
            entry_price, entry_date, capital = fill_px, bar_date, 0.0
        elif prev_sig == -1 and position > 0:
            fill_px = exec_open * (1 - slippage)
            pos_usd = round(position * entry_price, 2)
            capital = (position * fill_px) * (1 - FEE_RATE)
            gross_pnl = ((fill_px - entry_price) / entry_price) * 100
            net_pnl = gross_pnl - (FEE_RATE * 200)
            trades.append(net_pnl)
            AUDIT_TRADES.append({
                "record_type": "BACKTEST_SIMULATION",
                "data_source": "Binance Spot" if "Binance" in asset_name else "Yahoo Finance",
                "asset": asset_name,
                "strategy": strategy_name,
                "signal_date": signal_date,
                "entry_date": entry_date,
                "raw_entry_open": round(raw_entry_open, 4),
                "entry_price_slipped": round(entry_price, 4),
                "exit_date": bar_date,
                "raw_exit_open": round(exec_open, 4),
                "exit_price_slipped": round(fill_px, 4),
                "position_size_usd": pos_usd,
                "fee_roundtrip_pct": round(FEE_RATE * 200, 2),
                "slippage_roundtrip_pct": round(slippage * 200, 2),
                "gross_pnl_after_slippage_pct": round(gross_pnl, 2),
                "net_pnl_pct": round(net_pnl, 2),
                "equity_after_usd": round(capital, 2)
            })
            position = 0.0

        eq.append(capital if position == 0 else position * close_px)

    final_eq = eq[-1] if eq else INITIAL_CAPITAL
    net_ret = ((final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]

    wr = (len(wins) / len(trades) * 100) if trades else 0.0
    pf = (sum(wins) / abs(sum(losses))) if losses else (9.99 if wins else 0.0)

    eq_s = pd.Series(eq)
    dd = ((eq_s - eq_s.cummax()) / eq_s.cummax() * 100).min()
    max_dd = abs(dd) if not np.isnan(dd) else 0.0

    # Live Actionable Levels (ATR-14 based SL & TP + Exact Candle Date)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1]) if len(df) >= 14 else float(df["close"].iloc[-1] * 0.02)
    last_px = float(df["close"].iloc[-1])
    last_date = str(df["timestamp"].iloc[-1])[:10]
    last_sig = int(signals.iloc[-1])
    status = "BUY / LONG" if last_sig == 1 else ("EXIT / SELL" if last_sig == -1 else "FLAT / STANDBY")
    sl_px = round(last_px - (1.5 * atr), 4)
    tp_px = round(last_px + (2.5 * atr), 4)

    return {
        "asset": asset_name,
        "strategy": strategy_name,
        "price": round(float(df["close"].iloc[-1]), 2),
        "status": status, "candle_date": last_date, "stop_loss": sl_px, "take_profit": tp_px,
        "total_trades": len(trades),
        "win_rate": round(wr, 2),
        "profit_factor": round(pf, 2),
        "max_drawdown": round(max_dd, 2),
        "net_return": round(net_ret, 2),
        "final_equity": round(final_eq, 2),
        "max_risk_usd": round(INITIAL_CAPITAL * (MAX_RISK_PCT / 100.0), 2)
    }

import json, os

_SENT_FILE = "reports/sent_signals.json"

_FORWARD_CSV = "reports/live_forward_trades.csv"
_FORWARD_COLS = [
    "trade_id", "asset", "strategy", "status", "entry_date", "entry_price",
    "stop_loss", "take_profit", "exit_date", "exit_price", "net_pnl_pct", "result"
]

def _record_forward_trade(stats, candle_date):
    import csv
    os.makedirs("reports", exist_ok=True)
    rows = []
    if os.path.exists(_FORWARD_CSV):
        with open(_FORWARD_CSV, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    asset = stats.get("asset", "")
    action = stats.get("status", "")
    price = float(stats.get("price", 0))
    changed = False

    if action == "BUY / LONG":
        already_open = any(r["asset"] == asset and r["status"] == "OPEN" for r in rows)
        if not already_open:
            rows.append({
                "trade_id": f"FA-{len(rows)+1:04d}",
                "asset": asset,
                "strategy": stats.get("strategy", ""),
                "status": "OPEN",
                "entry_date": candle_date,
                "entry_price": round(price, 4),
                "stop_loss": round(float(stats.get("stop_loss", 0)), 4),
                "take_profit": round(float(stats.get("take_profit", 0)), 4),
                "exit_date": "",
                "exit_price": "",
                "net_pnl_pct": "",
                "result": "OPEN"
            })
            changed = True
    elif action == "EXIT / SELL":
        for r in reversed(rows):
            if r["asset"] == asset and r["status"] == "OPEN":
                entry_px = float(r["entry_price"])
                net_pnl = round(((price - entry_px) / entry_px) * 100 - 0.30, 2) if entry_px > 0 else 0.0
                r["status"] = "CLOSED"
                r["exit_date"] = candle_date
                r["exit_price"] = round(price, 4)
                r["net_pnl_pct"] = net_pnl
                r["result"] = "WIN" if net_pnl > 0 else "LOSS"
                changed = True
                break

    if changed or not os.path.exists(_FORWARD_CSV):
        with open(_FORWARD_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_FORWARD_COLS)
            w.writeheader()
            w.writerows(rows)


def _load_sent():
    if os.path.exists(_SENT_FILE):
        try:
            return json.load(open(_SENT_FILE, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_sent(data):
    os.makedirs("reports", exist_ok=True)
    with open(_SENT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def send_discord_report(stats, ai_verdict):
    if not DISCORD_WEBHOOK_URL or "YAHAN-PASTE-KARO" in DISCORD_WEBHOOK_URL:
        return

    sent = _load_sent()
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    asset = stats.get("asset", "")
    is_stock = "US Stock" in asset
    is_weekend = datetime.now(timezone.utc).weekday() >= 5
    raw_status = stats.get("status", "FLAT / STANDBY")

    # 1. Weekend US Stocks: send ONCE per weekend day
    if is_stock and is_weekend:
        key = f"WEEKEND_{today_utc}"
        if sent.get(key):
            return
        payload = {
            "username": "Future Admiral Signals",
            "embeds": [{
                "title": "US Stock Market Closed (Weekend)",
                "description": "US Stock Market is closed today. No US stock trades will be issued until Monday open. Crypto signals remain active 24/7.",
                "color": 0x6B7280
            }]
        }
        try:
            requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
            sent[key] = True
            _save_sent(sent)
        except Exception as e:
            print(f"Discord error: {e}")
        return

    # 2. Live BUY or EXIT signals: send ONCE per asset + status + candle date
    candle_date = str(stats.get("candle_date", today_utc))
    if "BUY" in raw_status or "LONG" in raw_status:
        sig_type = "BUY"
        title = f"BUY SIGNAL: {asset}"
        _record_forward_trade(stats, candle_date)
        color = 0x00E5A0
        desc = (
            f"**Action:** `BUY / LONG`\n"
            f"**Entry Price:** `${stats['price']:,.2f}`\n"
            f"**Stop Loss:** `${stats.get('stop_loss', 0):,.2f}`\n"
            f"**Take Profit:** `${stats.get('take_profit', 0):,.2f}`\n"
            f"**Candle Date:** `{candle_date}`"
        )
    elif "EXIT" in raw_status or "SELL" in raw_status:
        sig_type = "EXIT"
        title = f"EXIT / TAKE PROFIT: {asset}"
        _record_forward_trade(stats, candle_date)
        color = 0xF59E0B
        desc = (
            f"**Action:** `CLOSE LONG POSITION`\n"
            f"**Exit Price:** `${stats['price']:,.2f}`\n"
            f"**Candle Date:** `{candle_date}`"
        )
    else:
        return

    sig_key = f"{asset}|{sig_type}|{candle_date}"
    if sent.get(sig_key):
        return

    payload = {
        "username": "Future Admiral Signals",
        "embeds": [{
            "title": title,
            "description": desc,
            "color": color,
            "footer": {"text": "Future Admiral Live Signals"}
        }]
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        sent[sig_key] = True
        _save_sent(sent)
    except Exception as e:
        print(f"Discord error: {e}")


def evaluate_asset(df, asset_name):
    df = add_indicators(df)
    best_stats, best_score = None, -999999
    for name, fn in STRATEGY_VAULT.items():
        ml_sig = apply_ml_filter(df, fn(df))
        stats = run_backtest(df, ml_sig, asset_name, name)
        score = ((stats["win_rate"] * 0.6) + (stats["profit_factor"] * 12) + stats["net_return"] - (stats["max_drawdown"] * 1.2)) if (stats["total_trades"] >= 15 and stats["profit_factor"] >= 1.0) else -99999
        if score > best_score:
            best_score, best_stats = score, stats
    verdict = get_ai_verdict(asset_name, best_stats["strategy"], best_stats)
    best_stats["ai_verdict"] = verdict
    best_stats["updated_at"] = datetime.now(timezone.utc).isoformat()
    return best_stats

if __name__ == "__main__":
    print("\nSTARTING FUTURE ADMIRAL AI-QUANT ENGINE v2.0...\n")
    results = []
    TOP_CRYPTO = [
        "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","DOTUSDT","TRXUSDT",
        "TONUSDT","SUIUSDT","LTCUSDT","BCHUSDT","NEARUSDT","APTUSDT","ICPUSDT","UNIUSDT","ETCUSDT","XLMUSDT",
        "HBARUSDT","FILUSDT","ATOMUSDT","ARBUSDT","OPUSDT","INJUSDT","IMXUSDT","STXUSDT","RENDERUSDT","FETUSDT",
        "TAOUSDT","AAVEUSDT","MKRUSDT","GRTUSDT","VETUSDT","ALGOUSDT","FTMUSDT","SEIUSDT","TIAUSDT","RUNEUSDT",
        "THETAUSDT","FLOWUSDT","EGLDUSDT","AXSUSDT","SANDUSDT","MANAUSDT","GALAUSDT","CHZUSDT","QNTUSDT","KAVAUSDT"
    ]
    TOP_STOCKS = [
        "NVDA","AAPL","MSFT","GOOGL","AMZN","META","TSLA","AVGO","AMD","NFLX",
        "JPM","V","MA","UNH","LLY","XOM","COST","WMT","HD","PG",
        "CRM","ORCL","ADBE","SPY","QQQ", "BRK-B", "JNJ", "ABBV", "MRK", "PEP", "KO", "MCD", "DIS", "CSCO", "IBM", "TXN", "QCOM", "INTU", "NOW", "GS", "MS", "BLK", "CAT", "DE", "HON", "UNP", "LMT", "RTX", "LOW", "TJX", "NKE", "SBUX", "CVX", "COP", "NEE", "IWM", "DIA", "XLK", "XLF", "XLV", "XLE", "SMH"
    ]
    print("Batch-downloading 6Y data for all US Stocks in one request...")
    _STOCK_BATCH = yf.download(TOP_STOCKS, period="6y", interval="1d", group_by="ticker", threads=True, progress=False)

    def _get_batch_stock(sym):
        df = _STOCK_BATCH[sym].dropna(how="all").reset_index()
        if df.empty:
            df = yf.download(sym, period="6y", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(how="all").reset_index()
        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    TOP_CRYPTO_4H = TOP_CRYPTO[:10]
    assets = [(f"{c.replace('USDT','/USDT')} (Binance 1D)", lambda sym=c: fetch_binance_data(sym, "1d", 2200)) for c in TOP_CRYPTO] + \
             [(f"{c.replace('USDT','/USDT')} (Binance 4H)", lambda sym=c: fetch_binance_data(sym, "4h", 3000)) for c in TOP_CRYPTO_4H] + \
             [(f"{s} (US Stock 1D)", lambda sym=s: _get_batch_stock(sym)) for s in TOP_STOCKS]
    for label, fetcher in assets:
        try:
            winner = evaluate_asset(fetcher(), label)
            results.append(winner)
            print(f"WINNER {label}: {winner['strategy']} | Ret: {winner['net_return']}% | WR: {winner['win_rate']}% | PF: {winner['profit_factor']}")
            # send_discord_report(winner, winner["ai_verdict"])
        except Exception as e:
            print(f"Error on {label}: {e}")
    # CLEAN FILTER & TOP-5 DISCORD DISPATCH
    results = [r for r in results if r.get("total_trades", 0) >= 15 and r.get("net_return", 0) > 0 and r.get("profit_factor", 0) >= 1.5 and r.get("max_drawdown", 100) <= 25.0 and r.get("win_rate", 0) >= 75.0 and (sum(math.comb(int(r["total_trades"]), i) for i in range(round(int(r["total_trades"]) * (r["win_rate"]/100.0)), int(r["total_trades"]) + 1)) * (0.5 ** int(r["total_trades"])) * 100) < 5.0]
    for r in results:
        r["tier"] = "TIER 1 (80%+ ELITE)" if float(r.get("win_rate", 0)) >= 80.0 else "TIER 2 (75%-79%)"
    results.sort(key=lambda x: (x["win_rate"], x["profit_factor"]), reverse=True)
    for top in results[:5]:
        try:
            send_discord_report(top["asset"], top)
        except Exception:
            try:
                send_discord_report(top, top.get("ai_verdict", ""))
            except Exception:
                pass

    os.makedirs("reports", exist_ok=True)
    with open("reports/live_stats.json", "w", encoding="utf-8") as f:
        json.dump({"brand": "Future Admiral", "results": results}, f, indent=2)
    print("\nWebsite JSON saved to reports/live_stats.json")
    verified_pairs = {(r["asset"], r["strategy"]) for r in results}
    verified_trades = [t for t in AUDIT_TRADES if (t["asset"], t["strategy"]) in verified_pairs]
    pd.DataFrame(verified_trades).to_csv("reports/audit_trade_log.csv", index=False)
    print(f"Audit Trade Log saved to reports/audit_trade_log.csv ({len(verified_trades)} verified trades)")
