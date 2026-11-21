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

def fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=1000):
    print(f"Fetching Binance Data: {symbol} ({interval})...")
    url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=["timestamp","open","high","low","close","volume","ct","qav","nt","tb","tq","ig"])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df[["timestamp","open","high","low","close","volume"]]

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
    capital, position, entry_price, entry_date, trades, eq = INITIAL_CAPITAL, 0.0, 0.0, None, [], []
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
            entry_price, entry_date, capital = fill_px, bar_date, 0.0
        elif prev_sig == -1 and position > 0:
            fill_px = exec_open * (1 - slippage)
            capital = (position * fill_px) * (1 - FEE_RATE)
            net_pnl = ((fill_px - entry_price) / entry_price) * 100 - (total_cost * 200)
            trades.append(net_pnl)
            AUDIT_TRADES.append({
                "asset": asset_name,
                "strategy": strategy_name,
                "entry_date": entry_date,
                "entry_price": round(entry_price, 4),
                "exit_date": bar_date,
                "exit_price": round(fill_px, 4),
                "net_pnl_pct": round(net_pnl, 2)
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

def send_discord_report(stats, ai_verdict):
    if not DISCORD_WEBHOOK_URL or "YAHAN-PASTE-KARO" in DISCORD_WEBHOOK_URL:
        print(f"Discord Webhook not set in .env - skipping Discord for {stats['asset']}.")
        return

    is_stock = "US Stock" in stats.get("asset", "")
    is_weekend = datetime.now(timezone.utc).weekday() >= 5
    raw_status = stats.get("status", "FLAT / STANDBY")
    tier = stats.get("tier", "TIER 1 (80%+ ELITE)" if float(stats.get("win_rate", 0)) >= 80.0 else "TIER 2 (75%-79%)")

    if is_stock and is_weekend:
        action_header = "MARKET CLOSED (WEEKEND) - DO NOT TRADE"
        color = 0x6B7280
        trade_plan = f"**Last Close ({stats.get('candle_date', 'N/A')}):** `${stats['price']:,.2f}`\n**Action:** US Stock Market is closed for the weekend. Wait for Monday open."
    elif "BUY" in raw_status or "LONG" in raw_status:
        action_header = "ACTIVE BUY / LONG SIGNAL (OVERSOLD DIP)"
        color = 0x00E5A0
        trade_plan = (
            f"**Action:** TAKE TRADE (Next Bar Open)\n"
            f"**Entry / Close:** `${stats['price']:,.2f}` | **Stop Loss:** `${stats.get('stop_loss', 0):,.2f}` | **Take Profit:** `${stats.get('take_profit', 0):,.2f}`"
        )
    elif "EXIT" in raw_status or "SELL" in raw_status:
        action_header = "EXIT / TAKE PROFIT SIGNAL"
        color = 0xF59E0B
        trade_plan = f"**Last Close ({stats.get('candle_date', 'N/A')}):** `${stats['price']:,.2f}`\n**Action:** Close any open long position. Do NOT open a new buy."
    else:
        action_header = "NO TRADE - WAIT FOR OVERSOLD DIP"
        color = 0x3B82F6
        trade_plan = f"**Last Close ({stats.get('candle_date', 'N/A')}):** `${stats['price']:,.2f}`\n**Action:** No oversold setup today. Stay flat and preserve capital."

    payload = {
        "username": "Future Admiral AI-Quant Desk",
        "embeds": [{
            "title": f"[{tier}] {stats['asset']}",
            "description": (
                f"**Strategy:** `{stats['strategy']}`\n"
                f"**Status:** `{action_header}`\n"
                f"{trade_plan}"
            ),
            "color": color,
            "fields": [
                {"name": "Net Return", "value": f"**{stats['net_return']}%**", "inline": True},
                {"name": "Win Rate", "value": f"**{stats['win_rate']}%**", "inline": True},
                {"name": "Profit Factor", "value": f"**{stats['profit_factor']}**", "inline": True},
                {"name": "Max Drawdown", "value": f"**-{stats['max_drawdown']}%**", "inline": True},
                {"name": "Trades", "value": f"**{stats['total_trades']}**", "inline": True},
                {"name": "Max Risk/Trade", "value": f"**${stats['max_risk_usd']}**", "inline": True},
            ],
            "footer": {"text": "Future Admiral Institutional AI-Quant Engine v2.0"},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }]
    }
    res = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=15)
    print("Discord sent!" if res.status_code in [200, 204] else f"Discord Error: {res.status_code}")

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
        "CRM","ORCL","ADBE","SPY","QQQ"
    ]
    assets = [(f"{c.replace('USDT','/USDT')} (Binance 1D)", lambda sym=c: fetch_binance_data(sym, "1d", 1000)) for c in TOP_CRYPTO] + \
             [(f"{s} (US Stock 1D)", lambda sym=s: fetch_stock_data(sym, "3y", "1d")) for s in TOP_STOCKS]
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
