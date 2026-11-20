import os, json, requests, yfinance as yf, pandas as pd, numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

from strategies.vault import add_indicators, STRATEGY_VAULT
from ai_models.ml_filter import apply_ml_filter
from ai_models.llm_analyst import get_ai_verdict

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
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
    capital, position, entry_price, trades, eq = INITIAL_CAPITAL, 0.0, 0.0, [], []
    for i in range(len(df)):
        price, sig = float(df["close"].iloc[i]), int(signals.iloc[i])
        if sig == 1 and position == 0:
            position = (capital * (1 - FEE_RATE)) / price
            entry_price, capital = price, 0.0
        elif sig == -1 and position > 0:
            capital = (position * price) * (1 - FEE_RATE)
            trades.append(((price - entry_price) / entry_price) * 100 - (FEE_RATE * 200))
            position = 0.0
        eq.append(capital if position == 0 else position * price)

    final_eq = eq[-1] if eq else INITIAL_CAPITAL
    net_ret = ((final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]

    wr = (len(wins) / len(trades) * 100) if trades else 0.0
    pf = (sum(wins) / abs(sum(losses))) if losses else (9.99 if wins else 0.0)

    eq_s = pd.Series(eq)
    dd = ((eq_s - eq_s.cummax()) / eq_s.cummax() * 100).min()
    max_dd = abs(dd) if not np.isnan(dd) else 0.0
    status = "LONG / BULLISH" if int(signals.iloc[-1]) == 1 else "FLAT / BEARISH"

    return {
        "asset": asset_name,
        "strategy": strategy_name,
        "price": round(float(df["close"].iloc[-1]), 2),
        "status": status,
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
    color = 0x00E5A0 if stats["net_return"] >= 0 else 0xFF3B69
    payload = {
        "username": "Future Admiral AI-Quant Desk",
        "embeds": [{
            "title": f"TOP QUANT STRATEGY - {stats['asset']}",
            "description": f"**Winner:** `{stats['strategy']}`\n**Signal:** {stats['status']} | **Price:** `${stats['price']:,.2f}`\n\n**AI Analyst Verdict:**\n> {ai_verdict}",
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
        score = stats["net_return"] + (stats["profit_factor"] * 5) - (stats["max_drawdown"] * 1.5)
        if score > best_score:
            best_score, best_stats = score, stats
    verdict = get_ai_verdict(asset_name, best_stats["strategy"], best_stats)
    best_stats["ai_verdict"] = verdict
    best_stats["updated_at"] = datetime.now(timezone.utc).isoformat()
    return best_stats

if __name__ == "__main__":
    print("\nSTARTING FUTURE ADMIRAL AI-QUANT ENGINE v2.0...\n")
    results = []
    assets = [
        ("BTC/USDT (Binance 1H)", lambda: fetch_binance_data("BTCUSDT", "1h", 1000)),
        ("SOL/USDT (Binance 1H)", lambda: fetch_binance_data("SOLUSDT", "1h", 1000)),
        ("NVDA (US Stock 1H)",    lambda: fetch_stock_data("NVDA", "6mo", "1h")),
        ("AAPL (US Stock 1H)",    lambda: fetch_stock_data("AAPL", "6mo", "1h")),
    ]
    for label, fetcher in assets:
        try:
            winner = evaluate_asset(fetcher(), label)
            results.append(winner)
            print(f"WINNER {label}: {winner['strategy']} | Ret: {winner['net_return']}% | WR: {winner['win_rate']}% | PF: {winner['profit_factor']}")
            send_discord_report(winner, winner["ai_verdict"])
        except Exception as e:
            print(f"Error on {label}: {e}")
    os.makedirs("reports", exist_ok=True)
    with open("reports/live_stats.json", "w", encoding="utf-8") as f:
        json.dump({"brand": "Future Admiral", "results": results}, f, indent=2)
    print("\nWebsite JSON saved to reports/live_stats.json")