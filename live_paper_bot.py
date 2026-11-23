import os, json
from datetime import datetime, timezone
import pandas as pd

LIVE_JSON = "reports/live_stats.json"
WALLET_JSON = "reports/live_paper_wallet.json"
TRADES_CSV = "reports/live_paper_trades.csv"
START_BALANCE = 10.00
FEE_SLIPPAGE = 0.0015  # 0.10% exchange fee + 0.05% slippage per side

def load_wallet():
    if os.path.exists(WALLET_JSON):
        return json.load(open(WALLET_JSON, encoding="utf-8"))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "starting_balance_usd": START_BALANCE,
        "cash_usd": START_BALANCE,
        "open_position": None,
        "closed_trades": 0,
        "wins": 0,
        "realized_pnl_usd": 0.0,
    }

def step_live_paper_bot():
    if not os.path.exists(LIVE_JSON):
        print("[LIVE $10 PAPER-BOT] No live_stats.json found.")
        return

    data = json.load(open(LIVE_JSON, encoding="utf-8"))
    results = {r["asset"]: r for r in data.get("results", [])}
    wallet = load_wallet()
    now_iso = datetime.now(timezone.utc).isoformat()
    action_log = "HOLD (all 19 verified assets on standby)"

    # 1. Manage open position (exit on status flip, Stop-Loss, or Take-Profit)
    pos = wallet.get("open_position")
    if pos:
        asset_data = results.get(pos["asset"])
        if asset_data:
            status = str(asset_data.get("status", "FLAT / STANDBY")).upper()
            price = float(asset_data.get("price") or pos["entry_price"])
            sl = float(pos.get("stop_loss") or 0)
            tp = float(pos.get("take_profit") or 0)

            exit_reason = None
            if sl > 0 and price <= sl:
                exit_reason = "STOP_LOSS"
            elif tp > 0 and price >= tp:
                exit_reason = "TAKE_PROFIT"
            elif any(k in status for k in ("SELL", "EXIT", "FLAT", "STANDBY")):
                exit_reason = "SIGNAL_EXIT"

            if exit_reason:
                exit_px = price * (1.0 - FEE_SLIPPAGE)
                proceeds = pos["qty"] * exit_px
                pnl_usd = proceeds - pos["allocated_usd"]
                pnl_pct = (pnl_usd / pos["allocated_usd"]) * 100.0
                wallet["cash_usd"] = round(wallet["cash_usd"] + proceeds, 4)
                wallet["closed_trades"] += 1
                if pnl_usd > 0:
                    wallet["wins"] += 1
                wallet["realized_pnl_usd"] = round(wallet["realized_pnl_usd"] + pnl_usd, 4)
                wallet["open_position"] = None

                trade_row = pd.DataFrame([{
                    "asset": pos["asset"],
                    "strategy": pos["strategy"],
                    "exit_reason": exit_reason,
                    "entry_time": pos["entry_time"],
                    "exit_time": now_iso,
                    "entry_price": round(pos["entry_price"], 6),
                    "exit_price": round(exit_px, 6),
                    "allocated_usd": round(pos["allocated_usd"], 4),
                    "pnl_usd": round(pnl_usd, 4),
                    "pnl_pct": round(pnl_pct, 2),
                    "wallet_after_usd": wallet["cash_usd"],
                }])
                header = not os.path.exists(TRADES_CSV)
                trade_row.to_csv(TRADES_CSV, mode="a", header=header, index=False)
                action_log = f"CLOSED {pos['asset']} ({exit_reason}) | PnL: ${pnl_usd:+.4f} ({pnl_pct:+.2f}%) | Cash: ${wallet['cash_usd']:.2f}"

    # 2. If flat, enter highest-WR live BUY/LONG signal
    if wallet["open_position"] is None and wallet["cash_usd"] >= 5.0:
        buy_candidates = []
        for r in data.get("results", []):
            st = str(r.get("status", "")).upper()
            px = float(r.get("price") or 0)
            if px > 0 and any(k in st for k in ("BUY", "LONG")) and "FLAT" not in st and "STANDBY" not in st:
                buy_candidates.append(r)

        if buy_candidates:
            best = sorted(buy_candidates, key=lambda x: (x.get("win_rate", 0), x.get("profit_factor", 0)), reverse=True)[0]
            entry_px = float(best["price"]) * (1.0 + FEE_SLIPPAGE)
            alloc = wallet["cash_usd"]
            qty = alloc / entry_px
            wallet["cash_usd"] = 0.0
            wallet["open_position"] = {
                "asset": best["asset"],
                "strategy": best["strategy"],
                "entry_time": now_iso,
                "candle_date": best.get("candle_date", ""),
                "entry_price": round(entry_px, 6),
                "stop_loss": float(best.get("stop_loss") or 0),
                "take_profit": float(best.get("take_profit") or 0),
                "allocated_usd": round(alloc, 4),
                "qty": qty,
            }
            action_log = f"OPENED LONG {best['asset']} @ ${entry_px:.4f} (${alloc:.2f}) | SL: ${best.get('stop_loss', 0):.4f} | TP: ${best.get('take_profit', 0):.4f}"

    wallet["updated_at"] = now_iso
    with open(WALLET_JSON, "w", encoding="utf-8") as f:
        json.dump(wallet, f, indent=2)

    eq = wallet["cash_usd"]
    if wallet["open_position"]:
        p = wallet["open_position"]
        cur_px = float(results.get(p["asset"], {}).get("price") or p["entry_price"])
        eq += p["qty"] * cur_px

    print(f"\n[LIVE $10 PAPER-BOT] {action_log}")
    print(f"[LIVE $10 PAPER-BOT] Total Live Equity: ${eq:.2f} | Closed Trades: {wallet['closed_trades']} | Realized PnL: ${wallet['realized_pnl_usd']:+.4f}")

if __name__ == "__main__":
    step_live_paper_bot()
