import os, sys, json, requests
from datetime import datetime, timezone
import pandas as pd
from ai_models.news_shield import evaluate_trade_safety

LIVE_JSON = "reports/live_stats.json"
DUAL_WALLET_JSON = "reports/live_dual_wallet.json"
LEGACY_WALLET_JSON = "reports/live_paper_wallet.json"
TRADES_CSV = "reports/live_paper_trades.csv"
PROOF_MD = "reports/student_proof_summary.md"

FEE_SLIPPAGE = 0.0015       # 0.10% exchange fee + 0.05% slippage per side
MAX_SL_PCT = 0.045          # Hard cap: never risk more than 4.5% on any single trade
BREAKEVEN_TRIGGER = 0.015   # At +1.5% unrealized profit, lock SL at Entry + 0.2%
CIRCUIT_BREAKER_DD = 0.15   # Pause new entries if equity drops 15% from peak
MAX_HOLD_DAYS = 7           # Mean-reversion time-stop

HTTP = requests.Session()
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0.0.0"}

def fetch_live_price(asset_name, fallback_price):
    try:
        if "Binance" in asset_name:
            sym = asset_name.split(" ")[0].replace("/", "")
            r = HTTP.get(f"https://api.binance.com/api/v3/ticker/price?symbol={sym}", timeout=5)
            if r.status_code == 200:
                return float(r.json()["price"])
        else:
            sym = asset_name.split(" ")[0]
            r = HTTP.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                         params={"range": "1d", "interval": "5m"}, headers=HDRS, timeout=5)
            if r.status_code == 200:
                meta = (r.json().get("chart", {}).get("result") or [{}])[0].get("meta", {})
                if meta.get("regularMarketPrice"):
                    return float(meta["regularMarketPrice"])
    except Exception:
        pass
    return float(fallback_price)

def _new_tier(name, start_bal, max_slots):
    now_iso = datetime.now(timezone.utc).isoformat()
    min_wr = 80.0 if start_bal <= 10.0 else 70.0
    return {
        "tier_name": name,
        "min_win_rate": min_wr,
        "created_at": now_iso,
        "updated_at": now_iso,
        "starting_balance_usd": start_bal,
        "cash_usd": start_bal,
        "peak_equity_usd": start_bal,
        "circuit_breaker_tripped": False,
        "max_slots": max_slots,
        "open_positions": [],
        "closed_trades": 0,
        "wins": 0,
        "realized_pnl_usd": 0.0,
    }

def load_dual_wallet():
    if os.path.exists(DUAL_WALLET_JSON):
        return json.load(open(DUAL_WALLET_JSON, encoding="utf-8"))
    return {
        "student_10usd": _new_tier("Student Tier ($10 Start | 1 Slot)", 10.00, 1),
        "pro_100usd": _new_tier("Pro Tier ($100 Start | 4 Slots)", 100.00, 4),
    }

def _calc_equity(tier, price_map):
    eq = float(tier["cash_usd"])
    for p in tier["open_positions"]:
        px = price_map.get(p["asset"], p["entry_price"])
        eq += p["qty"] * px
    return round(eq, 4)

def _process_tier(tier_key, tier, results, price_map, allow_entries, now_dt):
    now_iso = now_dt.isoformat()
    logs = []
    remaining = []

    # 1. Manage Open Positions (Hard SL, TP, Breakeven Lock, Signal Exit, Time-Stop)
    for pos in tier["open_positions"]:
        asset = pos["asset"]
        cur_px = price_map.get(asset, pos["entry_price"])
        asset_data = results.get(asset, {})
        status = str(asset_data.get("status", "")).upper()

        # Breakeven Trailing Stop Protection
        unrealized_pct = (cur_px - pos["entry_price"]) / pos["entry_price"]
        if unrealized_pct >= BREAKEVEN_TRIGGER and not pos.get("breakeven_locked", False):
            be_sl = round(pos["entry_price"] * 1.002, 6)
            if be_sl > pos["stop_loss"]:
                pos["stop_loss"] = be_sl
                pos["breakeven_locked"] = True
                logs.append(f"[{tier_key}] BREAKEVEN LOCKED {asset} SL -> ${be_sl:.4f}")

        held_days = (now_dt - datetime.fromisoformat(pos["entry_time"])).total_seconds() / 86400.0
        exit_reason = None
        if pos["stop_loss"] > 0 and cur_px <= pos["stop_loss"]:
            exit_reason = "BREAKEVEN_STOP" if pos.get("breakeven_locked") else "STOP_LOSS"
        elif pos["take_profit"] > 0 and cur_px >= pos["take_profit"]:
            exit_reason = "TAKE_PROFIT"
        elif held_days >= MAX_HOLD_DAYS:
            exit_reason = "TIME_STOP_7D"
        elif allow_entries and any(k in status for k in ("SELL", "EXIT")):
            exit_reason = "SIGNAL_EXIT"

        if exit_reason:
            exit_px = cur_px * (1.0 - FEE_SLIPPAGE)
            proceeds = pos["qty"] * exit_px
            pnl_usd = proceeds - pos["allocated_usd"]
            pnl_pct = (pnl_usd / pos["allocated_usd"]) * 100.0
            tier["cash_usd"] = round(tier["cash_usd"] + proceeds, 4)
            tier["closed_trades"] += 1
            if pnl_usd > 0:
                tier["wins"] += 1
            tier["realized_pnl_usd"] = round(tier["realized_pnl_usd"] + pnl_usd, 4)

            row = pd.DataFrame([{
                "tier": tier_key,
                "asset": asset,
                "strategy": pos["strategy"],
                "exit_reason": exit_reason,
                "entry_time": pos["entry_time"],
                "exit_time": now_iso,
                "entry_price": round(pos["entry_price"], 6),
                "exit_price": round(exit_px, 6),
                "allocated_usd": round(pos["allocated_usd"], 4),
                "pnl_usd": round(pnl_usd, 4),
                "pnl_pct": round(pnl_pct, 2),
                "cash_after_usd": tier["cash_usd"],
            }])
            row.to_csv(TRADES_CSV, mode="a", header=not os.path.exists(TRADES_CSV), index=False)
            logs.append(f"[{tier_key}] CLOSED {asset} ({exit_reason}) | PnL: ${pnl_usd:+.4f} ({pnl_pct:+.2f}%)")
        else:
            remaining.append(pos)

    tier["open_positions"] = remaining

    # Update Peak Equity & Circuit Breaker
    eq_now = _calc_equity(tier, price_map)
    if eq_now > tier["peak_equity_usd"]:
        tier["peak_equity_usd"] = eq_now
    dd_pct = (tier["peak_equity_usd"] - eq_now) / tier["peak_equity_usd"] if tier["peak_equity_usd"] > 0 else 0.0
    if dd_pct >= CIRCUIT_BREAKER_DD:
        tier["circuit_breaker_tripped"] = True
        logs.append(f"[{tier_key}] CIRCUIT BREAKER TRIPPED (DD {dd_pct*100:.1f}% >= 15%) — new entries paused.")

    # 2. Open New Verified BUY Signals (only during full scan, never duplicate asset)
    if allow_entries and not tier["circuit_breaker_tripped"]:
        open_assets = {p["asset"] for p in tier["open_positions"]}
        free_slots = tier["max_slots"] - len(tier["open_positions"])
        if free_slots > 0 and tier["cash_usd"] >= 5.0:
            buys = [
                r for r in results.values()
                if float(r.get("price") or 0) > 0 and float(r.get("win_rate") or 0) >= float(tier.get("min_win_rate", 70.0))
                and r["asset"] not in open_assets
                and any(k in str(r.get("status", "")).upper() for k in ("BUY", "LONG"))
                and "FLAT" not in str(r.get("status", "")).upper()
                and "STANDBY" not in str(r.get("status", "")).upper()
            ]
            buys.sort(key=lambda x: (x.get("win_rate", 0), x.get("profit_factor", 0)), reverse=True)

            for sig in buys[:free_slots]:
                free_now = tier["max_slots"] - len(tier["open_positions"])
                alloc = round(tier["cash_usd"] / free_now, 4)
                if alloc < 5.0:
                    break
                shield = evaluate_trade_safety(sig["asset"])
                if not shield["allow"]:
                    logs.append(f"[{tier_key}] {shield['verdict']} {sig['asset']}: {shield['reason']}")
                    continue
                raw_px = fetch_live_price(sig["asset"], sig["price"])
                entry_px = raw_px * (1.0 + FEE_SLIPPAGE)

                # Hard-capped Stop-Loss (never risk > 4.5% on any single trade)
                engine_sl = float(sig.get("stop_loss") or (entry_px * (1.0 - MAX_SL_PCT)))
                hard_floor_sl = entry_px * (1.0 - MAX_SL_PCT)
                safe_sl = round(max(engine_sl, hard_floor_sl), 6)
                safe_tp = round(float(sig.get("take_profit") or (entry_px * 1.05)), 6)

                qty = alloc / entry_px
                tier["cash_usd"] = round(tier["cash_usd"] - alloc, 4)
                tier["open_positions"].append({
                    "asset": sig["asset"],
                    "strategy": sig["strategy"],
                    "entry_time": now_iso,
                    "entry_price": round(entry_px, 6),
                    "stop_loss": safe_sl,
                    "take_profit": safe_tp,
                    "breakeven_locked": False,
                    "allocated_usd": alloc,
                    "qty": qty,
                })
                logs.append(f"[{tier_key}] OPENED LONG {sig['asset']} @ ${entry_px:.4f} (${alloc:.2f}) | SL: ${safe_sl:.4f} | TP: ${safe_tp:.4f}")

    tier["updated_at"] = now_iso
    return logs

def step_live_paper_bot(guardian_only=False):
    if not os.path.exists(LIVE_JSON):
        print("[DUAL-TIER BOT] No reports/live_stats.json found.")
        return

    data = json.load(open(LIVE_JSON, encoding="utf-8"))
    results = {r["asset"]: r for r in data.get("results", [])}
    wallets = load_dual_wallet()
    now_dt = datetime.now(timezone.utc)

    # Fetch real-time prices only for currently open positions (< 3 seconds total)
    open_assets = set()
    for t in wallets.values():
        for p in t["open_positions"]:
            open_assets.add(p["asset"])

    price_map = {}
    for asset in open_assets:
        fallback = results.get(asset, {}).get("price", 0)
        price_map[asset] = fetch_live_price(asset, fallback)

    all_logs = []
    for key in ("student_10usd", "pro_100usd"):
        all_logs.extend(_process_tier(key, wallets[key], results, price_map, allow_entries=not guardian_only, now_dt=now_dt))

    with open(DUAL_WALLET_JSON, "w", encoding="utf-8") as f:
        json.dump(wallets, f, indent=2)

    # Keep legacy single-wallet JSON synced with $10 student tier for backwards compatibility
    st = wallets["student_10usd"]
    legacy = {
        "created_at": st["created_at"],
        "updated_at": st["updated_at"],
        "starting_balance_usd": st["starting_balance_usd"],
        "cash_usd": st["cash_usd"],
        "open_position": st["open_positions"][0] if st["open_positions"] else None,
        "closed_trades": st["closed_trades"],
        "wins": st["wins"],
        "realized_pnl_usd": st["realized_pnl_usd"],
    }
    with open(LEGACY_WALLET_JSON, "w", encoding="utf-8") as f:
        json.dump(legacy, f, indent=2)

    # Write clean Student Proof Summary
    eq10 = _calc_equity(wallets["student_10usd"], price_map)
    eq100 = _calc_equity(wallets["pro_100usd"], price_map)
    md = [
        "# Future Admiral — Live Forward-Test Proof (Original Rules)",
        f"- **Updated (UTC):** `{now_dt.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- **Risk Controls:** Hard Max `-4.5%` SL | Breakeven Lock at `+1.5%` | `15%` DD Circuit Breaker | `7D` Time-Stop",
        "",
        "| Tier | Start | Live Equity | Net Return | Open Trades | Closed Trades | Win Rate | Circuit Breaker |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for key, eq in (("student_10usd", eq10), ("pro_100usd", eq100)):
        t = wallets[key]
        ret = ((eq - t["starting_balance_usd"]) / t["starting_balance_usd"]) * 100.0
        wr = (t["wins"] / t["closed_trades"] * 100.0) if t["closed_trades"] else 0.0
        cb = "PAUSED" if t["circuit_breaker_tripped"] else "ACTIVE"
        md.append(f"| {t['tier_name']} | ${t['starting_balance_usd']:.2f} | **${eq:.2f}** | {ret:+.2f}% | {len(t['open_positions'])}/{t['max_slots']} | {t['closed_trades']} | {wr:.1f}% | {cb} |")
    with open(PROOF_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    mode = "15m RISK GUARDIAN" if guardian_only else "FULL ENGINE SCAN"
    print(f"\n=== [{mode}] DUAL-TIER LIVE BOT STATUS ===")
    for line in all_logs:
        print(" ", line)
    if not all_logs:
        print("  HOLD — All 19 verified assets on standby (0 open trades to adjust).")
    print(f"  $10 Student Vault : ${eq10:.2f} | Open: {len(wallets['student_10usd']['open_positions'])}/1 | Closed: {wallets['student_10usd']['closed_trades']}")
    print(f"  $100 Pro Vault    : ${eq100:.2f} | Open: {len(wallets['pro_100usd']['open_positions'])}/4 | Closed: {wallets['pro_100usd']['closed_trades']}")

if __name__ == "__main__":
    step_live_paper_bot(guardian_only=("--guardian" in sys.argv))
