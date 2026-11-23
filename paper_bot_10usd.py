import os, json
import pandas as pd
import numpy as np

STARTING_BALANCE = 10.00
AUDIT_CSV = "reports/audit_trade_log.csv"
LIVE_JSON = "reports/live_stats.json"
WEEKLY_CSV = "reports/bot_10usd_weekly_report.csv"
BOT_STATE_JSON = "reports/bot_10usd_state.json"

def run_10usd_simulation():
    if not os.path.exists(AUDIT_CSV):
        print("❌ reports/audit_trade_log.csv not found. Run fa_quant_engine.py first.")
        return

    df = pd.read_csv(AUDIT_CSV)
    if df.empty:
        print("❌ Audit trade log is empty.")
        return

    df["entry_dt"] = pd.to_datetime(df["entry_date"])
    df["exit_dt"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values(["entry_dt", "exit_dt"]).reset_index(drop=True)

    # Realistic $10 account execution: 1 position at a time (Binance MIN_NOTIONAL constraint)
    equity = STARTING_BALANCE
    busy_until = pd.Timestamp.min
    executed = []
    skipped = 0

    for _, row in df.iterrows():
        if row["entry_dt"] < busy_until:
            skipped += 1
            continue
        pnl_pct = float(row["net_pnl_pct"]) / 100.0
        pnl_usd = equity * pnl_pct
        equity_before = equity
        equity = max(0.01, equity + pnl_usd)
        busy_until = row["exit_dt"]
        executed.append({
            "asset": row["asset"],
            "strategy": row["strategy"],
            "entry_date": row["entry_date"],
            "exit_date": row["exit_date"],
            "week": row["exit_dt"].to_period("W").start_time.strftime("%Y-%m-%d"),
            "equity_before": round(equity_before, 4),
            "net_pnl_pct": round(float(row["net_pnl_pct"]), 2),
            "pnl_usd": round(pnl_usd, 4),
            "equity_after": round(equity, 4),
        })

    ex_df = pd.DataFrame(executed)
    total_weeks = max(1, int((df["exit_dt"].max() - df["entry_dt"].min()).days / 7))

    # Build full week-by-week ledger (including $0 quiet weeks)
    all_weeks = pd.date_range(
        start=df["entry_dt"].min().to_period("W").start_time,
        end=df["exit_dt"].max().to_period("W").start_time,
        freq="W-MON"
    ).strftime("%Y-%m-%d")

    wk_grp = ex_df.groupby("week").agg(
        trades=("pnl_usd", "count"),
        wins=("pnl_usd", lambda s: int((s > 0).sum())),
        weekly_pnl_usd=("pnl_usd", "sum"),
        end_equity=("equity_after", "last")
    ).reindex(all_weeks)

    wk_grp["trades"] = wk_grp["trades"].fillna(0).astype(int)
    wk_grp["wins"] = wk_grp["wins"].fillna(0).astype(int)
    wk_grp["weekly_pnl_usd"] = wk_grp["weekly_pnl_usd"].fillna(0.0)
    wk_grp["end_equity"] = wk_grp["end_equity"].ffill().fillna(STARTING_BALANCE)
    wk_grp["start_equity"] = wk_grp["end_equity"].shift(1).fillna(STARTING_BALANCE)
    wk_grp["weekly_ret_pct"] = ((wk_grp["end_equity"] - wk_grp["start_equity"]) / wk_grp["start_equity"]) * 100.0
    wk_grp.index.name = "week_start"
    wk_grp.to_csv(WEEKLY_CSV)

    # Stats
    avg_wk_usd = wk_grp["weekly_pnl_usd"].mean()
    avg_wk_pct = wk_grp["weekly_ret_pct"].mean()
    active_wks = wk_grp[wk_grp["trades"] > 0]
    best_wk = wk_grp["weekly_pnl_usd"].max()
    worst_wk = wk_grp["weekly_pnl_usd"].min()
    win_rate = (ex_df["pnl_usd"] > 0).mean() * 100.0
    total_ret_pct = ((equity - STARTING_BALANCE) / STARTING_BALANCE) * 100.0

    # Save persistent shadow-bot state for live background tracking
    live_signals = []
    if os.path.exists(LIVE_JSON):
        live_data = json.load(open(LIVE_JSON, encoding="utf-8"))
        for r in live_data.get("results", []):
            sig = str(r.get("latest_signal", "")).upper()
            if "BUY" in sig or "LONG" in sig:
                live_signals.append({"asset": r["asset"], "strategy": r["strategy"], "signal": sig, "win_rate": r["win_rate"]})

    state = {
        "mode": "PAPER_SHADOW_BOT",
        "starting_balance_usd": STARTING_BALANCE,
        "simulated_final_equity_usd": round(equity, 2),
        "total_oos_return_pct": round(total_ret_pct, 2),
        "executed_trades": len(ex_df),
        "skipped_due_to_10usd_lock": skipped,
        "avg_trades_per_week": round(len(ex_df) / total_weeks, 2),
        "avg_weekly_profit_usd": round(avg_wk_usd, 4),
        "avg_weekly_return_pct": round(avg_wk_pct, 2),
        "active_live_signals_now": live_signals,
    }
    with open(BOT_STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    print("\n=================================================================")
    print("  FUTURE ADMIRAL $10.00 SHADOW-BOT — ORIGINAL OOS REALITY REPORT")
    print("=================================================================")
    print(f"Starting Account Balance   : ${STARTING_BALANCE:.2f}")
    print(f"OOS Test Span              : {total_weeks} weeks ({df['entry_date'].min()} -> {df['exit_date'].max()})")
    print(f"Total Signals Available    : {len(df)} (~{len(df)/total_weeks:.2f} signals/week)")
    print(f"Trades Executed ($10 lock) : {len(ex_df)} (~{len(ex_df)/total_weeks:.2f}/week | {skipped} skipped while in a trade)")
    print(f"Executed Win Rate          : {win_rate:.1f}%")
    print("-----------------------------------------------------------------")
    print(f"Average Profit / Week      : ${avg_wk_usd:+.4f}  ({avg_wk_pct:+.2f}% / week)")
    print(f"Avg Active-Week Profit     : ${active_wks['weekly_pnl_usd'].mean():+.4f}  (in weeks with >=1 trade)")
    print(f"Best Single Week           : ${best_wk:+.4f}")
    print(f"Worst Single Week          : ${worst_wk:+.4f}")
    print(f"Final Equity After {total_weeks:3d} Wks : ${equity:.2f}  ({total_ret_pct:+.2f}% total compounded)")
    print("-----------------------------------------------------------------")
    print("LAST 8 WEEKS (ACTUAL ORIGINAL RESULTS ON $10 BASE):")
    tail = wk_grp.tail(8)
    for wk, r in tail.iterrows():
        print(f"  Week {wk} | Trades: {int(r['trades']):2d} | PnL: ${r['weekly_pnl_usd']:+7.4f} ({r['weekly_ret_pct']:+6.2f}%) | Equity: ${r['end_equity']:.2f}")
    print("=================================================================")
    print(f"Saved weekly ledger -> {WEEKLY_CSV}")
    print(f"Saved bot state     -> {BOT_STATE_JSON}")
    print(f"Active BUY signals right now: {len(live_signals)}")

if __name__ == "__main__":
    run_10usd_simulation()
