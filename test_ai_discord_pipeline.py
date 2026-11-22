import os, json, requests
from ai_committee import run_dual_ai_committee

webhook = os.getenv("DISCORD_WEBHOOK_URL", "")
print("=== 1. DISCORD WEBHOOK CHECK ===")
print("Webhook configured:", bool(webhook and "discord.com/api/webhooks" in webhook))

print("\n=== 2. CURRENT LIVE SIGNALS IN reports/live_stats.json ===")
if os.path.exists("reports/live_stats.json"):
    with open("reports/live_stats.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    res = data.get("results", [])
    buys = [r["asset"] for r in res if "BUY" in r.get("status", "")]
    exits = [r["asset"] for r in res if "EXIT" in r.get("status", "")]
    flats = [r["asset"] for r in res if "FLAT" in r.get("status", "")]
    print(f"Total Verified Assets: {len(res)} | BUY: {len(buys)} | EXIT: {len(exits)} | FLAT: {len(flats)}")
    if buys: print("Active BUY assets:", buys)
    if exits: print("Active EXIT assets:", exits)
else:
    print("reports/live_stats.json not found.")

print("\n=== 3. LIVE OLLAMA DUAL-AI COMMITTEE TEST (BTC/USDT) ===")
sample_metrics = {
    "price": 91500.0,
    "stop_loss": 88200.0,
    "take_profit": 97000.0,
    "win_rate": 82.1,
    "profit_factor": 8.56,
    "max_drawdown": 18.4
}
verdict = run_dual_ai_committee("BTC/USDT", "Connors RSI(2) Extreme (<5)", sample_metrics)
print(json.dumps(verdict, indent=2))

print("\n=== 4. SENDING TEST ALERT TO DISCORD ===")
if not webhook:
    raise SystemExit("ERROR: DISCORD_WEBHOOK_URL secret is EMPTY in GitHub Actions!")

status_label = "APPROVED" if verdict.get("consensus_approved") else "VETOED"
color = 0x00E5A0 if verdict.get("consensus_approved") else 0xFF4B4B
t_res = verdict.get("agent1_technical", {})
m_res = verdict.get("agent2_fundamental", {})

payload = {
    "embeds": [{
        "title": f"[LIVE SYSTEM TEST] BTC/USDT — AI COMMITTEE {status_label}",
        "color": color,
        "description": (
            f"**Strategy:** `Connors RSI(2) Extreme (<5)`\n"
            f"**Sample Price:** `$91,500.00` | **SL:** `$88,200.00` | **TP:** `$97,000.00`\n"
            f"**Agent 1 (Technical AI):** `{t_res.get('vote')}` ({t_res.get('confidence', 'N/A')}%) — {t_res.get('reason')}\n"
            f"**Agent 2 (Macro/News AI):** `{m_res.get('vote')}` (Risk: {m_res.get('risk_level', 'N/A')}) — {m_res.get('reason')}\n"
            f"**Final Decision:** `{status_label}`"
        )
    }]
}
r = requests.post(webhook, json=payload, timeout=15)
print(f"Discord HTTP Status: {r.status_code} (204 = Success!)")
if r.status_code not in (200, 204):
    print("Discord Error Response:", r.text)
    raise SystemExit(1)
print("✅ END-TO-END PIPELINE VERIFIED!")
