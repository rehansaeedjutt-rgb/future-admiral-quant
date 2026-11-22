import os, json, requests, xml.etree.ElementTree as ET

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

CRASH_KEYWORDS = [
    "sec sues", "exchange hack", "hacked", "bankrupt", "insolvent",
    "emergency rate hike", "tariff shock", "flash crash", "delisted", "war escalates"
]

def fetch_live_macro_news(asset_symbol, limit=6):
    query = f"{asset_symbol} crypto OR bitcoin OR federal reserve OR trump tariff"
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    headlines = []
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[:limit]:
                title = item.find("title").text
                pub = item.find("pubDate").text if item.find("pubDate") is not None else ""
                headlines.append(f"- {title} ({pub})")
    except Exception:
        pass
    return "\n".join(headlines) if headlines else "No major breaking news fetched."

def call_ollama(system_prompt, user_prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "system": system_prompt,
        "prompt": user_prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 140}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=90)
    r.raise_for_status()
    return json.loads(r.json()["response"])

def run_dual_ai_committee(asset, strategy, metrics: dict):
    """
    Fail-Closed Dual-AI Committee:
    - Python pre-calculates all geometry/math facts (zero LLM arithmetic).
    - Agent 1: Technical Quant Reviewer.
    - Agent 2: Independent Macro/News Risk Officer (headlines-only).
    - If Ollama fails or either agent vetoes -> VETO (Fail-Closed).
    """
    price = float(metrics.get("price", 0) or 0)
    sl = float(metrics.get("stop_loss", 0) or 0)
    tp = float(metrics.get("take_profit", 0) or 0)
    risk = max(price - sl, 1e-9)
    reward = max(tp - price, 0.0)
    rr_ratio = round(reward / risk, 2) if price > sl else 0.0
    valid_geometry = (sl < price < tp)

    if not valid_geometry:
        return {
            "consensus_approved": False,
            "agent1_technical": {"vote": "VETO", "confidence": 0, "reason": "Invalid SL/Price/TP geometry"},
            "agent2_fundamental": {"vote": "SKIPPED", "risk_level": "HIGH", "reason": "Blocked by geometry check"}
        }

    verified_facts = {
        "entry_price": price,
        "stop_loss_below_entry": sl,
        "take_profit_above_entry": tp,
        "risk_reward_ratio": rr_ratio,
        "historical_win_rate_pct": metrics.get("win_rate"),
        "profit_factor": metrics.get("profit_factor"),
        "max_drawdown_pct": metrics.get("max_drawdown")
    }

    news = fetch_live_macro_news(asset.split("/")[0])
    news_lower = news.lower()
    for kw in CRASH_KEYWORDS:
        if kw in news_lower:
            return {
                "consensus_approved": False,
                "agent1_technical": {"vote": "SKIPPED", "confidence": 0, "reason": "Blocked by Macro Kill-Switch"},
                "agent2_fundamental": {"vote": "VETO", "risk_level": "HIGH", "reason": f"Crash catalyst detected: '{kw}'"}
            }

    tech_sys = (
        "You are Agent 1: Institutional Technical Quant Reviewer. "
        "All numbers are pre-verified by Python. Do NOT recalculate math. "
        "VETO if max_drawdown_pct > 25 or risk_reward_ratio < 1.2. Otherwise APPROVE. "
        'Reply ONLY in JSON: {"vote": "APPROVE" or "VETO", "confidence": 0-100, "reason": "concise 1-sentence reason"}'
    )
    tech_user = f"Asset: {asset}\nStrategy: {strategy}\nPre-Verified Facts: {json.dumps(verified_facts)}"

    macro_sys = (
        "You are Agent 2: Institutional Macro & Fundamental Risk Officer. "
        "Analyze ONLY the provided news headlines for acute crash risks (Trump tariff shocks, Fed emergency hikes, SEC lawsuits, hacks, war). "
        "Do NOT mention technical metrics. If an acute crash catalyst exists, VETO; otherwise APPROVE. "
        'Reply ONLY in JSON: {"vote": "APPROVE" or "VETO", "risk_level": "LOW|MEDIUM|HIGH", "reason": "concise 1-sentence news summary"}'
    )
    macro_user = f"Asset: {asset}\nLatest News Headlines:\n{news}"

    try:
        agent1 = call_ollama(tech_sys, tech_user)
        agent2 = call_ollama(macro_sys, macro_user)
    except Exception as e:
        # Strict Fail-Closed: never approve a trade if the AI risk gate is offline
        return {
            "consensus_approved": False,
            "agent1_technical": {"vote": "VETO", "confidence": 0, "reason": f"Fail-Closed: Ollama error ({str(e)[:45]})"},
            "agent2_fundamental": {"vote": "VETO", "risk_level": "HIGH", "reason": "Fail-Closed: AI Risk Gate unavailable"}
        }

    consensus = (str(agent1.get("vote", "")).upper() == "APPROVE") and (str(agent2.get("vote", "")).upper() == "APPROVE")
    return {
        "consensus_approved": consensus,
        "agent1_technical": agent1,
        "agent2_fundamental": agent2
    }
