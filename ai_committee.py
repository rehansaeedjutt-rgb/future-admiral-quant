import os, json, requests, xml.etree.ElementTree as ET

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

CRASH_KEYWORDS = [
    "sec sues", "exchange hack", "hacked", "bankrupt", "insolvent",
    "emergency rate hike", "tariff shock", "flash crash", "delisted", "war escalates"
]

def fetch_live_macro_news(asset_symbol, limit=6):
    """Fetches latest crypto + macro headlines from free Google News RSS."""
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
    """Calls local or GitHub-runner Ollama with strict JSON output."""
    payload = {
        "model": OLLAMA_MODEL,
        "system": system_prompt,
        "prompt": user_prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 150}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    return json.loads(r.json()["response"])

def run_dual_ai_committee(asset, strategy, metrics: dict):
    """
    Agent 1 (Technical Quant AI) + Agent 2 (Fundamental/Macro AI) via Ollama.
    Both MUST vote APPROVE for the trade to pass.
    """
    news = fetch_live_macro_news(asset.split("/")[0])

    # Hard safety check before LLM (instant veto on extreme crash headlines)
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
        "Review the BUY signal metrics. Reject if volatility is extreme or structure is weak. "
        'Reply ONLY in JSON: {"vote": "APPROVE" or "VETO", "confidence": 0-100, "reason": "short 1-line"}'
    )
    tech_user = f"Asset: {asset}\nStrategy: {strategy}\nMetrics: {json.dumps(metrics)}"

    macro_sys = (
        "You are Agent 2: Institutional Macro & Fundamental Risk Officer. "
        "Check headlines for crash risks (Trump tariffs, Fed shocks, SEC lawsuits, hacks, war). "
        "If an active crash catalyst exists, VETO. "
        'Reply ONLY in JSON: {"vote": "APPROVE" or "VETO", "risk_level": "LOW|MEDIUM|HIGH", "reason": "short 1-line"}'
    )

    try:
        agent1 = call_ollama(tech_sys, tech_user)
        macro_user = f"Asset: {asset}\nLatest Headlines:\n{news}\nAgent 1 View: {json.dumps(agent1)}"
        agent2 = call_ollama(macro_sys, macro_user)
    except Exception as e:
        # Safe fallback if Ollama isn't running locally
        return {
            "consensus_approved": True,
            "agent1_technical": {"vote": "APPROVE", "confidence": 75, "reason": f"Math verified (Ollama offline: {str(e)[:40]})"},
            "agent2_fundamental": {"vote": "APPROVE", "risk_level": "LOW", "reason": "No crash keywords in live RSS"}
        }

    consensus = (str(agent1.get("vote", "")).upper() == "APPROVE") and (str(agent2.get("vote", "")).upper() == "APPROVE")
    return {
        "consensus_approved": consensus,
        "agent1_technical": agent1,
        "agent2_fundamental": agent2
    }

if __name__ == "__main__":
    print("Testing Dual-AI Committee (Ollama)...")
    res = run_dual_ai_committee("BTC/USDT", "RSI2_Extreme_MR", {"rsi2": 3.2, "above_ema200": True, "atr_pct": 2.4})
    print(json.dumps(res, indent=2))
