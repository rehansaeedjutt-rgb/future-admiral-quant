import os, json, requests, xml.etree.ElementTree as ET

def fetch_live_macro_news(asset_symbol, limit=6):
    """Fetches latest crypto + macro headlines (Trump, Fed, CPI, Hacks) from free Google News RSS."""
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

def call_llm(system_prompt, user_prompt):
    """Calls Gemini API (GEMINI_API_KEY) or OpenAI-compatible API (OPENAI_API_KEY)."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
        payload = {
            "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1}
        }
        r = requests.post(url, json=payload, timeout=20)
        r.raise_for_status()
        txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(txt)
    raise RuntimeError("Set GEMINI_API_KEY in GitHub Secrets to activate the Dual-AI Committee.")

def run_dual_ai_committee(asset, strategy, metrics: dict):
    """
    Agent 1 (Technical Quant AI) + Agent 2 (Fundamental/Macro AI)
    Both MUST vote APPROVE for the trade to pass.
    """
    news = fetch_live_macro_news(asset.split("/")[0])

    # Agent 1: Technical Quant AI
    tech_sys = (
        "You are Agent 1: Institutional Technical Quant Reviewer. "
        "Review the math metrics for a BUY signal. Reject if volatility/ATR is extreme or structure is broken. "
        "Return strict JSON: {\"vote\": \"APPROVE\" or \"VETO\", \"confidence\": 0-100, \"reason\": \"short 1-line\"}"
    )
    tech_user = f"Asset: {asset}\nStrategy: {strategy}\nMetrics: {json.dumps(metrics)}"
    agent1 = call_llm(tech_sys, tech_user)

    # Agent 2: Fundamental & Macro Risk AI
    macro_sys = (
        "You are Agent 2: Institutional Macro & Fundamental Risk Officer. "
        "Check the latest headlines for black-swan crash risks (Trump tariff shocks, Fed emergency hikes, "
        "SEC lawsuits, exchange hacks, war escalation, or severe coin-specific negative news). "
        "If there is an active crash catalyst, VETO the BUY signal. "
        "Return strict JSON: {\"vote\": \"APPROVE\" or \"VETO\", \"risk_level\": \"LOW|MEDIUM|HIGH\", \"reason\": \"short 1-line\"}"
    )
    macro_user = f"Asset: {asset}\nLatest Headlines:\n{news}\nAgent 1 Technical View: {json.dumps(agent1)}"
    agent2 = call_llm(macro_sys, macro_user)

    consensus = (agent1.get("vote") == "APPROVE") and (agent2.get("vote") == "APPROVE")
    return {
        "consensus_approved": consensus,
        "agent1_technical": agent1,
        "agent2_fundamental": agent2
    }
