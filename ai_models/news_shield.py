import os, json, re, urllib.parse, xml.etree.ElementTree as ET
import requests

HTTP = requests.Session()
HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0.0.0"}

WAR_SHOCK_KEYWORDS = [
    "war declared", "missile strike", "airstrike", "nuclear", "invasion",
    "military escalation", "martial law", "emergency rate hike", "market halt",
    "circuit breaker triggered", "banking crisis", "flash crash"
]

ASSET_SHOCK_KEYWORDS = [
    "ceo resigns", "ceo steps down", "ceo leaves", "ceo fired", "sec lawsuit",
    "fraud", "bankruptcy", "insolvent", "hacked", "exploit", "delist",
    "delisting", "accounting probe", "doj investigation", "recall", "sanctions"
]

def _fetch_rss_titles(query, limit=6):
    try:
        q = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={q}+when:2d&hl=en-US&gl=US&ceid=US:en"
        r = HTTP.get(url, headers=HDRS, timeout=6)
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        titles = [item.find("title").text for item in root.findall(".//item") if item.find("title") is not None]
        return titles[:limit]
    except Exception:
        return []

def _check_macro_panic():
    reasons = []
    # 1. Check live VIX (Stock fear gauge)
    try:
        r = HTTP.get("https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX",
                     params={"range": "5d", "interval": "1d"}, headers=HDRS, timeout=6)
        if r.status_code == 200:
            meta = (r.json().get("chart", {}).get("result") or [{}])[0].get("meta", {})
            vix = float(meta.get("regularMarketPrice") or 0)
            if vix >= 32.0:
                reasons.append(f"EXTREME VIX PANIC ({vix:.1f} >= 32)")
    except Exception:
        pass

    # 2. Check live BTC 24h flash crash
    try:
        r = HTTP.get("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT", timeout=5)
        if r.status_code == 200:
            chg = float(r.json().get("priceChangePercent") or 0)
            if chg <= -7.0:
                reasons.append(f"BTC 24H FLASH CRASH ({chg:.1f}%)")
    except Exception:
        pass

    return reasons

def _ask_gemini_optional(asset_name, headlines):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or not headlines:
        return None
    try:
        prompt = (
            f"You are an institutional risk officer. Asset: {asset_name}. "
            f"Recent 48h headlines:\n" + "\n".join(f"- {h}" for h in headlines[:10]) +
            "\nIs there a severe black-swan shock (war escalation, CEO sudden exit, fraud, hack, SEC ban, bankruptcy) "
            "that makes buying this dip dangerous? Reply ONLY in strict JSON: "
            '{"verdict":"APPROVE" or "BLOCK","risk_score":0-100,"reason":"short 1-line reason"}'
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        r = HTTP.post(url, json=payload, timeout=8)
        if r.status_code == 200:
            txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            m = re.search(r"\{[\s\S]*\}", txt)
            if m:
                return json.loads(m.group(0))
    except Exception:
        pass
    return None

def evaluate_trade_safety(asset_name):
    ticker = asset_name.split(" ")[0].replace("/USDT", "")
    is_crypto = "Binance" in asset_name
    query = f"{ticker} crypto" if is_crypto else f"{ticker} stock"

    macro_flags = _check_macro_panic()
    asset_news = _fetch_rss_titles(query, limit=6)
    war_news = _fetch_rss_titles("stock market war geopolitical crisis crash", limit=5)

    all_headlines = asset_news + war_news
    shock_hits = []
    for h in war_news:
        hl = h.lower()
        for kw in WAR_SHOCK_KEYWORDS:
            if kw in hl:
                shock_hits.append(f"MACRO_NEWS({kw}): {h[:85]}")
                break
    for h in asset_news:
        hl = h.lower()
        for kw in ASSET_SHOCK_KEYWORDS:
            if kw in hl:
                shock_hits.append(f"ASSET_NEWS({kw}): {h[:85]}")
                break

    # Optional Gemini LLM deep check if GEMINI_API_KEY is configured
    ai_res = _ask_gemini_optional(asset_name, all_headlines)
    if ai_res and str(ai_res.get("verdict", "")).upper() == "BLOCK":
        return {
            "allow": False,
            "verdict": "BLOCKED_BY_AI",
            "risk_score": int(ai_res.get("risk_score", 90)),
            "reason": str(ai_res.get("reason", "AI detected black-swan shock")),
        }

    if macro_flags or len(shock_hits) >= 2 or any(s.startswith("ASSET_NEWS") for s in shock_hits):
        reasons = macro_flags + shock_hits[:2]
        return {
            "allow": False,
            "verdict": "BLOCKED_BY_SHIELD",
            "risk_score": 85,
            "reason": " | ".join(reasons),
        }

    return {
        "allow": True,
        "verdict": "APPROVED_SAFE",
        "risk_score": 15 if not shock_hits else 40,
        "reason": "No war, macro crash, or corporate black-swan shock detected.",
    }

if __name__ == "__main__":
    for test_asset in ("NVDA (US Stock 1D)", "BTC/USDT (Binance 1D)"):
        res = evaluate_trade_safety(test_asset)
        print(f"[AI NEWS SHIELD TEST] {test_asset:24s} -> {res['verdict']} (Risk {res['risk_score']}/100) | {res['reason']}")
