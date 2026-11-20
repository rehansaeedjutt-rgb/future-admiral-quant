import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_ai_verdict(asset, strategy, stats):
    groq_key = os.getenv("GROQ_API_KEY", "")
    base_url = "https://api.groq.com/openai/v1" if groq_key else os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    api_key  = groq_key if groq_key else "ollama"
    model    = os.getenv("GROQ_MODEL_FAST", "llama-3.3-70b-versatile") if groq_key else os.getenv("OLLAMA_QUICK_MODEL", "llama3.2:3b")

    prompt = (
        f"You are the Chief Quant Analyst at Future Admiral. "
        f"Give a sharp 2-sentence institutional verdict for {asset} using '{strategy}'. "
        f"Stats: Net Return {stats['net_return']}%, Win Rate {stats['win_rate']}%, "
        f"Profit Factor {stats['profit_factor']}, Max Drawdown -{stats['max_drawdown']}%, "
        f"Signal: {stats['status']} at ${stats['price']}. Direct and risk-focused."
    )

    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=110
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        bias = "Defensive capital preservation advised." if "BEARISH" in stats["status"] else "Momentum alignment confirmed; enforce 1% risk cap."
        return f"Regime: {stats['status']} | Edge: PF {stats['profit_factor']} with -{stats['max_drawdown']}% max DD. {bias}"