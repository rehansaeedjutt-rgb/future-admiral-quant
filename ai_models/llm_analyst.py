import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_ai_verdict(asset, strategy, stats):
    """Uses local Ollama (llama3.2:3b) to generate a concise institutional trade verdict."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("OLLAMA_QUICK_MODEL", "llama3.2:3b")

    prompt = (
        f"You are the Chief Quant Analyst at Future Admiral. "
        f"Give a sharp 2-sentence institutional verdict for {asset} using strategy '{strategy}'. "
        f"Stats: Net Return {stats['net_return']}%, Win Rate {stats['win_rate']}%, "
        f"Profit Factor {stats['profit_factor']}, Max Drawdown -{stats['max_drawdown']}%, "
        f"Current Signal: {stats['status']} at ${stats['price']}. "
        f"Keep it professional, direct, and risk-focused."
    )

    try:
        client = OpenAI(base_url=base_url, api_key="ollama")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=120
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return f"Quant Filter: {stats['status']} | PF {stats['profit_factor']} | DD -{stats['max_drawdown']}% (Ollama offline - fallback mode)."
