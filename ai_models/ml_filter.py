import numpy as np
from sklearn.ensemble import RandomForestClassifier

def apply_ml_filter(df, raw_signals, threshold=0.52):
    df = df.copy()
    df["ema_dist"] = (df["close"] - df["ema50"]) / df["ema50"]
    features = ["rsi", "vol_ratio", "atr", "ema_dist"]
    df["target"] = (df["close"].shift(-6) > df["close"]).astype(int)
    valid = df.dropna().copy()

    split = int(len(valid) * 0.5)
    if split < 50:
        return raw_signals

    model = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42)
    model.fit(valid[features].iloc[:split], valid["target"].iloc[:split])

    probs = model.predict_proba(df[features])[:, 1]
    filtered = raw_signals.copy()
    filtered[(raw_signals == 1) & (probs < threshold)] = 0
    return filtered