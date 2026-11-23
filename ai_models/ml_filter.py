from sklearn.ensemble import RandomForestClassifier

def apply_ml_filter(df, raw_signals, threshold=0.42):
    df = df.copy()
    df["ema_dist"] = (df["close"] - df["ema50"]) / df["ema50"]
    features = ["rsi", "vol_ratio", "atr", "ema_dist"]
    df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
    valid = df.dropna().copy()

    split = int(len(valid) * 0.5)
    if split < 50:
        return raw_signals

    model = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=42)
    model.fit(valid[features].iloc[:split], valid["target"].iloc[:split])

    oos_start_idx = valid.index[min(split + 3, len(valid) - 1)]
    filtered = raw_signals.copy()
    filtered.loc[:oos_start_idx] = 0

    oos_mask = df.index > oos_start_idx
    if oos_mask.any():
        oos_features = df.loc[oos_mask, features].fillna(0)
        probs = model.predict_proba(oos_features)[:, 1]
        oos_sig = filtered.loc[oos_mask].copy()
        oos_sig[(oos_sig == 1) & (probs < threshold)] = 0
        filtered.loc[oos_mask] = oos_sig

    return filtered
