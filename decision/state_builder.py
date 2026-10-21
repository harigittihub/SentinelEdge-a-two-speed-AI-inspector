def build_state(row):
    score, tau = float(row["score"]), float(row["tau"])
    return {
        "inspection": "automated visual anomaly detection on a manufactured part",
        "part_type": row["category"],
        "anomaly_score": round(score, 3),
        "alarm_threshold": round(tau, 3),
        "score_to_threshold_ratio": round(score / tau, 2),
        "anomalous_area_pct": float(row["area_pct"]),
        "anomaly_location": row["location"],
    }