import numpy as np
import pandas as pd
from scipy.stats import spearmanr

df = pd.read_csv("results/diagnosed.csv")
df["ratio"] = df.score / df.tau


def matched_rule(g):
    """Plain threshold on score/tau, tuned to Jev's false-alarm rate."""
    good, bad = g[g.is_defect == 0], g[g.is_defect == 1]
    jev_fpr, jev_tpr = good.jev_flag.mean(), bad.jev_flag.mean()
    thr = np.quantile(good.ratio, 1 - jev_fpr)
    return pd.Series({
        "jev_fpr": jev_fpr, "jev_tpr": jev_tpr,
        "rule_fpr": (good.ratio > thr).mean(), "rule_tpr": (bad.ratio > thr).mean(),
    })


print("== Same false-alarm rate: does Jev catch more defects than a plain threshold? ==")
print(matched_rule(df).round(3).to_frame("overall").T)
print(df.groupby("category").apply(matched_rule).round(3))

print("\n== Spearman vs TRUE defect area, with 95% bootstrap CI (defects only) ==")
d = df[df.is_defect == 1].reset_index(drop=True)
rng = np.random.default_rng(0)
for name, col in {"Jev severity": "severity", "score/tau ratio": "ratio", "vision area_pct": "area_pct"}.items():
    rho = spearmanr(d[col], d.true_area_pct).statistic
    boots = []
    for _ in range(1000):
        s = d.sample(len(d), replace=True, random_state=int(rng.integers(1_000_000_000)))
        boots.append(spearmanr(s[col], s.true_area_pct).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"{name:18s} rho={rho:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")