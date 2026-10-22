from pathlib import Path

import numpy as np
import pandas as pd

AUTO = {"metal_nut"}   # frozen in FROZEN.md: categories Claude may auto-resolve

dec = pd.read_csv("results/decisions.csv")
split = pd.read_csv("results/split.csv")
vis = pd.concat([pd.read_csv(f) for f in Path("results").glob("*_vision.csv")])[["path", "tau"]]
esc = pd.read_csv("results/escalation_test.csv")[["path", "claude_flag"]]

df = dec.merge(split, on="path").merge(vis, on="path")
df = df[df.split == "test"].merge(esc, on="path", how="left").reset_index(drop=True)

review = df.action == "human_review"
has_claude = df.claude_flag.notna()
df["vision"] = (df.score > df.tau).astype(int)
df["jev"] = (df.action != "pass").astype(int)
df["jev_claude_all"] = np.where(review & has_claude, df.claude_flag, df.jev).astype(int)
df["jev_claude_policy"] = np.where(review & has_claude & df.category.isin(AUTO),
                                   df.claude_flag, df.jev).astype(int)
SYSTEMS = ["vision", "jev", "jev_claude_all", "jev_claude_policy"]


def metrics(d, col):
    good, bad = d[d.is_defect == 0], d[d.is_defect == 1]
    return {"acc": (d[col] == d.is_defect).mean(),
            "fa_rate": (good[col] == 1).mean(),
            "miss_rate": (bad[col] == 0).mean()}


def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"


rng = np.random.default_rng(0)
boot = {s: [] for s in SYSTEMS}
for _ in range(2000):
    s_ = df.iloc[rng.integers(0, len(df), len(df))]      # same resample for every system
    for s in SYSTEMS:
        boot[s].append(metrics(s_, s))
boot = {s: pd.DataFrame(v) for s, v in boot.items()}

print(f"TEST HALF: {len(df)} images ({int((df.is_defect == 0).sum())} good, {int(df.is_defect.sum())} defective)\n")
rows = []
for s in SYSTEMS:
    m = metrics(df, s)
    rows.append({
        "system": s,
        "false_alarms": int(((df[s] == 1) & (df.is_defect == 0)).sum()),
        "missed": int(((df[s] == 0) & (df.is_defect == 1)).sum()),
        "accuracy": f"{m['acc']:.3f} {ci(boot[s].acc)}",
        "FA rate": f"{m['fa_rate']:.3f} {ci(boot[s].fa_rate)}",
        "miss rate": f"{m['miss_rate']:.3f} {ci(boot[s].miss_rate)}",
    })
print(pd.DataFrame(rows).to_string(index=False))

print("\nPaired difference, frozen policy minus Jev alone (95% bootstrap CI):")
for k in ["acc", "fa_rate", "miss_rate"]:
    point = metrics(df, "jev_claude_policy")[k] - metrics(df, "jev")[k]
    d = boot["jev_claude_policy"][k] - boot["jev"][k]
    print(f"  {k:10s} {point:+.3f} {ci(d)}")

human_jev = int(review.sum())
human_policy = int((review & ~df.category.isin(AUTO)).sum())
print(f"\nImages sent to a human: Jev alone {human_jev} -> with frozen policy {human_policy}")

print("\nBy category (frozen policy):")
for c, g in df.groupby("category"):
    print(f"  {c:10s} n={len(g):3d} | Jev acc {metrics(g, 'jev')['acc']:.3f} "
          f"| policy acc {metrics(g, 'jev_claude_policy')['acc']:.3f}")