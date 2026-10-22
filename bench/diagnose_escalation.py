import pandas as pd
import sys

e = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/escalation_dev.csv")
dec = pd.read_csv("results/diagnosed.csv")[["path", "true_area_pct"]]
m = e.merge(dec, on="path")

print("== Claude verdict by category and truth ==")
print(pd.crosstab([m.category, m.is_defect], m.claude_verdict))

d = m[m.is_defect == 1]
print("\n== Median TRUE defect area (%): caught vs missed ==")
print(d.groupby("claude_verdict").true_area_pct.agg(["count", "median"]).round(2))

g = m[m.claude_verdict == "good"]
print("\n== Claude confidence on its 'good' verdicts ==")
print(g.groupby("is_defect").claude_conf.describe().round(2))

print("\n== Auto-clear only if Claude says 'good' with confidence >= c ==")
for c in [0.5, 0.6, 0.7, 0.8, 0.9]:
    s = g[g.claude_conf >= c]
    print(f"c={c}: cleared {len(s)} | good parts cleared {int((s.is_defect == 0).sum())}"
          f" | defects wrongly cleared {int((s.is_defect == 1).sum())}")


m["err"] = ((m.claude_verdict == "defect").astype(int) != m.is_defect).astype(int)
cat = m.groupby("category").agg(n=("err", "size"), errors=("err", "sum"))
cat["auto_resolve"] = (cat.n >= 10) & (cat.errors <= 2)
print("\n== Per-category auto-resolve rule (>=10 escalated, <=2 errors) ==")
print(cat)
    