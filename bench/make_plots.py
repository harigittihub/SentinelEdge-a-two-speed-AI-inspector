from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Path("docs").mkdir(exist_ok=True)

# 1) Errors on the held-out test half
ev = pd.read_csv("results/final_eval_test.csv")
names = {"vision": "Vision rule", "jev": "Jev alone",
         "jev_claude_all": "Jev + Claude\n(all categories)",
         "jev_claude_policy": "Jev + Claude\n(frozen policy)"}
x, w = list(range(len(ev))), 0.38
fig, ax = plt.subplots(figsize=(8.5, 4.6))
b1 = ax.bar([i - w / 2 for i in x], ev.false_alarms, w, label="False alarms (41 good parts)")
b2 = ax.bar([i + w / 2 for i in x], ev.missed, w, label="Missed defects (134 defective parts)")
ax.bar_label(b1)
ax.bar_label(b2)
ax.set_xticks(x)
ax.set_xticklabels([names[s] for s in ev.system])
ax.set_ylabel("Count")
ax.set_title("Errors on the held-out test half (175 images)")
ax.legend()
fig.tight_layout()
fig.savefig("docs/test_errors.png", dpi=160)

# 2) Latency and cost (numbers from bench/latency.py, sequential, persistent connection)
lat = pd.DataFrame({
    "system": ["Jev", "Claude Haiku 4.5", "Claude Sonnet 5.5"],
    "p50": [311, 840, 1510], "p95": [473, 1617, 2208],
    "cost_per_1k": [0.024, 0.47, 1.34],
})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
xi = list(range(3))
c1 = a1.bar([i - w / 2 for i in xi], lat.p50, w, label="p50")
c2 = a1.bar([i + w / 2 for i in xi], lat.p95, w, label="p95")
a1.bar_label(c1)
a1.bar_label(c2)
a1.set_xticks(xi)
a1.set_xticklabels(lat.system)
a1.set_ylabel("ms")
a1.set_title("Latency (45 sequential calls each)")
a1.legend()
c3 = a2.bar(lat.system, lat.cost_per_1k)
a2.bar_label(c3, fmt="$%.3f")
a2.set_yscale("log")
a2.set_ylabel("USD per 1,000 decisions (log scale)")
a2.set_title("Cost")
fig.tight_layout()
fig.savefig("docs/latency_cost.png", dpi=160)