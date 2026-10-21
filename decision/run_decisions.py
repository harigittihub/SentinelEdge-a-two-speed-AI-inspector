import csv
import glob
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from jev_client import JevClient
from questions import QUESTIONS
from state_builder import build_state

rows = []
for f in sorted(glob.glob("results/*_vision.csv")):
    with open(f, newline="") as fh:
        rows += list(csv.DictReader(fh))

if len(sys.argv) > 1:                      # quick test: python decision/run_decisions.py 20
    step = max(1, len(rows) // int(sys.argv[1]))
    rows = rows[::step]

client = JevClient()


def work(row):
    try:
        out = client.decide(build_state(row), QUESTIONS)
    except Exception as e:
        print("FAILED", row["path"], e)
        return None
    a = out["answers"]
    probs = a["action"].get("probabilities") or {}
    return {
        "category": row["category"], "path": row["path"],
        "defect_type": row["defect_type"], "is_defect": int(row["is_defect"]),
        "score": row["score"], "area_pct": row["area_pct"],
        "action": a["action"]["choice"],
        **{f"p_{k}": round(probs.get(k, float("nan")), 4)
           for k in ("pass", "rework", "scrap", "human_review")},
        "severity": round(a["severity"]["score"], 3),
        "stop_prob": a["stopLine"]["probability"],
        "latency_ms": round(out["latency_ms"]),
        "cost_usd": out["cost"],
    }


with ThreadPoolExecutor(max_workers=8) as ex:
    results = [r for r in ex.map(work, rows) if r]

df = pd.DataFrame(results)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/decisions.csv", index=False)

print(f"\n{len(df)}/{len(rows)} rows decided | total cost ${df.cost_usd.sum():.5f}")
print(pd.crosstab(df.is_defect, df.action))
print(df.groupby("is_defect")[["severity", "stop_prob"]].mean().round(3))