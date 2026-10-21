import csv
import random
import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score

from patchcore_lite import PatchCoreLite, describe

cat = sys.argv[1]
root = Path("data") / cat

good = sorted((root / "train" / "good").glob("*.png"))
random.Random(0).shuffle(good)
n_cal = max(5, len(good) // 10)
cal, fit = good[:n_cal], good[n_cal:]

pc = PatchCoreLite()
pc.fit(fit)
pc.calibrate(cal)
pc.save(f"artifacts/{cat}.pt")

rows = []
for p in sorted((root / "test").rglob("*.png")):
    defect = p.parent.name
    m = pc.anomaly_map(p)
    area, loc = describe(m, pc.tau)
    rows.append(dict(
        category=cat, path=str(p), defect_type=defect,
        is_defect=int(defect != "good"), score=round(float(m.max()), 4),
        tau=round(pc.tau, 4), area_pct=round(area, 2), location=loc,
    ))

auc = roc_auc_score([r["is_defect"] for r in rows], [r["score"] for r in rows])
print(f"{cat}: {len(rows)} test images | image-level AUROC = {auc:.3f}")

Path("results").mkdir(exist_ok=True)
with open(f"results/{cat}_vision.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)