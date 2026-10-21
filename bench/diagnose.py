from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

df = pd.read_csv("results/decisions.csv")
vis = pd.concat([pd.read_csv(f) for f in Path("results").glob("*_vision.csv")])
df = df.merge(vis[["path", "tau"]], on="path")


def true_area(row):
    if row.is_defect == 0:
        return 0.0
    p = Path(row.path)
    mask = p.parents[2] / "ground_truth" / p.parent.name / f"{p.stem}_mask.png"
    return float((np.array(Image.open(mask)) > 0).mean() * 100)


df["true_area_pct"] = df.apply(true_area, axis=1)
df["vision_flag"] = (df.score > df.tau).astype(int)
df["jev_flag"] = (df.action != "pass").astype(int)

print("\n== Pass vs flag accuracy ==")
print("vision rule (score > tau):", round((df.vision_flag == df.is_defect).mean(), 3))
print("Jev (action != pass):     ", round((df.jev_flag == df.is_defect).mean(), 3))
print(df.groupby("category").apply(
    lambda g: pd.Series({"vision_rule": (g.vision_flag == g.is_defect).mean(),
                         "jev": (g.jev_flag == g.is_defect).mean()})).round(3))

print("\n== Error breakdown: rows = (truth, vision flag), cols = Jev action ==")
print(pd.crosstab([df.is_defect, df.vision_flag], df.action))

d = df[df.is_defect == 1]
print("\n== Does severity track the TRUE defect size? (Spearman) ==")
print("Jev severity     vs true area:", round(spearmanr(d.severity, d.true_area_pct).statistic, 3))
print("anomaly score    vs true area:", round(spearmanr(d.score, d.true_area_pct).statistic, 3))
print("\nmedian true defect area (%) by Jev action:")
print(d.groupby("action").true_area_pct.median().round(2))
df.to_csv("results/diagnosed.csv", index=False)