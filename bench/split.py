import random

import pandas as pd

df = pd.read_csv("results/decisions.csv")
rng = random.Random(42)

rows = []
for (cat, defect), g in df.groupby(["category", "defect_type"]):
    paths = sorted(g.path)
    rng.shuffle(paths)
    n_dev = (len(paths) + 1) // 2          # dev gets the extra image on odd groups
    for i, p in enumerate(paths):
        rows.append({"path": p, "split": "dev" if i < n_dev else "test"})

split = pd.DataFrame(rows)
split.to_csv("results/split.csv", index=False)

check = df.merge(split, on="path")
print(pd.crosstab([check.category, check.is_defect], check.split))
print(split.split.value_counts())