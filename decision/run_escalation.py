import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, "vision")
from claude_client import ClaudeClient
from patchcore_lite import PatchCoreLite

SIZE = 512


def load_img(p):
    return Image.open(p).convert("RGB").resize((SIZE, SIZE))


def overlay(pc, path):
    img = np.array(load_img(path)).astype(float)
    m = pc.anomaly_map(path)
    a = np.clip((m - 0.8 * pc.tau) / (0.6 * pc.tau), 0, 1)
    a = np.array(Image.fromarray((a * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR)) / 255.0
    w = 0.6 * a[..., None]
    red = np.zeros_like(img)
    red[..., 0] = 255
    return Image.fromarray((img * (1 - w) + red * w).astype(np.uint8))

def zoom_crop(pc, path, frac=0.3, out=SIZE):
    m = pc.anomaly_map(path)
    py, px = np.unravel_index(np.argmax(m), m.shape)
    img = Image.open(path).convert("RGB")
    W, H = img.size
    s = int(frac * min(W, H))
    cx, cy = int((px + 0.5) / m.shape[1] * W), int((py + 0.5) / m.shape[0] * H)
    x0 = min(max(cx - s // 2, 0), W - s)
    y0 = min(max(cy - s // 2, 0), H - s)
    return img.crop((x0, y0, x0 + s, y0 + s)).resize((out, out), Image.LANCZOS)

CROP = "--crop" in sys.argv
SPLIT = "test" if "--test" in sys.argv else "dev"
nums = [a for a in sys.argv[1:] if a.isdigit()]
limit = int(nums[0]) if nums else None
OUT = f"results/escalation_{SPLIT}{'_crop' if CROP else ''}.csv"

dec = pd.read_csv("results/decisions.csv").merge(pd.read_csv("results/split.csv"), on="path")
dev = dec[dec.split == SPLIT]
todo = dev[dev.action == "human_review"]
if limit:
    todo = todo.sample(limit, random_state=0)

models, refs = {}, {}
for cat in todo.category.unique():
    pc = PatchCoreLite()
    pc.load(f"artifacts/{cat}.pt")
    models[cat] = pc
    refs[cat] = load_img(sorted((Path("data") / cat / "train" / "good").glob("*.png"))[0])

jobs = [(r, load_img(r.path), overlay(models[r.category], r.path),
         zoom_crop(models[r.category], r.path) if CROP else None)
        for _, r in todo.iterrows()]
client = ClaudeClient()


def work(job):
    r, test, heat, crop = job
    try:
        out = client.inspect(r.category, refs[r.category], test, heat, crop)
    except Exception as e:
        print("FAILED", r.path, e)
        return None
    p = out["parsed"]
    return {
        "path": r.path, "category": r.category, "is_defect": int(r.is_defect),
        "claude_verdict": p.get("verdict"), "claude_conf": p.get("confidence"),
        "claude_severity": p.get("severity"),
        "defect_description": p.get("defect_description"),
        "incident_report": p.get("incident_report"),
        "prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"],
        "latency_ms": round(out["latency_ms"]),
    }


with ThreadPoolExecutor(max_workers=4) as ex:
    res = [x for x in ex.map(work, jobs) if x]

df = pd.DataFrame(res)
df["claude_flag"] = (df.claude_verdict == "defect").astype(int)
df.to_csv(OUT, index=False)

print(f"\n{len(df)} escalated images")
print("always-'defect' baseline:", round(df.is_defect.mean(), 3))
print("Claude accuracy:         ", round((df.claude_flag == df.is_defect).mean(), 3))
print(pd.crosstab(df.is_defect, df.claude_verdict))
print(df.groupby("category").apply(lambda g: (g.claude_flag == g.is_defect).mean()).round(3))
print("avg latency ms:", int(df.latency_ms.mean()),
      "| avg tokens in/out:", int(df.prompt_tokens.mean()), int(df.completion_tokens.mean()))

if not limit:
    both = dev.merge(df[["path", "claude_flag"]], on="path", how="left")
    both["jev_flag"] = (both.action != "pass").astype(int)
    both["final_flag"] = both.claude_flag.fillna(both.jev_flag).astype(int)

    def summarize(flag):
        fa = int(((flag == 1) & (both.is_defect == 0)).sum())
        miss = int(((flag == 0) & (both.is_defect == 1)).sum())
        return f"acc {(flag == both.is_defect).mean():.3f} | false alarms {fa} | missed defects {miss}"

    print(f"\nWhole {SPLIT} pipeline (pass vs flag):")
    print("Jev alone :", summarize(both.jev_flag))
    print("Jev+Claude:", summarize(both.final_flag))