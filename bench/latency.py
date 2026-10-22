import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, "decision")
from jev_client import JevClient
from questions import QUESTIONS
from state_builder import build_state

load_dotenv()
KEY = os.getenv("AI_GATEWAY_API_KEY")
HEAD = {"Authorization": f"Bearer {KEY}"}
http = httpx.Client(headers=HEAD, timeout=60)
CHAT_URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
N, WARM = 45, 5

PROMPT = """You are an inspection triage system. Measurement report for one manufactured part:
{state}

Reply with ONLY a JSON object: {{"action": "pass" | "rework" | "scrap" | "human_review", "severity": integer 0-3 (0 none, 1 minor cosmetic, 2 major, 3 critical), "stop_line": true or false}}
pass = no meaningful anomaly; rework = minor localized defect that can be repaired; scrap = large or serious defect; human_review = borderline or ambiguous. stop_line = true only for severe defects that risk more bad parts."""


def chat(model, state):
    body = {"model": model, "max_tokens": 100,
            "messages": [{"role": "user", "content": PROMPT.format(state=json.dumps(state))}]}
    t0 = time.perf_counter()
    r = http.post(CHAT_URL, json=body)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    d = r.json()
    m = re.search(r"\{.*\}", d["choices"][0]["message"]["content"], re.S)
    usage = d.get("usage", {})
    cost = usage.get("cost") or d.get("providerMetadata", {}).get("gateway", {}).get("cost")
    return ms, (json.loads(m.group(0)) if m else {}), usage, cost


models = sys.argv[1:] or ["anthropic/claude-haiku-4.5", "anthropic/claude-sonnet-5.5"]
systems = ["jev"] + models

vis = pd.concat([pd.read_csv(f) for f in Path("results").glob("*_vision.csv")])
vis = vis.merge(pd.read_csv("results/split.csv"), on="path")
rows = vis[vis.split == "dev"].sample(N + WARM, random_state=0).to_dict("records")

floor = []
for _ in range(21):
    t0 = time.perf_counter()
    http.get("https://ai-gateway.vercel.sh/v1/models/typesafe-ai/jev")
    floor.append((time.perf_counter() - t0) * 1000)
floor = floor[1:]          # first request pays the handshake, drop it

jev = JevClient()
rec = {s: {"ms": [], "ok": [], "cost": [], "tin": [], "tout": []} for s in systems}
for i, row in enumerate(rows):
    state = build_state(row)
    for s in systems:
        try:
            if s == "jev":
                o = jev.decide(state, QUESTIONS)
                ms, action, cost, tin, tout = o["latency_ms"], o["answers"]["action"]["choice"], o["cost"], 0, 0
            else:
                ms, parsed, usage, cost = chat(s, state)
                action, tin, tout = parsed.get("action"), usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        except Exception as e:
            print("FAILED", s, e)
            continue
        if i >= WARM:
            rec[s]["ms"].append(ms)
            rec[s]["ok"].append(int((action != "pass") == bool(row["is_defect"])))
            rec[s]["cost"].append(cost if cost is not None else np.nan)
            rec[s]["tin"].append(tin)
            rec[s]["tout"].append(tout)

print(f"\nGateway round-trip floor (small GET, from here): median {np.median(floor):.0f} ms\n")
out = []
for s in (systems if i % 2 == 0 else systems[::-1]):
    r = rec[s]
    out.append({
        "system": s, "n": len(r["ms"]),
        "p50_ms": round(np.percentile(r["ms"], 50)), "p95_ms": round(np.percentile(r["ms"], 95)),
        "mean_ms": round(np.mean(r["ms"])),
        "cost_per_call_usd": round(float(np.nanmean(r["cost"])), 6) if not np.all(np.isnan(r["cost"])) else "n/a",
        "tokens_in/out": f"{int(np.mean(r['tin']))}/{int(np.mean(r['tout']))}",
        "pass_vs_flag_acc (dev, exploratory)": round(float(np.mean(r["ok"])), 3),
    })
print(pd.DataFrame(out).to_string(index=False))