import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vision"))
sys.path.insert(0, str(ROOT / "decision"))

from claude_client import ClaudeClient
from jev_client import JevClient
from patchcore_lite import PatchCoreLite, describe
from questions import QUESTIONS
from state_builder import build_state

AUTO = {"metal_nut"}   # frozen policy (FROZEN.md): categories Claude may auto-resolve
SIZE = 512
_models, _refs = {}, {}
_jev, _claude = JevClient(), ClaudeClient()


def categories():
    return sorted(p.stem for p in (ROOT / "artifacts").glob("*.pt"))


def _load(cat):
    if cat not in _models:
        pc = PatchCoreLite()
        pc.load(str(ROOT / "artifacts" / f"{cat}.pt"))
        _models[cat] = pc
        ref = sorted((ROOT / "data" / cat / "train" / "good").glob("*.png"))[0]
        _refs[cat] = Image.open(ref).convert("RGB").resize((SIZE, SIZE))
    return _models[cat], _refs[cat]


def _overlay(img, m, tau):
    arr = np.array(img).astype(float)
    a = np.clip((m - 0.8 * tau) / (0.6 * tau), 0, 1)
    a = np.array(Image.fromarray((a * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR)) / 255.0
    w = 0.6 * a[..., None]
    red = np.zeros_like(arr)
    red[..., 0] = 255
    return Image.fromarray((arr * (1 - w) + red * w).astype(np.uint8))


def inspect_image(path, category, escalate=True):
    pc, ref = _load(category)
    m = pc.anomaly_map(path)
    area, loc = describe(m, pc.tau)
    row = {"category": category, "score": float(m.max()), "tau": float(pc.tau),
           "area_pct": round(area, 2), "location": loc}

    jev = _jev.decide(build_state(row), QUESTIONS)
    a = jev["answers"]
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE))
    heat = _overlay(img, m, pc.tau)

    out = {"vision": row, "jev_action": a["action"]["choice"],
           "jev_probs": a["action"].get("probabilities", {}),
           "severity": a["severity"]["score"], "stop_prob": a["stopLine"]["probability"],
           "jev_ms": jev["latency_ms"], "image": img, "heatmap": heat,
           "final": a["action"]["choice"], "claude": None}

    if out["jev_action"] == "human_review":
        if escalate and category in AUTO:
            c = _claude.inspect(category, ref, img, heat)
            p = c["parsed"]
            out["claude"] = {**p, "latency_ms": c["latency_ms"]}
            out["final"] = "defect_confirmed_by_claude" if p.get("verdict") == "defect" else "cleared_by_claude"
        else:
            out["final"] = "human_review"
    return out