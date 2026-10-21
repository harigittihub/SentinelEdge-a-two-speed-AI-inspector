import json
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("AI_GATEWAY_API_KEY")
assert key, "AI_GATEWAY_API_KEY is missing from .env"

payload = {
    "model": "typesafe-ai/jev",
    "state": (
        "Defect: scratch, anomaly score 0.91, affected area 2.3% of surface, "
        "location: edge of a metal nut."
    ),
    "questions": {
        "stopLine": {
            "type": "boolean",
            "instructions": "Should the production line be stopped?",
        }
    },
}

t0 = time.perf_counter()
r = httpx.post(
    "https://ai-gateway.vercel.sh/v1/evaluate",
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    json=payload,
    timeout=30,
)
ms = (time.perf_counter() - t0) * 1000

print(f"status: {r.status_code} | {ms:.0f} ms round trip")
try:
    print(json.dumps(r.json(), indent=2))
except ValueError:
    print(r.text)