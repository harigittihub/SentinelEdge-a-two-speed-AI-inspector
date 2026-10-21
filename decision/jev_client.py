import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()
URL = "https://ai-gateway.vercel.sh/v1/evaluate"


class JevClient:
    def __init__(self, model="typesafe-ai/jev", timeout=30):
        key = os.getenv("AI_GATEWAY_API_KEY")
        assert key, "AI_GATEWAY_API_KEY is missing from .env"
        self.model = model
        self.http = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=timeout)

    def decide(self, state, questions, retries=3):
        for attempt in range(retries):
            t0 = time.perf_counter()
            r = self.http.post(URL, json={"model": self.model, "state": state, "questions": questions})
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            data = r.json()
            gw = data.get("providerMetadata", {}).get("gateway", {})
            return {
                "answers": data["answers"],
                "latency_ms": ms,
                "cost": float(gw.get("cost") or 0),
                "usage": data.get("usage", {}),
            }