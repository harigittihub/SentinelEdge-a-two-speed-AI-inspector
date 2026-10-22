import base64
import io
import json
import os
import re
import time

import httpx
from dotenv import load_dotenv

load_dotenv()
URL = "https://ai-gateway.vercel.sh/v1/chat/completions"

PROMPT = """You are a quality inspector for manufactured {category} parts.
You will see these images: (1) a reference part known to be defect-free, (2) the part to inspect, (3) the same part with an automated anomaly heatmap (red = unusual), and sometimes (4) a zoomed full-resolution crop of the part around the strongest heatmap signal. The heatmap can be wrong; trust what you can actually see.
Reply with ONLY a JSON object:
{{"verdict": "defect" or "good", "confidence": number 0-1, "severity": integer 0-3 (0 none, 1 cosmetic, 2 functional risk, 3 critical), "defect_description": short text, "incident_report": 2-3 sentence report for a production log}}"""


def data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


class ClaudeClient:
    def __init__(self, model="anthropic/claude-sonnet-5.5", timeout=120):
        key = os.getenv("AI_GATEWAY_API_KEY")
        assert key, "AI_GATEWAY_API_KEY is missing from .env"
        self.model = model
        self.http = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=timeout)

    def inspect(self, category, reference, test, heat, crop=None, retries=3):
        def img(label, im):
            return [{"type": "text", "text": label},
                    {"type": "image_url", "image_url": {"url": data_url(im)}}]

        content = (
            [{"type": "text", "text": PROMPT.format(category=category.replace("_", " "))}]
            + img("Image 1: reference part, defect-free.", reference)
            + img("Image 2: the part to inspect.", test)
            + img("Image 3: the part to inspect with anomaly heatmap overlay.", heat)
        )
        if crop is not None:
            content += img("Image 4: zoomed full resolution crop around the strongest anomaly sigmal.", crop)
        body = {"model": self.model, "max_tokens": 800,
                "messages": [{"role": "user", "content": content}]}
        for attempt in range(retries):
            t0 = time.perf_counter()
            r = self.http.post(URL, json=body)
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            data = r.json()
            usage = data.get("usage", {})
            return {
                "parsed": parse_json(data["choices"][0]["message"]["content"]),
                "latency_ms": ms,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }