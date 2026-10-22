import os

import httpx
from dotenv import load_dotenv

load_dotenv()
r = httpx.get(
    "https://ai-gateway.vercel.sh/v1/models",
    headers={"Authorization": f"Bearer {os.getenv('AI_GATEWAY_API_KEY')}"},
    timeout=30,
)
for m in r.json()["data"]:
    if m["id"].startswith("anthropic/"):
        print(m["id"])
        