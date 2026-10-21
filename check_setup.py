import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
print("API key loaded:", bool(os.getenv("AI_GATEWAY_API_KEY")))

root = Path("data/mvtec_ad")
for category in ["metal_nut", "bottle", "screw"]:
    print(category, "test folder found:", (root / category / "test").exists())