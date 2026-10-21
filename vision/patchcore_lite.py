from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 224

tf = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class PatchCoreLite:
    def __init__(self, keep_ratio=0.1, seed=0):
        weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        self.net = models.wide_resnet50_2(weights=weights).to(DEVICE).eval()
        self.keep_ratio = keep_ratio
        self.rng = np.random.default_rng(seed)
        self.bank = None
        self.tau = None

    @torch.no_grad()
    def _features(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer1(x)
        f2 = n.layer2(x)
        f3 = n.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)   # local aggregation
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)  # B, 1536, 28, 28

    def fit(self, paths, bs=16):
        chunks = []
        for i in range(0, len(paths), bs):
            batch = [tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]
            f = self._features(torch.stack(batch).to(DEVICE))
            c = f.shape[1]
            f = f.permute(0, 2, 3, 1).reshape(-1, c)
            k = int(len(f) * self.keep_ratio)          # random subsample of patches
            idx = torch.from_numpy(self.rng.choice(len(f), k, replace=False)).to(DEVICE)
            chunks.append(f[idx])
        self.bank = torch.cat(chunks)

    @torch.no_grad()
    def anomaly_map(self, path):
        x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
        f = self._features(x)[0]
        c, h, w = f.shape
        p = f.permute(1, 2, 0).reshape(-1, c)
        d = torch.cdist(p, self.bank).min(dim=1).values.reshape(1, 1, h, w)
        m = F.interpolate(d, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        return gaussian_filter(m[0, 0].cpu().numpy(), sigma=4)

    def calibrate(self, good_paths):
        # threshold = highest score seen on held-out defect-free images
        self.tau = float(max(self.anomaly_map(p).max() for p in good_paths))

    def save(self, path):
        Path(path).parent.mkdir(exist_ok=True)
        torch.save({"bank": self.bank.cpu(), "tau": self.tau}, path)

    def load(self, path):
        s = torch.load(path)
        self.bank, self.tau = s["bank"].to(DEVICE), s["tau"]


def describe(m, tau):
    """Turn a heatmap into area (% of image) and a coarse location word."""
    mask = m > tau
    if not mask.any():
        return 0.0, "none"
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean() / m.shape[0], xs.mean() / m.shape[1]
    row = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
    col = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
    loc = "center" if (row, col) == ("middle", "center") else f"{row}-{col}"
    return float(mask.mean() * 100), loc