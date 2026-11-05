# Shared helpers for the A2 project: image encoders, occlusion heatmap, nearest neighbours.
# Used by both the Colab notebook and the Gradio app (app.py).
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw


def load_image(path):
    im = Image.open(path)
    im.load()
    return im.convert("RGB")


class _Encoder:
    """Base class: turns a list of PIL images into a feature matrix."""
    l2 = False

    def _init_device(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    def move(self, device):
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.model.to(device=device, dtype=self.dtype)
        return self

    @torch.no_grad()
    def encode(self, imgs, bs=64):
        out = []
        for i in range(0, len(imgs), bs):
            x = torch.stack([self.preprocess(im) for im in imgs[i:i + bs]])
            x = x.to(self.device, self.dtype)
            f = self._forward(x).float()
            if self.l2:
                f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu().numpy())
        return np.concatenate(out)


class ClipEncoder(_Encoder):
    """Frozen CLIP image encoder (open_clip). L2-normalised embeddings."""
    l2 = True

    def __init__(self, arch="ViT-L-14", pretrained="openai", device=None):
        import open_clip
        self._init_device(device)
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model = model.eval().to(device=self.device, dtype=self.dtype)
        self.preprocess = preprocess
        self.name = f"CLIP {arch} ({pretrained})"

    def _forward(self, x):
        return self.model.encode_image(x)


class ResNetEncoder(_Encoder):
    """ImageNet ResNet50 feature extractor (2048-d). Used as the A1-style baseline."""

    def __init__(self, device=None):
        import torchvision
        self._init_device(device)
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        m = torchvision.models.resnet50(weights=w)
        m.fc = torch.nn.Identity()
        self.model = m.eval().to(device=self.device, dtype=self.dtype)
        self.preprocess = w.transforms()
        self.name = "ResNet50 (ImageNet)"

    def _forward(self, x):
        return self.model(x)


def occlusion_heatmap(encoder, clf, img, grid=7, target=None, fill=(124, 116, 104)):
    """Model-agnostic explanation: slide a grey square over the image and measure
    how much the probability of the target class drops. Big drop = important region.
    Returns (heat[grid, grid], base_image_224)."""
    base = ImageOps.fit(img.convert("RGB"), (224, 224), Image.BICUBIC)
    p0 = clf.predict_proba(encoder.encode([base]))[0]
    t = int(p0.argmax()) if target is None else int(target)
    step = 224 / grid
    variants = []
    for r in range(grid):
        for c in range(grid):
            v = base.copy()
            ImageDraw.Draw(v).rectangle(
                [int(c * step), int(r * step), int((c + 1) * step) - 1, int((r + 1) * step) - 1], fill=fill)
            variants.append(v)
    ps = clf.predict_proba(encoder.encode(variants))[:, t]
    heat = (p0[t] - ps).reshape(grid, grid)
    return heat, base


def heat_overlay(base, heat, strength=0.65):
    """Blend a coarse heatmap (positive = supports the decision) onto the image."""
    import matplotlib
    h = np.clip(heat, 0, None)
    h = h / (h.max() + 1e-8)
    h_img = Image.fromarray((h * 255).astype(np.uint8)).resize(base.size, Image.BICUBIC)
    a = np.asarray(h_img, dtype=np.float32)[..., None] / 255.0
    rgb = matplotlib.colormaps["jet"](a[..., 0])[..., :3] * 255.0
    out = np.asarray(base.convert("RGB"), dtype=np.float32) * (1 - a * strength) + rgb * (a * strength)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def nearest_neighbors(q, gallery, k=5):
    """Cosine-similarity neighbours (embeddings are L2-normalised)."""
    sims = gallery @ q
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]
