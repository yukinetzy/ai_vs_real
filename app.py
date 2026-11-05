"""Gradio demo: upload or take a photo -> AI/Real prediction -> occlusion heatmap + nearest training images.

Run:  python app.py        (needs the artifacts/ folder exported by the notebook)
Env:  SHARE=1 for a public link, GRID=7 for the occlusion grid size.
"""
import os
import platform
import time

import gradio as gr
import joblib
import numpy as np
import torch

from detector import ClipEncoder, heat_overlay, nearest_neighbors, occlusion_heatmap
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
GRID = int(os.environ.get("GRID", "7"))

bundle = joblib.load(os.path.join(ART, "probe.joblib"))
clf, names, ai_idx = bundle["clf"], bundle["class_names"], bundle["ai_idx"]
enc = ClipEncoder(bundle["arch"], bundle["pretrained"])
gal = np.load(os.path.join(ART, "gallery.npz"))
G, G_labels, G_files = gal["emb"], gal["labels"], gal["files"]

DEVICE = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
ENV = f"{platform.system()} {platform.release()}, {DEVICE}, torch {torch.__version__}"
pred_times = []

# warm-up so the first user request is not slower than the rest
enc.encode([Image.new("RGB", (224, 224))])


def analyze(img):
    if img is None:
        return None, None, [], "Please upload or capture an image."
    pil = img.convert("RGB")

    t0 = time.perf_counter()
    q = enc.encode([pil])
    proba = clf.predict_proba(q)[0]
    t_pred = (time.perf_counter() - t0) * 1000
    pred_times.append(t_pred)

    t0 = time.perf_counter()
    heat, base = occlusion_heatmap(enc, clf, pil, grid=GRID, target=int(proba.argmax()))
    overlay = heat_overlay(base, heat)
    t_expl = (time.perf_counter() - t0) * 1000

    idx, sims = nearest_neighbors(q[0], G, k=5)
    gallery = [(Image.open(os.path.join(ART, "gallery", str(G_files[i]))), f"{names[G_labels[i]]} | sim {s:.2f}")
               for i, s in zip(idx, sims)]

    label = {names[i]: float(proba[i]) for i in range(len(names))}
    info = (
        f"**Prediction latency:** {t_pred:.0f} ms (average over {len(pred_times)} requests: {np.mean(pred_times):.0f} ms)  \n"
        f"**Explanation latency:** {t_expl:.0f} ms ({GRID * GRID + 1} forward passes)  \n"
        f"**Environment:** {ENV}  \n"
        f"**Model:** {enc.name} + logistic regression probe"
    )
    return label, overlay, gallery, info


with gr.Blocks(title="AI vs Real photo detector") as demo:
    gr.Markdown(
        "# AI-generated vs Real photo detector\n"
        "Frozen CLIP ViT-L/14 features + linear probe (course project, Computer Vision, Astana IT University).\n\n"
        "**Notice:** this is a statistical estimate trained on a small dataset, not proof. It can be wrong, especially for new generators, "
        "heavily compressed images or screenshots. Do not use it to accuse anyone. Uploaded images are processed in memory and not stored. "
        "Only upload images you have the right to use, and do not upload photos of people without their consent."
    )
    with gr.Row():
        inp = gr.Image(type="pil", label="Upload or take a photo", sources=["upload", "webcam"])
        with gr.Column():
            label = gr.Label(num_top_classes=2, label="Prediction")
            info = gr.Markdown()
    btn = gr.Button("Analyze", variant="primary")
    with gr.Row():
        heat = gr.Image(label="Occlusion heatmap (red = regions that support the prediction)")
        nn = gr.Gallery(label="Nearest training images (label | cosine similarity)", columns=5, height=240)
    gr.Markdown("Training images in the gallery are real photos or **AI-generated (synthetic)** images, as shown by their label.")
    btn.click(analyze, inp, [label, heat, nn, info])

if __name__ == "__main__":
    demo.launch(share=os.environ.get("SHARE") == "1")
