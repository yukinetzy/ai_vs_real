# AI-generated vs Real photo detection (Computer Vision, Assignment 2)

Upgrade of Assignment 1 (Baseline CNN and ResNet50 transfer learning) with a SOTA component and a working demo.

**Method:** frozen CLIP ViT-L/14 image embeddings + logistic regression (UnivFD-style linear probe).
**Explainability:** occlusion heatmap and nearest-neighbour gallery.
**Compared against:** A1 Keras ResNet50 and a reproduced ResNet50-feature baseline with the same head.

## Files

| File | Purpose |
|---|---|
| `A2_simple.ipynb` | Main notebook (Colab, GPU): baseline vs CLIP, ablation, explainability, latency, export for the app |
| `extras/A2_full_with_extras.ipynb` | Optional longer version: also data checks, learning curves, robustness, cross-generator test (SDXL-Turbo) |
| `detector.py` | Encoders, occlusion heatmap, nearest neighbours (shared by notebook and app) |
| `app.py` | Gradio web demo: upload/camera, prediction, heatmap, neighbours, latency and environment |
| `artifacts/` | Created by the notebook (section 11): `probe.joblib`, `gallery.npz`, `gallery/` |

## How to reproduce

1. Open the notebook in Google Colab, set the runtime to GPU, put the A1 dataset on Google Drive (it is not stored in this repo), point `DATA_DIR` to it (`ai_vs_real/<class folders>`), run all cells. Tables and figures are saved to `ai_vs_real_A2_results/` on Drive.
2. Download `artifacts.zip` (last cell) and unzip it next to `app.py` so that `artifacts/probe.joblib` exists.
3. Run the app locally:

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                         # open http://127.0.0.1:7860
```

The first run downloads the CLIP weights (about 900 MB). On CPU the explanation takes several seconds. Use `GRID=5` to make it faster.
For a public link use `SHARE=1 python app.py` (Gradio temporary link) or deploy `app.py`, `detector.py`, `requirements.txt` and `artifacts/` as a Hugging Face Space (Gradio SDK).

## Ethics and limitations

- Output is a probability estimate, not proof that an image is fake or real. Do not use it to accuse people.
- Small dataset (about 200 images), limited generators; accuracy on new generators and on heavily compressed images may be much lower (see robustness and cross-generator results in the report).
- Synthetic images are labelled as AI-generated everywhere (dataset folders, gallery captions, SDXL-Turbo test images). No real persons are depicted in the generated prompts.
- Real photos may show people. Get consent before publishing them, or remove `artifacts/gallery/` from the public repo.
- Uploaded images in the demo are processed in memory and not stored.

## References

- Radford et al., Learning Transferable Visual Models From Natural Language Supervision (CLIP), 2021.
- Ojha et al., Towards Universal Fake Image Detectors that Generalize Across Generative Models, CVPR 2023.
- Sauer et al., Adversarial Diffusion Distillation (SDXL-Turbo), 2023.
- open_clip: https://github.com/mlfoundations/open_clip
