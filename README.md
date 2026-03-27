# ProtoNet Embedding Explorer — Visual Analytics Dashboard

An interactive visual analytics dashboard for exploring and manipulating Prototypical Network (ProtoNet) embeddings trained on the Google QuickDraw dataset. Built with Dash/Plotly, it lets you inspect few-shot sketch classification in real time — drag prototypes, swap support sets, draw new sketches, and watch accuracy change live.

---

## A Note on Development

The codebase for this project was largely written with the assistance of AI tools. However, all decisions regarding methodology, model architecture, UI design, application workflow, and system integration were made by us. The AI served as an implementation aid; the direction, reasoning, and design choices throughout are our own.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Data Setup](#data-setup)
4. [Running the Dashboard](#running-the-dashboard)
5. [Dashboard Features](#dashboard-features)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python** 3.9 or higher
- **pip** (Python package manager)
- **Git** (to clone the repository)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Visual-Analytics.git
cd Visual-Analytics
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch, Dash, Plotly, scikit-learn, UMAP, and all other required packages.

---

## Data Setup

The project uses the **Google QuickDraw** dataset (bitmap format). A download script is included that fetches 29 sketch classes and splits them into train/val/test sets.

```bash
python download_quickdraw.py
```

This will:
- Download 500 samples per class from Google Cloud Storage.
- Resize images to 64×64 pixels.
- Save them under `data/quickdraw/{train,val,test}/<class_name>/`.

The 29 classes span animals, food, vehicles, nature, objects, and geometric shapes (cat, dog, bird, fish, horse, apple, banana, cake, pizza, square, triangle, hexagon, diamond, car, bus, bicycle, truck, flower, cloud, lightning, mountain, clock, key, scissors, eyeglasses, door, table, chair, ladder).

> **Note:** We are skipping the training step as it not relevant to the problem statement. The dashboard only needs the `data/quickdraw/test/` split and a trained checkpoint. We have the needed checkpoint under `checkpoints/best_model.pt`.

### Adding new classes later

To download additional QuickDraw classes (that were not part of training) for few-shot evaluation:

```bash
python download_new_classes.py
```

Edit the class list at the bottom of that script to choose which new classes to fetch.

---

## Running the Dashboard

Once you have a trained checkpoint with prototypes and test data in place, launch the dashboard:

```bash
python -m dashboard.app
```

Then open your browser at **http://localhost:8050**.

On startup the dashboard will:
1. Load the trained ProtoNet encoder from `checkpoints/best_model.pt`.
2. Load test images from `data/quickdraw/test/` for the active classes.
3. Compute 512-dimensional embeddings for all images.
4. Fit a UMAP reducer to project embeddings to 2D.
5. Render the interactive scatter plot with decision boundaries.

The first launch takes 30–60 seconds (UMAP fitting). Subsequent launches restore from `session.json` if saved.

### Quick-start summary

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download data
python download_quickdraw.py

# 3. Launch dashboard
python -m dashboard.app
```

---

## Dashboard Features

- **UMAP Embedding Scatter** — 2D projection of all sketch embeddings with topographical decision boundaries showing class territories and confidence gradients.
- **Draggable Prototypes** — Drag ★ prototype markers to reposition them; the engine re-weights support items to match and accuracy updates instantly.
- **Support Set Editing** — Add/remove support images per class, adjust individual weights, and see leave-one-out accuracy impact before committing.
- **Temperature (τ) Control** — Slider to sharpen or soften the softmax classification boundary in real time.
- **Point Inspector** — Click any query point to see its true label, prediction, confidence distribution, and top-3 high-dimensional influencers with co-activation heatmaps.
- **Candidate Pool** — Browse all images for a class ranked by cosine similarity and competitor distance; toggle candidates in/out of the support set with projected accuracy deltas.
- **Draw Panel** — Sketch directly on a canvas, see the drawing embedded live on the scatter plot, and add it as a support or query point.
- **Class Management** — Toggle active classes, add new pre-trained or custom-drawn classes, and cycle per-class colors. Changes rebuild the UMAP projection.
- **Undo / Redo** — Full action history with step-by-step rollback.
- **Session Persistence** — Save/restore the full state (support sets, classes, colors) to `session.json`.

---

## Troubleshooting

**`FileNotFoundError: checkpoints/best_model.pt`**
You need to train the model first. Run `python main.py train --config config/config.yaml`, then `python generate_proto.py`.

**`No module named 'umap'`**
Install the UMAP package: `pip install umap-learn`.

**Dashboard is slow to start**
The initial UMAP fit over all test embeddings takes 30–60 seconds. This is normal. Subsequent interactions are fast because the reducer is cached.

**`CUDA out of memory` during training**
Reduce `n_way` or `n_query` in `config/config.yaml`, or set `system.device: cpu` to train on CPU.

**Port 8050 already in use**
Either stop the existing process or run with a different port: `python -m dashboard.app` and edit the `app.run(port=...)` call in `dashboard/app.py`.

**Empty scatter plot / "No data"**
Ensure test images exist under `data/quickdraw/test/<class_name>/` for the active classes. Run `python download_quickdraw.py` if the directory is empty.