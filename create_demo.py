"""
generate_demo_session.py
========================
Generates session.json pre-configured for the demo video.

Run from the project root:
    python generate_demo_session.py

This loads the engine with exactly the 10 demo classes, computes the
default 5-shot support set for each, and writes session.json.
Open the dashboard immediately after — it will restore this session.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Demo configuration ────────────────────────────────────────────────────────
# Edit this list if you want a different class order.
# The order here is the order classes appear in the UMAP legend.
DEMO_CLASSES = [
    "dog",       # T1 primary confusion class (24% default accuracy)
    "horse",     # T1 pair with dog
    "truck",     # T3 volatility demo (20%–80% swing)
    "car",       # T1 secondary pair with truck
    "cat",       # T3 curation (+20pp headroom)
    "scissors",  # T3 curation (can reach 100%)
    "banana",    # T4 context (moon confusion partner)
    "key",       # competitor context for scissors
    "pizza",     # UMAP visual variety
    "triangle",  # anchor class (100% stable reference)
]

SESSION_PATH = Path("session.json")

# ── Load engine ───────────────────────────────────────────────────────────────
logger.info("Loading engine with %d demo classes...", len(DEMO_CLASSES))

from dashboard.engine import AnalyticsEngine

engine = AnalyticsEngine()
engine.init_demo(active_classes=DEMO_CLASSES)

logger.info("Engine ready — %d images across %d classes",
            len(engine.images), engine.n_classes)

# ── Verify all classes loaded ─────────────────────────────────────────────────
missing = [c for c in DEMO_CLASSES if c not in engine.class_names]
if missing:
    logger.warning("WARNING: these classes were not found in data/quickdraw/test/: %s", missing)
    logger.warning("Session will be written without them.")

loaded = engine.class_names
logger.info("Loaded classes: %s", loaded)

# ── Log default support indices for inspection ────────────────────────────────
logger.info("\nDefault support set (first 5 images per class):")
logger.info("  %-12s  %-5s  %-30s  %s", "class", "n_img", "support indices", "image paths")
logger.info("  " + "-"*80)
for cname in loaded:
    items = engine.default_support.get(cname, [])
    idxs  = [it["idx"] for it in items]
    paths = [str(engine.image_paths[i].name) for i in idxs]
    n_img = sum(1 for l in engine.labels if engine.class_names[l] == cname)
    logger.info("  %-12s  %-5d  %-30s  %s", cname, n_img, str(idxs), str(paths))

# ── Build session ─────────────────────────────────────────────────────────────
session = {
    "classes":        loaded,
    "sc":             engine.default_support,
    "colors":         {},
    # Store the 2D layout so the dashboard skips UMAP refit on load.
    # Comment this line out if you want a fresh UMAP layout each time.
    "embeddings_2d":  engine.embeddings_2d.tolist(),
}

SESSION_PATH.write_text(json.dumps(session, indent=2))
logger.info("\nSession written to: %s", SESSION_PATH.resolve())
logger.info("Start the dashboard with:  python -m dashboard.app")
logger.info("It will restore this 10-class session automatically.")