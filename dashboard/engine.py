"""
Analytics engine for the ProtoNet Visual Analytics Dashboard.

Uses the real trained ProtoNet model to compute embeddings from
actual QuickDraw test images.  Same public interface as before so
all dashboard callbacks remain unchanged.
"""

import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)

# ── Subset of trained classes shown in the dashboard ──────────────
# Edit this list to show more/fewer of the 29 available classes.
CLASS_NAMES = [
    'cat', 'dog', 'fish', 'car', 'flower',
    'bicycle', 'bird', 'pizza', 'clock', 'lightning',
]

CLASS_COLORS = [
    '#58a6ff',   # cat
    '#3fb950',   # dog
    '#d29922',   # fish
    '#f85149',   # car
    '#bc8cff',   # flower
    '#79c0ff',   # bicycle
    '#56d364',   # bird
    '#e3b341',   # pizza
    '#ff7b72',   # clock
    '#ffa657',   # lightning
]

# ── Paths ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINT = _ROOT / 'checkpoints' / 'best_model.pt'
_DATA_DIR = _ROOT / 'data' / 'quickdraw' / 'test'
_MAX_PER_CLASS = 40   # images to load per class (out of ~75 available)
_INITIAL_SUPPORT = 5


class AnalyticsEngine:
    """Backend engine that wraps the real ProtoNet model."""

    def __init__(self):
        self.class_names: List[str] = list(CLASS_NAMES)
        self.n_classes: int = len(self.class_names)
        self.class_colors: List[str] = list(CLASS_COLORS[:self.n_classes])

        self.images: List[np.ndarray] = []
        self.labels: np.ndarray = np.array([], dtype=int)
        self.embeddings_hd: Optional[np.ndarray] = None   # (N, 512)
        self.embeddings_2d: Optional[np.ndarray] = None    # (N, 2)
        self.default_support: Dict[str, list] = {}

    # ------------------------------------------------------------------
    # Initialisation — loads real model, images, computes embeddings
    # ------------------------------------------------------------------

    def init_demo(self) -> "AnalyticsEngine":
        """Load real model & QuickDraw data, compute embeddings, run UMAP."""
        import torch
        import torchvision.transforms as T
        from models.encoder import get_encoder

        # ── load checkpoint ───────────────────────────────────────
        cp = torch.load(str(_CHECKPOINT), map_location='cpu', weights_only=False)

        encoder = get_encoder(
            encoder_type='conv4',
            config={
                'num_channels': 1,
                'embedding_dim': 512,
                'conv4': {
                    'channels': [64, 128, 256, 512],
                    'use_batchnorm': True,
                    'dropout': 0.1,
                },
            },
        )
        # Extract encoder weights from the full ProtoNet state dict
        if 'model_state_dict' in cp:
            sd = {
                k.replace('encoder.', '', 1): v
                for k, v in cp['model_state_dict'].items()
                if k.startswith('encoder.')
            }
            encoder.load_state_dict(sd, strict=True)
        elif 'encoder_state_dict' in cp:
            encoder.load_state_dict(cp['encoder_state_dict'])
        encoder.eval()

        transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        # ── load images ──────────────────────────────────────────
        self.images = []
        labels: List[int] = []

        for ci, cname in enumerate(self.class_names):
            cls_dir = _DATA_DIR / cname
            if not cls_dir.exists():
                logger.warning("Class directory not found: %s", cls_dir)
                continue
            img_files = sorted(cls_dir.glob('*.png'))[:_MAX_PER_CLASS]
            for f in img_files:
                arr = np.array(Image.open(f).convert('L'))
                self.images.append(arr)
                labels.append(ci)

        self.labels = np.array(labels, dtype=int)
        n = len(self.images)
        logger.info("Loaded %d images across %d classes", n, self.n_classes)

        # ── compute embeddings with real encoder ──────────────────
        all_embs: List[np.ndarray] = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch_imgs = self.images[i:i + batch_size]
                tensors = torch.stack([
                    transform(Image.fromarray(img.astype(np.uint8), mode='L'))
                    for img in batch_imgs
                ])
                embs = encoder(tensors)           # (B, 512)
                all_embs.append(embs.cpu().numpy())

        self.embeddings_hd = np.concatenate(all_embs, axis=0)

        # ── UMAP / t-SNE for 2-D scatter ────────────────────────
        self._compute_umap()

        # ── default support set (first K per class) ──────────────
        self.default_support = {}
        for ci, cname in enumerate(self.class_names):
            idxs = np.where(self.labels == ci)[0][:_INITIAL_SUPPORT].tolist()
            self.default_support[cname] = [
                {'idx': idx, 'weight': 1.0} for idx in idxs
            ]

        return self

    # ------------------------------------------------------------------
    # UMAP
    # ------------------------------------------------------------------

    def _compute_umap(self):
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42,
                           n_neighbors=15, min_dist=0.3)
        except ImportError:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        self.embeddings_2d = reducer.fit_transform(self.embeddings_hd)

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def image_to_base64(self, idx: int, size: int = 56) -> str:
        arr = self.images[idx]
        img = Image.fromarray(arr.astype(np.uint8), mode='L')
        img = img.resize((size, size), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

    def support_indices(self, sc: Dict) -> set:
        out: set = set()
        for items in sc.values():
            for item in items:
                out.add(item['idx'])
        return out

    def query_mask(self, sc: Dict) -> np.ndarray:
        si = self.support_indices(sc)
        return np.array([i not in si for i in range(len(self.labels))])

    # ------------------------------------------------------------------
    # Prototype computation (HD for accuracy, 2D for scatter)
    # ------------------------------------------------------------------

    def compute_prototypes(
        self, sc: Dict
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Return (protos_2d, protos_hd, class_order)."""
        order, p2d, phd = [], [], []
        for cname in self.class_names:
            items = sc.get(cname, [])
            if not items:
                continue
            idxs = [it['idx'] for it in items]
            wts = np.array([it['weight'] for it in items])
            wts = wts / wts.sum()
            order.append(cname)
            p2d.append(np.average(self.embeddings_2d[idxs], axis=0, weights=wts))
            phd.append(np.average(self.embeddings_hd[idxs], axis=0, weights=wts))
        return np.array(p2d), np.array(phd), order

    # ------------------------------------------------------------------
    # Classification (uses HD embeddings for real model accuracy)
    # ------------------------------------------------------------------

    def classify(
        self,
        sc: Dict,
        temperature: float = 1.0,
        proto_overrides: Optional[Dict] = None,
    ) -> Dict:
        p2d, phd, order = self.compute_prototypes(sc)
        if not order:
            return self._empty_result()

        # 2D overrides only affect the scatter; HD prototypes stay real
        if proto_overrides:
            for cname, pos in proto_overrides.items():
                if cname in order:
                    p2d[order.index(cname)] = pos

        qm = self.query_mask(sc)
        qhd = self.embeddings_hd[qm]
        qlabels = self.labels[qm]
        qidxs = np.where(qm)[0]

        # Classify using *high-dimensional* distances (real model behaviour)
        dists = cdist(qhd, phd, metric='euclidean')
        temp = max(temperature, 0.01)
        logits = -dists / temp
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)

        name2idx = {n: i for i, n in enumerate(order)}
        true_idx = np.array(
            [name2idx.get(self.class_names[l], -1) for l in qlabels]
        )

        nc = len(order)
        cm = np.zeros((nc, nc), dtype=int)
        for t, p in zip(true_idx, preds):
            if t >= 0:
                cm[t, p] += 1

        accs = {}
        for i, cn in enumerate(order):
            s = cm[i].sum()
            accs[cn] = cm[i, i] / s if s > 0 else 0.0

        total = cm.sum()
        overall = sum(cm[i, i] for i in range(nc)) / total if total > 0 else 0.0

        return {
            'preds': preds,
            'true_idx': true_idx,
            'probs': probs,
            'dists': dists,
            'cm': cm,
            'order': order,
            'accs': accs,
            'overall': overall,
            'query_idxs': qidxs,
            'p2d': p2d,
            'phd': phd,
        }

    def _empty_result(self):
        return {
            'preds': np.array([]),
            'true_idx': np.array([]),
            'probs': np.array([]),
            'dists': np.array([]),
            'cm': np.array([]),
            'order': [],
            'accs': {},
            'overall': 0.0,
            'query_idxs': np.array([]),
            'p2d': np.array([]),
            'phd': np.array([]),
        }

    def query_distances(self, qidx: int, sc: Dict, temperature: float = 1.0) -> Dict:
        """Distance breakdown for a single query (HD-based)."""
        _, phd, order = self.compute_prototypes(sc)
        if not order:
            return {'order': [], 'dists': [], 'probs': [], 'true': '', 'pred': ''}
        pt = self.embeddings_hd[qidx]
        d = np.sqrt(((phd - pt) ** 2).sum(axis=1))
        temp = max(temperature, 0.01)
        logits = -d / temp
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        true_name = self.class_names[self.labels[qidx]]
        pred_name = order[np.argmax(probs)]
        return {
            'order': order,
            'dists': d,
            'probs': probs,
            'true': true_name,
            'pred': pred_name,
        }

    # ------------------------------------------------------------------
    # Decision boundary mesh (2-D approximation for scatter overlay)
    # ------------------------------------------------------------------

    def decision_mesh(
        self, sc: Dict, temperature: float,
        proto_overrides: Optional[Dict] = None, res: int = 80,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p2d, _, order = self.compute_prototypes(sc)
        if not order:
            return np.array([]), np.array([]), np.array([])
        if proto_overrides:
            for cn, pos in proto_overrides.items():
                if cn in order:
                    p2d[order.index(cn)] = pos

        pad = 1.5
        xmin = self.embeddings_2d[:, 0].min() - pad
        xmax = self.embeddings_2d[:, 0].max() + pad
        ymin = self.embeddings_2d[:, 1].min() - pad
        ymax = self.embeddings_2d[:, 1].max() + pad
        xs = np.linspace(xmin, xmax, res)
        ys = np.linspace(ymin, ymax, res)
        xx, yy = np.meshgrid(xs, ys)
        grid = np.c_[xx.ravel(), yy.ravel()]
        d = cdist(grid, p2d)
        temp = max(temperature, 0.01)
        zz = np.argmin(d / temp, axis=1).reshape(xx.shape)
        return xx, ys, zz
