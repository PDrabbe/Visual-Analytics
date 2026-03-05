"""
Analytics engine for the ProtoNet Visual Analytics Dashboard.

Handles demo data generation, embedding computation, UMAP projection,
prototype computation, and classification logic.
"""

import numpy as np
from PIL import Image, ImageDraw
import io
import base64
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist

CLASS_NAMES = ['cat', 'dog', 'fish', 'car', 'tree', 'house', 'star', 'moon', 'flower']
CLASS_COLORS = [
    '#58a6ff', '#3fb950', '#d29922', '#f85149', '#bc8cff',
    '#79c0ff', '#56d364', '#e3b341', '#ff7b72',
]


class AnalyticsEngine:
    """Backend engine for the ProtoNet visual analytics dashboard."""

    def __init__(self):
        self.class_names: List[str] = list(CLASS_NAMES)
        self.n_classes: int = len(self.class_names)
        self.class_colors: List[str] = list(CLASS_COLORS[: self.n_classes])

        self.images: List[np.ndarray] = []
        self.labels: np.ndarray = np.array([], dtype=int)
        self.embeddings_hd: Optional[np.ndarray] = None
        self.embeddings_2d: Optional[np.ndarray] = None

        self.embedding_dim = 64
        self.n_per_class = 80
        self.initial_support = 5
        self.image_size = 28

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_demo(self) -> "AnalyticsEngine":
        """Generate synthetic demo data."""
        rng = np.random.RandomState(42)

        self.images = []
        labels = []
        for ci, cname in enumerate(self.class_names):
            for i in range(self.n_per_class):
                img = self._generate_sketch(cname, seed=ci * 10000 + i)
                self.images.append(img)
                labels.append(ci)
        self.labels = np.array(labels, dtype=int)

        self._generate_embeddings(rng)
        self._compute_umap()

        support_config: Dict[str, list] = {}
        for ci, cname in enumerate(self.class_names):
            idxs = np.where(self.labels == ci)[0][: self.initial_support].tolist()
            support_config[cname] = [{"idx": idx, "weight": 1.0} for idx in idxs]
        self.default_support = support_config
        return self

    # ------------------------------------------------------------------
    # Sketch generation
    # ------------------------------------------------------------------

    def _generate_sketch(self, class_name: str, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        sz = self.image_size
        img = Image.new("L", (sz, sz), 255)
        draw = ImageDraw.Draw(img)
        ox, oy = rng.randint(-2, 3), rng.randint(-2, 3)
        s = 1.0 + rng.uniform(-0.15, 0.15)
        cx, cy = sz // 2 + ox, sz // 2 + oy

        if class_name == "cat":
            r = int(7 * s)
            draw.ellipse([cx - r, cy - r + 2, cx + r, cy + r + 2], outline=0, width=1)
            draw.polygon(
                [(cx - r + 1, cy - r + 2), (cx - r - 1, cy - r - 4), (cx - r + 5, cy - r + 2)],
                outline=0,
            )
            draw.polygon(
                [(cx + r - 1, cy - r + 2), (cx + r + 1, cy - r - 4), (cx + r - 5, cy - r + 2)],
                outline=0,
            )
            draw.ellipse([cx - 4, cy - 1, cx - 2, cy + 1], fill=0)
            draw.ellipse([cx + 2, cy - 1, cx + 4, cy + 1], fill=0)

        elif class_name == "dog":
            r = int(7 * s)
            draw.ellipse([cx - r, cy - r + 2, cx + r, cy + r + 2], outline=0, width=1)
            draw.ellipse([cx - r - 2, cy - 2, cx - r + 3, cy + 6], outline=0, width=1)
            draw.ellipse([cx + r - 3, cy - 2, cx + r + 2, cy + 6], outline=0, width=1)
            draw.ellipse([cx - 3, cy, cx - 1, cy + 2], fill=0)
            draw.ellipse([cx + 1, cy, cx + 3, cy + 2], fill=0)
            draw.ellipse([cx - 1, cy + 3, cx + 1, cy + 5], fill=0)

        elif class_name == "fish":
            rw, rh = int(9 * s), int(5 * s)
            draw.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], outline=0, width=1)
            draw.polygon([(cx + rw - 1, cy), (cx + rw + 5, cy - 4), (cx + rw + 5, cy + 4)], outline=0)
            draw.ellipse([cx - rw + 3, cy - 2, cx - rw + 5, cy], fill=0)

        elif class_name == "car":
            w, h = int(10 * s), int(5 * s)
            draw.rectangle([cx - w, cy - h + 3, cx + w, cy + h], outline=0, width=1)
            draw.rectangle([cx - 5, cy - h - 2, cx + 5, cy - h + 3], outline=0, width=1)
            draw.ellipse([cx - w + 1, cy + h - 2, cx - w + 5, cy + h + 2], outline=0, fill=0)
            draw.ellipse([cx + w - 5, cy + h - 2, cx + w - 1, cy + h + 2], outline=0, fill=0)

        elif class_name == "tree":
            draw.rectangle([cx - 2, cy + 2, cx + 2, cy + 10], outline=0, fill=0)
            draw.polygon([(cx, cy - 10), (cx - 8, cy + 2), (cx + 8, cy + 2)], outline=0, width=1)

        elif class_name == "house":
            w, h = int(8 * s), int(6 * s)
            draw.rectangle([cx - w, cy - h + 4, cx + w, cy + h + 2], outline=0, width=1)
            draw.polygon([(cx - w - 2, cy - h + 4), (cx, cy - h - 5), (cx + w + 2, cy - h + 4)], outline=0, width=1)
            draw.rectangle([cx - 2, cy + h - 2, cx + 2, cy + h + 2], outline=0, width=1)

        elif class_name == "star":
            r_out, r_in = int(9 * s), int(4 * s)
            pts = []
            for i in range(10):
                a = np.pi / 2 + i * np.pi / 5
                r = r_out if i % 2 == 0 else r_in
                pts.append((cx + int(r * np.cos(a)), cy - int(r * np.sin(a))))
            draw.polygon(pts, outline=0, width=1)

        elif class_name == "moon":
            r = int(8 * s)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=1)
            r2 = int(7 * s)
            draw.ellipse([cx - r2 + 4, cy - r2, cx + r2 + 4, cy + r2], fill=255, outline=255)
            draw.arc([cx - r, cy - r, cx + r, cy + r], 0, 360, fill=0, width=1)

        elif class_name == "flower":
            r = int(3 * s)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, fill=0)
            for a in np.linspace(0, 2 * np.pi, 6, endpoint=False):
                px = cx + int(6 * s * np.cos(a))
                py = cy + int(6 * s * np.sin(a))
                pr = int(3 * s)
                draw.ellipse([px - pr, py - pr, px + pr, py + pr], outline=0, width=1)
            draw.line([(cx, cy + int(6 * s)), (cx, cy + 12)], fill=0, width=1)

        return np.array(img)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _generate_embeddings(self, rng: np.random.RandomState):
        n = len(self.labels)
        D = self.embedding_dim
        centroids = rng.randn(self.n_classes, D) * 1.5
        # Make similar classes overlap heavily
        centroids[1] = centroids[0] + rng.randn(D) * 0.5   # dog~cat (high overlap)
        centroids[8] = centroids[6] + rng.randn(D) * 0.6   # flower~star
        centroids[3] = centroids[5] + rng.randn(D) * 0.6   # car~house
        centroids[7] = centroids[8] + rng.randn(D) * 0.7   # moon~flower
        centroids[2] = centroids[7] + rng.randn(D) * 0.9   # fish~moon

        emb = np.zeros((n, D))
        for i in range(n):
            noise = 1.2 + rng.uniform(0, 1.2)
            emb[i] = centroids[self.labels[i]] + rng.randn(D) * noise
        self.embeddings_hd = emb

    def _compute_umap(self):
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.3)
        except ImportError:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        self.embeddings_2d = reducer.fit_transform(self.embeddings_hd)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def image_to_base64(self, idx: int, size: int = 56) -> str:
        arr = self.images[idx]
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
        img = img.resize((size, size), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def support_indices(self, sc: Dict) -> set:
        out: set = set()
        for items in sc.values():
            for item in items:
                out.add(item["idx"])
        return out

    def query_mask(self, sc: Dict) -> np.ndarray:
        si = self.support_indices(sc)
        return np.array([i not in si for i in range(len(self.labels))])

    # ------------------------------------------------------------------
    # Prototype computation
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
            idxs = [it["idx"] for it in items]
            wts = np.array([it["weight"] for it in items])
            wts = wts / wts.sum()
            order.append(cname)
            p2d.append(np.average(self.embeddings_2d[idxs], axis=0, weights=wts))
            phd.append(np.average(self.embeddings_hd[idxs], axis=0, weights=wts))
        return np.array(p2d), np.array(phd), order

    # ------------------------------------------------------------------
    # Classification
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

        # Apply 2D overrides
        if proto_overrides:
            for cname, pos in proto_overrides.items():
                if cname in order:
                    p2d[order.index(cname)] = pos

        qm = self.query_mask(sc)
        q2d = self.embeddings_2d[qm]
        qlabels = self.labels[qm]
        qidxs = np.where(qm)[0]

        dists = cdist(q2d, p2d, metric="euclidean")
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
            "preds": preds,
            "true_idx": true_idx,
            "probs": probs,
            "dists": dists,
            "cm": cm,
            "order": order,
            "accs": accs,
            "overall": overall,
            "query_idxs": qidxs,
            "p2d": p2d,
            "phd": phd,
        }

    def _empty_result(self):
        return {
            "preds": np.array([]),
            "true_idx": np.array([]),
            "probs": np.array([]),
            "dists": np.array([]),
            "cm": np.array([]),
            "order": [],
            "accs": {},
            "overall": 0.0,
            "query_idxs": np.array([]),
            "p2d": np.array([]),
            "phd": np.array([]),
        }

    def query_distances(self, qidx: int, sc: Dict, temperature: float = 1.0) -> Dict:
        p2d, _, order = self.compute_prototypes(sc)
        if not order:
            return {"order": [], "dists": [], "probs": [], "true": "", "pred": ""}
        pt = self.embeddings_2d[qidx]
        d = np.sqrt(((p2d - pt) ** 2).sum(axis=1))
        temp = max(temperature, 0.01)
        logits = -d / temp
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        true_name = self.class_names[self.labels[qidx]]
        pred_name = order[np.argmax(probs)]
        return {
            "order": order,
            "dists": d,
            "probs": probs,
            "true": true_name,
            "pred": pred_name,
        }

    # ------------------------------------------------------------------
    # Decision boundary mesh
    # ------------------------------------------------------------------

    def decision_mesh(
        self, sc: Dict, temperature: float, proto_overrides: Optional[Dict] = None, res: int = 80
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (xx, yy, zz) for decision boundary contour."""
        p2d, _, order = self.compute_prototypes(sc)
        if not order:
            return np.array([]), np.array([]), np.array([])
        if proto_overrides:
            for cn, pos in proto_overrides.items():
                if cn in order:
                    p2d[order.index(cn)] = pos

        pad = 1.5
        xmin, xmax = self.embeddings_2d[:, 0].min() - pad, self.embeddings_2d[:, 0].max() + pad
        ymin, ymax = self.embeddings_2d[:, 1].min() - pad, self.embeddings_2d[:, 1].max() + pad
        xs = np.linspace(xmin, xmax, res)
        ys = np.linspace(ymin, ymax, res)
        xx, yy = np.meshgrid(xs, ys)
        grid = np.c_[xx.ravel(), yy.ravel()]
        d = cdist(grid, p2d)
        temp = max(temperature, 0.01)
        zz = np.argmin(d / temp, axis=1).reshape(xx.shape)
        return xx, ys, zz
