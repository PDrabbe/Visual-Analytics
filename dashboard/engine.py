import numpy as np
from PIL import Image
import io
import uuid
import shutil
import json
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
    '#f25a5a',  # red
    '#f2975a',  # orange
    '#f2d35a',  # amber
    '#d3f25a',  # yellow-green
    '#97f25a',  # lime
    '#5af25a',  # green
    '#5af297',  # spring-green
    '#5af2d3',  # cyan-mint
    '#5ad3f2',  # sky
    '#5a97f2',  # blue
    '#5a5af2',  # blue-violet
    '#975af2',  # violet
    '#d35af2',  # magenta
    '#f25ad3',  # pink
    '#f25a97',  # rose
]

# ── Paths ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINT = _ROOT / 'checkpoints' / 'best_model.pt'
_DATA_DIR = _ROOT / 'data' / 'quickdraw' / 'test'
_CUSTOM_DRAWINGS_DIR = _ROOT / 'custom_drawings'
_EXCLUDED_PATH = _ROOT / 'excluded_images.json'
_MAX_PER_CLASS = 40   # images to load per class (out of ~75 available)
_INITIAL_SUPPORT = 5


class AnalyticsEngine:
    """Backend engine that wraps the ProtoNet model."""

    def __init__(self):
        self.class_names: List[str] = list(CLASS_NAMES)
        self.n_classes: int = len(self.class_names)
        self.class_colors: List[str] = list(CLASS_COLORS[:self.n_classes])

        self.images: List[np.ndarray] = []
        self.labels: np.ndarray = np.array([], dtype=int)
        self.embeddings_hd: Optional[np.ndarray] = None   # (N, 512)
        self.embeddings_2d: Optional[np.ndarray] = None    # (N, 2)
        self.default_support: Dict[str, list] = {}
        self.image_paths: List[Path] = []
        self.excluded_indices: set = set()

        # Set by init_demo
        self._encoder = None        # torch.nn.Module (conv4)
        self._transform = None      # torchvision transform pipeline
        self._umap_reducer = None   # fitted UMAP reducer for .transform()
        self._kd_tree = None        # KDTree over embeddings_2d for mesh lookup

        # In-process caches (cleared when arrays are mutated)
        self._b64_cache: dict = {}      # (idx, size) -> base64 string
        self._coact_cache: dict = {}    # (q_idx, s_idx, size) -> (str, str)

    # ------------------------------------------------------------------
    # Initialisation — loads real model, images, computes embeddings
    # ------------------------------------------------------------------

    def init_demo(self, active_classes: Optional[List[str]] = None,
                  loaded_embeddings_2d: Optional[List[List[float]]] = None) -> "AnalyticsEngine":
        """Load real model & QuickDraw data, compute embeddings, run UMAP.

        If *active_classes* is provided it overrides CLASS_NAMES.
        """
        if active_classes:
            self.class_names = [c for c in active_classes if (_DATA_DIR / c).exists()]
        
        self.n_classes = len(self.class_names)
        self.class_colors = list(CLASS_COLORS[:self.n_classes])

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
        if 'model_state_dict' in cp:
            sd = {k.replace('encoder.', '', 1): v for k, v in cp['model_state_dict'].items() if k.startswith('encoder.')}
            encoder.load_state_dict(sd, strict=True)
        elif 'encoder_state_dict' in cp:
            encoder.load_state_dict(cp['encoder_state_dict'])
        encoder.eval()

        self._encoder = encoder
        transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
        self._transform = transform

        # ── load images ──────────────────────────────────────────
        self.images: List[np.ndarray] = []
        self.labels: np.ndarray = np.array([], dtype=int)
        self.image_paths: List[str] = []
        self.excluded_indices = set()

        if _EXCLUDED_PATH.exists():
            try:
                with open(_EXCLUDED_PATH) as f:
                    self.excluded_indices = set(json.load(f))
            except Exception: pass

        n = 0
        for ci, cname in enumerate(self.class_names):
            cls_dir = _DATA_DIR / cname
            if not cls_dir.exists(): continue
            img_files = sorted(cls_dir.glob('*.png'))
            valid_count = 0
            for f in img_files:
                name = str(f.relative_to(_DATA_DIR))
                self.images.append(np.array(Image.open(f).convert('L')))
                self.labels = np.append(self.labels, ci)
                self.image_paths.append(f)
                if name not in self.excluded_indices:
                    valid_count += 1
                n += 1
                if valid_count >= _MAX_PER_CLASS:
                    break

        logger.info("Loaded %d images across %d classes", n, self.n_classes)

        # ── compute embeddings ──────────────────
        all_embs: List[np.ndarray] = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch_imgs = self.images[i:i + batch_size]
                tensors = torch.stack([transform(Image.fromarray(img.astype(np.uint8), mode='L')) for img in batch_imgs])
                embs = encoder(tensors)
                all_embs.append(embs.cpu().numpy())

        self.embeddings_hd = np.concatenate(all_embs, axis=0)

        # ── UMAP ────────────────────────
        if loaded_embeddings_2d is not None and len(loaded_embeddings_2d) == n:
            logger.info("Applying saved layout from session.")
            self.embeddings_2d = np.array(loaded_embeddings_2d)
            from scipy.spatial import cKDTree
            self._kd_tree = cKDTree(self.embeddings_2d)
            self._umap_reducer = None
        else:
            self._compute_umap()

        # ── default support set ──────────────
        self.default_support = {}
        for ci, cname in enumerate(self.class_names):
            idxs = np.where(self.labels == ci)[0][:_INITIAL_SUPPORT].tolist()
            self.default_support[cname] = [{'idx': idx, 'weight': 1.0} for idx in idxs]

        return self

    # ------------------------------------------------------------------
    # UMAP
    # ------------------------------------------------------------------

    def _compute_umap(self):
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError(
                "umap-learn is required. Install it with: pip install umap-learn"
            )
        reducer = UMAP(n_components=2, random_state=42,
                       n_neighbors=15, min_dist=0.3)
        self._umap_reducer = reducer
        self.embeddings_2d = reducer.fit_transform(self.embeddings_hd)
        from scipy.spatial import cKDTree
        self._kd_tree = cKDTree(self.embeddings_2d)
        self._b64_cache = {}
        self._coact_cache = {}


    # ------------------------------------------------------------------
    # Persistence — session
    # ------------------------------------------------------------------

    def save_excluded(self) -> None:
        """Persist the current excluded_indices set to excluded_images.json."""
        try:
            paths = []
            for i in self.excluded_indices:
                if i < len(self.image_paths):
                    try:
                        paths.append(str(self.image_paths[i].relative_to(_DATA_DIR)))
                    except ValueError:
                        pass
            with open(_EXCLUDED_PATH, 'w') as f:
                json.dump(paths, f)
        except Exception as e:
            logger.warning("save_excluded failed: %s", e)

    def save_session(self, path: str, state: Dict) -> None:
        """Serialize central application state to disk."""
        try:
            with open(path, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.warning("save_session failed: %s", e)

    def load_session(self, path: str) -> Optional[Dict]:
        """Return full saved session data or None if not found."""
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def available_classes(data_dir: Optional[str] = None) -> List[str]:
        """Return all class names found in the QuickDraw test directory."""
        d = Path(data_dir) if data_dir else _DATA_DIR
        if not d.exists():
            return list(CLASS_NAMES)
        return sorted(p.name for p in d.iterdir() if p.is_dir())

    @staticmethod
    def available_custom_drawing_classes() -> List[dict]:
        """Return info about classes in custom_drawings/ available for import.

        Returns a list of dicts with keys: name, count, preview_paths.
        """
        if not _CUSTOM_DRAWINGS_DIR.exists():
            return []
        result = []
        for p in sorted(_CUSTOM_DRAWINGS_DIR.iterdir()):
            if not p.is_dir():
                continue
            imgs = sorted(p.glob('*.png'))
            if not imgs:
                continue
            result.append({
                "name": p.name,
                "count": len(imgs),
                "preview_paths": [str(f) for f in imgs[:5]],
            })
        return result

    def add_class_from_custom_drawings(self, name: str, source_folder: Optional[str] = None) -> bool:
        """Load images from custom_drawings/<source_folder>/ and add as class <name>.

        *source_folder* defaults to *name* if not provided.
        Returns True on success, False on failure.
        """
        if name in self.class_names:
            logger.warning("Class %r already loaded", name)
            return False
        if len(self.class_names) >= len(CLASS_COLORS):
            logger.warning("Class cap reached (%d)", len(CLASS_COLORS))
            return False

        folder = source_folder or name
        cls_dir = _CUSTOM_DRAWINGS_DIR / folder
        if not cls_dir.exists():
            logger.warning("Custom drawings folder not found: %s", cls_dir)
            return False

        # Persistence: Copy images to data/quickdraw/test/{name} with drawn_ prefix
        target_dir = _DATA_DIR / name
        target_dir.mkdir(parents=True, exist_ok=True)

        import torch
        ci = len(self.class_names)  # new class index
        new_images: List[np.ndarray] = []
        new_labels: List[int] = []
        new_image_paths: List[Path] = []

        img_files = sorted(cls_dir.glob('*.png'))
        for i, f in enumerate(img_files[:_MAX_PER_CLASS]):
            target_path = target_dir / f"drawn_{i:04d}.png"
            shutil.copy2(f, target_path)
            
            arr = np.array(Image.open(target_path).convert('L'))
            new_images.append(arr)
            new_labels.append(ci)
            new_image_paths.append(target_path)

        if not new_images:
            logger.warning("No PNG images found in %s", cls_dir)
            return False

        # Embed with frozen encoder
        all_embs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(new_images), 64):
                batch = new_images[i:i + 64]
                tensors = torch.stack([
                    self._transform(
                        Image.fromarray(img.astype(np.uint8), mode='L'))
                    for img in batch
                ])
                all_embs.append(self._encoder(tensors).cpu().numpy())
        new_hd = np.concatenate(all_embs, axis=0)

        # Project into existing 2D UMAP space
        new_2d = self._umap_reducer.transform(new_hd)

        # Append to all arrays
        self.images.extend(new_images)
        self.labels = np.concatenate(
            [self.labels, np.array(new_labels, dtype=int)])
        self.embeddings_hd = np.vstack([self.embeddings_hd, new_hd])
        self.embeddings_2d = np.vstack([self.embeddings_2d, new_2d])
        self.image_paths.extend(new_image_paths)

        # Rebuild KD-tree
        from scipy.spatial import cKDTree
        self._kd_tree = cKDTree(self.embeddings_2d)
        self._b64_cache = {}
        self._coact_cache = {}

        self.class_names.append(name)
        self.n_classes = len(self.class_names)
        self.class_colors = list(CLASS_COLORS[:self.n_classes])

        # Default support for new class (first K images)
        idxs = np.where(self.labels == ci)[0][:_INITIAL_SUPPORT].tolist()
        self.default_support[name] = [
            {'idx': int(i), 'weight': 1.0} for i in idxs
        ]
        logger.info("Imported %d images from custom_drawings/%s as class %r",
                    len(new_images), folder, name)
        return True

    # ------------------------------------------------------------------
    # Dynamic class management
    # ------------------------------------------------------------------

    def add_class(self, name: str) -> bool:
        """Load images + embeddings for *name* and append to all arrays.

        Returns True on success, False if the data directory is missing
        or the class limit is reached.
        """
        if name in self.class_names:
            return True
        if len(self.class_names) >= len(CLASS_COLORS):
            logger.warning("Class cap reached (%d)", len(CLASS_COLORS))
            return False

        cls_dir = _DATA_DIR / name
        if not cls_dir.exists():
            logger.warning("Class directory not found: %s", cls_dir)
            return False

        import torch
        new_images: List[np.ndarray] = []
        new_labels: List[int] = []
        ci = len(self.class_names)   # new class index

        excluded = set()
        if _EXCLUDED_PATH.exists():
            try:
                with open(_EXCLUDED_PATH) as f:
                    excluded = set(json.load(f))
            except Exception:
                pass

        img_files = sorted(cls_dir.glob('*.png'))
        valid_count = 0
        new_image_paths = []
        for f in img_files:
            rel_path = str(f.relative_to(_DATA_DIR))
            arr = np.array(Image.open(f).convert('L'))
            new_images.append(arr)
            new_labels.append(ci)
            new_image_paths.append(f)
            
            if rel_path in excluded:
                pass
            else:
                valid_count += 1
            if valid_count >= _MAX_PER_CLASS:
                break

        if not new_images:
            return False

        # Embed with frozen encoder
        all_embs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(new_images), 64):
                batch = new_images[i:i + 64]
                tensors = torch.stack([
                    self._transform(
                        Image.fromarray(img.astype(np.uint8), mode='L'))
                    for img in batch
                ])
                all_embs.append(self._encoder(tensors).cpu().numpy())
        new_hd = np.concatenate(all_embs, axis=0)

        # Project into existing 2D space using the fitted UMAP reducer
        new_2d = self._umap_reducer.transform(new_hd)

        # Append to all arrays
        start_idx = len(self.images)
        self.images.extend(new_images)
        self.labels = np.concatenate(
            [self.labels, np.array(new_labels, dtype=int)])
        self.embeddings_hd = np.vstack([self.embeddings_hd, new_hd])
        self.embeddings_2d = np.vstack([self.embeddings_2d, new_2d])

        for i, f in enumerate(new_image_paths):
            rel_path = str(f.relative_to(_DATA_DIR))
            if rel_path in excluded:
                self.excluded_indices.add(start_idx + i)
        self.image_paths.extend(new_image_paths)

        # Rebuild KD-tree
        from scipy.spatial import cKDTree
        self._kd_tree = cKDTree(self.embeddings_2d)
        self._b64_cache = {}
        self._coact_cache = {}

        self.class_names.append(name)
        self.n_classes = len(self.class_names)
        self.class_colors = list(CLASS_COLORS[:self.n_classes])

        # Default support for new class
        idxs = np.where(self.labels == ci)[0][:_INITIAL_SUPPORT].tolist()
        self.default_support[name] = [
            {'idx': int(i), 'weight': 1.0} for i in idxs
        ]
        return True

    def remove_classes(self, names: List[str]) -> Dict[int, int]:
        """Remove all images/embeddings for multiple classes.
        
        Returns a dictionary mapping old image indices to new image indices.
        """
        if not names:
            return {}
            
        valid_names = [n for n in names if n in self.class_names]
        if not valid_names:
            return {}
        if len(self.class_names) - len(valid_names) < 1:
            logger.warning("Cannot remove all classes")
            return {}
            
        remove_cis = [self.class_names.index(n) for n in valid_names]
        
        # Slicing mask
        keep = np.ones(len(self.labels), dtype=bool)
        for ci in remove_cis:
            keep[self.labels == ci] = False
            
        # Map old indices to new
        old_to_new = {}
        new_exc = set()
        new_idx = 0
        for i, k in enumerate(keep):
            if k:
                old_to_new[i] = new_idx
                if i in self.excluded_indices:
                    new_exc.add(new_idx)
                new_idx += 1
                
        self.excluded_indices = new_exc
        
        # Slice arrays
        keep_idxs = [i for i, k in enumerate(keep) if k]
        self.image_paths = [self.image_paths[i] for i in keep_idxs]
        self.images = [self.images[i] for i in keep_idxs]
        
        # Re-index labels
        new_labels = self.labels[keep].copy()
        for ci in sorted(remove_cis, reverse=True):
            new_labels[new_labels > ci] -= 1
        self.labels = new_labels
        
        self.embeddings_hd = self.embeddings_hd[keep]
        self.embeddings_2d = self.embeddings_2d[keep]
        
        # Rebuild KD-tree
        from scipy.spatial import cKDTree
        self._kd_tree = cKDTree(self.embeddings_2d)
        self._b64_cache = {}
        self._coact_cache = {}
        
        # Update class lists
        for n in valid_names:
            self.class_names.remove(n)
            self.default_support.pop(n, None)
            
        self.n_classes = len(self.class_names)
        self.class_colors = list(CLASS_COLORS[:self.n_classes])
        
        return old_to_new

    def sync_to_classes(self, target: List[str]) -> tuple:
        """Bring the engine's class set in line with *target*.

        Removes classes that are currently loaded but absent from *target*,
        then adds classes that are in *target* but not yet loaded.

        Returns (idx_map, readded) where:
        - idx_map: old-to-new index map from any remove_classes call
        - readded: set of class names that were re-added via add_class.
          Their images are at brand-new indices (appended at the end), so
          any sc indices from an old snapshot are invalid for them — callers
          must fall back to engine.default_support for these classes.
        """
        current = set(self.class_names)
        wanted  = list(target)

        to_remove = [c for c in current if c not in set(wanted)]
        to_add    = [c for c in wanted  if c not in current]

        idx_map: Dict[int, int] = {}
        if to_remove:
            idx_map = self.remove_classes(to_remove)

        readded: set = set()
        for name in to_add:
            if self.add_class(name):
                readded.add(name)

        return idx_map, readded


    # ------------------------------------------------------------------

    def support_diagnostics(
        self,
        cname: str,
        sc: Dict,
        temperature: float = 1.0,
        fast: bool = False,
    ) -> List[Dict]:
        """
        Compute diagnostics for every support image of *cname*.

        Returns a list (same order as ``sc[cname]``) of dicts::

            {
                "idx":        int,       # image index
                "loo_delta":  float,     # overall-accuracy change when this
                                         #   image is removed  (negative = hurts)
                "dist":       float,     # Euclidean distance to own prototype
                                         #   (computed *with* this image included)
                "competitor_dist": float, # euclidean distance to nearest
                                         #   competing prototype
                "competitor_name": str,  # which class is closest competitor
            }
        """
        items = sc.get(cname, [])
        if not items:
            return []

        # Baseline overall accuracy with the full support set
        base_res = self.classify(sc, temperature)
        base_acc = base_res["overall"]

        # Current prototypes (HD) — needed for own and competitor Euclidean distances
        _, phd, order = self.compute_prototypes(sc)
        if cname not in order:
            return [{"idx": it["idx"], "loo_delta": 0.0,
                     "dist": 0.0, "competitor_dist": 0.0,
                     "competitor_name": ""} for it in items]

        ci = order.index(cname)
        own_proto = phd[ci]          # (512,)

        # Competing prototypes (all except own class)
        other_idxs = [i for i in range(len(order)) if i != ci]
        other_protos = phd[other_idxs]                       # (C-1, 512)
        other_names = [order[i] for i in other_idxs]

        results = []
        for item in items:
            idx = item["idx"]
            emb = self.embeddings_hd[idx]                    # (512,)

            # ── Euclidean distance to own prototype ────────────
            dist = float(np.sqrt(((emb - own_proto) ** 2).sum()))

            # ── distance to nearest competing prototype ───────
            if len(other_protos) > 0:
                dists_to_others = np.linalg.norm(
                    other_protos - emb[None, :], axis=1)
                nearest_i = int(np.argmin(dists_to_others))
                comp_dist = float(dists_to_others[nearest_i])
                comp_name = other_names[nearest_i]
            else:
                comp_dist = float("inf")
                comp_name = ""

            # ── leave-one-out accuracy delta ──────────────────
            if fast:
                loo_delta = None
            elif len(items) <= 1:
                # Can't remove the only support image — delta undefined
                loo_delta = 0.0
            else:
                loo_sc = {c: list(v) for c, v in sc.items()}
                loo_sc[cname] = [it for it in loo_sc[cname]
                                 if it["idx"] != idx]
                loo_res = self.classify(loo_sc, temperature)
                loo_delta = float(loo_res["overall"] - base_acc)

            results.append({
                "idx": idx,
                "loo_delta": loo_delta,
                "dist": dist,
                "competitor_dist": comp_dist,
                "competitor_name": comp_name,
            })

        return results

    def support_loo_delta(self, cname: str, idx: int, sc: Dict, temp: float = 1.0) -> float:
        """
        Compute LOO delta for a single support image.
        """
        items = sc.get(cname, [])
        if len(items) <= 1:
            return 0.0

        base_res = self.classify(sc, temp)
        base_acc = base_res["overall"]

        loo_sc = {c: list(v) for c, v in sc.items()}
        loo_sc[cname] = [it for it in loo_sc[cname] if it["idx"] != idx]
        
        loo_res = self.classify(loo_sc, temp)
        return float(loo_res["overall"] - base_acc)

    def class_images_pool(self, cname: str, sc: Dict) -> List[Dict]:
        """
        Returns ALL images of cname with:
        {idx, dist, competitor_dist, is_support}

        dist             — Euclidean distance to own prototype (lower = better candidate)
        competitor_dist  — Euclidean distance to nearest competitor prototype (higher = better)
        """
        _, phd, order = self.compute_prototypes(sc)
        if cname not in order:
            return []

        ci = order.index(cname)
        own_proto = phd[ci]

        other_idxs = [i for i in range(len(order)) if i != ci]
        other_protos = phd[other_idxs] if other_idxs else np.array([])

        all_idxs = np.where(self.labels == self.class_names.index(cname))[0]
        sc_idxs = self.support_indices(sc)

        results = []
        for idx in all_idxs:
            emb = self.embeddings_hd[idx]

            # Euclidean distance to own prototype
            dist = float(np.sqrt(((emb - own_proto) ** 2).sum()))

            if len(other_protos) > 0:
                dists_to_others = np.linalg.norm(other_protos - emb[None, :], axis=1)
                comp_dist = float(np.min(dists_to_others))
            else:
                comp_dist = float("inf")

            results.append({
                "idx": int(idx),
                "dist": dist,
                "competitor_dist": comp_dist,
                "is_support": int(idx) in sc_idxs
            })

        return results

    def candidate_add_delta(self, cname: str, cand_idx: int, sc: Dict, temp: float = 1.0) -> float:
        """
        Runs one trial classify with cand_idx added, returns accuracy delta
        """
        base_res = self.classify(sc, temp)
        base_acc = base_res["overall"]

        trial_sc = {c: list(v) for c, v in sc.items()}
        trial_sc[cname] = trial_sc.get(cname, []) + [{"idx": cand_idx, "weight": 1.0}]

        trial_res = self.classify(trial_sc, temp)
        return float(trial_res["overall"] - base_acc)

    # ------------------------------------------------------------------
    # Image helpers
    def encode_image(self, pil_image):
        """
        Runs _transform → _encoder → _umap_reducer.transform()
        Returns (hd_embedding, position_2d)
        """
        import torch
        img_t = self._transform(pil_image)
        if len(img_t.shape) == 3:
            img_t = img_t.unsqueeze(0)
            
        self._encoder.eval()
        with torch.no_grad():
            emb_hd = self._encoder(img_t)
            emb_hd = emb_hd.cpu().numpy()[0]
            
        pos_2d = self._umap_reducer.transform(emb_hd.reshape(1, -1))[0]
        return emb_hd, pos_2d

    def add_custom_image(self, cname: str, img_arr: np.ndarray, hd_emb: np.ndarray, pos_2d: np.ndarray) -> int:
        """
        Appends single drawn image to global matrices.
        Returns the new global index, or -1 on error.
        """
        if cname not in self.class_names:
            return -1
        ci = self.class_names.index(cname)
        
        from pathlib import Path
        import time

        idx = len(self.images)
        self.images.append(img_arr)
        self.labels = np.concatenate([self.labels, np.array([ci], dtype=int)])
        self.embeddings_hd = np.vstack([self.embeddings_hd, hd_emb.reshape(1, -1)])
        self.embeddings_2d = np.vstack([self.embeddings_2d, pos_2d.reshape(1, -1)])
        
        if hasattr(self, 'image_paths'):
            self.image_paths.append(Path(f"drawings/{cname}/{uuid.uuid4().hex[:8]}.png"))
            
        from scipy.spatial import cKDTree
        self._kd_tree = cKDTree(self.embeddings_2d)
        # New image added — invalidate thumbnail cache for this index
        # (coact cache is unaffected since new idx has no prior entry)
        self._b64_cache.pop((idx, 56), None)
        self._b64_cache.pop((idx, 80), None)
        self._b64_cache.pop((idx, 32), None)
        
        return idx

    # ------------------------------------------------------------------

    def image_to_base64(self, idx: int, size: int = 56) -> str:
        key = (idx, size)
        cached = self._b64_cache.get(key)
        if cached is not None:
            return cached
        arr = self.images[idx]
        img = Image.fromarray(arr.astype(np.uint8), mode='L')
        img = img.resize((size, size), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        result = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
        self._b64_cache[key] = result
        return result

    def support_indices(self, sc: Dict) -> set:
        out: set = set()
        for items in sc.values():
            for item in items:
                out.add(item['idx'])
        return out

    def query_mask(self, sc: Dict) -> np.ndarray:
        si = self.support_indices(sc)
        exc = getattr(self, 'excluded_indices', set())
        return np.array([i not in si and i not in exc for i in range(len(self.labels))])

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
            wts = np.array([it['weight'] for it in items])
            if wts.sum() < 1e-9:
                continue
            wts = wts / wts.sum()
            order.append(cname)
            
            vecs_2d, vecs_hd = [], []
            for it in items:
                idx = it['idx']
                if isinstance(idx, int):
                    vecs_2d.append(self.embeddings_2d[idx])
                    vecs_hd.append(self.embeddings_hd[idx])
                else:
                    if 'emb_hd' in it and 'pos2d' in it:
                        vecs_2d.append(np.array(it['pos2d']))
                        vecs_hd.append(np.array(it['emb_hd']))
            
            p2d.append(np.average(vecs_2d, axis=0, weights=wts))
            phd.append(np.average(vecs_hd, axis=0, weights=wts))

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

        # 2D overrides: move the visual position AND reroute the HD
        # prototype to the nearest real data point in 2D space so that
        # accuracy / decision boundary reflect the drag.
        if proto_overrides:
            for cname, pos in proto_overrides.items():
                if cname in order:
                    i = order.index(cname)
                    p2d[i] = pos
                    dists_2d = np.linalg.norm(
                        self.embeddings_2d - np.array(pos), axis=1)
                    nearest = int(np.argmin(dists_2d))
                    phd[i] = self.embeddings_hd[nearest]

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
        cm = np.zeros((nc, nc), dtype=float)
        for t, prob_vec in zip(true_idx, probs):
            if t >= 0:
                cm[t] += prob_vec

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

    def fit_weights_to_target(
        self, cname: str, sc: Dict, target_2d: List[float]
    ) -> Dict:
        """
        Solve for support-item weights such that the weighted mean of
        their 2D embeddings is as close as possible to `target_2d`.

        The prototype is defined as:
            p = sum(w_i * e_i) / sum(w_i)
        so the reachable region is the convex hull of the support
        embeddings in 2D.  Weights are bounded to [0.1, 3.0].

        Returns a new sc dict with updated weights for `cname`.
        """
        from scipy.optimize import minimize

        items = sc.get(cname, [])
        if not items:
            return sc

        idxs   = [it["idx"]    for it in items]
        w0     = np.array([it["weight"] for it in items], dtype=float)
        E      = self.embeddings_2d[idxs].astype(float)   # (k, 2)
        target = np.array(target_2d, dtype=float)

        def objective(w):
            S = w.sum()
            if S < 1e-9:
                return 1e9
            proto = (E * w[:, None]).sum(axis=0) / S
            return float(np.sum((proto - target) ** 2))

        result = minimize(
            objective, w0, method="L-BFGS-B",
            bounds=[(0.1, 3.0)] * len(idxs),
            options={"maxiter": 500, "ftol": 1e-12},
        )
        w_opt = np.clip(result.x, 0.1, 3.0)

        new_sc = {c: list(v) for c, v in sc.items()}
        new_sc[cname] = [
            {**it, "weight": round(float(w_opt[i]), 2)}
            for i, it in enumerate(items)
        ]
        return new_sc

    def exclude_image(self, idx: int) -> Optional[int]:
        """Exclude an image by index and load a replacement if possible.
        
        Saves relative path to excluded_images.json.
        Returns the index of the newly added replacement image, or None.
        """
        if idx >= len(self.image_paths):
            return None
            
        path = self.image_paths[idx]
        try:
            rel_path = str(path.relative_to(_DATA_DIR))
        except ValueError:
            return None
            
        # Save to file
        excluded = set()
        if _EXCLUDED_PATH.exists():
            try:
                with open(_EXCLUDED_PATH) as f:
                    excluded = set(json.load(f))
            except Exception:
                pass
        
        excluded.add(rel_path)
        try:
            with open(_EXCLUDED_PATH, 'w') as f:
                json.dump(list(excluded), f)
        except Exception as e:
            logger.warning("Failed to save excluded_images.json: %s", e)
            return None
            
        self.excluded_indices.add(idx)
        
        # Load replacement
        cname = self.class_names[self.labels[idx]]
        cls_dir = _DATA_DIR / cname
        if not cls_dir.exists():
            return None
            
        img_files = sorted(cls_dir.glob('*.png'))
        loaded_paths = {str(p) for p in self.image_paths}
        
        replacement_file = None
        for f in img_files:
            rel = str(f.relative_to(_DATA_DIR))
            if str(f) not in loaded_paths and rel not in excluded:
                replacement_file = f
                break
                
        if replacement_file:
            logger.info("Loading replacement for index %d: %s", idx, replacement_file)
            arr = np.array(Image.open(replacement_file).convert('L'))
            
            import torch
            with torch.no_grad():
                tensor = self._transform(Image.fromarray(arr.astype(np.uint8), mode='L')).unsqueeze(0)
                new_hd = self._encoder(tensor).cpu().numpy()
            
            new_2d = self._umap_reducer.transform(new_hd)
            
            # Append to memory arrays
            self.images.append(arr)
            self.labels = np.concatenate([self.labels, np.array([self.labels[idx]], dtype=int)])
            self.embeddings_hd = np.vstack([self.embeddings_hd, new_hd])
            self.embeddings_2d = np.vstack([self.embeddings_2d, new_2d])
            self.image_paths.append(replacement_file)
            
            from scipy.spatial import cKDTree
            self._kd_tree = cKDTree(self.embeddings_2d)
            # Replaced image — evict the old index from thumbnail cache
            for _size in (32, 56, 80):
                self._b64_cache.pop((idx, _size), None)
            self._coact_cache = {
                k: v for k, v in self._coact_cache.items()
                if k[0] != idx and k[1] != idx
            }
            
            return len(self.images) - 1
            
        return None

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

    def get_co_activation_images(self, q_idx: int, s_idx: int, size: int = 120) -> Tuple[str, str]:
        """Compute spatial Dense Co-Activation Correspondence between query and support image.
        Intercepts internal Conv4 layers to return glowing visual analytic heatmaps.
        """
        key = (q_idx, s_idx, size)
        cached = self._coact_cache.get(key)
        if cached is not None:
            return cached
        import torch
        import torch.nn.functional as F
        import matplotlib.pyplot as plt
        
        # 1. Fetch raw PyTorch grids
        img_q = self.images[q_idx]
        img_s = self.images[s_idx]
        
        t_q = self._transform(Image.fromarray(img_q.astype(np.uint8), mode='L')).unsqueeze(0)
        t_s = self._transform(Image.fromarray(img_s.astype(np.uint8), mode='L')).unsqueeze(0)
        
        with torch.no_grad():
            # 2. Extract deep 4x4 Spatial Features cleanly
            feat_q = self._encoder.encoder(t_q)  
            feat_s = self._encoder.encoder(t_s)
        
        B, C, H, W = feat_q.shape  # 1, 512, 4, 4
        
        # 3. Compute pairwise Cosine Similarity Matrix globally
        vec_q = feat_q.view(C, -1).t()
        vec_s = feat_s.view(C, -1).t()
        
        vec_q_norm = F.normalize(vec_q, p=2, dim=1)
        vec_s_norm = F.normalize(vec_s, p=2, dim=1)
        
        sim_matrix = torch.mm(vec_q_norm, vec_s_norm.t())
        sim_matrix = F.relu(sim_matrix)
        
        # 4. Filter spatial strongest-match intensities
        q_scores, _ = sim_matrix.max(dim=1)
        s_scores, _ = sim_matrix.max(dim=0)
        
        q_grid = q_scores.view(1, 1, H, W)
        s_grid = s_scores.view(1, 1, H, W)
        
        # 5. Bicubic Upsample cleanly to 64x64 HD layout
        q_hm = F.interpolate(q_grid, size=(64, 64), mode='bicubic', align_corners=False).squeeze().numpy()
        s_hm = F.interpolate(s_grid, size=(64, 64), mode='bicubic', align_corners=False).squeeze().numpy()
        
        q_hm = np.clip((q_hm - q_hm.min()) / (q_hm.max() - q_hm.min() + 1e-9), 0, 1)
        s_hm = np.clip((s_hm - s_hm.min()) / (s_hm.max() - s_hm.min() + 1e-9), 0, 1)
        
        cmap = plt.colormaps.get_cmap('magma')
        
        def apply_heatmap(img_arr, hm_arr):
            colored_hm = cmap(hm_arr)[:, :, :3] * 255
            base_img = np.stack((img_arr,) * 3, axis=-1).astype(float)
            base_img = base_img * 0.4  # Dim background structural grid
            
            # Blend maps heavily enforcing the visual focus glow geometry
            alpha = hm_arr[:, :, np.newaxis] ** 1.8 
            blended = (base_img * (1 - alpha)) + (colored_hm * alpha)
            
            final_img = Image.fromarray(blended.astype(np.uint8))
            if size != 64:
                final_img = final_img.resize((size, size), Image.NEAREST)
                
            buf = io.BytesIO()
            final_img.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            
        result = apply_heatmap(img_q, q_hm), apply_heatmap(img_s, s_hm)
        self._coact_cache[key] = result
        return result

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
    # Decision boundary mesh (2-D Voronoi approximation)
    # ------------------------------------------------------------------

    def decision_mesh(
        self, sc: Dict, temperature: float,
        proto_overrides: Optional[Dict] = None, res: int = 60,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (xx, ys, zz_class, zz_dummy) — 2D Voronoi over prototype
        positions.  zz_dummy is a zeros array kept for call-site compatibility.
        The mesh is a spatial approximation only; actual classification uses
        HD embeddings.  Explanation of individual decisions lives in the
        inspector, not the mesh.
        """
        p2d, _, order = self.compute_prototypes(sc)
        if not order:
            e = np.array([])
            return e, e, e, e
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
        
        logits = -d / temp
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        
        zz_class = np.argmax(probs, axis=1).reshape(xx.shape)
        zz_alpha = probs.max(axis=1).reshape(xx.shape)
        
        return xx, ys, zz_class, zz_alpha