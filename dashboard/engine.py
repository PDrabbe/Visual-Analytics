import numpy as np
from PIL import Image
import io
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
    '#ff6b6b',   # cat      — coral red
    '#51cf66',   # dog      — green
    '#339af0',   # fish     — blue
    '#fcc419',   # car      — yellow
    '#cc5de8',   # flower   — purple
    '#20c997',   # bicycle  — teal
    '#ff922b',   # bird     — orange
    '#74c0fc',   # pizza    — sky blue
    '#f783ac',   # clock    — pink
    '#a9e34b',   # lightning — lime
]

# ── Paths ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINT = _ROOT / 'checkpoints' / 'best_model.pt'
_DATA_DIR = _ROOT / 'data' / 'quickdraw' / 'test'
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

        # Set by init_demo — needed for Grad-CAM and masked embeddings
        self._encoder = None       # torch.nn.Module (conv4)
        self._transform = None     # torchvision transform pipeline
        self._last_conv = None     # reference to last Conv2d layer
        self._pre_gap_layer = None # last module outputting 4-D tensor (before GAP)
        self._fmap_hw = (4, 4)     # spatial dims of feature map before GAP
        self._masked_emb_cache: Dict[str, np.ndarray] = {}  # (idx, mask_key) → emb
        self._gradcam_tile_cache: Dict[str, Dict[str, str]] = {}  # (idx, sc_key) → tiles

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

        # Persist for Grad-CAM / masking later
        self._encoder = encoder
        self._last_conv = self._find_last_conv(encoder)
        self._pre_gap_layer, self._fmap_hw = self._find_pre_gap(encoder)

        transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
        self._transform = transform

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
    # Encoder introspection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_last_conv(model) -> "torch.nn.Module":
        """Walk the module tree and return the last Conv2d layer."""
        import torch.nn as nn
        last = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is None:
            raise RuntimeError("No Conv2d layer found in encoder")
        return last

    @staticmethod
    def _find_pre_gap(model) -> Tuple:
        """
        Probe the encoder to find the last module that outputs a 4-D
        tensor (B, C, H, W).  That module's output is the feature map
        right before global average pooling.

        Returns (module, (H, W)).
        """
        import torch

        candidates = []       # [(module, (H, W))]
        hooks = []

        def _make_hook(mod):
            def _hook(_m, _inp, out):
                if isinstance(out, torch.Tensor) and out.dim() == 4:
                    candidates.append((mod, (out.shape[2], out.shape[3])))
            return _hook

        for m in model.modules():
            hooks.append(m.register_forward_hook(_make_hook(m)))

        with torch.no_grad():
            model(torch.randn(1, 1, 64, 64))

        for h in hooks:
            h.remove()

        if not candidates:
            raise RuntimeError("No 4-D feature map found in encoder")

        # Last 4-D output = feature map immediately before GAP
        layer, hw = candidates[-1]
        logger.info("Pre-GAP layer: %s  feature map: %s", layer, hw)
        return layer, hw

    # ------------------------------------------------------------------
    # Grad-CAM  (Phase 1)
    # ------------------------------------------------------------------

    def compute_gradcam(
        self,
        idx: int,
        sc: Dict,
        target_class: Optional[str] = None,
        contrastive: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute a Grad-CAM heatmap for image *idx*.

        Parameters
        ----------
        idx : int
            Index into ``self.images``.
        sc : dict
            Current support-set configuration (needed to build prototypes).
        target_class : str or None
            Class to explain.  If None, uses the image's ground-truth class.

        Returns
        -------
        heatmap : np.ndarray, shape (64, 64), values in [0, 1]
            Upsampled, normalised Grad-CAM heatmap aligned to the input image.
        """
        import torch

        # ── resolve target class ──────────────────────────────────
        if target_class is None:
            target_class = self.class_names[self.labels[idx]]

        _, phd, order = self.compute_prototypes(sc)
        if target_class not in order:
            return np.zeros((64, 64), dtype=np.float32)
        ti = order.index(target_class)
        # All prototypes as a single tensor — needed for softmax score
        all_protos = torch.tensor(phd, dtype=torch.float32)  # (C, 512)

        # ── prepare input tensor ──────────────────────────────────
        img_pil = Image.fromarray(self.images[idx].astype(np.uint8), mode='L')
        x = self._transform(img_pil).unsqueeze(0)          # (1, 1, 64, 64)
        x.requires_grad_(True)

        # ── register hooks on last conv layer ─────────────────────
        fmaps = []    # forward:  feature maps  (1, C, H, W)
        grads = []    # backward: gradients      (1, C, H, W)

        def _fwd_hook(_mod, _inp, out):
            fmaps.append(out)

        def _bwd_hook(_mod, _gin, gout):
            grads.append(gout[0])

        h_fwd = self._pre_gap_layer.register_forward_hook(_fwd_hook)
        h_bwd = self._pre_gap_layer.register_full_backward_hook(_bwd_hook)

        try:
            # ── forward pass ──────────────────────────────────────
            self._encoder.zero_grad()
            embedding = self._encoder(x)                    # (1, 512)

            # ── score: log_softmax (contrastive) or -dist (single) ──
            emb = embedding.squeeze()                           # (512,)
            dists = torch.sum(
                (all_protos - emb.unsqueeze(0)) ** 2, dim=1
            )                                                   # (C,)
            if contrastive and len(all_protos) > 1:
                # log P(target | x) — gradient flows through all
                # competing prototypes proportionally.  Best for
                # comparing classes side-by-side in the inspector.
                score = torch.log_softmax(-dists, dim=0)[ti]
                logger.debug(
                    "gradcam idx=%d target=%s  log_prob=%.4f  "
                    "dists(min=%.1f max=%.1f)",
                    idx, target_class,
                    score.item(),
                    dists.min().item(), dists.max().item(),
                )
            else:
                # -dist_target — always produces a non-trivial map,
                # ideal for single-class mask editor / support view.
                score = -dists[ti]
                logger.debug(
                    "gradcam idx=%d target=%s  non-contrastive "
                    "dist_target=%.4f",
                    idx, target_class, dists[ti].item(),
                )
            score.backward()
        finally:
            h_fwd.remove()
            h_bwd.remove()

        # ── Grad-CAM computation ──────────────────────────────────
        feat = fmaps[0].detach()[0]   # (C, H, W)  — [0] removes batch only
        grad = grads[0].detach()[0]   # (C, H, W)

        # Channel weights = global-average-pooled gradients
        weights = grad.mean(dim=(1, 2))      # (C,)

        # Weighted combination → ReLU
        cam = torch.relu((weights[:, None, None] * feat).sum(dim=0))  # (H, W)

        # Upsample to image resolution then smooth out 4×4 block artefacts
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        cam = torch.nn.functional.interpolate(
            cam, size=(64, 64), mode='bicubic', align_corners=False
        ).squeeze().clamp(min=0)             # (64, 64) — bicubic can go negative
        # Gaussian blur to suppress 16-px grid seams from 4×4 feature maps
        cam_np_tmp = cam.numpy()
        from scipy.ndimage import gaussian_filter
        cam_np_tmp = gaussian_filter(cam_np_tmp, sigma=2.0)
        cam_np = cam_np_tmp

        # Normalise to [0, 1] (optional)
        if normalize:
            cmin, cmax = cam_np.min(), cam_np.max()
            if cmax - cmin > 1e-8:
                cam_np = (cam_np - cmin) / (cmax - cmin)
            else:
                logger.debug(
                    "gradcam idx=%d target=%s  flat heatmap (cmax-cmin=%.2e) "
                    "— returning zeros",
                    idx, target_class, float(cmax - cmin),
                )
                cam_np = np.zeros_like(cam_np)

        logger.debug(
            "gradcam idx=%d target=%s  heatmap stats: "
            "mean=%.4f  max=%.4f  nonzero_frac=%.2f  "
            "feat_shape=%s  grad_shape=%s",
            idx, target_class,
            float(cam_np.mean()), float(cam_np.max()),
            float((cam_np > 0.1).mean()),
            tuple(feat.shape), tuple(grad.shape),
        )
        return cam_np.astype(np.float32)

    # ------------------------------------------------------------------

    def gradcam_overlay_base64(
        self,
        idx: int,
        sc: Dict,
        target_class: Optional[str] = None,
        size: int = 128,
        alpha: float = 0.55,
    ) -> str:
        """
        Return a base-64 PNG of the Grad-CAM heatmap overlaid on the
        original sketch, ready for <img src=...> embedding.

        Parameters
        ----------
        idx   : image index
        sc    : support-set config
        target_class : class to explain (None → ground truth)
        size  : output pixel size (square)
        alpha : overlay opacity (0 = sketch only, 1 = heatmap only)
        """
        import matplotlib.cm as cm

        heatmap = self.compute_gradcam(idx, sc, target_class, contrastive=False)  # (64, 64) [0..1]

        # Sketch as grey RGB
        sketch = self.images[idx].astype(np.float32)
        sketch = np.stack([sketch] * 3, axis=-1)               # (64, 64, 3)
        sketch /= 255.0

        # Heatmap as RGB via 'viridis' colourmap (perceptually uniform)
        colored = cm.viridis(heatmap)[:, :, :3].astype(np.float32) # (64, 64, 3)

        # Blend with activation-scaled opacity (data-ink ratio)
        dynamic_alpha = heatmap[..., np.newaxis] * alpha
        blended = (1 - dynamic_alpha) * sketch + dynamic_alpha * colored
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

        img = Image.fromarray(blended, mode='RGB').resize(
            (size, size), Image.BILINEAR
        )
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


    def compute_gradcam_multi(
        self,
        idx: int,
        sc: Dict,
    ) -> Dict[str, np.ndarray]:
        """
        Run ``compute_gradcam`` once per class in *sc* and return all
        heatmaps in a single dict.

        The contrastive score inside ``compute_gradcam`` means each map
        answers "what made this image look like <class> rather than its
        nearest competitor?" -- so the maps are directly comparable.

        Returns
        -------
        heatmaps : dict  { class_name -> np.ndarray (64, 64) [0..1] }
        """
        import time
        _, _, order = self.compute_prototypes(sc)
        logger.debug(
            "gradcam_multi idx=%d  computing %d heatmaps for classes: %s",
            idx, len(order), order,
        )
        if not order:
            return {}

        t0 = time.perf_counter()

        # Raw class heatmaps first; normalise globally across classes
        raw_heatmaps = {
            cname: self.compute_gradcam(
                idx, sc, target_class=cname, normalize=False
            )
            for cname in order
        }

        global_max = max(hm.max() for hm in raw_heatmaps.values())
        global_min = min(hm.min() for hm in raw_heatmaps.values())
        heatmaps: Dict[str, np.ndarray] = {}
        for cname, hm in raw_heatmaps.items():
            if global_max - global_min > 1e-8:
                heatmaps[cname] = (hm - global_min) / (global_max - global_min)
            else:
                heatmaps[cname] = np.zeros_like(hm)

        logger.debug(
            "gradcam_multi idx=%d  done in %.3fs",
            idx, time.perf_counter() - t0,
        )
        return heatmaps

    def gradcam_tiled_overlays(
        self,
        idx: int,
        sc: Dict,
        size: int = 128,
        alpha: float = 0.55,
    ) -> Dict[str, str]:
        """
        Return one base-64 PNG per class -- the Grad-CAM overlay rendered
        with the contrastive heatmap for that class.  Ready to tile
        directly in the dashboard.

        Parameters
        ----------
        idx   : image index
        sc    : support-set config
        size  : output pixel size per tile (square)
        alpha : heatmap blend opacity

        Returns
        -------
        tiles : dict  { class_name -> 'data:image/png;base64,...' }
        """
        import matplotlib.cm as cm

        key = self._tile_cache_key(idx, sc)
        if key in self._gradcam_tile_cache:
            logger.debug(
                "gradcam_tiled_overlays idx=%d  cache HIT  "
                "(cache size=%d)",
                idx, len(self._gradcam_tile_cache),
            )
            return self._gradcam_tile_cache[key]

        logger.debug(
            "gradcam_tiled_overlays idx=%d  cache MISS  "
            "(cache size=%d)  computing tiles...",
            idx, len(self._gradcam_tile_cache),
        )
        heatmaps = self.compute_gradcam_multi(idx, sc)

        sketch = self.images[idx].astype(np.float32) / 255.0
        sketch_rgb = np.stack([sketch] * 3, axis=-1)           # (64, 64, 3)

        tiles: Dict[str, str] = {}
        for cname, heatmap in heatmaps.items():
            colored = cm.viridis(heatmap)[:, :, :3].astype(np.float32)
            dynamic_alpha = heatmap[..., np.newaxis] * alpha
            blended = np.clip(
                ((1 - dynamic_alpha) * sketch_rgb + dynamic_alpha * colored) * 255,
                0,
                255,
            ).astype(np.uint8)
            img = Image.fromarray(blended, mode='RGB').resize(
                (size, size), Image.BILINEAR
            )
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            tiles[cname] = (
                'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
            )

        self._gradcam_tile_cache[key] = tiles
        logger.debug(
            "gradcam_tiled_overlays idx=%d  stored %d tiles  "
            "(cache size now=%d)",
            idx, len(tiles), len(self._gradcam_tile_cache),
        )
        return tiles

    # ------------------------------------------------------------------
    # Spatial attention masking  (Phase 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_cache_key(idx: int, mask: List[List[int]]) -> str:
        """Stable string key for the masked-embedding cache."""
        return f"{idx}:{json.dumps(mask)}"

    @staticmethod
    def _tile_cache_key(idx: int, sc: Dict) -> str:
        """Stable string key for the Grad-CAM tile cache.

        Encodes the image index plus the support-set structure (class →
        sorted list of (image-idx, weight, mask) tuples) so the cache
        invalidates automatically whenever the support set changes.
        """
        sc_repr = {
            cname: sorted(
                (it["idx"], round(it.get("weight", 1.0), 3),
                 json.dumps(it.get("mask")))
                for it in items
            )
            for cname, items in sc.items()
            if items
        }
        return f"{idx}:{json.dumps(sc_repr, sort_keys=True)}"

    def compute_masked_embedding(
        self,
        idx: int,
        mask: List[List[int]],
    ) -> np.ndarray:
        """
        Run image *idx* through the encoder and apply a spatial mask to
        the feature map before global-average pooling.

        Parameters
        ----------
        idx  : image index
        mask : 2-D list (H × W matching ``self._fmap_hw``),
               values 0 (exclude) or 1 (keep).
               An all-ones mask reproduces the original embedding.

        Returns
        -------
        embedding : np.ndarray, shape (512,)
        """
        import torch

        # ── cache lookup ──────────────────────────────────────────
        key = self._mask_cache_key(idx, mask)
        if key in self._masked_emb_cache:
            return self._masked_emb_cache[key]

        # ── forward pass — capture pre-GAP feature map ────────────
        img_pil = Image.fromarray(self.images[idx].astype(np.uint8), mode='L')
        x = self._transform(img_pil).unsqueeze(0)      # (1, 1, 64, 64)

        fmaps = []

        def _fwd_hook(_mod, _inp, out):
            fmaps.append(out)

        h = self._pre_gap_layer.register_forward_hook(_fwd_hook)
        try:
            with torch.no_grad():
                self._encoder(x)
        finally:
            h.remove()

        feat = fmaps[0].detach().squeeze(0)             # (C, H, W)

        # ── build mask tensor ─────────────────────────────────────
        fh, fw = self._fmap_hw
        mask_t = torch.tensor(mask, dtype=torch.float32) # (H', W')
        if mask_t.shape != (fh, fw):
            mask_t = torch.nn.functional.interpolate(
                mask_t.unsqueeze(0).unsqueeze(0),
                size=(fh, fw), mode='nearest',
            ).squeeze()

        # ── masked global-average pooling ─────────────────────────
        masked = feat * mask_t.unsqueeze(0)              # (C, H, W)
        mask_sum = mask_t.sum().item()
        if mask_sum > 0:
            emb = masked.sum(dim=(1, 2)) / mask_sum      # (C,)
        else:
            emb = torch.zeros(feat.shape[0])

        result = emb.numpy().astype(np.float32)
        self._masked_emb_cache[key] = result
        return result

    def invalidate_mask_cache(self):
        """Clear the masked-embedding cache and Grad-CAM tile cache
        (call after model changes or support-set edits)."""
        self._masked_emb_cache.clear()
        self._gradcam_tile_cache.clear()

    # ------------------------------------------------------------------
    # Per-support-image diagnostics  (Phase 2)
    # ------------------------------------------------------------------

    def support_diagnostics(
        self,
        cname: str,
        sc: Dict,
        temperature: float = 1.0,
    ) -> List[Dict]:
        """
        Compute diagnostics for every support image of *cname*.

        Returns a list (same order as ``sc[cname]``) of dicts::

            {
                "idx":        int,       # image index
                "loo_delta":  float,     # overall-accuracy change when this
                                         #   image is removed  (negative = hurts)
                "cos_sim":    float,     # cosine similarity to own prototype
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

        # Current prototypes (HD) — needed for cos-sim and competitor dist
        _, phd, order = self.compute_prototypes(sc)
        if cname not in order:
            return [{"idx": it["idx"], "loo_delta": 0.0,
                     "cos_sim": 0.0, "competitor_dist": 0.0,
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

            # ── cosine similarity to own prototype ────────────
            norm_e = np.linalg.norm(emb)
            norm_p = np.linalg.norm(own_proto)
            if norm_e > 1e-9 and norm_p > 1e-9:
                cos = float(np.dot(emb, own_proto) / (norm_e * norm_p))
            else:
                cos = 0.0

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
            if len(items) <= 1:
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
                "cos_sim": cos,
                "competitor_dist": comp_dist,
                "competitor_name": comp_name,
            })

        return results

    # ------------------------------------------------------------------
    # Support set suggestions  (Phase 4)
    # ------------------------------------------------------------------

    def suggest_support_changes(
        self,
        cname: str,
        sc: Dict,
        temperature: float = 1.0,
        max_candidates: int = 5,
    ) -> Dict[str, list]:
        """
        Suggest swaps and additions for class *cname*'s support set.

        Scans all non-support images of the same class, simulates each
        change, and ranks by overall-accuracy delta.

        Returns
        -------
        {
            "swaps":  [ { "remove_idx", "add_idx", "acc_delta",
                          "new_acc", "cos_sim" }, … ],
            "adds":   [ { "add_idx", "acc_delta",
                          "new_acc", "cos_sim" }, … ],
        }

        Lists are sorted best-first (highest acc_delta), capped at
        *max_candidates* each.
        """
        items = sc.get(cname, [])
        if not items:
            return {"swaps": [], "adds": []}

        base_res = self.classify(sc, temperature)
        base_acc = base_res["overall"]

        # Current prototype for cos-sim computation
        _, phd, order = self.compute_prototypes(sc)
        ci = order.index(cname) if cname in order else -1

        support_idxs = {it["idx"] for it in items}
        # Non-support images of the same class
        class_mask = self.labels == self.class_names.index(cname)
        candidates = [
            int(i) for i in np.where(class_mask)[0]
            if i not in support_idxs
        ]

        if not candidates:
            return {"swaps": [], "adds": []}

        # Pre-compute cosine similarity of each candidate to prototype
        def _cos(idx):
            if ci < 0:
                return 0.0
            emb = self.embeddings_hd[idx]
            proto = phd[ci]
            ne, np_ = np.linalg.norm(emb), np.linalg.norm(proto)
            return float(np.dot(emb, proto) / (ne * np_ + 1e-9))

        # ── swap suggestions ──────────────────────────────────────
        # Only try swapping support images that have positive LOO delta
        # (removing them helps or is neutral) — no point swapping the
        # best support images.  But if all are positive, try all.
        diags = self.support_diagnostics(cname, sc, temperature)
        # Sort worst-first (highest LOO delta = most harmful)
        worst = sorted(diags, key=lambda d: d["loo_delta"], reverse=True)
        # Try swapping the worst support images (up to 3 to bound cost)
        swap_targets = worst[:min(3, len(worst))]

        swap_results = []
        for diag in swap_targets:
            rm_idx = diag["idx"]
            # Build sc without this image
            stripped = [it for it in items if it["idx"] != rm_idx]
            for cand_idx in candidates:
                trial_sc = {c: list(v) for c, v in sc.items()}
                trial_sc[cname] = stripped + [
                    {"idx": cand_idx, "weight": 1.0}
                ]
                trial_res = self.classify(trial_sc, temperature)
                delta = trial_res["overall"] - base_acc
                swap_results.append({
                    "remove_idx": rm_idx,
                    "add_idx": cand_idx,
                    "acc_delta": float(delta),
                    "new_acc": float(trial_res["overall"]),
                    "cos_sim": _cos(cand_idx),
                })

        swap_results.sort(key=lambda r: r["acc_delta"], reverse=True)

        # ── add suggestions ───────────────────────────────────────
        add_results = []
        for cand_idx in candidates:
            trial_sc = {c: list(v) for c, v in sc.items()}
            trial_sc[cname] = list(items) + [
                {"idx": cand_idx, "weight": 1.0}
            ]
            trial_res = self.classify(trial_sc, temperature)
            delta = trial_res["overall"] - base_acc
            add_results.append({
                "add_idx": cand_idx,
                "acc_delta": float(delta),
                "new_acc": float(trial_res["overall"]),
                "cos_sim": _cos(cand_idx),
            })

        add_results.sort(key=lambda r: r["acc_delta"], reverse=True)

        return {
            "swaps": swap_results[:max_candidates],
            "adds": add_results[:max_candidates],
        }

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
        """Return (protos_2d, protos_hd, class_order).

        If any support item has a ``"mask"`` key, its HD embedding is
        replaced by the masked version (computed via
        ``compute_masked_embedding``).  The 2-D embedding is left
        unchanged — UMAP positions stay fixed; only the HD prototype
        (which drives accuracy) is affected.
        """
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

            # HD embeddings — use masked version when a mask is present
            hd_vecs = []
            for it in items:
                mask = it.get('mask')
                if mask is not None:
                    hd_vecs.append(self.compute_masked_embedding(it['idx'], mask))
                else:
                    hd_vecs.append(self.embeddings_hd[it['idx']])
            hd_vecs = np.array(hd_vecs)
            phd.append(np.average(hd_vecs, axis=0, weights=wts))

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