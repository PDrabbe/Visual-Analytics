"""
diagnose_va.py
==============
Comprehensive diagnostic script for ProtoNet Explorer.

Goals
-----
1. Identify the best BASE CLASS SUBSET (8-12 classes) to load in the dashboard
   for the demo video, maximising visible inter-class confusion and boundary drama.
2. Identify the best NOVEL CLASS to add live during the demo (T4), ideally one
   that shows the "high accuracy / wrong features" heatmap finding.
3. Measure SUPPORT-SET VOLATILITY per class so we can recommend which class to
   use for the T3 curation demonstration.
4. Export a JSON artefact with all numbers used in the written report.

VA tasks being scored
---------------------
  T1 - Cluster overlap detection      (UMAP confusion, inter-class proximity)
  T2 - Misclassification diagnosis    (accuracy paradox proxy, embedding isotropy)
  T3 - Support-set curation           (accuracy variance across support configs)
  T4 - Novel-class evaluation         (separation from base, accuracy with default support)

Usage
-----
  python diagnose_va.py

Output
------
  - Printed report to stdout (pipe to a .txt if you want to save it)
  - diagnose_va_results.json  (all numbers for the written report)
"""

import os
import json
import random
import itertools
import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── class lists ───────────────────────────────────────────────────────────────
BASE_CLASSES = [
    'apple', 'banana', 'bicycle', 'bird', 'bus', 'cake', 'car', 'cat',
    'chair', 'clock', 'cloud', 'diamond', 'dog', 'door', 'eyeglasses',
    'fish', 'flower', 'hexagon', 'horse', 'key', 'ladder', 'lightning',
    'mountain', 'pizza', 'scissors', 'square', 'table', 'triangle', 'truck',
]

NOVEL_CLASSES = [
    'airplane', 'book', 'tree', 'house', 'umbrella', 'guitar', 'moon', 'star',
]

# Paths
BASE_DATA_DIR   = 'data/quickdraw/test'
NOVEL_DATA_DIR  = 'custom_drawings'
CHECKPOINT_PATH = 'checkpoints/best_model.pt'

# Diagnostic parameters
N_IMAGES_PER_CLASS   = 30   # embeddings to collect per class
N_SUPPORT            = 5    # k-shot
N_SUPPORT_CONFIGS    = 20   # random support configurations for volatility test
N_QUERY_VOLATILITY   = 15   # query images used in volatility test (after support)
TOP_N_CONFUSION      = 5    # confusion pairs to print

OUTPUT_JSON = 'diagnose_va_results.json'


# =============================================================================
# Section 0 – Helpers
# =============================================================================

def _banner(title: str):
    width = 68
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _section(title: str):
    print("\n" + "-" * 68)
    print(f"  {title}")
    print("-" * 68)


def _score_label(score: float, thresholds=(0.6, 0.35)) -> str:
    """Three-tier qualitative label."""
    if score >= thresholds[0]:
        return "★★★  EXCELLENT"
    elif score >= thresholds[1]:
        return "★★   GOOD"
    else:
        return "★    WEAK"


def _safe_mean(lst):
    return float(np.mean(lst)) if lst else 0.0


# =============================================================================
# Section 1 – Load model and collect embeddings
# =============================================================================

def load_predictor():
    from inference.predictor import DrawingPredictor
    print(f"  Loading checkpoint: {CHECKPOINT_PATH}")
    predictor = DrawingPredictor(CHECKPOINT_PATH)
    return predictor


def collect_embeddings(predictor):
    """Return dict: class_name -> np.ndarray (N, 512)."""
    from inference.predictor import DrawingPredictor

    all_embs  = {}
    all_paths = {}   # class_name -> list of image paths (in order)

    print("\n  Base classes:")
    for cname in BASE_CLASSES:
        embs, paths = [], []
        for i in range(N_IMAGES_PER_CLASS):
            p = os.path.join(BASE_DATA_DIR, cname, f'{i:04d}.png')
            if os.path.exists(p):
                embs.append(predictor.get_embedding(p))
                paths.append(p)
        if embs:
            all_embs[cname]  = np.array(embs)
            all_paths[cname] = paths
            print(f"    {cname:12}: {len(embs):>3} images")
        else:
            print(f"    {cname:12}: NOT FOUND")

    print("\n  Novel classes:")
    for cname in NOVEL_CLASSES:
        embs, paths = [], []
        for i in range(N_IMAGES_PER_CLASS):
            p = os.path.join(NOVEL_DATA_DIR, cname, f'{i:04d}.png')
            if os.path.exists(p):
                embs.append(predictor.get_embedding(p))
                paths.append(p)
        if embs:
            all_embs[cname]  = np.array(embs)
            all_paths[cname] = paths
            print(f"    {cname:12}: {len(embs):>3} images")
        else:
            print(f"    {cname:12}: NOT FOUND (run download_new_classes.py first)")

    return all_embs, all_paths


# =============================================================================
# Section 2 – Prototype utilities
# =============================================================================

def compute_prototype(embs: np.ndarray, indices=None, weights=None) -> np.ndarray:
    """Weighted mean embedding over given indices (default: all, uniform weight)."""
    if indices is None:
        indices = list(range(len(embs)))
    subset = embs[indices]
    if weights is None:
        return subset.mean(axis=0)
    w = np.array(weights, dtype=float)
    w /= w.sum()
    return (subset * w[:, None]).sum(axis=0)


def classify_query(query_emb: np.ndarray, prototypes: dict) -> str:
    """Nearest-prototype Euclidean classification."""
    min_d, pred = float('inf'), None
    for cname, proto in prototypes.items():
        d = float(np.linalg.norm(query_emb - proto))
        if d < min_d:
            min_d, pred = d, cname
    return pred


# =============================================================================
# Section 3 – Intra / inter-class distance analysis  (T1 proxy)
# =============================================================================

def analyse_distances(all_embs: dict) -> dict:
    """
    For every class compute:
      intra_mean  - average pairwise distance within class
      inter_min   - distance to nearest other class (by prototype)
      ratio       - inter_min / intra_mean  (>1 means separable)
      nearest     - name of nearest other class

    Returns dict: class_name -> metrics_dict
    """
    # Compute per-class prototypes first
    protos = {c: e.mean(axis=0) for c, e in all_embs.items()}
    cnames = list(protos.keys())
    proto_mat = np.array([protos[c] for c in cnames])
    proto_dists = cdist(proto_mat, proto_mat, 'euclidean')

    results = {}
    for ci, cname in enumerate(cnames):
        embs = all_embs[cname]
        intra = cdist(embs, embs, 'euclidean')
        np.fill_diagonal(intra, np.nan)
        intra_mean = float(np.nanmean(intra))

        # Nearest prototype among all OTHER classes
        row = proto_dists[ci].copy()
        row[ci] = np.inf
        nearest_idx = int(np.argmin(row))
        inter_min   = float(row[nearest_idx])
        nearest     = cnames[nearest_idx]

        ratio = inter_min / intra_mean if intra_mean > 0 else 0.0

        results[cname] = {
            'intra_mean': intra_mean,
            'inter_min':  inter_min,
            'ratio':      ratio,
            'nearest':    nearest,
        }
    return results


# =============================================================================
# Section 4 – Full confusion matrix over all base classes  (T1, T3)
# =============================================================================

def full_confusion_matrix(all_embs: dict, classes: list) -> dict:
    """
    Build a confusion matrix using a leave-one-out single-support-image
    approach (fast, no randomness needed for a global picture).

    Uses first N_SUPPORT images as support, remaining as queries.
    Returns dict: true_class -> pred_class -> count
    """
    # Build support prototypes from first N_SUPPORT images of each class
    support_protos = {}
    for cname in classes:
        if cname in all_embs and len(all_embs[cname]) >= N_SUPPORT:
            support_protos[cname] = compute_prototype(all_embs[cname],
                                                      indices=list(range(N_SUPPORT)))

    if len(support_protos) < 2:
        return {}

    cm = {c: defaultdict(int) for c in support_protos}

    for true_class in support_protos:
        embs = all_embs[true_class]
        query_embs = embs[N_SUPPORT:]   # everything after support
        for qe in query_embs:
            pred = classify_query(qe, support_protos)
            cm[true_class][pred] += 1

    return cm


def confusion_accuracy(cm: dict) -> dict:
    """Per-class accuracy and top confusion partner from a confusion matrix."""
    acc = {}
    for true_class, preds in cm.items():
        total  = sum(preds.values())
        correct = preds.get(true_class, 0)
        top_conf = sorted(
            [(p, n) for p, n in preds.items() if p != true_class],
            key=lambda x: -x[1]
        )
        acc[true_class] = {
            'accuracy':     correct / total if total > 0 else 0.0,
            'correct':      correct,
            'total':        total,
            'top_confused': top_conf[:3],
        }
    return acc


# =============================================================================
# Section 5 – Support-set volatility  (T3 proxy)
# =============================================================================

def support_volatility(all_embs: dict, classes: list) -> dict:
    """
    For each class, sample N_SUPPORT_CONFIGS random support configurations and
    measure the accuracy of queries (images after the fixed support pool).
    Returns per-class accuracy mean, std, and min (= worst configuration).

    High std => the class is strongly affected by which support images are chosen
               => ideal class for the T3 curation demo.
    """
    results = {}
    for target_class in classes:
        if target_class not in all_embs:
            continue
        embs = all_embs[target_class]
        if len(embs) < N_SUPPORT + 5:
            continue

        pool_size = min(len(embs), N_IMAGES_PER_CLASS)
        pool_indices = list(range(pool_size))

        # Build a fixed "all other classes" prototype dict (deterministic)
        other_protos = {}
        for cname in classes:
            if cname == target_class or cname not in all_embs:
                continue
            if len(all_embs[cname]) >= N_SUPPORT:
                other_protos[cname] = compute_prototype(
                    all_embs[cname], indices=list(range(N_SUPPORT))
                )

        accs = []
        for _ in range(N_SUPPORT_CONFIGS):
            sup_indices = random.sample(pool_indices, N_SUPPORT)
            query_indices = [i for i in pool_indices if i not in sup_indices]
            query_indices = random.sample(query_indices,
                                          min(N_QUERY_VOLATILITY, len(query_indices)))

            proto = compute_prototype(embs, indices=sup_indices)
            protos = {**other_protos, target_class: proto}

            correct = sum(
                1 for qi in query_indices
                if classify_query(embs[qi], protos) == target_class
            )
            accs.append(correct / len(query_indices) if query_indices else 0.0)

        results[target_class] = {
            'acc_mean': float(np.mean(accs)),
            'acc_std':  float(np.std(accs)),
            'acc_min':  float(np.min(accs)),
            'acc_max':  float(np.max(accs)),
        }
    return results


# =============================================================================
# Section 6 – Embedding isotropy  (T2 proxy for heatmap interest)
# =============================================================================

def embedding_isotropy(all_embs: dict) -> dict:
    """
    Low isotropy (few dominant PCA directions) => the encoder's response
    to a class is concentrated in a small sub-space => the heatmap will show
    a clear localised activation => good for the T2 co-activation demo.

    Returns per-class dict with:
      participation_ratio  - PR = (sum eigenvalues)^2 / (d * sum eigenvalues^2)
                             Range [1/d, 1]. Low = anisotropic (concentrated).
      top1_var_explained   - fraction of variance in the first PC
    """
    results = {}
    for cname, embs in all_embs.items():
        if len(embs) < 4:
            continue
        centered = embs - embs.mean(axis=0)
        # PCA via SVD
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            eigenvalues = s ** 2
            total = eigenvalues.sum()
            if total == 0:
                continue
            pr   = (total ** 2) / (len(eigenvalues) * (eigenvalues ** 2).sum())
            top1 = eigenvalues[0] / total
            results[cname] = {
                'participation_ratio': float(pr),
                'top1_var_explained':  float(top1),
            }
        except Exception:
            continue
    return results


# =============================================================================
# Section 7 – Novel-class separation from base  (T4)
# =============================================================================

def novel_class_separation(all_embs: dict) -> dict:
    """
    For each novel class, measure:
      - 5-shot accuracy when added alongside all available base classes
      - Distance to nearest base prototype
      - Whether it is likely to be confused with a specific base class
    """
    # Compute base prototypes (first N_SUPPORT images)
    base_protos = {}
    for cname in BASE_CLASSES:
        if cname in all_embs and len(all_embs[cname]) >= N_SUPPORT:
            base_protos[cname] = compute_prototype(
                all_embs[cname], indices=list(range(N_SUPPORT))
            )

    results = {}
    for novel in NOVEL_CLASSES:
        if novel not in all_embs or len(all_embs[novel]) < N_SUPPORT + 3:
            continue
        embs = all_embs[novel]

        # 5-shot prototype for novel class
        novel_proto = compute_prototype(embs, indices=list(range(N_SUPPORT)))

        # Distance to every base prototype
        dists_to_base = {
            bc: float(np.linalg.norm(novel_proto - bp))
            for bc, bp in base_protos.items()
        }
        nearest_base  = min(dists_to_base, key=dists_to_base.get)
        nearest_dist  = dists_to_base[nearest_base]

        # Accuracy on query images
        protos = {**base_protos, novel: novel_proto}
        query_embs = embs[N_SUPPORT:]
        correct = sum(
            1 for qe in query_embs
            if classify_query(qe, protos) == novel
        )
        accuracy = correct / len(query_embs) if len(query_embs) > 0 else 0.0

        # Confusion breakdown
        confusion = defaultdict(int)
        for qe in query_embs:
            pred = classify_query(qe, protos)
            if pred != novel:
                confusion[pred] += 1

        results[novel] = {
            'accuracy':     accuracy,
            'correct':      correct,
            'total_queries': len(query_embs),
            'nearest_base': nearest_base,
            'nearest_dist': nearest_dist,
            'confusion':    dict(sorted(confusion.items(), key=lambda x: -x[1])[:5]),
        }
    return results


# =============================================================================
# Section 8 – Confusion pair scoring  (T1 demo value)
# =============================================================================

def score_confusion_pairs(cm_acc: dict, dist_results: dict) -> list:
    """
    Score every class pair by its demo value for T1:
      - High mutual confusion between the two classes
      - Both classes have nearby prototypes (geometric confusion)

    Returns sorted list of (score, class_a, class_b, details).
    """
    classes = list(cm_acc.keys())
    pairs = []
    for ca, cb in itertools.combinations(classes, 2):
        # Mutual confusion rate
        total_a = cm_acc[ca]['total']
        total_b = cm_acc[cb]['total']
        conf_a_to_b = dict(cm_acc[ca]['top_confused']).get(cb, 0)
        conf_b_to_a = dict(cm_acc[cb]['top_confused']).get(ca, 0)
        mutual_conf = (
            (conf_a_to_b / total_a if total_a > 0 else 0) +
            (conf_b_to_a / total_b if total_b > 0 else 0)
        ) / 2.0

        # Prototype proximity (normalised): lower ratio = more confusable
        ratio_a = dist_results.get(ca, {}).get('ratio', 99)
        ratio_b = dist_results.get(cb, {}).get('ratio', 99)
        # Nearest neighbours mutually pointing at each other?
        mutual_nearest = (
            dist_results.get(ca, {}).get('nearest') == cb and
            dist_results.get(cb, {}).get('nearest') == ca
        )

        # Score: emphasise mutual confusion + mutual nearest-neighbour bonus
        score = mutual_conf * 0.7 + (0.3 if mutual_nearest else 0.0)

        pairs.append({
            'score':          score,
            'class_a':        ca,
            'class_b':        cb,
            'mutual_conf':    mutual_conf,
            'conf_a_to_b':    conf_a_to_b,
            'conf_b_to_a':    conf_b_to_a,
            'mutual_nearest': mutual_nearest,
        })

    pairs.sort(key=lambda x: -x['score'])
    return pairs


# =============================================================================
# Section 9 – Recommended dashboard class subset
# =============================================================================

# =============================================================================
# Section 9 – Subset-aware re-evaluation
# =============================================================================

def evaluate_subset(
    all_embs: dict,
    subset: list,
    novel_class: str,
    n_support_configs: int = 20,
    n_query: int = 15,
) -> dict:
    """
    Re-run all diagnostics restricted to exactly the chosen class subset
    (plus the novel class when evaluating T4).

    This matters because ProtoNet is a geometric model: which classes are
    present in the session changes every prototype's nearest competitor,
    every Voronoi boundary, and therefore every accuracy / confusion number.
    Numbers from the global 29-class run are approximations; these are the
    numbers that will match the actual dashboard behaviour.
    """
    subset = [c for c in subset if c in all_embs]

    # ── 1. Distances within subset ───────────────────────────────────────────
    protos = {c: all_embs[c].mean(axis=0) for c in subset}
    cnames = list(protos.keys())
    proto_mat = np.array([protos[c] for c in cnames])
    proto_dists = cdist(proto_mat, proto_mat, 'euclidean')

    subset_dist = {}
    for ci, cname in enumerate(cnames):
        embs = all_embs[cname]
        intra = cdist(embs, embs, 'euclidean')
        np.fill_diagonal(intra, np.nan)
        intra_mean = float(np.nanmean(intra))
        row = proto_dists[ci].copy()
        row[ci] = np.inf
        nearest_idx = int(np.argmin(row))
        inter_min   = float(row[nearest_idx])
        subset_dist[cname] = {
            'intra_mean': intra_mean,
            'inter_min':  inter_min,
            'ratio':      inter_min / intra_mean if intra_mean > 0 else 0.0,
            'nearest':    cnames[nearest_idx],
        }

    # ── 2. Confusion matrix within subset ───────────────────────────────────
    cm      = full_confusion_matrix(all_embs, subset)
    cm_acc  = confusion_accuracy(cm)
    overall = _safe_mean([v['accuracy'] for v in cm_acc.values()])

    # ── 3. Volatility within subset ──────────────────────────────────────────
    # Re-compute using only subset classes as competitors
    volatility = {}
    for target in subset:
        embs = all_embs[target]
        if len(embs) < N_SUPPORT + 5:
            continue
        pool = list(range(min(len(embs), N_IMAGES_PER_CLASS)))
        other_protos = {
            c: compute_prototype(all_embs[c], list(range(N_SUPPORT)))
            for c in subset if c != target and len(all_embs.get(c, [])) >= N_SUPPORT
        }
        accs = []
        for _ in range(n_support_configs):
            sup = random.sample(pool, N_SUPPORT)
            qry = random.sample([i for i in pool if i not in sup],
                                min(n_query, len(pool) - N_SUPPORT))
            proto  = compute_prototype(embs, sup)
            protos_run = {**other_protos, target: proto}
            correct = sum(1 for qi in qry if classify_query(embs[qi], protos_run) == target)
            accs.append(correct / len(qry) if qry else 0.0)
        volatility[target] = {
            'acc_mean': float(np.mean(accs)),
            'acc_std':  float(np.std(accs)),
            'acc_min':  float(np.min(accs)),
            'acc_max':  float(np.max(accs)),
        }

    # ── 4. Novel class within subset ─────────────────────────────────────────
    novel_result = None
    if novel_class and novel_class in all_embs and len(all_embs[novel_class]) >= N_SUPPORT + 3:
        embs = all_embs[novel_class]
        novel_proto = compute_prototype(embs, list(range(N_SUPPORT)))
        base_protos_sub = {
            c: compute_prototype(all_embs[c], list(range(N_SUPPORT)))
            for c in subset if len(all_embs.get(c, [])) >= N_SUPPORT
        }
        protos_with_novel = {**base_protos_sub, novel_class: novel_proto}
        query_embs = embs[N_SUPPORT:]
        confusion  = defaultdict(int)
        correct    = 0
        for qe in query_embs:
            pred = classify_query(qe, protos_with_novel)
            if pred == novel_class:
                correct += 1
            else:
                confusion[pred] += 1
        dists_to_base = {
            bc: float(np.linalg.norm(novel_proto - base_protos_sub[bc]))
            for bc in base_protos_sub
        }
        nearest_base = min(dists_to_base, key=dists_to_base.get)
        novel_result = {
            'accuracy':      correct / len(query_embs) if len(query_embs) > 0 else 0.0,
            'correct':       correct,
            'total_queries': len(query_embs),
            'nearest_base':  nearest_base,
            'nearest_dist':  dists_to_base[nearest_base],
            'confusion':     dict(sorted(confusion.items(), key=lambda x: -x[1])[:5]),
        }

    return {
        'subset':          subset,
        'novel_class':     novel_class,
        'distance':        subset_dist,
        'confusion_acc':   cm_acc,
        'overall_acc':     overall,
        'volatility':      volatility,
        'novel_eval':      novel_result,
    }


def recommend_class_subset(
    available_base: list,
    cm_acc: dict,
    volatility: dict,
    dist_results: dict,
    confusion_pairs: list,
    n_classes: int = 10,
) -> list:
    """
    Greedy selection of n_classes base classes that maximise:
      - At least one high-confusion pair        (T1/T3 demo)
      - At least one high-volatility class      (T3 curation demo)
      - Exactly one stable anchor class         (scientific reference point:
          lets the audience see what a well-separated class looks like,
          making the confused classes more legible by contrast)
      - Remaining slots filled by demo-value score
    """
    scores = {}
    for cname in available_base:
        if cname not in cm_acc:
            continue
        acc   = cm_acc[cname]['accuracy']
        vol   = volatility.get(cname, {}).get('acc_std', 0)
        ratio = dist_results.get(cname, {}).get('ratio', 99)
        acc_score  = 1.0 - acc
        vol_score  = vol * 5.0
        prox_score = max(0, 1.0 - ratio / 3.0)
        scores[cname] = acc_score * 0.35 + vol_score * 0.35 + prox_score * 0.30

    must_include = set()

    # Primary T1 confusion pair
    if confusion_pairs:
        must_include.add(confusion_pairs[0]['class_a'])
        must_include.add(confusion_pairs[0]['class_b'])

    # Most volatile class for T3 curation demo
    if volatility:
        most_volatile = max(volatility, key=lambda c: volatility[c]['acc_std'])
        must_include.add(most_volatile)

    # One stable anchor: highest accuracy + lowest volatility std, not already included.
    # Serves as a scientific reference point (contrast makes confused classes more readable).
    anchor_candidates = sorted(
        [c for c in available_base if c in cm_acc and c not in must_include],
        key=lambda c: (-cm_acc[c]['accuracy'], volatility.get(c, {}).get('acc_std', 99))
    )
    if anchor_candidates:
        must_include.add(anchor_candidates[0])

    # Fill remaining slots by demo-value score
    selected = list(must_include)
    remaining = sorted(
        [c for c in scores if c not in must_include],
        key=lambda c: -scores[c]
    )
    for c in remaining:
        if len(selected) >= n_classes:
            break
        selected.append(c)

    return selected


# =============================================================================
# Main
# =============================================================================

def main():
    _banner("ProtoNet Explorer — VA Diagnostic Suite")
    print("""
  This script scores all base and novel classes across four dimensions:
    T1  Cluster overlap (confusion pair detection)
    T2  Misclassification diagnosis (embedding anisotropy proxy)
    T3  Support-set curation (accuracy volatility)
    T4  Novel-class evaluation (separation from base + accuracy)

  All numbers are saved to  diagnose_va_results.json  for the report.
""")

    # ── 0. Load model ──────────────────────────────────────────────────────────
    _section("0. Loading model and collecting embeddings")
    predictor = load_predictor()
    all_embs, all_paths = collect_embeddings(predictor)

    available_base  = [c for c in BASE_CLASSES  if c in all_embs]
    available_novel = [c for c in NOVEL_CLASSES if c in all_embs]

    print(f"\n  Available base classes : {len(available_base)}")
    print(f"  Available novel classes: {len(available_novel)}")

    if len(available_base) < 3:
        print("\n  ERROR: Not enough base classes found. Check your data directory.")
        return

    # ── 1. Distance analysis ───────────────────────────────────────────────────
    _section("1. Intra / inter-class distance analysis")
    dist_results = analyse_distances(all_embs)

    print(f"\n  {'Class':12} {'Intra':>7} {'Inter_min':>10} {'Ratio':>7}  {'Nearest':12}  Quality")
    print(f"  {'-'*66}")
    for cname in sorted(available_base, key=lambda c: dist_results[c]['ratio']):
        r = dist_results[cname]
        q = "GOOD" if r['ratio'] > 1.5 else ("WEAK" if r['ratio'] > 1.0 else "BAD ")
        print(f"  {cname:12} {r['intra_mean']:7.2f} {r['inter_min']:10.2f} "
              f"{r['ratio']:7.3f}  {r['nearest']:12}  {q}")

    # ── 2. Confusion matrix ────────────────────────────────────────────────────
    _section("2. Confusion matrix over base classes")
    cm = full_confusion_matrix(all_embs, available_base)
    cm_acc = confusion_accuracy(cm)

    print(f"\n  {'Class':12} {'Acc':>7}  {'Confused with (top 3)':40}")
    print(f"  {'-'*66}")
    for cname in sorted(cm_acc, key=lambda c: cm_acc[c]['accuracy']):
        a = cm_acc[cname]
        top = ", ".join(f"{p}({n})" for p, n in a['top_confused'][:3])
        flag = " <<< LOW ACC" if a['accuracy'] < 0.5 else ""
        print(f"  {cname:12} {a['accuracy']:7.1%}  {top}{flag}")

    overall_acc = _safe_mean([v['accuracy'] for v in cm_acc.values()])
    print(f"\n  Overall accuracy (base classes, fixed support): {overall_acc:.1%}")

    # ── 3. Confusion pairs ─────────────────────────────────────────────────────
    _section("3. Most confusable class pairs (T1 demo candidates)")
    conf_pairs = score_confusion_pairs(cm_acc, dist_results)

    print(f"\n  {'#':>3}  {'Class A':12} {'Class B':12} {'Mut.Conf':>9}  "
          f"{'A→B':>5}  {'B→A':>5}  {'Mut.NN':>7}  Score")
    print(f"  {'-'*72}")
    for i, p in enumerate(conf_pairs[:TOP_N_CONFUSION]):
        nn = "YES" if p['mutual_nearest'] else "no"
        print(f"  {i+1:>3}  {p['class_a']:12} {p['class_b']:12} "
              f"{p['mutual_conf']:9.1%}  {p['conf_a_to_b']:5d}  {p['conf_b_to_a']:5d}  "
              f"{nn:>7}  {p['score']:.3f}")

    # ── 4. Support-set volatility ──────────────────────────────────────────────
    _section(f"4. Support-set volatility ({N_SUPPORT_CONFIGS} random configs each)")
    print("     High std = large performance swing across support choices = best T3 demo.\n")
    volatility = support_volatility(all_embs, available_base)

    print(f"  {'Class':12} {'Mean acc':>9} {'Std':>7} {'Min':>7} {'Max':>7}  T3 demo value")
    print(f"  {'-'*66}")
    for cname in sorted(volatility, key=lambda c: -volatility[c]['acc_std']):
        v = volatility[cname]
        label = _score_label(v['acc_std'], thresholds=(0.15, 0.08))
        print(f"  {cname:12} {v['acc_mean']:9.1%} {v['acc_std']:7.1%} "
              f"{v['acc_min']:7.1%} {v['acc_max']:7.1%}  {label}")

    # ── 5. Embedding isotropy ──────────────────────────────────────────────────
    _section("5. Embedding isotropy (T2 heatmap interest proxy)")
    print("     Low participation ratio = concentrated activations = vivid heatmaps.\n")
    isotropy = embedding_isotropy(all_embs)

    print(f"  {'Class':12} {'Part.Ratio':>11} {'Top-1 var':>10}  T2 demo value")
    print(f"  {'-'*56}")
    all_classes_iso = sorted(
        [c for c in isotropy if c in available_base or c in available_novel],
        key=lambda c: isotropy[c]['participation_ratio']
    )
    for cname in all_classes_iso:
        iso = isotropy[cname]
        kind = "[NOVEL]" if cname in available_novel else "       "
        # Low PR = more anisotropic = better heatmap
        label = _score_label(1.0 - iso['participation_ratio'], thresholds=(0.65, 0.40))
        print(f"  {cname:12} {kind} {iso['participation_ratio']:11.4f} "
              f"{iso['top1_var_explained']:10.1%}  {label}")

    # ── 6. Novel class separation ──────────────────────────────────────────────
    _section("6. Novel class evaluation (T4 demo candidates)")
    print("     Best T4 demo: high accuracy (model works) BUT anisotropic/background-focused")
    print("     heatmap (found via T2). Or low accuracy + clear confusion pair.\n")
    novel_sep = novel_class_separation(all_embs)

    print(f"  {'Novel class':12} {'Acc':>7}  {'Nearest base':14} {'Dist':>7}  Top confusion")
    print(f"  {'-'*72}")
    for cname in sorted(novel_sep, key=lambda c: -novel_sep[c]['accuracy']):
        ns  = novel_sep[cname]
        iso = isotropy.get(cname, {})
        pr  = iso.get('participation_ratio', 99)
        acc_warning = " <<< HIGH-ACC/ANISO?" if ns['accuracy'] > 0.7 and pr < 0.3 else ""
        top_conf = ", ".join(f"{p}({n})" for p, n in list(ns['confusion'].items())[:2])
        print(f"  {cname:12} {ns['accuracy']:7.1%}  {ns['nearest_base']:14} "
              f"{ns['nearest_dist']:7.1f}  {top_conf}{acc_warning}")

    # ── 7. Recommended class subset ───────────────────────────────────────────
    _section("7. Recommended dashboard class subset for demo video")
    recommended = recommend_class_subset(
        available_base, cm_acc, volatility, dist_results, conf_pairs, n_classes=10
    )

    print(f"\n  Recommended {len(recommended)} base classes:\n")
    for cname in recommended:
        acc  = cm_acc.get(cname, {}).get('accuracy', 0)
        vstd = volatility.get(cname, {}).get('acc_std', 0)
        rat  = dist_results.get(cname, {}).get('ratio', 99)
        near = dist_results.get(cname, {}).get('nearest', '?')
        print(f"    {cname:12}  acc={acc:.0%}  vol_std={vstd:.0%}  "
              f"proto_ratio={rat:.2f}  nearest={near}")

    # Best novel class recommendation
    if novel_sep:
        best_novel = max(
            novel_sep,
            # Prefer classes that are near a base class AND have low isotropy
            key=lambda c: (
                (1.0 - isotropy.get(c, {}).get('participation_ratio', 1.0)) * 0.6 +
                (1.0 - novel_sep[c]['accuracy']) * 0.4
            )
        )
        print(f"\n  Recommended NOVEL class to add in demo (T4): {best_novel.upper()}")
        ns = novel_sep[best_novel]
        iso = isotropy.get(best_novel, {})
        print(f"    accuracy={ns['accuracy']:.0%}  nearest_base={ns['nearest_base']}  "
              f"part_ratio={iso.get('participation_ratio', 99):.4f}")

    # ── 8. Video workflow summary ──────────────────────────────────────────────
    _section("8. Suggested demo video workflow")
    top_pair        = conf_pairs[0] if conf_pairs else {}
    top_volatile    = max(volatility, key=lambda c: volatility[c]['acc_std']) if volatility else None
    top_vol_std     = volatility[top_volatile]['acc_std'] if top_volatile else 0.0
    best_novel_name = best_novel if novel_sep else "???"
    pair_a          = top_pair.get('class_a', '?').upper()
    pair_b          = top_pair.get('class_b', '?').upper()
    pair_conf       = top_pair.get('mutual_conf', 0)
    n_rec           = len(recommended)

    print(f"""
  SEGMENT 1 - Problem intro  (~40 s)
    "Few-shot ProtoNets are powerful but fragile: one bad support image can
     corrupt an entire class prototype. Standard accuracy metrics hide this.
     ProtoNet Explorer makes it visible and fixable."

  SEGMENT 2 - T1: Show cluster confusion  (~50 s)
    Load recommended {n_rec}-class session.
    UMAP: point out overlapping clusters.
    Zoom in on the {pair_a} / {pair_b} region
    (mutual confusion: {pair_conf:.0%}) and lower tau to sharpen boundary.

  SEGMENT 3 - T2: Inspector + heatmaps  (~60 s)
    Click a misclassified point in that confusion region.
    Inspector opens: show distance bars, then click a support sketch
    to trigger the co-activation heatmap.
    Show contrast: correct class (tight object focus) vs confused class.

  SEGMENT 4 - T3: Support curation  (~70 s)
    Open Candidate Explorer for {top_volatile or '???'} (highest volatility:
    std={top_vol_std:.0%} across configurations).
    Point out that top-right candidates (naive choice) are NOT optimal.
    Replace one support sketch, watch prototype move in UMAP in real time.
    Show accuracy improvement in confusion matrix.

  SEGMENT 5 - T4: Novel class  (~50 s)
    Add {best_novel_name.upper()} as novel class.
    Canvas: draw a sketch live (or import from file).
    System plots it in UMAP, shows prediction.
    Open inspector: show heatmap where encoder focuses on background/texture
    rather than object outline. This is invisible without the VA system.

  SEGMENT 6 - Wrap-up  (~30 s)
    Return to overview UMAP with all classes.
    "ProtoNet Explorer surfaces the geometry behind few-shot decisions,
     enabling practitioners to diagnose, curate, and validate without
     retraining the encoder."
""")

    # ── 10. Subset-aware re-evaluation ────────────────────────────────────────
    _section("10. Subset-aware re-evaluation (numbers that match the actual dashboard)")
    print("""
  WARNING: all sections above used all 29 base classes as competitors.
  ProtoNet classifies by nearest prototype among ACTIVE classes only.
  Removing 19 classes restructures every Voronoi boundary, so confusion
  pairs, accuracy, and volatility can all shift significantly.
  The numbers below are what you will actually see in the dashboard.
""")
    novel_for_subset = best_novel if novel_sep else None
    subset_eval = evaluate_subset(
        all_embs, recommended, novel_for_subset,
        n_support_configs=N_SUPPORT_CONFIGS,
        n_query=N_QUERY_VOLATILITY,
    )

    # Distance within subset
    print(f"  Distances within {len(recommended)}-class subset:\n")
    print(f"  {'Class':12} {'Intra':>7} {'Inter_min':>10} {'Ratio':>7}  {'Nearest (subset)':18}  Quality")
    print(f"  {'-'*72}")
    for cname in sorted(subset_eval['distance'], key=lambda c: subset_eval['distance'][c]['ratio']):
        r = subset_eval['distance'][cname]
        q = "GOOD" if r['ratio'] > 1.5 else ("WEAK" if r['ratio'] > 1.0 else "BAD ")
        print(f"  {cname:12} {r['intra_mean']:7.2f} {r['inter_min']:10.2f} "
              f"{r['ratio']:7.3f}  {r['nearest']:18}  {q}")

    # Confusion within subset
    print(f"\n  Confusion within {len(recommended)}-class subset:\n")
    print(f"  {'Class':12} {'Acc':>7}  {'Confused with':40}")
    print(f"  {'-'*66}")
    for cname in sorted(subset_eval['confusion_acc'], key=lambda c: subset_eval['confusion_acc'][c]['accuracy']):
        a   = subset_eval['confusion_acc'][cname]
        top = ", ".join(f"{p}({n})" for p, n in a['top_confused'][:3])
        anchor_flag = " [ANCHOR]" if a['accuracy'] >= 0.98 and subset_eval['volatility'].get(cname, {}).get('acc_std', 1) < 0.05 else ""
        print(f"  {cname:12} {a['accuracy']:7.1%}  {top}{anchor_flag}")
    print(f"\n  Overall accuracy in subset: {subset_eval['overall_acc']:.1%}")

    # Volatility within subset
    print(f"\n  Volatility within subset ({N_SUPPORT_CONFIGS} configs):\n")
    print(f"  {'Class':12} {'Mean':>8} {'Std':>7} {'Min':>7} {'Max':>7}  Gap (max-default)")
    print(f"  {'-'*72}")
    sub_cm_acc = subset_eval['confusion_acc']
    for cname in sorted(subset_eval['volatility'], key=lambda c: -subset_eval['volatility'][c]['acc_std']):
        v     = subset_eval['volatility'][cname]
        d_acc = sub_cm_acc.get(cname, {}).get('accuracy', 0)
        gap   = v['acc_max'] - d_acc
        print(f"  {cname:12} {v['acc_mean']:8.1%} {v['acc_std']:7.1%} "
              f"{v['acc_min']:7.1%} {v['acc_max']:7.1%}  +{gap:.0%}")

    # Novel class within subset
    if subset_eval['novel_eval']:
        nv = subset_eval['novel_eval']
        print(f"\n  Novel class '{novel_for_subset}' evaluated within subset:")
        print(f"    accuracy={nv['accuracy']:.1%}  nearest_base={nv['nearest_base']}  "
              f"dist={nv['nearest_dist']:.1f}")
        top_conf = ", ".join(f"{p}({n})" for p, n in list(nv['confusion'].items())[:3])
        print(f"    top confusion: {top_conf}")

    # ── 11. Export JSON for report ─────────────────────────────────────────────
    _section("11. Exporting results to JSON")

    report_data = {
        'meta': {
            'n_support':          N_SUPPORT,
            'n_images_per_class': N_IMAGES_PER_CLASS,
            'n_support_configs':  N_SUPPORT_CONFIGS,
            'n_query_volatility': N_QUERY_VOLATILITY,
            'seed':               SEED,
        },
        'available_base_classes':  available_base,
        'available_novel_classes': available_novel,
        'distance_results':        dist_results,
        'confusion_accuracy':      {
            c: {**v, 'top_confused': v['top_confused']}
            for c, v in cm_acc.items()
        },
        'overall_accuracy_base':   overall_acc,
        'top_confusion_pairs':     conf_pairs[:10],
        'support_volatility':      volatility,
        'embedding_isotropy':      isotropy,
        'novel_class_separation':  novel_sep,
        'recommended_base_subset': recommended,
        'recommended_novel_class': best_novel if novel_sep else None,
        'subset_eval':             subset_eval,
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)

    print(f"\n  Results saved to:  {OUTPUT_JSON}")
    _banner("Done")


if __name__ == '__main__':
    main()