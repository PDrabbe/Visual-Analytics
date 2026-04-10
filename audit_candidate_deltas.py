"""
audit_candidate_deltas.py
=========================
For every class in the demo session, computes the full per-class accuracy
delta for every candidate image and prints a ranked table.

This verifies the hypothesis:
  The best candidate (highest overall delta) is NOT always the one
  closest to the current prototype. A support image choice shifts the
  prototype, reshaping ALL decision boundaries, so the optimal pick
  must be evaluated globally — not per-class.

Run from project root:
    python audit_candidate_deltas.py

Output:
  - Printed tables per class
  - audit_candidate_deltas.json  (full data for further analysis)
"""

import json
import logging
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.WARNING)

DEMO_CLASSES = [
    "dog", "horse", "truck", "car", "cat",
    "scissors", "banana", "key", "pizza", "triangle",
]

SESSION_PATH = Path("session.json")
OUTPUT_JSON  = Path("audit_candidate_deltas.json")

# ── Load engine ───────────────────────────────────────────────────────────────
print("Loading engine...")
from dashboard.engine import AnalyticsEngine

engine = AnalyticsEngine()

# Restore session if available
saved = {}
if SESSION_PATH.exists():
    try:
        saved = json.loads(SESSION_PATH.read_text())
    except Exception:
        pass

init_classes   = saved.get("classes", DEMO_CLASSES)
init_layout    = saved.get("embeddings_2d")
engine.init_demo(active_classes=init_classes, loaded_embeddings_2d=init_layout)

# Build sc from session or use defaults
sc = {}
saved_sc = saved.get("sc", {})
for cname in engine.class_names:
    if cname in saved_sc and saved_sc[cname]:
        sc[cname] = saved_sc[cname]
    else:
        sc[cname] = engine.default_support.get(cname, [])

print(f"Session: {len(engine.class_names)} classes, {len(engine.images)} images\n")

# ── Helpers ───────────────────────────────────────────────────────────────────

def full_candidate_delta(cname, cand_idx, sc):
    """
    Returns a dict of per-class accuracy deltas for every class,
    plus overall delta, when cand_idx is added to cname's support.
    """
    base_res = engine.classify(sc)
    base_accs = base_res["accs"]
    base_overall = base_res["overall"]

    trial_sc = {c: list(v) for c, v in sc.items()}
    trial_sc[cname] = trial_sc.get(cname, []) + [{"idx": cand_idx, "weight": 1.0}]
    trial_res = engine.classify(trial_sc)

    per_class = {
        cn: float(trial_res["accs"].get(cn, 0.0) - base_accs.get(cn, 0.0))
        for cn in engine.class_names
    }
    return {
        "overall":   float(trial_res["overall"] - base_overall),
        "per_class": per_class,
    }


def dist_to_prototype(cand_idx, cname, sc):
    """Euclidean distance from candidate embedding to current class prototype."""
    _, phd, order = engine.compute_prototypes(sc)
    if cname not in order:
        return float("inf")
    proto = phd[order.index(cname)]
    return float(np.linalg.norm(engine.embeddings_hd[cand_idx] - proto))


def dist_to_competitor(cand_idx, cname, sc):
    """Euclidean distance from candidate to nearest competitor prototype."""
    _, phd, order = engine.compute_prototypes(sc)
    if cname not in order:
        return float("inf")
    ci = order.index(cname)
    others = [i for i in range(len(order)) if i != ci]
    if not others:
        return float("inf")
    dists = np.linalg.norm(phd[others] - engine.embeddings_hd[cand_idx], axis=1)
    return float(dists.min())


# ── Baseline accuracy ─────────────────────────────────────────────────────────
base = engine.classify(sc)
print("=" * 72)
print("BASELINE ACCURACY")
print("=" * 72)
print(f"  Overall: {base['overall']:.1%}\n")
for cn in engine.class_names:
    acc = base["accs"].get(cn, 0.0)
    bar = "█" * int(acc * 20)
    print(f"  {cn:12} {acc:6.1%}  {bar}")

# ── Per-class candidate audit ─────────────────────────────────────────────────
all_results = {}

for target_class in engine.class_names:
    pool = engine.class_images_pool(target_class, sc)
    candidates = [c for c in pool if not c["is_support"]]

    if not candidates:
        continue

    print(f"\n{'=' * 72}")
    print(f"CLASS: {target_class.upper()}  ({len(candidates)} candidates)")
    print(f"{'=' * 72}")

    rows = []
    for c in candidates:
        idx = c["idx"]
        d_proto = c["dist"]
        d_comp  = c["competitor_dist"]
        result  = full_candidate_delta(target_class, idx, sc)
        overall = result["overall"]
        per_cls = result["per_class"]

        # Most helped and most hurt OTHER class
        other_deltas = {cn: v for cn, v in per_cls.items() if cn != target_class}
        most_helped  = max(other_deltas, key=lambda cn: other_deltas[cn])
        most_hurt    = min(other_deltas, key=lambda cn: other_deltas[cn])

        rows.append({
            "idx":          idx,
            "d_proto":      d_proto,
            "d_comp":       d_comp,
            "own_delta":    per_cls.get(target_class, 0.0),
            "overall":      overall,
            "most_helped":  most_helped,
            "help_val":     other_deltas[most_helped],
            "most_hurt":    most_hurt,
            "hurt_val":     other_deltas[most_hurt],
            "per_class":    per_cls,
        })

    # Sort by overall delta descending
    rows.sort(key=lambda r: -r["overall"])

    # Print top 10 and bottom 5
    print(f"\n  {'Rank':>4}  {'idx':>5}  {'d_proto':>8}  {'d_comp':>8}  "
          f"{'own Δ':>8}  {'overall Δ':>10}  {'most helped':>16}  {'most hurt':>16}")
    print(f"  {'-'*100}")

    for rank, r in enumerate(rows[:10], 1):
        helped_str = f"{r['most_helped']}({r['help_val']:+.1%})"
        hurt_str   = f"{r['most_hurt']}({r['hurt_val']:+.1%})"
        marker = " <<<  BEST" if rank == 1 else ""
        print(f"  {rank:>4}  #{r['idx']:<4}  {r['d_proto']:8.2f}  {r['d_comp']:8.2f}  "
              f"{r['own_delta']:8.1%}  {r['overall']:10.1%}  "
              f"{helped_str:>16}  {hurt_str:>16}{marker}")

    if len(rows) > 10:
        print(f"  {'...':>4}")
        for r in rows[-3:]:
            helped_str = f"{r['most_helped']}({r['help_val']:+.1%})"
            hurt_str   = f"{r['most_hurt']}({r['hurt_val']:+.1%})"
            print(f"  {'bot':>4}  #{r['idx']:<4}  {r['d_proto']:8.2f}  {r['d_comp']:8.2f}  "
                  f"{r['own_delta']:8.1%}  {r['overall']:10.1%}  "
                  f"{helped_str:>16}  {hurt_str:>16}")

    # ── Key insight check ─────────────────────────────────────────────────────
    best        = rows[0]
    naive_best  = min(rows, key=lambda r: r["d_proto"])  # closest to own prototype

    print(f"\n  INSIGHT CHECK for {target_class.upper()}:")
    print(f"    Best overall candidate:   #{best['idx']}  "
          f"d_proto={best['d_proto']:.2f}  overall={best['overall']:+.1%}  "
          f"own={best['own_delta']:+.1%}")
    print(f"    Naive choice (min d_proto): #{naive_best['idx']}  "
          f"d_proto={naive_best['d_proto']:.2f}  overall={naive_best['overall']:+.1%}  "
          f"own={naive_best['own_delta']:+.1%}")

    if best["idx"] != naive_best["idx"]:
        diff = best["overall"] - naive_best["overall"]
        print(f"    THEORY CONFIRMED: best != naive. Overall gap = {diff:+.1%}")
    else:
        print(f"    Theory not confirmed for this class: naive choice is also the best.")

    # ── Correlation check ─────────────────────────────────────────────────────
    d_protos  = np.array([r["d_proto"]  for r in rows])
    overalls  = np.array([r["overall"]  for r in rows])
    own_deltas= np.array([r["own_delta"] for r in rows])

    corr_proto_overall = float(np.corrcoef(d_protos, overalls)[0, 1])
    corr_own_overall   = float(np.corrcoef(own_deltas, overalls)[0, 1])

    print(f"\n    Correlation(d_proto, overall_delta) = {corr_proto_overall:+.3f}"
          f"  {'(weak/no link)' if abs(corr_proto_overall) < 0.4 else '(some link)'}")
    print(f"    Correlation(own_delta, overall_delta) = {corr_own_overall:+.3f}"
          f"  {'(weak/no link)' if abs(corr_own_overall) < 0.4 else '(some link)'}")

    if abs(corr_proto_overall) < 0.4:
        print(f"    => Prototype distance is a poor predictor of overall delta.")
        print(f"       Supporting the theory: you cannot optimise globally by looking at per-class geometry alone.")

    all_results[target_class] = rows

# ── Summary across all classes ────────────────────────────────────────────────
print(f"\n{'=' * 72}")
print("SUMMARY — Theory confirmation across all classes")
print("=" * 72)
print(f"\n  {'Class':12}  {'Best≠Naive':>10}  {'Overall gap':>12}  {'Corr(d_proto,Δ)':>16}")
print(f"  {'-'*60}")

confirmed_count = 0
for cname, rows in all_results.items():
    if not rows:
        continue
    best       = rows[0]
    naive_best = min(rows, key=lambda r: r["d_proto"])
    confirmed  = best["idx"] != naive_best["idx"]
    if confirmed:
        confirmed_count += 1
    gap        = best["overall"] - naive_best["overall"]
    d_protos   = np.array([r["d_proto"]  for r in rows])
    overalls   = np.array([r["overall"]  for r in rows])
    corr       = float(np.corrcoef(d_protos, overalls)[0, 1]) if len(rows) > 2 else 0.0
    tick       = "YES" if confirmed else "no"
    print(f"  {cname:12}  {tick:>10}  {gap:>+12.1%}  {corr:>+16.3f}")

print(f"\n  Theory confirmed for {confirmed_count}/{len(all_results)} classes.")

# ── Export ────────────────────────────────────────────────────────────────────
export = {}
for cname, rows in all_results.items():
    export[cname] = [
        {k: v for k, v in r.items() if k != "per_class"}
        for r in rows
    ]
    # Include full per_class for top 5 only
    for i, r in enumerate(rows[:5]):
        export[cname][i]["per_class"] = r["per_class"]

OUTPUT_JSON.write_text(json.dumps({
    "baseline": {cn: base["accs"].get(cn, 0.0) for cn in engine.class_names},
    "baseline_overall": base["overall"],
    "results": export,
}, indent=2))

print(f"\nFull results saved to: {OUTPUT_JSON}")