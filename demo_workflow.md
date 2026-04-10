# ProtoNet Dashboard: Demo Workflow

This workflow is designed to showcase the full capabilities of the **ProtoNet Embedding Explorer**, specifically focusing on how visual analytics can improve few-shot classification.

## 🚀 Setup
Before starting, run the setup script to initialize the 10 optimal base classes:
```bash
python prepare_demo.py
python -m dashboard.app
```

---

## 🟢 Part 1: Exploration (The "Static" AI)
**Goal:** Show the initial state of the pre-trained classifier.

1.  **Overview**: Point out the 10 base classes in the UMAP (e.g., `bird`, `clock`, `dog`, `fish`).
2.  **Accuracy**: Click the **ACCURACY** tab in the sidebar. Note that some classes (like `bird`) might have lower accuracy due to diversity in the drawings.
3.  **Confusion**: Switch to the **CONFUSION** tab. See if `dog` and `bird` show any overlap.
4.  **Inspect**: Click a correct query point (○) to see the sketch and its top-k predictions.

---

## 🟠 Part 2: The Challenge (Out-of-Distribution)
**Goal:** Show what happens when the AI sees something new.

1.  **Select Query**: Find a point in the "bird" cluster that looks different, or prepare to draw.
2.  **The "Airplane" Problem**:
    - An `airplane` is NOT in our base classes.
    - If you draw an airplane (or find one in the `custom_drawings` pool), the AI will likely classify it as a `bird`.
    - **Visual Proof**: In the UMAP, the airplane point will land near the `bird` prototype (★) because that's the closest thing it knows.

---

## 🔵 Part 3: Few-Shot Learning (Adding a Class)
**Goal:** Teach the AI a new concept in seconds.

1.  **Import**: Go to the **IMPORT** panel (sidebar).
2.  **Add Airplane**: Select `airplane` from the list.
3.  **Visible Magic**:
    - A new cluster of points appears!
    - A new prototype (★) for `airplane` is created.
    - Watch the **Decision Mesh** (the colored background) shift in real-time. The `bird` territory shrinks to make room for `airplane`.
4.  **Verification**: The previous "misclassified" point is now correctly labeled as `airplane`.

---

## 🟣 Part 4: Interactive Alignment (The "Visual" in Visual Analytics)
**Goal:** Human-in-the-loop refinement.

1.  **Draggable Prototypes**:
    - Notice a `bird` sketch that is dangerously close to the `airplane` border.
    - **Action**: Drag the `airplane` prototype (★) slightly away from the `bird` cluster.
    - **Result**: The decision boundary moves, and the `bird` sketch is now safely in the `bird` zone.
2.  **Weight Adjustment**:
    - Go to the **Support Set** for `airplane`.
    - Find a "weirdly drawn" airplane.
    - **Action**: Lower its weight to `0.2`.
    - **Result**: The prototype (★) nudges away from that messy example, centering itself on the better ones.

---

## 🔴 Part 5: Cleaning the Space (Performance Optimization)
**Goal:** Improve accuracy by removing "noise".

1.  **Diagnostics**: Click **DIAGNOSTICS** on the `airplane` class.
2.  **Identify Harmful Data**:
    - Look for images with a **Negative LOO Delta** (Leave-One-Out). These are hurting overall accuracy.
    - Look for images with **High Competitor Similarity** (e.g., an airplane that looks too much like a `scissors`).
3.  **Exclude**: Click **EXCLUDE** on the worst performers.
4.  **Fill the Gap**: Go to the **CANDIDATES** tab. Identify the image with the highest **Predicted Accuracy Delta** and add it to the support set.

---

## ✨ Conclusion
In 5 minutes, we:
1.  Identified a classification weakness.
2.  Added a new category with only 5 examples.
3.  Refined the decision boundary visually (dragging).
4.  Optimized the model by "cleaning" the embedding space based on diagnostics.
