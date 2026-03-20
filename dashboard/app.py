"""
ProtoNet Embedding Explorer
Run:  python -m dashboard.app
"""

import hashlib
import json
import re
import numpy as np
import plotly.graph_objects as go
from dash import (
    Dash, html, dcc,
    Input, Output, State,
    callback_context, ALL, no_update,
)
from dash.exceptions import PreventUpdate

from dashboard.engine import AnalyticsEngine, CLASS_COLORS, CLASS_NAMES

# =====================================================================
# Engine initialisation
# =====================================================================
print("Loading model + QuickDraw data + UMAP …")
engine = AnalyticsEngine().init_demo()
print(f"Ready — {len(engine.images)} sketches, {engine.n_classes} classes")

# =====================================================================
# Theme — Lab Instrument
# =====================================================================
DARK_BG = "#1a1d17"
CARD_BG = "#232820"
BORDER  = "#3a4235"
TEXT    = "#c8c4a8"
TEXT2   = "#7a7860"
ACCENT  = "#e8a420"
OK_COL  = "#50a050"
ERR_COL = "#c84040"
FONT    = "Consolas, 'Courier New', 'Liberation Mono', monospace"

SCATTER_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=DARK_BG,
    font=dict(color=TEXT, size=11, family=FONT),
    margin=dict(l=48, r=16, t=40, b=48),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, title="UMAP-1",
               gridwidth=1),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, title="UMAP-2",
               scaleanchor="x", scaleratio=1, gridwidth=1),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10),
                x=0.01, y=0.99, xanchor="left", yanchor="top"),
    dragmode="pan",
)


def _base_fig(**kw) -> go.Figure:
    layout = {**SCATTER_LAYOUT, **kw}
    return go.Figure(layout=layout)


def _hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)'"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _uirev_hash(sc, temp, sel_q, uirev) -> str:
    """Stable hash of all scatter inputs *except* pov.
    """
    key = json.dumps(sc, sort_keys=True) + str(temp) + str(sel_q) + str(uirev)
    return hashlib.md5(key.encode()).hexdigest()[:8]


# =====================================================================
# Layout
# =====================================================================
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    assets_folder="assets",
    title="ProtoNet Embedding Explorer",
)

app.layout = html.Div(
    [
        # ── header ──────────────────────────────────────────────
        html.Div(
            [
                html.Div(
                    [
                        html.H1("PROTONET EXPLORER",
                                className="header-title"),
                        html.Span("few-shot sketch classifier",
                                  className="header-sub"),
                    ],
                    className="header-left",
                ),
                html.Div(
                    [
                        html.Label("τ", className="temp-label"),
                        dcc.Slider(
                            id="temp-slider",
                            min=0.1, max=5.0, step=0.1, value=1.0,
                            marks={0.1: "0.1", 1: "1", 3: "3", 5: "5"},
                            tooltip={"placement": "bottom",
                                     "always_visible": True},
                            className="temp-slider",
                        ),
                    ],
                    className="header-temp",
                ),
                html.Div(id="overall-stat", className="header-stat"),
            ],
            className="header",
        ),

        # ── body ────────────────────────────────────────────────
        html.Div(
            [
                # ── hero scatter ───────────────────────────────
                html.Div(
                    [
                        dcc.Graph(
                            id="scatter",
                            config={
                                "scrollZoom": True,
                                "displaylogo": False,
                                "edits": {
                                    "annotationPosition": True,
                                    "annotationText": False,
                                    "annotationTail": False,
                                    "titleText": False,
                                    "axisTitleText": False,
                                    "legendPosition": False,
                                    "legendText": False,
                                    "colorbarPosition": False,
                                    "colorbarTitleText": False,
                                },
                                "modeBarButtonsToRemove": [
                                    "select2d", "lasso2d",
                                    "autoScale2d"],
                            },
                            style={"height": "100%", "width": "100%"},
                        ),
                        # symbol key
                        html.Div(
                            [
                                html.Span("○", className="leg-sym"),
                                html.Span("correct  ", className="leg-txt"),
                                html.Span("✕", className="leg-sym leg-wrong"),
                                html.Span("wrong  ", className="leg-txt"),
                                html.Span("◆", className="leg-sym leg-sup"),
                                html.Span("support  ", className="leg-txt"),
                                html.Span("★", className="leg-sym leg-proto"),
                                html.Span("prototype (drag to move)",
                                          className="leg-txt"),
                            ],
                            className="scatter-key",
                        ),
                    ],
                    className="scatter-container",
                ),

                # ── sidebar ───────────────────────────────────
                html.Div(
                    [
                        # classes
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("CLASSES",
                                                   className="ctrl-label"),
                                        html.Button("RESET",
                                                     id="reset-btn",
                                                     className="btn-small"),
                                    ],
                                    className="section-header",
                                ),
                                html.Div(id="class-legend",
                                         className="class-legend"),
                            ],
                            className="sidebar-section",
                        ),

                        # active class
                        html.Div(
                            [
                                html.Label("ACTIVE CLASS",
                                           className="ctrl-label"),
                                dcc.Dropdown(
                                    id="class-selector",
                                    options=[{"label": c, "value": c}
                                             for c in CLASS_NAMES],
                                    value=CLASS_NAMES[0],
                                    clearable=False,
                                    className="class-dropdown",
                                ),
                            ],
                            className="sidebar-section",
                        ),

                        # prototype overrides
                        html.Div(
                            [
                                html.Label("PROTOTYPE OVERRIDES",
                                           className="ctrl-label"),
                                html.Div(id="override-list",
                                         className="override-list"),
                                        html.Div(
                                    [
                                        html.Button(
                                            "↩ UNDO",
                                            id="undo-btn",
                                            className="btn-small",
                                            n_clicks=0,
                                        ),
                                        html.Button(
                                            "↪ REDO",
                                            id="redo-btn",
                                            className="btn-small",
                                            n_clicks=0,
                                        ),
                                    ],
                                    className="undo-redo-row",
                                ),
                                html.Div(
                                    "Drag ★ labels to reposition prototypes.",
                                    className="hint-text",
                                ),
                            ],
                            className="sidebar-section",
                        ),

                        # detail panel (inspector OR support set)
                        html.Div(
                            [
                                html.Div(id="detail-panel",
                                         className="detail-panel"),
                            ],
                            className="sidebar-section support-section",
                        ),
                    ],
                    className="sidebar",
                ),
            ],
            className="body-wrapper",
        ),

        # ── mask editor modal (overlays everything) ─────────────
        html.Div(
            id="mask-modal",
            className="mask-modal",
            style={"display": "none"},
            children=[
                html.Div(
                    id="mask-modal-content",
                    className="mask-modal-body",
                ),
            ],
        ),

        # ── stores ──────────────────────────────────────────────
        dcc.Store(id="sc-store",          data=engine.default_support),
        dcc.Store(id="proto-ov-store",    data={}),
        dcc.Store(id="proto-undo-store",  data=[]),
        dcc.Store(id="proto-redo-store",  data=[]),
        dcc.Store(id="proto-order-store", data=[]),
        dcc.Store(id="sel-query-store",   data=None),
        dcc.Store(id="sel-support-store", data=None),  # highlighted support idx
        dcc.Store(id="gradcam-store",     data=False),  # Grad-CAM overlay toggle
        dcc.Store(id="mask-edit-store",   data=None),   # idx being mask-edited
        dcc.Store(id="suggest-store",     data=False),  # suggestion panel toggle
        dcc.Store(id="whatif-store",      data=None),   # trial sc for preview
        dcc.Store(id="uirev-store",       data=0),
    ],
    className="root",
)


# =====================================================================
# Callbacks
# =====================================================================

# ── 1. Scatter + class legend + overall stat ──────────────────────


@app.callback(
    Output("scatter", "figure"),
    Output("class-legend", "children"),
    Output("overall-stat", "children"),
    Output("proto-order-store", "data"),
    Input("sc-store", "data"),
    Input("temp-slider", "value"),
    Input("sel-query-store", "data"),
    Input("whatif-store", "data"),
    State("uirev-store", "data"),
)
def update_scatter(sc, temp, sel_q, whatif_sc, uirev):
    res = engine.classify(sc, temp)
    order = res["order"]
    if not order:
        empty = _base_fig(title="No data")
        return empty, "No classes", "", []

    sfig = _base_fig(title="Embedding Space  (UMAP)",
                     uirevision=_uirev_hash(sc, temp, sel_q, uirev))

    # decision mesh
    xx, ys, zz = engine.decision_mesh(sc, temp, res=60)
    if len(xx) > 0:
        nc = len(order)
        cvals = []
        for i in range(nc):
            lo = i / nc
            hi = (i + 1) / nc
            col = CLASS_COLORS[CLASS_NAMES.index(order[i]) % len(CLASS_COLORS)]
            cvals.append([lo, col])
            cvals.append([hi, col])
        sfig.add_trace(
            go.Heatmap(
                x=xx[0], y=ys, z=zz,
                colorscale=cvals, opacity=0.08,
                showscale=False, hoverinfo="skip",
            )
        )

    # query points
    qm = engine.query_mask(sc)
    qidxs = np.where(qm)[0]
    name2oi = {n: i for i, n in enumerate(order)}

    for ci, cname in enumerate(CLASS_NAMES):
        if cname not in name2oi:
            continue
        mask = engine.labels[qidxs] == ci
        if not mask.any():
            continue
        pts = qidxs[mask]
        q_positions = np.where(mask)[0]
        correct = res["true_idx"][q_positions] == res["preds"][q_positions]
        col = CLASS_COLORS[ci % len(CLASS_COLORS)]

        # Correct points — circle, shown in Plotly legend
        c_pts = pts[correct]
        if len(c_pts):
            sfig.add_trace(go.Scatter(
                x=engine.embeddings_2d[c_pts, 0],
                y=engine.embeddings_2d[c_pts, 1],
                mode="markers",
                marker=dict(size=7, color=col, symbol="circle",
                            opacity=0.85,
                            line=dict(width=0.5,
                                      color="rgba(255,255,255,0.15)")),
                name=cname,
                customdata=c_pts.tolist(),
                text=[f"<b>{cname}</b> (#{i})<br>✓ correct" for i in c_pts],
                hovertemplate="%{text}<extra></extra>",
            ))

        # Wrong points — X marker, hidden from Plotly legend
        w_pts = pts[~correct]
        if len(w_pts):
            sfig.add_trace(go.Scatter(
                x=engine.embeddings_2d[w_pts, 0],
                y=engine.embeddings_2d[w_pts, 1],
                mode="markers",
                marker=dict(size=8, color=col, symbol="x",
                            opacity=0.9,
                            line=dict(width=1.5, color=col)),
                name=cname,
                showlegend=False,
                customdata=w_pts.tolist(),
                text=[f"<b>{cname}</b> (#{i})<br>✗ WRONG" for i in w_pts],
                hovertemplate="%{text}<extra></extra>",
            ))

    # support convex hulls (reachable region for prototype)
    for cname, items in sc.items():
        if len(items) < 2:
            continue
        ci = CLASS_NAMES.index(cname) if cname in CLASS_NAMES else 0
        col = CLASS_COLORS[ci % len(CLASS_COLORS)]
        idxs_h = [it["idx"] for it in items]
        pts = engine.embeddings_2d[idxs_h]
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
            v = np.append(hull.vertices, hull.vertices[0])
            hx, hy = pts[v, 0].tolist(), pts[v, 1].tolist()
        except Exception:
            hx, hy = pts[:, 0].tolist() + [pts[0, 0]], \
                     pts[:, 1].tolist() + [pts[0, 1]]
        sfig.add_trace(go.Scatter(
            x=hx, y=hy,
            mode="lines",
            line=dict(color=col, width=1.2, dash="dot"),
            fill="toself",
            fillcolor=_hex_rgba(col, 0.09),
            showlegend=False,
            hoverinfo="skip",
        ))

    # support diamonds
    for cname, items in sc.items():
        if not items:
            continue
        ci = CLASS_NAMES.index(cname)
        col = CLASS_COLORS[ci % len(CLASS_COLORS)]
        idxs = [it["idx"] for it in items]
        sfig.add_trace(
            go.Scatter(
                x=engine.embeddings_2d[idxs, 0],
                y=engine.embeddings_2d[idxs, 1],
                mode="markers",
                marker=dict(size=10, color=col, symbol="diamond",
                            line=dict(width=1.5, color="white")),
                name=f"{cname} (support)",
                showlegend=False,
                hoverinfo="text",
                text=[f"◆ Support: {cname}<br>"
                      f"weight {it['weight']:.1f}"
                      for it in items],
                customdata=[[idx, "support"] for idx in idxs],
            )
        )

    # prototypes — draggable annotations
    p2d = res["p2d"]
    for i, cname in enumerate(order):
        ci = CLASS_NAMES.index(cname)
        col = CLASS_COLORS[ci % len(CLASS_COLORS)]
        sfig.add_annotation(
            x=float(p2d[i, 0]),
            y=float(p2d[i, 1]),
            text=f"★ {cname}",
            font=dict(size=13, color=col, family=FONT),
            showarrow=False,
            bgcolor="rgba(26,29,23,0.8)",
            bordercolor=col,
            borderwidth=2,
            borderpad=4,
            opacity=0.95,
        )

    # highlight selected query
    if sel_q is not None and sel_q < len(engine.labels):
        sfig.add_trace(
            go.Scatter(
                x=[engine.embeddings_2d[sel_q, 0]],
                y=[engine.embeddings_2d[sel_q, 1]],
                mode="markers",
                marker=dict(size=16, color="rgba(0,0,0,0)",
                            line=dict(width=3, color=ERR_COL)),
                showlegend=False, hoverinfo="skip",
            )
        )

    # ── what-if ghost prototypes ──────────────────────────────────
    whatif_res = None
    if whatif_sc:
        whatif_res = engine.classify(whatif_sc, temp)
        w_p2d, _, w_order = engine.compute_prototypes(whatif_sc)
        for i, cname in enumerate(w_order):
            ci = CLASS_NAMES.index(cname)
            col = CLASS_COLORS[ci % len(CLASS_COLORS)]
            sfig.add_annotation(
                x=float(w_p2d[i, 0]),
                y=float(w_p2d[i, 1]),
                text=f"☆ {cname}",
                font=dict(size=12, color=col, family=FONT),
                showarrow=False,
                bgcolor="rgba(26,29,23,0.5)",
                bordercolor=col,
                borderwidth=1,
                borderpad=3,
                opacity=0.55,
            )

    # ── class legend ──────────────────────────────────────────
    accs = res["accs"]
    legend_items = []
    for cname in order:
        ci = CLASS_NAMES.index(cname)
        col = CLASS_COLORS[ci % len(CLASS_COLORS)]
        acc = accs.get(cname, 0.0)
        legend_items.append(
            html.Div(
                [
                    html.Span("■ ",
                              style={"color": col, "fontSize": "14px"}),
                    html.Span(cname, className="legend-name"),
                    html.Div(
                        html.Div(
                            className="acc-fill",
                            style={"width": f"{acc * 100:.0f}%",
                                   "background": col},
                        ),
                        className="acc-bar",
                    ),
                    html.Span(f"{acc:.0%}", className="acc-pct"),
                ],
                className="legend-row",
            )
        )

    # ── overall stat (with what-if projection) ─────────────────
    if whatif_res:
        delta = whatif_res["overall"] - res["overall"]
        delta_col = OK_COL if delta > 0.005 else (
            ERR_COL if delta < -0.005 else TEXT2)
        stat = html.Div(
            [
                html.Span(f"{res['overall']:.1%}", className="stat-value"),
                html.Span(" → ", className="stat-label"),
                html.Span(f"{whatif_res['overall']:.1%}",
                          className="stat-value",
                          style={"color": delta_col}),
                html.Span(f" ({delta:+.1%})",
                          className="stat-label",
                          style={"color": delta_col}),
            ]
        )
    else:
        stat = html.Div(
            [
                html.Span(f"{res['overall']:.1%}", className="stat-value"),
                html.Span(" acc", className="stat-label"),
            ]
        )

    return sfig, legend_items, stat, list(order)


# ── 2. Annotation drag → reweight support items ──────────────────


@app.callback(
    Output("sc-store", "data", allow_duplicate=True),
    Output("proto-undo-store", "data", allow_duplicate=True),
    Output("proto-redo-store", "data", allow_duplicate=True),
    Input("scatter", "relayoutData"),
    State("sc-store", "data"),
    State("proto-undo-store", "data"),
    State("proto-order-store", "data"),
    prevent_initial_call=True,
)
def handle_annotation_drag(relayout, sc, undo_stack, order):
    if not relayout or not order:
        raise PreventUpdate

    # Parse dragged annotation positions (fires once on mouseup)
    updates = {}  # annotation_index -> {"x": ..., "y": ...}
    for key, val in relayout.items():
        m = re.match(r"annotations\[(\d+)\]\.([xy])", key)
        if m:
            i = int(m.group(1))
            axis = m.group(2)
            if i not in updates:
                updates[i] = {}
            updates[i][axis] = val

    if not updates:
        raise PreventUpdate

    new_sc = dict(sc or {})
    changed = False
    for ann_idx, pos in updates.items():
        if ann_idx >= len(order) or "x" not in pos or "y" not in pos:
            continue
        cname = order[ann_idx]
        new_sc = engine.fit_weights_to_target(cname, new_sc, [pos["x"], pos["y"]])
        changed = True

    if not changed:
        raise PreventUpdate

    undo_stack = list(undo_stack or [])
    undo_stack.append(sc)          # push full sc snapshot for undo
    return new_sc, undo_stack, []  # clear redo on new drag


# ── 3. Click → select query ──────────────────────────────────────


@app.callback(
    Output("sel-query-store",   "data", allow_duplicate=True),
    Output("sel-support-store", "data", allow_duplicate=True),
    Output("class-selector",    "value", allow_duplicate=True),
    Output("mask-edit-store",   "data", allow_duplicate=True),
    Output("suggest-store",     "data", allow_duplicate=True),
    Output("whatif-store",      "data", allow_duplicate=True),
    Input("scatter", "clickData"),
    State("sc-store", "data"),
    prevent_initial_call=True,
)
def scatter_click(click, sc):
    if not click or not click.get("points"):
        return None, None, no_update, None, False, None
    pt = click["points"][0]
    cd = pt.get("customdata")
    if cd is None:
        return None, None, no_update, None, False, None
    # Support diamond: customdata = [idx, "support"]
    if isinstance(cd, list) and len(cd) == 2 and cd[1] == "support":
        idx = int(cd[0])
        cname = next(
            (c for c, items in sc.items()
             if any(it["idx"] == idx for it in items)),
            no_update,
        )
        return None, idx, cname, None, False, None
    # Query point: customdata = idx (int or single-element list)
    idx = int(cd) if not isinstance(cd, list) else (int(cd[0]) if cd else None)
    if idx is not None:
        cname = engine.class_names[engine.labels[idx]]
        return idx, None, cname, None, False, None
    return None, None, no_update, None, False, None


# ── 3b. Clear selection button ──────────────────────────────────


@app.callback(
    Output("sel-query-store", "data", allow_duplicate=True),
    Output("sel-support-store", "data", allow_duplicate=True),
    Input("clear-sel-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_selection(n):
    if not n:
        raise PreventUpdate
    return None, None


# ── 4. Undo button ──────────────────────────────────────────


@app.callback(
    Output("sc-store", "data", allow_duplicate=True),
    Output("proto-undo-store", "data", allow_duplicate=True),
    Output("proto-redo-store", "data", allow_duplicate=True),
    Output("uirev-store", "data", allow_duplicate=True),
    Input("undo-btn", "n_clicks"),
    State("sc-store", "data"),
    State("proto-undo-store", "data"),
    State("proto-redo-store", "data"),
    State("uirev-store", "data"),
    prevent_initial_call=True,
)
def handle_undo(n_clicks, sc, undo_stack, redo_stack, uirev):
    if not n_clicks or not undo_stack:
        raise PreventUpdate
    undo_stack = list(undo_stack)
    redo_stack = list(redo_stack or [])
    redo_stack.append(sc)          # push current sc for redo
    prev_sc = undo_stack.pop()
    return prev_sc, undo_stack, redo_stack, (uirev or 0) + 1


# ── 5. Redo button ───────────────────────────────────────────────


@app.callback(
    Output("sc-store", "data", allow_duplicate=True),
    Output("proto-undo-store", "data", allow_duplicate=True),
    Output("proto-redo-store", "data", allow_duplicate=True),
    Output("uirev-store", "data", allow_duplicate=True),
    Input("redo-btn", "n_clicks"),
    State("sc-store", "data"),
    State("proto-undo-store", "data"),
    State("proto-redo-store", "data"),
    State("uirev-store", "data"),
    prevent_initial_call=True,
)
def handle_redo(n_clicks, sc, undo_stack, redo_stack, uirev):
    if not n_clicks or not redo_stack:
        raise PreventUpdate
    undo_stack = list(undo_stack or [])
    redo_stack = list(redo_stack)
    undo_stack.append(sc)          # push current sc for undo
    next_sc = redo_stack.pop()
    return next_sc, undo_stack, redo_stack, (uirev or 0) + 1


# ── 6. Override list ──────────────────────────────────────────────


@app.callback(
    Output("override-list", "children"),
    Input("sc-store", "data"),
)
def render_overrides(sc):
    """Show classes whose support weights have been adjusted from uniform."""
    sc = sc or {}
    items = []
    for cname in CLASS_NAMES:
        class_items = sc.get(cname, [])
        if not class_items:
            continue
        weights = [it["weight"] for it in class_items]
        deviation = max(weights) - min(weights)
        if deviation < 0.05:
            continue
        ci = CLASS_NAMES.index(cname)
        col = CLASS_COLORS[ci % len(CLASS_COLORS)]
        items.append(
            html.Div(
                [
                    html.Span("■ ", style={"color": col}),
                    html.Span(cname, className="ov-class"),
                    html.Span(f" Δw={deviation:.1f}", className="ov-status"),
                ],
                className="ov-row",
            )
        )
    if not items:
        return html.Div("— uniform weights —", className="hint-text")
    return items


# ── 7. Detail panel (inspector OR support set) ───────────────────


@app.callback(
    Output("detail-panel", "children"),
    Input("sel-query-store", "data"),
    Input("sc-store", "data"),
    Input("temp-slider", "value"),
    Input("class-selector", "value"),
    Input("sel-support-store", "data"),
    Input("gradcam-store", "data"),
    Input("mask-edit-store", "data"),
    Input("suggest-store", "data"),
)
def render_detail_panel(sel_q, sc, temp, sel_class, sel_sup, gradcam_on,
                        mask_edit, suggest_on):
    """Show inspector > suggestions > support set."""

    # ── INSPECTOR MODE ────────────────────────────────────────────
    if sel_q is not None:
        return _render_inspector(sel_q, sc, temp)

    # ── SUGGESTION MODE ───────────────────────────────────────────
    if suggest_on:
        return _render_suggestions(sel_class, sc, temp)

    # ── SUPPORT SET MODE ──────────────────────────────────────────
    return _render_support(sel_class, sc, sel_sup, gradcam_on, temp)


def _render_inspector(sel_q, sc, temp):
    idx = int(sel_q)
    true_class = engine.class_names[engine.labels[idx]]
    info = engine.query_distances(idx, sc, temp)
    is_support = idx in engine.support_indices(sc)
    is_correct = info["true"] == info["pred"]

    header = html.Div(
        [
            html.Label("INSPECTOR", className="ctrl-label"),
            html.Button(
                "✕ CLOSE", id="clear-sel-btn",
                className="btn-small", n_clicks=0,
            ),
        ],
        className="section-header",
    )

    # ── image row: original + contrastive Grad-CAM tile per class ──
    # gradcam_tiled_overlays runs one contrastive CAM per prototype,
    # answering "why this class rather than its nearest competitor?"
    cam_tiles = engine.gradcam_tiled_overlays(idx, sc, size=80, alpha=0.55)
    print(f"DEBUG cam_tiles keys: {list(cam_tiles.keys())}  sc classes: {list(sc.keys()) if sc else 'EMPTY'}")

    img_children = [
        html.Div(
            [
                html.Img(src=engine.image_to_base64(idx, size=80),
                          className="inspector-img"),
                html.Span("original", className="inspector-img-label"),
            ],
            className="inspector-img-cell",
        ),
    ]
    for cname, tile_src in cam_tiles.items():
        is_true = (cname == true_class)
        is_pred = (cname == info.get("pred"))
        if is_true and is_pred:
            label     = f"✓ {cname}"
            label_cls = "inspector-img-label inspector-img-label--correct"
        elif is_true:
            label     = f"✓ {cname}"
            label_cls = "inspector-img-label inspector-img-label--true"
        elif is_pred:
            label     = f"✗ {cname}"
            label_cls = "inspector-img-label inspector-img-label--wrong"
        else:
            label     = cname
            label_cls = "inspector-img-label"
        img_children.append(
            html.Div(
                [
                    html.Img(src=tile_src, className="inspector-img"),
                    html.Span(label, className=label_cls),
                ],
                className="inspector-img-cell",
            )
        )

    top = html.Div(
        [
            html.Div(img_children, className="inspector-img-row"),
            html.Div(
                [
                    html.Div(f"#{idx}", className="inspector-id"),
                    html.Div(
                        [html.Span("TRUE ", className="inspector-dim"),
                         html.Span(true_class)],
                        className="inspector-field",
                    ),
                    html.Div(
                        [html.Span("PRED ", className="inspector-dim"),
                         html.Span(info["pred"])],
                        className="inspector-field",
                    ),
                    html.Div(
                        "CORRECT" if is_correct else "WRONG",
                        className="inspector-tag " + (
                            "tag-ok" if is_correct else "tag-err"),
                    ),
                ],
                className="inspector-info",
            ),
        ],
        className="inspector-row",
    )

    dist_bars = []
    if info["order"]:
        names = info["order"]
        probs = info["probs"]
        dists = info["dists"]
        for n, p, d in zip(names, probs, dists):
            ci = CLASS_NAMES.index(n)
            col = CLASS_COLORS[ci % len(CLASS_COLORS)]
            dist_bars.append(
                html.Div(
                    [
                        html.Span(n, className="dist-name"),
                        html.Div(
                            html.Div(
                                className="dist-fill",
                                style={"width": f"{p * 100:.0f}%",
                                       "background": col},
                            ),
                            className="dist-bar",
                        ),
                        html.Span(f"{p:.0%}", className="dist-pct"),
                    ],
                    className="dist-row",
                )
            )

    btn = []
    if not is_support:
        btn.append(
            html.Button(
                f"+ ADD TO {true_class.upper()} SUPPORT",
                id="add-support-btn",
                className="btn-accent",
                n_clicks=0,
            )
        )

    return html.Div(
        [header, html.Div([top] + dist_bars + btn,
                          className="inspector-content")],
        className="inspector-panel",
    )


def _render_mask_editor(edit_idx, sc, sel_class, temp):
    """Render the spatial mask editor content for the modal dialog."""

    items = sc.get(sel_class, [])
    item = next((it for it in items if it["idx"] == edit_idx), None)
    if item is None:
        return html.Div("Image not found", className="hint-text")

    fh, fw = engine._fmap_hw
    mask = item.get("mask") or [[1] * fw for _ in range(fh)]
    excluded = sum(1 for r in range(fh) for c in range(fw) if mask[r][c] == 0)
    total = fh * fw

    # ── compute accuracy with and without mask ────────────────────
    base_res = engine.classify(sc, temp)
    base_acc = base_res["overall"]

    # accuracy if mask were cleared (for comparison)
    nomask_sc = {cn: list(v) for cn, v in sc.items()}
    for it in nomask_sc.get(sel_class, []):
        if it["idx"] == edit_idx:
            it.pop("mask", None)
    nomask_res = engine.classify(nomask_sc, temp)
    nomask_acc = nomask_res["overall"]
    mask_delta = base_acc - nomask_acc

    # ── images: original | Grad-CAM | mask preview ────────────────
    img_size = 256

    original_src = engine.image_to_base64(edit_idx, size=img_size)
    gradcam_src = engine.gradcam_overlay_base64(
        edit_idx, sc, target_class=sel_class,
        size=img_size, alpha=0.55,
    )

    cell_w = img_size // fw
    cell_h = img_size // fh

    grid_cells = []
    for r in range(fh):
        for c in range(fw):
            active = mask[r][c] == 1
            grid_cells.append(
                html.Div(
                    id={"type": "mask-cell", "row": r, "col": c},
                    n_clicks=0,
                    className="mask-cell" + ("" if active
                                             else " mask-cell--excluded"),
                    style={
                        "left": f"{c * cell_w}px",
                        "top": f"{r * cell_h}px",
                        "width": f"{cell_w}px",
                        "height": f"{cell_h}px",
                    },
                )
            )

    # ── header ────────────────────────────────────────────────────
    ci = CLASS_NAMES.index(sel_class) if sel_class in CLASS_NAMES else 0
    col = CLASS_COLORS[ci % len(CLASS_COLORS)]

    if mask_delta > 0.005:
        delta_col, delta_label = OK_COL, "mask helps"
    elif mask_delta < -0.005:
        delta_col, delta_label = ERR_COL, "mask hurts"
    else:
        delta_col, delta_label = TEXT2, "mask neutral"

    header = html.Div(
        [
            html.Div(
                [
                    html.Span("MASK EDITOR", className="modal-title"),
                    html.Span(f"  #{edit_idx}  ",
                              className="modal-subtitle"),
                    html.Span(sel_class,
                              className="modal-subtitle",
                              style={"color": col}),
                ],
                className="modal-title-row",
            ),
            html.Button(
                "✕", id="mask-done-btn",
                className="modal-close-btn", n_clicks=0,
            ),
        ],
        className="modal-header",
    )

    # ── stats bar ─────────────────────────────────────────────────
    stats = html.Div(
        [
            html.Div(
                [
                    html.Span("CELLS EXCLUDED", className="mask-stat-label"),
                    html.Span(f"{excluded}/{total}",
                              className="mask-stat-value"),
                ],
                className="mask-stat",
            ),
            html.Div(
                [
                    html.Span("ACCURACY", className="mask-stat-label"),
                    html.Span(f"{base_acc:.1%}", className="mask-stat-value"),
                ],
                className="mask-stat",
            ),
            html.Div(
                [
                    html.Span("MASK EFFECT", className="mask-stat-label"),
                    html.Span(f"{mask_delta:+.1%}",
                              className="mask-stat-value",
                              style={"color": delta_col},
                              title=delta_label),
                ],
                className="mask-stat",
            ),
        ],
        className="mask-stats-bar",
    )

    # ── image panels ──────────────────────────────────────────────
    panels = html.Div(
        [
            # left: original sketch
            html.Div(
                [
                    html.Div("ORIGINAL", className="mask-panel-label"),
                    html.Img(
                        src=original_src,
                        className="mask-panel-img",
                        style={"width": f"{img_size}px",
                               "height": f"{img_size}px"},
                    ),
                ],
                className="mask-panel",
            ),
            # center: Grad-CAM + clickable grid
            html.Div(
                [
                    html.Div("GRAD-CAM + MASK", className="mask-panel-label"),
                    html.Div(
                        [
                            html.Img(
                                src=gradcam_src,
                                style={"width": f"{img_size}px",
                                       "height": f"{img_size}px",
                                       "display": "block",
                                       "imageRendering": "pixelated"},
                            ),
                            html.Div(
                                grid_cells,
                                className="mask-grid",
                                style={"width": f"{img_size}px",
                                       "height": f"{img_size}px"},
                            ),
                        ],
                        className="mask-img-wrapper",
                        style={"width": f"{img_size}px",
                               "height": f"{img_size}px"},
                    ),
                ],
                className="mask-panel",
            ),
        ],
        className="mask-panels",
    )

    # ── explanation + controls ──────────────────────────────────
    fmap_px = 64 // fh  # each cell covers this many pixels

    explanation = html.Div(
        [
            html.Div("HOW THIS WORKS", className="mask-explain-title"),
            html.Div(
                f"The {fh}×{fw} grid = the spatial resolution of the last "
                f"conv layer's feature map. Each cell covers a "
                f"{fmap_px}×{fmap_px}px region of the input image.",
                className="mask-explain-text",
            ),
            html.Div([
                html.Span("Opaque, warm colours ", style={"color": "#e8a420"}),
                html.Span(
                    "= regions the model actively used to distinguish this "
                    "class from competitors. Fully transparent areas mean the "
                    "model ignored those pixels. If a warm region covers "
                    "background noise, mask it out (click the grid) to force the "
                    "model to rely on the actual sketch."
                ),
            ], className="mask-explain-text"),
            html.Div([
                html.Span("Red cells ", style={"color": ERR_COL}),
                html.Span(
                    "= excluded. Those spatial positions are zeroed "
                    "in the feature map before global average pooling. "
                    "This changes the 512-d embedding, which shifts "
                    "the class prototype, which changes classification."
                ),
            ], className="mask-explain-text"),
            html.Div([
                html.Span("MASK EFFECT ", style={"color": ACCENT,
                                                  "fontWeight": "700"}),
                html.Span(
                    "shows the accuracy impact of your current mask. "
                    "Positive = the mask is helping the model classify "
                    "better by removing distracting features."
                ),
            ], className="mask-explain-text"),
        ],
        className="mask-explanation",
    )

    controls = html.Div(
        [
            html.Button(
                "CLEAR ALL", id="mask-clear-btn",
                className="btn-small", n_clicks=0,
            ) if excluded > 0 else None,
        ],
        className="mask-modal-controls",
    )

    return html.Div([header, stats, explanation, panels, controls])


def _render_suggestions(sel_class, sc, temp):
    """Render swap and add suggestions for the selected class."""

    header = html.Div(
        [
            html.Label(f"SUGGESTIONS — {sel_class}", className="ctrl-label"),
            html.Div(
                [
                    html.Button(
                        "CLEAR PREVIEW", id="suggest-clear-preview-btn",
                        className="btn-small", n_clicks=0,
                    ),
                    html.Button(
                        "✕ CLOSE", id="suggest-close-btn",
                        className="btn-small", n_clicks=0,
                    ),
                ],
                style={"display": "flex", "gap": "4px"},
            ),
        ],
        className="section-header",
    )

    data = engine.suggest_support_changes(sel_class, sc, temp, max_candidates=5)

    children = []

    # ── swap suggestions ──────────────────────────────────────────
    swaps = data["swaps"]
    if swaps:
        children.append(html.Div("SWAP", className="suggest-section-label"))
        for s in swaps:
            delta = s["acc_delta"]
            delta_col = OK_COL if delta > 0.005 else (
                ERR_COL if delta < -0.005 else TEXT2)

            children.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Img(
                                    src=engine.image_to_base64(
                                        s["remove_idx"], size=32),
                                    className="suggest-thumb",
                                    title=f"remove #{s['remove_idx']}",
                                ),
                                html.Span("→", className="suggest-arrow"),
                                html.Img(
                                    src=engine.image_to_base64(
                                        s["add_idx"], size=32),
                                    className="suggest-thumb",
                                    title=f"add #{s['add_idx']}",
                                ),
                            ],
                            className="suggest-imgs",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    f"{delta:+.1%}",
                                    className="suggest-delta",
                                    style={"color": delta_col},
                                ),
                                html.Span(
                                    f"→ {s['new_acc']:.1%}",
                                    className="suggest-acc",
                                ),
                            ],
                            className="suggest-stats",
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "👁",
                                    id={"type": "preview-swap-btn",
                                        "rm": s["remove_idx"],
                                        "add": s["add_idx"]},
                                    className="btn-small suggest-preview",
                                    n_clicks=0,
                                    title="Preview on scatter",
                                ),
                                html.Button(
                                    "APPLY",
                                    id={"type": "apply-swap-btn",
                                        "rm": s["remove_idx"],
                                        "add": s["add_idx"]},
                                    className="btn-small suggest-apply",
                                    n_clicks=0,
                                ),
                            ],
                            className="suggest-btn-group",
                        ),
                    ],
                    className="suggest-row",
                )
            )

    # ── add suggestions ───────────────────────────────────────────
    adds = data["adds"]
    if adds:
        children.append(html.Div("ADD", className="suggest-section-label"))
        for a in adds:
            delta = a["acc_delta"]
            delta_col = OK_COL if delta > 0.005 else (
                ERR_COL if delta < -0.005 else TEXT2)

            children.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Img(
                                    src=engine.image_to_base64(
                                        a["add_idx"], size=32),
                                    className="suggest-thumb",
                                    title=f"add #{a['add_idx']}",
                                ),
                            ],
                            className="suggest-imgs",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    f"{delta:+.1%}",
                                    className="suggest-delta",
                                    style={"color": delta_col},
                                ),
                                html.Span(
                                    f"→ {a['new_acc']:.1%}",
                                    className="suggest-acc",
                                ),
                            ],
                            className="suggest-stats",
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "👁",
                                    id={"type": "preview-add-btn",
                                        "add": a["add_idx"]},
                                    className="btn-small suggest-preview",
                                    n_clicks=0,
                                    title="Preview on scatter",
                                ),
                                html.Button(
                                    "APPLY",
                                    id={"type": "apply-add-btn",
                                        "add": a["add_idx"]},
                                    className="btn-small suggest-apply",
                                    n_clicks=0,
                                ),
                            ],
                            className="suggest-btn-group",
                        ),
                    ],
                    className="suggest-row",
                )
            )

    if not children:
        children.append(
            html.Div("No improvements found.", className="hint-text"))

    return html.Div(
        [header, html.Div(children, className="suggest-list")],
        className="suggest-panel",
    )


def _render_support(sel_class, sc, sel_sup, gradcam_on, temp):
    header = html.Div(
        [
            html.Label("SUPPORT SET", className="ctrl-label"),
            html.Div(
                [
                    html.Button(
                        "SUGGEST", id="suggest-btn",
                        className="btn-small", n_clicks=0,
                    ),
                    html.Button(
                        "GRAD-CAM", id="gradcam-btn",
                        className="btn-small gradcam-active" if gradcam_on
                                  else "btn-small",
                        n_clicks=0,
                    ),
                ],
                style={"display": "flex", "gap": "4px"},
            ),
        ],
        className="section-header",
    )

    items = sc.get(sel_class, [])
    if not items:
        return html.Div(
            [header, html.Div("No support sketches", className="hint-text")],
        )

    # ── compute diagnostics for all support images ────────────────
    diags = engine.support_diagnostics(sel_class, sc, temp)
    diag_by_idx = {d["idx"]: d for d in diags}

    ci = CLASS_NAMES.index(sel_class) if sel_class in CLASS_NAMES else 0
    col = CLASS_COLORS[ci % len(CLASS_COLORS)]
    weights = [it["weight"] for it in items]
    max_w = max(weights) if weights else 1.0

    children = []
    for item in items:
        idx = item["idx"]
        w = item["weight"]
        bar_pct = f"{w / max_w * 100:.0f}%"
        is_active = (sel_sup == idx)
        item_cls = "support-item support-item--active" if is_active \
            else "support-item"

        if gradcam_on:
            thumb_src = engine.gradcam_overlay_base64(
                idx, sc, target_class=sel_class, size=56, alpha=0.55,
            )
        else:
            thumb_src = engine.image_to_base64(idx)

        # ── diagnostics row ───────────────────────────────────
        d = diag_by_idx.get(idx, {})
        loo = d.get("loo_delta", 0.0)
        cos = d.get("cos_sim", 0.0)
        cdist_val = d.get("competitor_dist", 0.0)
        comp = d.get("competitor_name", "")

        # colour the LOO delta: green if removing hurts (negative),
        # red if removing helps (positive), dim if near zero
        if loo < -0.005:
            loo_col = OK_COL
        elif loo > 0.005:
            loo_col = ERR_COL
        else:
            loo_col = TEXT2

        diag_row = html.Div(
            [
                html.Span(
                    f"LOO {loo:+.1%}",
                    className="diag-chip",
                    style={"color": loo_col},
                    title="Overall accuracy change if this image is removed",
                ),
                html.Span(
                    f"cos {cos:.2f}",
                    className="diag-chip",
                    title="Cosine similarity to class prototype",
                ),
                html.Span(
                    f"→{comp} {cdist_val:.1f}",
                    className="diag-chip",
                    title=f"Euclidean distance to nearest competitor ({comp})",
                ) if comp else None,
            ],
            className="diag-row",
        )

        # ── mask badge ─────────────────────────────────────────
        has_mask = item.get("mask") is not None

        children.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src=thumb_src,
                                className="support-thumb",
                                title=f"sketch #{idx}  "
                                      + ("(Grad-CAM)" if gradcam_on else ""),
                            ),
                            html.Span(f"#{idx}", className="support-idx"),
                            html.Span("M", className="mask-badge",
                                      title="Spatial mask active")
                            if has_mask else None,
                        ],
                        className="support-thumb-col",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("W", className="wt-label"),
                                    dcc.Input(
                                        id={"type": "wt-input", "index": idx},
                                        type="number",
                                        min=0.1, max=3.0, step=0.1,
                                        value=round(w, 1),
                                        debounce=True,
                                        className="wt-input",
                                    ),
                                ],
                                className="wt-row",
                            ),
                            html.Div(
                                html.Div(
                                    className="wt-bar-fill",
                                    style={"width": bar_pct, "background": col},
                                ),
                                className="wt-bar",
                                title=f"relative weight: {w:.1f} / {max_w:.1f}",
                            ),
                            diag_row,
                        ],
                        className="wt-col",
                    ),
                    html.Div(
                        [
                            html.Button(
                                "✎",
                                id={"type": "mask-edit-btn", "index": idx},
                                className="mask-edit-btn",
                                n_clicks=0,
                                title="Edit spatial mask",
                            ),
                            html.Button(
                                "×",
                                id={"type": "rm-btn", "index": idx},
                                className="rm-btn",
                                n_clicks=0,
                            ),
                        ],
                        className="support-btn-col",
                    ),
                ],
                className=item_cls,
                style={"borderLeftColor": col},
            )
        )
    return html.Div(
        [header, html.Div(children, className="support-list")],
        className="support-panel",
    )


# ── 8. Add to support ────────────────────────────────────────────


@app.callback(
    Output("sc-store", "data", allow_duplicate=True),
    Input("add-support-btn", "n_clicks"),
    State("sel-query-store", "data"),
    State("sc-store", "data"),
    prevent_initial_call=True,
)
def add_to_support(n, sel_q, sc):
    if not n or sel_q is None:
        raise PreventUpdate
    idx = int(sel_q)
    cname = engine.class_names[engine.labels[idx]]
    existing = {it["idx"] for it in sc.get(cname, [])}
    if idx in existing:
        raise PreventUpdate
    sc.setdefault(cname, []).append({"idx": idx, "weight": 1.0})
    return sc


# ── 9b. Grad-CAM toggle ─────────────────────────────────────────


@app.callback(
    Output("gradcam-store", "data"),
    Input("gradcam-btn", "n_clicks"),
    State("gradcam-store", "data"),
    prevent_initial_call=True,
)
def toggle_gradcam(n, current):
    if not n:
        raise PreventUpdate
    return not current


# ── 9b-2. Suggest toggle ────────────────────────────────────────


@app.callback(
    Output("suggest-store", "data"),
    Output("whatif-store", "data", allow_duplicate=True),
    Input("suggest-btn", "n_clicks"),
    State("suggest-store", "data"),
    prevent_initial_call=True,
)
def toggle_suggest(n, current):
    if not n:
        raise PreventUpdate
    return (not current), None


# ── 9b-3. Close suggestions ─────────────────────────────────────


@app.callback(
    Output("suggest-store", "data", allow_duplicate=True),
    Output("whatif-store", "data", allow_duplicate=True),
    Input("suggest-close-btn", "n_clicks"),
    prevent_initial_call=True,
)
def close_suggestions(n):
    if not n:
        raise PreventUpdate
    return False, None


# ── 9b-3b. Clear suggestion preview (keep panel open) ───────────


@app.callback(
    Output("whatif-store", "data", allow_duplicate=True),
    Input("suggest-clear-preview-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_suggestion_preview(n):
    if not n:
        raise PreventUpdate
    return None


# ── 9b-4. Preview swap suggestion ────────────────────────────────


@app.callback(
    Output("whatif-store", "data", allow_duplicate=True),
    Input({"type": "preview-swap-btn", "rm": ALL, "add": ALL}, "n_clicks"),
    State({"type": "preview-swap-btn", "rm": ALL, "add": ALL}, "id"),
    State("class-selector", "value"),
    State("sc-store", "data"),
    State("whatif-store", "data"),
    prevent_initial_call=True,
)
def preview_swap(clicks, ids, sel_class, sc, current_whatif):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trig = ctx.triggered[0]
    trig_val = trig.get("value")
    if trig_val is None or trig_val < 1:
        raise PreventUpdate
    try:
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        rm_idx = int(triggered["rm"])
        add_idx = int(triggered["add"])
    except Exception:
        raise PreventUpdate

    trial = {c: list(v) for c, v in sc.items()}
    trial[sel_class] = [
        it if it["idx"] != rm_idx else {"idx": add_idx, "weight": 1.0}
        for it in trial.get(sel_class, [])
    ]
    if current_whatif == trial:
        return None
    return trial


# ── 9b-5. Preview add suggestion ─────────────────────────────────


@app.callback(
    Output("whatif-store", "data", allow_duplicate=True),
    Input({"type": "preview-add-btn", "add": ALL}, "n_clicks"),
    State({"type": "preview-add-btn", "add": ALL}, "id"),
    State("class-selector", "value"),
    State("sc-store", "data"),
    State("whatif-store", "data"),
    prevent_initial_call=True,
)
def preview_add(clicks, ids, sel_class, sc, current_whatif):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trig = ctx.triggered[0]
    trig_val = trig.get("value")
    if trig_val is None or trig_val < 1:
        raise PreventUpdate
    try:
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        add_idx = int(triggered["add"])
    except Exception:
        raise PreventUpdate

    trial = {c: list(v) for c, v in sc.items()}
    items = list(trial.get(sel_class, []))
    existing = {it["idx"] for it in items}
    if add_idx not in existing:
        items.append({"idx": add_idx, "weight": 1.0})
    trial[sel_class] = items
    if current_whatif == trial:
        return None
    return trial


# ── 9b-6. Apply swap suggestion ──────────────────────────────────


@app.callback(
    Output("sc-store", "data", allow_duplicate=True),
    Output("suggest-store", "data", allow_duplicate=True),
    Output("whatif-store", "data", allow_duplicate=True),
    Input({"type": "apply-swap-btn", "rm": ALL, "add": ALL}, "n_clicks"),
    State({"type": "apply-swap-btn", "rm": ALL, "add": ALL}, "id"),
    State("class-selector", "value"),
    State("sc-store", "data"),
    prevent_initial_call=True,
)
def apply_swap(clicks, ids, sel_class, sc):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    try:
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        rm_idx = triggered["rm"]
        add_idx = triggered["add"]
    except Exception:
        raise PreventUpdate

    items = sc.get(sel_class, [])
    sc[sel_class] = [
        it if it["idx"] != rm_idx else {"idx": add_idx, "weight": 1.0}
        for it in items
    ]
    return sc, False, None


# ── 9b-7. Apply add suggestion ───────────────────────────────────


@app.callback(
    Output("sc-store", "data", allow_duplicate=True),
    Output("suggest-store", "data", allow_duplicate=True),
    Output("whatif-store", "data", allow_duplicate=True),
    Input({"type": "apply-add-btn", "add": ALL}, "n_clicks"),
    State({"type": "apply-add-btn", "add": ALL}, "id"),
    State("class-selector", "value"),
    State("sc-store", "data"),
    prevent_initial_call=True,
)
def apply_add(clicks, ids, sel_class, sc):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    try:
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        add_idx = triggered["add"]
    except Exception:
        raise PreventUpdate

    items = sc.get(sel_class, [])
    existing = {it["idx"] for it in items}
    if add_idx in existing:
        raise PreventUpdate
    items.append({"idx": add_idx, "weight": 1.0})
    sc[sel_class] = items
    return sc, False, None


# ── 9c. Mask editor modal ─────────────────────────────────────────


@app.callback(
    Output("mask-modal", "style"),
    Output("mask-modal-content", "children"),
    Input("mask-edit-store", "data"),
    Input("sc-store", "data"),
    State("class-selector", "value"),
    State("temp-slider", "value"),
)
def render_mask_modal(edit_idx, sc, sel_class, temp):
    if edit_idx is None:
        return {"display": "none"}, []
    content = _render_mask_editor(int(edit_idx), sc, sel_class, temp)
    return {"display": "flex"}, content


# ── 9c-2. Enter mask editor ──────────────────────────────────────


@app.callback(
    Output("mask-edit-store", "data"),
    Input({"type": "mask-edit-btn", "index": ALL}, "n_clicks"),
    State({"type": "mask-edit-btn", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def enter_mask_editor(clicks, ids):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trig = ctx.triggered[0]
    trig_val = trig.get("value")
    if isinstance(trig_val, list):
        if not any(v is not None and v > 0 for v in trig_val):
            raise PreventUpdate
    elif trig_val is None or trig_val < 1:
        raise PreventUpdate
    # Find which button was actually clicked (n_clicks > 0)
    try:
        triggered = ctx.triggered_id
        if isinstance(triggered, dict):
            return triggered["index"]
    except Exception:
        pass
    raise PreventUpdate


# ── 9d. Mask cell toggle ─────────────────────────────────────────


@app.callback(
    Output("sc-store", "data", allow_duplicate=True),
    Input({"type": "mask-cell", "row": ALL, "col": ALL}, "n_clicks"),
    State({"type": "mask-cell", "row": ALL, "col": ALL}, "id"),
    State("mask-edit-store", "data"),
    State("class-selector", "value"),
    State("sc-store", "data"),
    prevent_initial_call=True,
)
def toggle_mask_cell(clicks, ids, edit_idx, sel_class, sc):
    ctx = callback_context
    if not ctx.triggered or edit_idx is None or not clicks:
        raise PreventUpdate

    # ── robust guard against re-render side effects ───────────────
    # When the modal re-renders (e.g. after clear), all cells are
    # recreated with n_clicks=0.  Dash fires this callback because
    # the component list changed.  We must only act when the user
    # genuinely clicked a cell (n_clicks went from N to N+1).
    triggered_val = ctx.triggered[0].get("value")
    if not triggered_val or triggered_val < 1:
        raise PreventUpdate

    # Extra safety: ensure total clicks > 0 (not all zeros from init)
    if all(c is None or c == 0 for c in clicks):
        raise PreventUpdate

    # Identify which cell was clicked
    try:
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        r, c = triggered["row"], triggered["col"]
    except Exception:
        raise PreventUpdate

    edit_idx = int(edit_idx)
    fh, fw = engine._fmap_hw

    # Find and update the item's mask
    items = sc.get(sel_class, [])
    for item in items:
        if item["idx"] == edit_idx:
            mask = item.get("mask") or [[1] * fw for _ in range(fh)]
            # Deep copy to ensure Dash detects the change
            mask = [row[:] for row in mask]
            mask[r][c] = 0 if mask[r][c] == 1 else 1
            item["mask"] = mask
            # Invalidate cache for this image
            engine._masked_emb_cache.pop(
                engine._mask_cache_key(edit_idx, mask), None)
            return sc

    raise PreventUpdate


# ── 9e. Clear mask ───────────────────────────────────────────────


@app.callback(
    Output("sc-store", "data", allow_duplicate=True),
    Input("mask-clear-btn", "n_clicks"),
    State("mask-edit-store", "data"),
    State("class-selector", "value"),
    State("sc-store", "data"),
    prevent_initial_call=True,
)
def clear_mask(n, edit_idx, sel_class, sc):
    if not n or n < 1 or edit_idx is None:
        raise PreventUpdate
    edit_idx = int(edit_idx)
    items = sc.get(sel_class, [])
    for item in items:
        if item["idx"] == edit_idx and "mask" in item:
            item.pop("mask")
            engine._masked_emb_cache.clear()
            return sc
    raise PreventUpdate


# ── 9f. Exit mask editor ─────────────────────────────────────────


@app.callback(
    Output("mask-edit-store", "data", allow_duplicate=True),
    Input("mask-done-btn", "n_clicks"),
    prevent_initial_call=True,
)
def exit_mask_editor(n):
    if not n:
        raise PreventUpdate
    return None


# ── 10. Weight change ────────────────────────────────────────────


@app.callback(
    Output("sc-store", "data", allow_duplicate=True),
    Input({"type": "wt-input", "index": ALL}, "value"),
    State({"type": "wt-input", "index": ALL}, "id"),
    State("sc-store", "data"),
    prevent_initial_call=True,
)
def update_weights(values, ids, sc):
    if not values:
        raise PreventUpdate
    changed = False
    for val, id_dict in zip(values, ids):
        idx = id_dict["index"]
        if val is None or val <= 0:
            continue
        for cname, items in sc.items():
            for item in items:
                if item["idx"] == idx and abs(item["weight"] - val) > 0.001:
                    item["weight"] = float(val)
                    changed = True
    if not changed:
        raise PreventUpdate
    return sc


# ── 11. Remove from support ──────────────────────────────────────


@app.callback(
    Output("sc-store", "data", allow_duplicate=True),
    Input({"type": "rm-btn", "index": ALL}, "n_clicks"),
    State({"type": "rm-btn", "index": ALL}, "id"),
    State("sc-store", "data"),
    prevent_initial_call=True,
)
def remove_support(clicks, ids, sc):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trig = ctx.triggered[0]
    if trig["value"] is None or trig["value"] == 0:
        raise PreventUpdate
    prop_id = trig["prop_id"]
    try:
        id_dict = json.loads(prop_id.split(".")[0])
        idx = id_dict["index"]
    except Exception:
        raise PreventUpdate
    for cname, items in sc.items():
        sc[cname] = [it for it in items if it["idx"] != idx]
    return sc


# ── 12. Reset ─────────────────────────────────────────────────────


@app.callback(
    Output("sc-store", "data", allow_duplicate=True),
    Output("proto-ov-store", "data", allow_duplicate=True),
    Output("proto-undo-store", "data", allow_duplicate=True),
    Output("proto-redo-store", "data", allow_duplicate=True),
    Output("sel-query-store", "data", allow_duplicate=True),
    Output("mask-edit-store", "data", allow_duplicate=True),
    Output("suggest-store", "data", allow_duplicate=True),
    Output("whatif-store", "data", allow_duplicate=True),
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_all(n):
    if not n:
        raise PreventUpdate
    engine.invalidate_mask_cache()
    return engine.default_support, {}, [], [], None, None, False, None


# =====================================================================
# Entry
# =====================================================================
server = app.server

if __name__ == "__main__":
    app.run(debug=True, dev_tools_ui=False, port=8050)