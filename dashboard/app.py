"""
ProtoNet Embedding Explorer
Run:  python -m dashboard.app
"""

import hashlib
import json
import re
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from dash import (
    Dash, html, dcc,
    Input, Output, State,
    callback_context, ALL, no_update, Patch,
)
from dash.exceptions import PreventUpdate

from dashboard.engine import AnalyticsEngine, CLASS_COLORS, CLASS_NAMES

# =====================================================================
# Engine initialisation
# =====================================================================
_SESSION_PATH = str(Path(__file__).resolve().parent.parent / 'session.json')

import os

_engine_tmp = AnalyticsEngine()

# Load central session if present
_saved_session = _engine_tmp.load_session(_SESSION_PATH) or {}
_saved_classes = _saved_session.get("classes")
_saved_sc = _saved_session.get("sc", _saved_session.get("default_support", {}))
_saved_colors = _saved_session.get("colors", {})

is_worker = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
is_main_file = __name__ == "__main__"

if not is_main_file or is_worker:
    print("Loading model + QuickDraw data + UMAP …")
    engine = _engine_tmp.init_demo(active_classes=_saved_classes)

    print(f"Ready — {len(engine.images)} sketches, {engine.n_classes} classes")

    if _saved_sc:
        n = len(engine.images)
        valid_session = {}
        for cname, items in _saved_sc.items():
            if cname not in engine.class_names:
                continue
            ci = engine.class_names.index(cname)
            valid_items = [
                it for it in items
                if it.get("idx", n) < n
                and isinstance(it.get("idx"), int) and int(engine.labels[it["idx"]]) == ci
            ]
            if valid_items:
                valid_session[cname] = valid_items
        for cname in engine.class_names:
            if cname not in valid_session:
                valid_session[cname] = engine.default_support.get(cname, [])
        engine.default_support = valid_session
        print(f"Session restored ({len(valid_session)} classes)")
else:
    if _saved_classes:
        _engine_tmp.class_names = list(_saved_classes)
    engine = _engine_tmp

# Discover all available classes
ALL_CLASSES = AnalyticsEngine.available_classes()


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
    """Stable hash explicitly locked to manual view resets and undo/redo."""
    return str(uirev)


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
                html.Div(
                    [
                        html.Div(id="overall-stat", className="header-stat"),
                        html.Button(
                            "SAVE",
                            id="manual-save-btn",
                            className="btn-small",
                            n_clicks=0,
                            title="Save session to disk",
                        ),
                        html.Span(id="save-indicator",
                                  className="save-indicator"),
                    ],
                    style={"display": "flex", "alignItems": "center",
                           "gap": "8px", "flexShrink": "0"},
                ),
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
                        # class configuration (collapsible)
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            id="class-config-label",
                                            className="ctrl-label",
                                            style={"cursor": "pointer",
                                                   "userSelect": "none"},
                                            n_clicks=0,
                                        ),
                                        html.Button("RESET",
                                                     id="reset-btn",
                                                     className="btn-small"),
                                    ],
                                    className="section-header",
                                ),
                                html.Div(
                                    id="class-config-panel",
                                    className="class-config-panel",
                                    style={"display": "none"},
                                ),
                                html.Div(id="class-legend",
                                         className="class-legend"),
                                html.Div(id="changelog-display",
                                         className="changelog-display"),
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
                                             for c in engine.class_names],
                                    value=engine.class_names[0],
                                    clearable=False,
                                    className="class-dropdown",
                                ),
                            ],
                            className="sidebar-section",
                        ),

                        # undo / redo + drag hint
                        html.Div(
                            [
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
                                html.Details(
                                    [
                                        html.Summary(id="history-summary", children="History: No actions"),
                                        html.Ul(id="history-list", className="history-list")
                                    ],
                                    className="history-log-panel",
                                    style={"marginTop": "8px", "fontSize": "11px", "cursor": "pointer", "userSelect": "none"}
                                ),
                                html.Div(
                                    "Drag ★ to reposition prototypes.",
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
                                # draw-panel status elements moved into _render_canvas_tool
                            ],
                            className="sidebar-section support-section",
                        ),
                    ],
                    id="sidebar-container",
                    className="sidebar",
                ),
            ],
            className="body-wrapper",
        ),

        # ── stores ──────────────────────────────────────────────
        dcc.Store(id="sc-store",            data=engine.default_support),
        dcc.Store(id="proto-ov-store",      data={}),
        dcc.Store(id="master-undo-store",   data=[]),
        dcc.Store(id="master-redo-store",   data=[]),
        dcc.Store(id="master-view-id",      data=0),
        dcc.Store(id="proto-order-store",   data=[]),
        dcc.Store(id="sel-query-store",     data=None),
        dcc.Store(id="sel-support-store",   data=None),
        dcc.Store(id="whatif-store",        data=None),
        dcc.Store(id="pending-remove-store",data=None),
        dcc.Store(id="changelog-store",     data=[]),
        dcc.Store(id="active-classes-store",  data=list(engine.class_names)),
        dcc.Store(id="pending-classes-store",  data=list(engine.class_names)),
        dcc.Store(id="color-map-store",        data={}),
        dcc.Store(id="session-saved-store",    data=False),
        dcc.Store(id="uirev-store",            data=0),
        dcc.Store(id="sidebar-mode-store",     data="support"),
        dcc.Store(id="candidate-delta-store",  data={}),
        dcc.Interval(id="delta-interval", interval=1000, n_intervals=0, disabled=True),
        dcc.Store(id="drawing-strokes-store", data=[]),
        dcc.Store(id="drawing-ghost-store", data=None),
        dcc.Store(id="draw-interval-active", data=False),
        dcc.Interval(id="draw-interval", interval=600, n_intervals=0, disabled=True),
        dcc.Store(id="new-class-drawings-store", data=[]),
        dcc.Store(id="new-class-name-store", data=""),
        dcc.Store(id="engine-version-store", data=0),
        # proto drag undo/redo (were referenced by handle_annotation_drag but never declared)
        dcc.Store(id="proto-undo-store",  data=[]),
        dcc.Store(id="proto-redo-store",  data=[]),
        # filtered relay store — clientside callback writes here only when
        # annotation keys are present, eliminating zoom/pan server round-trips
        dcc.Store(id="annotation-relayout-store", data=None),
    ],
    className="root",
)


# =====================================================================
# Clientside callback — filter relayoutData before it hits the server.
# Only annotation drag events contain keys like "annotations[0].x".
# Zoom, pan, and autorange events are swallowed client-side, eliminating
# the round-trip that was causing zoom jitter.
# =====================================================================
app.clientside_callback(
    """
    function(relayoutData) {
        if (!relayoutData) return window.dash_clientside.no_update;
        var hasAnnotation = Object.keys(relayoutData).some(
            function(k) { return k.indexOf('annotations[') === 0; }
        );
        return hasAnnotation ? relayoutData : window.dash_clientside.no_update;
    }
    """,
    Output("annotation-relayout-store", "data"),
    Input("scatter", "relayoutData"),
)

# =====================================================================
# Decision-mesh cache — reused when sc and temp haven't changed.
# Avoids 800×800 cdist + PNG re-encode on selection clicks, ghost
# sketch updates, and color-only changes.
# =====================================================================
_mesh_cache: dict = {"key": None, "result": None}

# Maximum entries retained in any undo stack (prevents unbounded growth)
_MAX_UNDO = 50


def _mesh_cache_key(sc: dict, temp: float) -> str:
    """Stable string key for the decision mesh inputs."""
    return hashlib.md5(
        json.dumps(sc, sort_keys=True, default=str).encode() +
        str(round(temp, 3)).encode()
    ).hexdigest()


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
    Input("whatif-store", "data"),
    Input("color-map-store", "data"),
    Input("drawing-ghost-store", "data"),
    Input({"type": "assign-class-dropdown", "index": ALL}, "value"),
    Input("engine-version-store", "data"),
    State("sel-query-store", "data"),
    State("master-view-id", "data"),
)
def update_scatter(sc, temp, whatif_sc, color_map, drawing_ghost, active_class_list, _engine_ver, sel_q, uirev):
    color_map = color_map or {}
    active_class = active_class_list[0] if active_class_list else None

    res = engine.classify(sc, temp)
    order = res["order"]
    if not order:
        empty = _base_fig(title="No data")
        return empty, "No classes", "", []

    sfig = _base_fig(title="Embedding Space  (UMAP)",
                     uirevision=_uirev_hash(sc, temp, sel_q, uirev))

    # ── Decision mesh — cached by (sc, temp) so zoom/color/ghost updates
    # don't re-run the 800×800 cdist and PNG encode
    global _mesh_cache
    _mk = _mesh_cache_key(sc, temp)
    if _mesh_cache["key"] == _mk:
        cached_mesh = _mesh_cache["result"]
    else:
        cached_mesh = None

    if cached_mesh is None:
        xx, ys, zz_class, zz_alpha = engine.decision_mesh(sc, temp, res=800)
        nc = len(order)
        res_val = zz_class.shape[0]
        img_rgba = np.zeros((res_val, res_val, 4), dtype=np.uint8)
        base_alpha = 115  # ~45% max opacity
        
        def hex_to_rgb(hx):
            hx = hx.lstrip('#')
            return tuple(int(hx[i:i+2], 16) for i in (0, 2, 4))
            
        # Calculate topographical elevation bands
        STEPS = 6
        norm_alpha = (zz_alpha - 1.0/nc) / (1.0 - 1.0/nc + 1e-9)
        norm_alpha = np.clip(norm_alpha, 0, 1) ** 0.8
        zz_band = np.ceil(norm_alpha * STEPS)
        
        # Edge detection for contour lines (np.roll shifts the array to compare neighbors)
        edge_mask = (
            (zz_band != np.roll(zz_band, 1, axis=0)) |
            (zz_band != np.roll(zz_band, -1, axis=0)) |
            (zz_band != np.roll(zz_band, 1, axis=1)) |
            (zz_band != np.roll(zz_band, -1, axis=1)) |
            (zz_class != np.roll(zz_class, 1, axis=0)) |
            (zz_class != np.roll(zz_class, -1, axis=0)) |
            (zz_class != np.roll(zz_class, 1, axis=1)) |
            (zz_class != np.roll(zz_class, -1, axis=1))
        )
        
        # Prevent boundary rendering from wrap-around roll
        edge_mask[0, :] = False
        edge_mask[-1, :] = False
        edge_mask[:, 0] = False
        edge_mask[:, -1] = False

        for i in range(nc):
            col_str = _resolve_color(order[i], list(engine.class_names), color_map)
            try:
                r, g, b = hex_to_rgb(col_str)
            except Exception:
                r, g, b = (128, 128, 128)
            
            mask = (zz_class == i)
            # Use discrete banded alpha
            alpha_mask = zz_band[mask] / STEPS
            
            # Boost opacity heavily on contour edges for solid topographical lines
            final_alpha = alpha_mask * base_alpha
            edge_boost = np.where(edge_mask[mask], 180, final_alpha)
            
            img_rgba[mask, 0] = r
            img_rgba[mask, 1] = g
            img_rgba[mask, 2] = b
            img_rgba[mask, 3] = np.clip(edge_boost, 0, 255).astype(np.uint8)

        # Plotly layout images draw from top-left. Our ys array goes bottom-to-top.
        # Flip vertically to render properly in Cartesian coordinates.
        img_rgba = img_rgba[::-1, :, :]
        
        import io, base64
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(img_rgba)
        
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        img_str = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        cached_mesh = {
            "img_str": img_str,
            "x0": float(xx[0][0]), "y1": float(ys[-1]),
            "sizex": float(xx[0][-1] - xx[0][0]),
            "sizey": float(ys[-1] - ys[0]),
        }
        _mesh_cache["key"] = _mk
        _mesh_cache["result"] = cached_mesh

    if cached_mesh:
        sfig.add_layout_image(
            dict(
                source=cached_mesh["img_str"],
                xref="x", yref="y",
                x=cached_mesh["x0"], y=cached_mesh["y1"],
                sizex=cached_mesh["sizex"],
                sizey=cached_mesh["sizey"],
                sizing="stretch",
                opacity=1,
                layer="below"
            )
        )

    # query points
    qm = engine.query_mask(sc)
    qidxs = np.where(qm)[0]
    name2oi = {n: i for i, n in enumerate(order)}

    for ci, cname in enumerate(engine.class_names):
        if cname not in name2oi:
            continue
        mask = engine.labels[qidxs] == ci
        if not mask.any():
            continue
        pts = qidxs[mask]
        q_positions = np.where(mask)[0]
        correct = res["true_idx"][q_positions] == res["preds"][q_positions]
        col = _resolve_color(cname, list(engine.class_names), color_map)

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
        ci = engine.class_names.index(cname) if cname in engine.class_names else 0
        col = _resolve_color(cname, list(engine.class_names), color_map)
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
        ci = engine.class_names.index(cname)
        col = _resolve_color(cname, list(engine.class_names), color_map)
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

    # ── Drawing Ghost Sketch ───────────────────────────────────────
    if drawing_ghost and "pos2d" in drawing_ghost:
        sfig.add_trace(
            go.Scatter(
                x=[drawing_ghost["pos2d"][0]],
                y=[drawing_ghost["pos2d"][1]],
                mode="markers+text",
                marker=dict(size=11, color="white", symbol="circle-open",
                            line=dict(width=2, dash="dash", color="#ffffff")),
                name="Sketch Position",
                text=["≈ Sketch"],
                textposition="top center",
                textfont=dict(color="white", size=10),
                hoverinfo="skip",
                showlegend=False
            )
        )

    # prototypes — draggable annotations
    p2d = res["p2d"]
    for i, cname in enumerate(order):
        ci = engine.class_names.index(cname)
        col = _resolve_color(cname, list(engine.class_names), color_map)
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

    # ── what-if ghost prototypes ──────────────────────────────────
    whatif_res = None
    if whatif_sc:
        whatif_res = engine.classify(whatif_sc, temp)
        w_p2d, _, w_order = engine.compute_prototypes(whatif_sc)
    elif drawing_ghost and "hd" in drawing_ghost and active_class:
        temp_sc = {c: list(v) for c, v in sc.items()}
        class_items = temp_sc.setdefault(active_class, [])
        class_items.append({
            "idx": "ghost_temp",
            "weight": 1.0,
            "emb_hd": drawing_ghost["hd"],
            "pos2d": drawing_ghost["pos2d"]
        })
        whatif_res = engine.classify(temp_sc, temp)
        w_p2d, _, w_order = engine.compute_prototypes(temp_sc)
        for i, cname in enumerate(w_order):
            ci = engine.class_names.index(cname)
            col = _resolve_color(cname, list(engine.class_names), color_map)
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
        ci = engine.class_names.index(cname)
        col = _resolve_color(cname, list(engine.class_names), color_map)
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


# ── 1b. Selection ring — lightweight Patch, no full figure rebuild ─
# sel-query-store is now only an Input here (removed from update_scatter).
# Uses Patch() to splice just the ring trace onto the existing figure,
# so zoom state, viewport, and the background mesh are untouched.

_SEL_TRACE_ID = "selection-ring"


@app.callback(
    Output("scatter", "figure", allow_duplicate=True),
    Input("sel-query-store", "data"),
    State("scatter", "figure"),
    prevent_initial_call=True,
)
def update_selection_ring(sel_q, current_fig):
    p = Patch()
    # Remove any existing ring traces by rebuilding the data list without them.
    # Patch supports item-level list mutations via p["data"] index assignment.
    # Simpler: append the ring at a known position and overwrite it.
    if current_fig is None:
        raise PreventUpdate

    # Strip old ring traces (identified by hoverinfo="skip" + size=16 marker)
    # We use a Patch list-append approach: always append, and keep only the last.
    # Clearing old rings requires knowing their index; instead we embed a uid.
    # Cleanest approach: use allow_duplicate and rebuild just the ring element
    # by comparing existing traces. Since Patch doesn't support list filtering,
    # we do a targeted no_update when nothing changed and let update_scatter
    # handle ring removal when sc changes (ring naturally disappears since
    # update_scatter no longer adds it and figure is replaced wholesale).
    if sel_q is not None and sel_q < len(engine.labels):
        ring = go.Scatter(
            x=[engine.embeddings_2d[sel_q, 0]],
            y=[engine.embeddings_2d[sel_q, 1]],
            mode="markers",
            marker=dict(size=16, color="rgba(0,0,0,0)",
                        line=dict(width=3, color=ERR_COL)),
            showlegend=False, hoverinfo="skip",
            uid=_SEL_TRACE_ID,
        )
        # Remove any previous ring by filtering existing traces, then append
        existing = current_fig.get("data", [])
        filtered = [t for t in existing if t.get("uid") != _SEL_TRACE_ID]
        p["data"] = filtered + [ring]
    else:
        # Deselect — strip ring trace if present
        existing = current_fig.get("data", [])
        filtered = [t for t in existing if t.get("uid") != _SEL_TRACE_ID]
        if len(filtered) == len(existing):
            raise PreventUpdate  # no ring was present, nothing to do
        p["data"] = filtered
    return p


# ── 2. Annotation drag → reweight support items ──────────────────
# Now listens to annotation-relayout-store (pre-filtered by clientside
# callback) instead of scatter.relayoutData directly.  Zoom and pan
# events never reach this callback.


@app.callback(
    Output("sc-store", "data", allow_duplicate=True),
    Output("proto-undo-store", "data", allow_duplicate=True),
    Output("proto-redo-store", "data", allow_duplicate=True),
    Input("annotation-relayout-store", "data"),
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
    undo_stack.append(sc)
    undo_stack = undo_stack[-_MAX_UNDO:]
    return new_sc, undo_stack, []  # clear redo on new drag


# ── 3. Click → select query ──────────────────────────────────────


@app.callback(
    Output("sel-query-store",   "data", allow_duplicate=True),
    Output("sel-support-store", "data", allow_duplicate=True),
    Output("class-selector",    "value", allow_duplicate=True),
    Output("whatif-store",      "data", allow_duplicate=True),
    Input("scatter", "clickData"),
    State("sc-store", "data"),
    prevent_initial_call=True,
)
def scatter_click(click, sc):
    if not click or not click.get("points"):
        raise PreventUpdate
    pt = click["points"][0]
    cd = pt.get("customdata")
    if cd is None:
        return None, None, no_update, None
    # Support diamond: customdata = [idx, "support"]
    if isinstance(cd, list) and len(cd) == 2 and cd[1] == "support":
        idx = int(cd[0])
        cname = next(
            (c for c, items in sc.items()
             if any(it["idx"] == idx for it in items)),
            no_update,
        )
        return None, idx, cname, None
    # Query point: customdata = idx (int or single-element list)
    idx = int(cd) if not isinstance(cd, list) else (int(cd[0]) if cd else None)
    if idx is not None:
        cname = engine.class_names[engine.labels[idx]]
        return idx, None, cname, None
    return None, None, no_update, None


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
    Output("active-classes-store", "data", allow_duplicate=True),
    Output("color-map-store", "data", allow_duplicate=True),
    Output("master-undo-store", "data", allow_duplicate=True),
    Output("master-redo-store", "data", allow_duplicate=True),
    Output("master-view-id", "data", allow_duplicate=True),
    Input("undo-btn", "n_clicks"),
    State("sc-store", "data"),
    State("active-classes-store", "data"),
    State("color-map-store", "data"),
    State("master-undo-store", "data"),
    State("master-redo-store", "data"),
    State("master-view-id", "data"),
    prevent_initial_call=True,
)
def handle_undo(n_clicks, sc, classes, colors, undo_stack, redo_stack, view_id):
    if not n_clicks or not undo_stack:
        raise PreventUpdate
    undo_stack = list(undo_stack)
    redo_stack = list(redo_stack or [])
    
    # Store the exact state before we undo
    current_action_label = undo_stack[-1].get("desc", "?")
    redo_stack.append({
        "desc": f"Redo: {current_action_label}",
        "sc": sc, 
        "classes": classes, 
        "colors": colors,
        "excluded": list(getattr(engine, "excluded_indices", set()))
    })
    
    prev_state = undo_stack.pop()
    if "excluded" in prev_state:
        engine.excluded_indices = set(prev_state["excluded"])
        
    return prev_state.get("sc", {}), prev_state.get("classes", []), prev_state.get("colors", {}), undo_stack, redo_stack, (view_id or 0) + 1


# ── 5. Redo button ───────────────────────────────────────────────


@app.callback(
    Output("sc-store", "data", allow_duplicate=True),
    Output("active-classes-store", "data", allow_duplicate=True),
    Output("color-map-store", "data", allow_duplicate=True),
    Output("master-undo-store", "data", allow_duplicate=True),
    Output("master-redo-store", "data", allow_duplicate=True),
    Output("master-view-id", "data", allow_duplicate=True),
    Input("redo-btn", "n_clicks"),
    State("sc-store", "data"),
    State("active-classes-store", "data"),
    State("color-map-store", "data"),
    State("master-undo-store", "data"),
    State("master-redo-store", "data"),
    State("master-view-id", "data"),
    prevent_initial_call=True,
)
def handle_redo(n_clicks, sc, classes, colors, undo_stack, redo_stack, view_id):
    if not n_clicks or not redo_stack:
        raise PreventUpdate
    undo_stack = list(undo_stack or [])
    redo_stack = list(redo_stack)
    
    next_state = redo_stack.pop()
    
    current_action_label = next_state.get("desc", "?").replace("Redo: ", "")
    undo_stack.append({
        "desc": current_action_label,
        "sc": sc, 
        "classes": classes, 
        "colors": colors,
        "excluded": list(getattr(engine, "excluded_indices", set()))
    })
    undo_stack = undo_stack[-_MAX_UNDO:]
    
    if "excluded" in next_state:
        engine.excluded_indices = set(next_state["excluded"])
        
    return next_state.get("sc", {}), next_state.get("classes", []), next_state.get("colors", {}), undo_stack, redo_stack, (view_id or 0) + 1


@app.callback(
    Output("history-summary", "children"),
    Output("history-list", "children"),
    Input("master-undo-store", "data"),
)
def update_history_ui(undo_stack):
    undo_stack = undo_stack or []
    if not undo_stack:
        return "History: Initial State", [html.Li("Clean session (actions will appear here)")]
    
    # Latest action logic
    latest_action = undo_stack[-1].get("desc", "Unknown action")
    if latest_action.startswith("Redo: "):
        latest_action = latest_action[6:]
    
    list_items = []
    # Reverse the stack to show newest history at the top
    for i, state in enumerate(reversed(undo_stack)):
        desc = state.get("desc", f"Action {len(undo_stack) - i}")
        list_items.append(html.Li(desc))
        
    return f"History: {latest_action}", list_items


# ── 7. Detail panel (inspector OR support set) ───────────────────


@app.callback(
    Output("detail-panel", "children"),
    Input("sel-query-store",      "data"),
    Input("sc-store",             "data"),
    Input("temp-slider",          "value"),
    Input("class-selector",       "value"),
    Input("sel-support-store",    "data"),
    Input("whatif-store",         "data"),
    Input("pending-remove-store", "data"),
    Input("sidebar-mode-store", "data"),
    State("new-class-name-store", "data"),
)
def render_detail_panel(sel_q, sc, temp, sel_class, sel_sup,
                        whatif_sc, pending_rm, sidebar_mode, saved_class_name):
    """Show inspector or support set."""
    # Don't rebuild the canvas when sc-store or whatif-store update while the
    # draw panel is open — that would wipe the canvas and reset controls.
    if sidebar_mode == "draw":
        triggered_props = {t["prop_id"] for t in callback_context.triggered}
        draw_safe = {"sc-store.data", "whatif-store.data", "engine-version-store.data"}
        if triggered_props and triggered_props.issubset(draw_safe):
            raise PreventUpdate

    if sel_q is not None:
        return _render_inspector(sel_q, sc, temp, whatif_sc)
    if sidebar_mode == "candidates":
        return _render_candidates(sel_class, sc, temp, whatif_sc)
    elif sidebar_mode == "draw":
        return _render_canvas_tool(sel_class, sc, whatif_sc, saved_class_name)
    return _render_support(sel_class, sc, sel_sup, temp, whatif_sc, pending_rm)


def _render_inspector(sel_q, sc, temp, whatif_sc=None):
    idx = int(sel_q)
    true_class = engine.class_names[engine.labels[idx]]
    info = engine.query_distances(idx, sc, temp)
    is_support = idx in engine.support_indices(sc)
    is_correct = info["true"] == info["pred"]

    # Check if this idx is a pending staged add
    staged_add = False
    if whatif_sc and not is_support:
        added, _ = _diff_sc(sc, whatif_sc)
        staged_add = any(i == idx for _, i in added)

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

    top = html.Div(
        [
            html.Div(
                [
                    html.Img(src=engine.image_to_base64(idx, size=80),
                             className="inspector-img"),
                    html.Span("original", className="inspector-img-label"),
                ],
                className="inspector-img-cell",
            ),
        ],
        className="inspector-img-row",
    )

    meta = html.Div(
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
    )

    dist_bars = []
    if info["order"]:
        for n, p, d in zip(info["order"], info["probs"], info["dists"]):
            ci = engine.class_names.index(n) if n in engine.class_names else 0
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

    # ── GLOBAL 512D INFLUENCERS section ─────────────────────────────────────────
    # Evaluates the Top 5 High-Dimensional neighbors dragging this point
    # across ALL classes dynamically to explain multi-class probability splits.
    influencers = []
    query_hd = engine.embeddings_hd[idx]

    sims = []
    for cname, items in sc.items():
        if not items: continue
        c_ci = engine.class_names.index(cname) if cname in engine.class_names else 0
        c_col = CLASS_COLORS[c_ci % len(CLASS_COLORS)]
        
        for it in items:
            s_hd = engine.embeddings_hd[it["idx"]]
            norm_q = np.linalg.norm(query_hd)
            norm_s = np.linalg.norm(s_hd)
            if norm_q > 1e-9 and norm_s > 1e-9:
                cos = float(np.dot(query_hd, s_hd) / (norm_q * norm_s))
            else:
                cos = 0.0
            sims.append((cos, it["idx"], cname, c_col))

    # Top 3 most similar globally (restricted to 3 to keep UI clean with heatmaps)
    sims.sort(key=lambda x: x[0], reverse=True)
    top_items = sims[:3]

    why_rows = []
    for cos, s_idx, s_cname, s_col in top_items:
        bar_w = f"{max(0, cos) * 100:.0f}%"
        
        # Geometrical Co-Activation Match Rendering
        try:
            hm_q, hm_s = engine.get_co_activation_images(idx, s_idx, size=40)
        except Exception:
            hm_q, hm_s = "", ""
            
        why_rows.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src=engine.image_to_base64(s_idx, size=32),
                                className="why-thumb",
                                title=f"{s_cname} support #{s_idx}",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        html.Div(
                                            className="why-bar-fill",
                                            style={"width": bar_w,
                                                   "background": s_col},
                                        ),
                                        className="why-bar",
                                        title=f"cos sim: {cos:.3f}",
                                    ),
                                    html.Span(
                                        f"{cos:.2f}",
                                        className="why-sim",
                                    ),
                                ],
                                className="why-bar-col",
                            ),
                            html.Button(
                                f"#{s_idx}",
                                id={"type": "view-wrong-btn",
                                    "idx": s_idx,
                                    "cls": s_cname},
                                className="btn-small why-view-btn",
                                n_clicks=0,
                                title=f"go to {s_cname} #{s_idx}",
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center", "gap": "6px", "width": "100%"}
                    ),
                    # Co-Activation Visualization Dual Row
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("Query Match", style={"fontSize": "8px", "color": "#888", "marginBottom": "2px"}),
                                    html.Img(src=hm_q, style={"width": "40px", "height": "40px", "borderRadius": "4px", "border": "1px solid #444"})
                                ],
                                style={"display": "flex", "flexDirection": "column", "alignItems": "center"}
                            ),
                            html.Span("↔", style={"color": s_col, "fontSize": "16px", "fontWeight": "bold", "margin": "0 10px", "opacity": "0.8"}),
                            html.Div(
                                [
                                    html.Span("Support Match", style={"fontSize": "8px", "color": "#888", "marginBottom": "2px"}),
                                    html.Img(src=hm_s, style={"width": "40px", "height": "40px", "borderRadius": "4px", "border": f"1px solid {s_col}"})
                                ],
                                style={"display": "flex", "flexDirection": "column", "alignItems": "center"}
                            )
                        ],
                        style={"display": "flex", "justifyContent": "center", "alignItems": "center", "width": "100%", "marginTop": "6px", "backgroundColor": "rgba(0,0,0,0.15)", "padding": "6px", "borderRadius": "6px"}
                    ) if hm_q else None
                ],
                className="why-row",
                style={"display": "flex", "flexDirection": "column", "marginBottom": "10px", "borderBottom": "1px solid #1a1a1a", "paddingBottom": "8px"}
            )
        )

    why_wrong = [
        html.Div(
            [
                html.Div(
                    [
                        html.Span(
                            "TOP 512D INFLUENCERS",
                            className="why-header",
                        ),
                        html.Span(
                            "support cos sim →",
                            className="why-sub",
                        ),
                    ],
                    className="why-title-row",
                ),
                html.Div(why_rows, className="why-list"),
            ],
            className="why-section",
        )
    ] if why_rows else []

    # Bottom action area
    ci = engine.class_names.index(true_class) if true_class in engine.class_names else 0
    class_col = CLASS_COLORS[ci % len(CLASS_COLORS)]

    if staged_add:
        try:
            cur_acc  = engine.classify(sc, temp)["overall"]
            proj_acc = engine.classify(whatif_sc, temp)["overall"]
            delta    = proj_acc - cur_acc
        except Exception:
            delta = 0.0
        delta_col = OK_COL if delta > 0.005 else (
            ERR_COL if delta < -0.005 else TEXT2)

        action = html.Div(
            [
                html.Span(
                    f"add to {true_class}",
                    style={"color": class_col, "fontWeight": "700",
                           "fontSize": "11px"},
                ),
                html.Span(
                    f" {delta:+.1%}",
                    style={"color": delta_col, "fontSize": "11px",
                           "fontFamily": FONT},
                ),
                html.Button(
                    "✓",
                    id="confirm-add-btn",
                    className="inline-confirm",
                    n_clicks=0,
                    title="Confirm add",
                ),
                html.Button(
                    "✕",
                    id="cancel-add-btn",
                    className="inline-cancel",
                    n_clicks=0,
                    title="Cancel",
                ),
            ],
            className="inline-action-row",
        )
    elif not is_support:
        action = html.Div([
            html.Button(
                f"+ ADD SUPPORT",
                id="add-support-btn",
                className="btn-accent",
                n_clicks=0,
                style={"flex": "1", "fontSize": "10px", "padding": "6px 2px"},
                title=f"Add to {true_class} support",
            ),
            html.Button(
                "❌ REPLACE",
                id="replace-image-btn",
                className="btn-small",
                n_clicks=0,
                style={"flex": "1", "fontSize": "10px", "background": "#c84040", "color": "white", "padding": "6px 2px"},
                title="Exclude this image and load a replacement"
            )
        ], style={"display": "flex", "gap": "8px", "width": "100%", "marginTop": "8px"})
    else:
        action = None

    body = [top, meta] + dist_bars + why_wrong
    if action:
        body.append(action)

    return html.Div(
        [header, html.Div(body, className="inspector-content")],
        className="inspector-panel",
    )


def _render_candidates(sel_class, sc, temp, whatif_sc=None):
    cands = engine.class_images_pool(sel_class, sc)
    header = html.Div(
        [
            html.Button(
                "← BACK",
                id={"type": "sidebar-nav", "id": "support"},
                className="btn-small",
                style={"marginRight": "8px"},
                n_clicks=0,
            ),
            html.Label(f"{sel_class.upper()} GRAPH SUPPORT", className="ctrl-label"),
        ],
        className="section-header",
    )
    if not cands:
        return html.Div([header, html.Div("No images found.", className="hint-text")])

    sc_idxs = {it["idx"] for it in sc.get(sel_class, [])}
    whatif_sc = whatif_sc or sc
    whatif_idxs = {it["idx"] for it in whatif_sc.get(sel_class, [])}

    staged_add = [i for i in whatif_idxs if i not in sc_idxs]
    staged_rm  = [i for i in sc_idxs if i not in whatif_idxs]
    is_dirty = len(staged_add) > 0 or len(staged_rm) > 0

    footer_content = []
    if is_dirty:
        txts = []
        if staged_add: txts.append(f"Add {len(staged_add)}")
        if staged_rm:  txts.append(f"Remove {len(staged_rm)}")
        footer_content = [
            html.Div(", ".join(txts), className="hint-text", style={"color": "#50a050", "fontWeight": "bold"}),
            html.Button(
                "APPLY CHANGES",
                id="apply-graph-btn",
                className="btn-accent",
                style={"marginTop": "8px", "padding": "6px"},
                n_clicks=0
            )
        ]
    else:
        footer_content = [html.Div("Click points to toggle support selection.", className="hint-text")]

    footer = html.Div(
        footer_content,
        id="candidate-footer",
        style={"marginTop": "auto", "borderTop": "1px solid #3a4235", "padding": "8px", "minHeight": "60px"}
    )

    return html.Div(
        [
            header,
            dcc.Graph(
                id="candidate-scatter",
                config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]},
                style={"height": "380px", "marginTop": "8px"},
            ),
            footer
        ],
        className="candidates-panel",
        style={"display": "flex", "flexDirection": "column", "height": "460px"}
    )


def _render_canvas_tool(sel_class, sc, whatif_sc=None, saved_class_name=""):
    use_sc = whatif_sc if whatif_sc else sc
    header = html.Div(
        [
            html.Button(
                "← BACK",
                id={"type": "sidebar-nav", "id": "support"},
                className="btn-small",
                style={"marginRight": "8px"},
                n_clicks=0,
            ),
            html.Label("DRAW SKETCH", className="ctrl-label"),
        ],
        className="section-header",
    )
    
    canvas_container = html.Div(
        [
            html.Canvas(
                id="draw-canvas",
                width=240, height=240,
                style={"background": "#000000", "border": "1px solid #3a4235", "cursor": "crosshair"}
            ),
            html.Div(id="draw-live-confidence",
                     className="hint-text",
                     style={"textAlign": "center", "marginTop": "6px", "color": "#aaaaaa"}),
        ],
        style={"display": "flex", "flexDirection": "column", "alignItems": "center", "marginTop": "12px"}
    )
    
    controls = html.Div([
        html.Button("CLEAR", id="clear-canvas-btn", className="btn-small", n_clicks=0),
        html.Button("UNDO", id="undo-canvas-btn", className="btn-small", n_clicks=0, style={"marginLeft": "4px"}),
        html.Button("REDO", id="redo-canvas-btn", className="btn-small", n_clicks=0, style={"marginLeft": "4px"}),
        html.Div(style={"flex": "1"}),
        html.Button("BRUSH", id="brush-tool-btn", className="btn-small btn-accent", n_clicks=0),
        html.Button("ERASER", id="eraser-tool-btn", className="btn-small", n_clicks=0, style={"marginLeft": "4px"}),
    ], style={"display": "flex", "gap": "6px", "alignItems": "center", "marginTop": "8px"})
    
    new_class_controls = html.Div([
        dcc.Input(id="new-class-name", value=saved_class_name, placeholder="Enter class name...", debounce=True, style={"width": "100%", "marginTop": "8px", "background": "#2a2a2a", "color": "white", "border": "1px solid #444", "padding": "4px", "borderRadius": "4px"}),
        html.Div("0 of 2 drawn", id="new-class-progress", className="hint-text", style={"marginTop": "4px", "color": "#ffaa00", "textAlign": "center"}),
        html.Button("CREATE CLASS", id="create-class-btn", className="btn-accent", style={"width": "100%", "marginTop": "8px", "padding": "6px"}, disabled=True, n_clicks=0),
    ], id="new-class-controls", style={"display": "none"})

    assign_row = html.Div([
        html.Label("Assign to:", className="ctrl-label", style={"marginTop": "12px"}),
        dcc.Dropdown(
            id={"type": "assign-class-dropdown", "index": "main"},
            options=[{"label": f"{c.upper()} ({len(use_sc.get(c, []))})", "value": c} for c in engine.class_names] + [{"label": "+ Create New Class", "value": "CREATE_NEW"}],
            value=sel_class,
            className="class-dropdown",
            clearable=False
        ),
        new_class_controls,
        html.Button("STAGE ADD", id="stage-draw-btn", className="btn-accent", style={"width": "100%", "marginTop": "12px"}, n_clicks=0),
        html.Button(
            "ADD TO DATASET",
            id="add-query-btn",
            className="btn-small",
            style={"width": "100%", "marginTop": "6px", "display": "none" if sel_class == "CREATE_NEW" else "block"},
            n_clicks=0,
            title="Add this drawing as a query/test point (not support)"
        ),
    ], style={"marginTop": "auto"})

    return html.Div(
        [header, canvas_container, controls, assign_row],
        className="draw-panel",
        style={"display": "flex", "flexDirection": "column", "height": "460px", "padding": "8px"}
    )


def _render_support(sel_class, sc, sel_sup, temp,
                    whatif_sc=None, pending_rm=None):
    header = html.Div(
        [
            html.Label("SUPPORT SET", className="ctrl-label"),
            html.Button(
                "CANDIDATES",
                id={"type": "sidebar-nav", "id": "candidates"},
                className="btn-accent",
                style={"marginLeft": "auto", "fontSize": "11px", "padding": "4px 8px"},
                n_clicks=0,
            ),
            html.Button(
                "DRAW",
                id={"type": "sidebar-nav", "id": "draw"},
                className="btn-accent",
                style={"marginLeft": "8px", "fontSize": "11px", "padding": "4px 8px"},
                n_clicks=0,
            ),
        ],
        className="section-header",
    )

    items = list(sc.get(sel_class, []))
    print(f"\n[_render_support] Class: {sel_class}")
    print(f"[_render_support] Base items count: {len(items)}")
    if whatif_sc:
        staged_items = whatif_sc.get(sel_class, [])
        print(f"[_render_support] Staged items for {sel_class}: {[it.get('idx') for it in staged_items]}")
        drawn_items = [it for it in staged_items if isinstance(it.get("idx"), str) and it["idx"].startswith("drawn_")]
        print(f"[_render_support] Appending {len(drawn_items)} drawn items")
        items.extend(drawn_items)

    if not items:
        return html.Div(
            [header, html.Div("No support sketches.", className="hint-text")],
        )

    diags = engine.support_diagnostics(sel_class, sc, temp)
    diag_by_idx = {d["idx"]: d for d in diags}
    ci = engine.class_names.index(sel_class) if sel_class in engine.class_names else 0
    col = CLASS_COLORS[ci % len(CLASS_COLORS)]
    weights = [it["weight"] for it in items]
    max_w = max(weights) if weights else 1.0

    # Staged add for this class
    staged_add_idx = None
    if whatif_sc:
        added, _ = _diff_sc(sc, whatif_sc)
        for cname, aidx in added:
            if cname == sel_class:
                staged_add_idx = aidx
                break

    # Compute whatif accuracy delta once — reused for both remove and add overlays
    _whatif_delta = None
    if whatif_sc and (pending_rm is not None or staged_add_idx is not None):
        try:
            _cur_acc  = engine.classify(sc, temp)["overall"]
            _proj_acc = engine.classify(whatif_sc, temp)["overall"]
            _whatif_delta = _proj_acc - _cur_acc
        except Exception:
            _whatif_delta = 0.0

    # Delta for pending-remove overlay
    remove_delta = _whatif_delta if pending_rm is not None and whatif_sc else None

    children = []

    # Pending-add item at top
    if staged_add_idx is not None:
        add_thumb = engine.image_to_base64(staged_add_idx)
        add_delta = _whatif_delta
        delta_col = OK_COL if (add_delta or 0) > 0.005 else (
            ERR_COL if (add_delta or 0) < -0.005 else TEXT2)
        children.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(src=add_thumb, className="support-thumb",
                                     title=f"sketch #{staged_add_idx}"),
                            html.Span(f"#{staged_add_idx}", className="support-idx"),
                        ],
                        className="support-thumb-col",
                    ),
                    html.Div(
                        [
                            html.Span("PENDING ADD", className="pending-label"),
                            html.Span(
                                f"{add_delta:+.1%}" if add_delta is not None else "…",
                                style={"color": delta_col, "fontSize": "10px",
                                       "fontFamily": FONT, "fontWeight": "700"},
                            ),
                        ],
                        className="pending-meta",
                    ),
                    html.Div(
                        [
                            html.Button("✓", id="confirm-add-btn",
                                        className="inline-confirm",
                                        n_clicks=0, title="Confirm add"),
                            html.Button("✕", id="cancel-add-btn",
                                        className="inline-cancel",
                                        n_clicks=0, title="Cancel"),
                        ],
                        className="inline-btn-col",
                    ),
                ],
                className="support-item support-item--pending-add",
                style={"borderLeftColor": col},
            )
        )

    for item in items:
        idx = item["idx"]
        w = item["weight"]
        is_drawn = isinstance(idx, str) and idx.startswith("drawn_")
        thumb_src = item.get("base64") if is_drawn else engine.image_to_base64(idx)
        display_idx = "Sketch" if is_drawn else f"#{idx}"
        bar_pct = f"{w / max_w * 100:.0f}%"
        is_active = (sel_sup == idx)
        is_pending_rm = (pending_rm == idx)

        d = diag_by_idx.get(idx, {})
        loo = d.get("loo_delta", 0.0)
        cos = d.get("cos_sim", 0.0)
        comp = d.get("competitor_name", "")
        cdist_val = d.get("competitor_dist", 0.0)
        loo_col = OK_COL if loo < -0.005 else (ERR_COL if loo > 0.005 else TEXT2)

        diag_row = html.Div(
            [
                html.Span(f"LOO {loo:+.1%}", className="diag-chip",
                          style={"color": loo_col},
                          title="Accuracy change if removed"),
                html.Span(f"cos {cos:.2f}", className="diag-chip",
                          title="Cosine sim to prototype"),
                html.Span(f"→{comp} {cdist_val:.1f}", className="diag-chip",
                          title=f"Distance to nearest competitor ({comp})")
                if comp else None,
            ],
            className="diag-row",
        )

        if is_pending_rm:
            delta_col = OK_COL if (remove_delta or 0) > 0.005 else (
                ERR_COL if (remove_delta or 0) < -0.005 else TEXT2)
            children.append(
                html.Div(
                    [
                        html.Div(
                        [html.Img(src=thumb_src,
                                      className="support-thumb",
                                      title=f"sketch {display_idx}"),
                             html.Span(display_idx, className="support-idx")],
                            className="support-thumb-col",
                            style={"opacity": "0.25"},
                        ),
                        html.Div(
                            [html.Div([html.Span("W", className="wt-label"),
                                       html.Span(f"{w:.1f}", className="wt-label")],
                                      className="wt-row"),
                             diag_row],
                            className="wt-col",
                            style={"opacity": "0.25"},
                        ),
                        html.Div(
                            [
                                html.Span(
                                    f"{remove_delta:+.1%}"
                                    if remove_delta is not None else "",
                                    style={"color": delta_col,
                                           "fontSize": "10px",
                                           "fontFamily": FONT,
                                           "fontWeight": "700",
                                           "marginRight": "4px"},
                                ),
                                html.Button("✓",
                                    id={"type": "confirm-rm-btn", "idx": idx},
                                    className="inline-confirm--lg",
                                    n_clicks=0, title="Confirm remove"),
                                html.Button("✕",
                                    id={"type": "cancel-rm-btn", "idx": idx},
                                    className="inline-cancel--lg",
                                    n_clicks=0, title="Cancel"),
                            ],
                            className="pending-rm-overlay",
                        ),
                    ],
                    className="support-item support-item--pending-rm",
                    style={"borderLeftColor": ERR_COL, "position": "relative"},
                )
            )
        else:
            item_cls = ("support-item support-item--active"
                        if is_active else "support-item")
            children.append(
                html.Div(
                    [
                        html.Div(
                            [html.Img(src=thumb_src,
                                      className="support-thumb",
                                      title=f"sketch {display_idx}"),
                             html.Span(display_idx, className="support-idx")],
                            className="support-thumb-col",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [html.Span("W", className="wt-label"),
                                     dcc.Input(id={"type": "wt-input", "index": idx},
                                               type="number", min=0.1, max=3.0,
                                               step=0.1, value=round(w, 1),
                                               debounce=True, className="wt-input")],
                                    className="wt-row",
                                ),
                                html.Div(
                                    html.Div(className="wt-bar-fill",
                                             style={"width": bar_pct, "background": col}),
                                    className="wt-bar",
                                    title=f"relative weight: {w:.1f}/{max_w:.1f}",
                                ),
                                diag_row,
                            ],
                            className="wt-col",
                        ),
                        html.Div(
                            [html.Button("×", id={"type": "rm-btn", "index": idx},
                                         className="rm-btn", n_clicks=0)],
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



# ── 8. Stage: add to support ────────────────────────────────────


def _diff_sc(sc, whatif_sc):
    """Return (added, removed) as lists of (cname, idx) tuples."""
    added, removed = [], []
    all_classes = set(list(sc.keys()) + list(whatif_sc.keys()))
    for cname in all_classes:
        sc_idxs = {it["idx"] for it in sc.get(cname, [])}
        wi_idxs = {it["idx"] for it in whatif_sc.get(cname, [])}
        for idx in wi_idxs - sc_idxs:
            added.append((cname, idx))
        for idx in sc_idxs - wi_idxs:
            removed.append((cname, idx))
    return added, removed


@app.callback(
    Output("whatif-store", "data", allow_duplicate=True),
    Input("add-support-btn", "n_clicks"),
    State("sel-query-store", "data"),
    State("sc-store", "data"),
    prevent_initial_call=True,
)
def stage_add_support(n, sel_q, sc):
    if not n or sel_q is None:
        raise PreventUpdate
    idx = int(sel_q)
    cname = engine.class_names[engine.labels[idx]]
    existing = {it["idx"] for it in sc.get(cname, [])}
    if idx in existing:
        raise PreventUpdate
    trial = {c: list(v) for c, v in sc.items()}
    trial.setdefault(cname, []).append({"idx": idx, "weight": 1.0})
    return trial


@app.callback(
    Output("sel-query-store", "data", allow_duplicate=True),
    Output("uirev-store", "data", allow_duplicate=True),
    Output("master-undo-store", "data", allow_duplicate=True),
    Input("replace-image-btn", "n_clicks"),
    State("sel-query-store", "data"),
    State("uirev-store", "data"),
    State("sc-store", "data"),
    State("active-classes-store", "data"),
    State("color-map-store", "data"),
    State("master-undo-store", "data"),
    prevent_initial_call=True,
)
def replace_image_callback(n, sel_q, uirev, sc, classes, colors, undo_stack):
    if not n or sel_q is None:
        raise PreventUpdate
    idx = int(sel_q)
    cname = engine.class_names[engine.labels[idx]]
    old_excluded = list(getattr(engine, "excluded_indices", set()))
    
    new_idx = engine.exclude_image(idx)
    print(f"DEBUG: replace_image_callback: idx={idx}, new_idx={new_idx}")
    
    undo_stack = list(undo_stack or [])
    undo_stack.append({
        "desc": f"Replaced #{idx} ({cname})",
        "sc": sc, 
        "classes": classes or list(engine.class_names), 
        "colors": colors or {},
        "excluded": old_excluded
    })
    undo_stack = undo_stack[-_MAX_UNDO:]
    
    return new_idx, (uirev or 0) + 1, undo_stack


# ── 8b. Navigate to wrong-class support item ────────────────────


@app.callback(
    Output("class-selector",    "value",  allow_duplicate=True),
    Output("sel-support-store", "data",   allow_duplicate=True),
    Output("sel-query-store",   "data",   allow_duplicate=True),
    Input({"type": "view-wrong-btn", "idx": ALL, "cls": ALL}, "n_clicks"),
    State({"type": "view-wrong-btn", "idx": ALL, "cls": ALL}, "id"),
    prevent_initial_call=True,
)
def navigate_to_wrong_support(clicks, ids):
    ctx = callback_context
    if not ctx.triggered or not any(v for v in clicks if v):
        raise PreventUpdate
    try:
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        s_idx = triggered["idx"]
        cls   = triggered["cls"]
    except Exception:
        raise PreventUpdate
    # Switch class selector to the wrong class, highlight the support item,
    # and close the inspector so the support panel becomes visible
    return cls, s_idx, None


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



# ── 11b. Stage: set pending-remove ──────────────────────────────


@app.callback(
    Output("pending-remove-store", "data", allow_duplicate=True),
    Output("whatif-store",         "data", allow_duplicate=True),
    Input({"type": "rm-btn", "index": ALL}, "n_clicks"),
    State({"type": "rm-btn", "index": ALL}, "id"),
    State("sc-store", "data"),
    prevent_initial_call=True,
)
def set_pending_remove(clicks, ids, sc):
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
    # Guard: can't remove the last item from a class
    for cname, items in sc.items():
        if any(it["idx"] == idx for it in items) and len(items) <= 1:
            raise PreventUpdate
    trial = {c: list(v) for c, v in sc.items()}
    for cname in trial:
        trial[cname] = [it for it in trial[cname] if it["idx"] != idx]
    return idx, trial


# ── 13. Confirm remove ───────────────────────────────────────────


def _do_commit(whatif_sc, sc, classes, colors, temp, undo_stack, changelog, view_id, drawn_url=None):
    """Shared commit logic used by both confirm-rm and confirm-add."""
    # Intercept drawn items in added list
    final_whatif = {c: list(v) for c, v in whatif_sc.items()}
    for cname, items in final_whatif.items():
        for it in items:
            idx = it["idx"]
            if isinstance(idx, str) and idx.startswith("drawn"):
                url = it.get("base64") or drawn_url
                if not url or "base64," not in url:
                    continue
                try:
                    header, encoded = url.split("base64,")
                    decoded = __import__('base64').b64decode(encoded)
                    import time
                    from pathlib import Path
                    save_dir = Path("drawings") / cname
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f"{int(time.time())}.png"
                    
                    img = Image.open(__import__('io').BytesIO(decoded)).convert('L')
                    img.save(save_path)
                    
                    img_arr = np.array(img.resize((64, 64)))
                    new_idx = engine.add_custom_image(cname, img_arr, np.array(it["emb_hd"]), np.array(it["pos2d"]))
                    if new_idx != -1:
                        it["idx"] = new_idx
                        it.pop("emb_hd", None)
                        it.pop("pos2d", None)
                except Exception as e:
                    print(f"Error committing drawn image: {e}")

    added, removed = _diff_sc(sc, final_whatif)
    whatif_sc = final_whatif

    parts = ([f"+ {c} #{i}" for c, i in added] +
             [f"− {c} #{i}" for c, i in removed])
    label = "  ".join(parts) if parts else "Modified Support configuration"
    try:
        cur_acc  = engine.classify(sc, temp)["overall"]
        proj_acc = engine.classify(whatif_sc, temp)["overall"]
        delta    = proj_acc - cur_acc
    except Exception:
        proj_acc = delta = 0.0
    entry = {"label": label, "delta": round(delta, 4),
             "acc": round(proj_acc, 4)}
    new_log  = ([entry] + list(changelog or []))[:20]
    new_undo = (list(undo_stack or []) + [{"desc": label, "sc": sc, "classes": classes, "colors": colors}])[-_MAX_UNDO:]
    new_view_id = (view_id or 0) + 1

    # Persist session
    engine.default_support = whatif_sc
    engine.save_session(_SESSION_PATH, {"sc": whatif_sc, "classes": classes, "colors": colors})

    return whatif_sc, new_undo, [], None, new_log, new_view_id



@app.callback(
    Output("sc-store",             "data", allow_duplicate=True),
    Output("master-undo-store",     "data", allow_duplicate=True),
    Output("master-redo-store",     "data", allow_duplicate=True),
    Output("whatif-store",         "data", allow_duplicate=True),
    Output("changelog-store",      "data", allow_duplicate=True),
    Output("master-view-id",          "data", allow_duplicate=True),
    Output("pending-remove-store", "data", allow_duplicate=True),
    Input({"type": "confirm-rm-btn", "idx": ALL}, "n_clicks"),
    State("whatif-store",      "data"),
    State("sc-store",          "data"),
    State("active-classes-store", "data"),
    State("color-map-store",   "data"),
    State("temp-slider",       "value"),
    State("master-undo-store",  "data"),
    State("changelog-store",   "data"),
    State("master-view-id",       "data"),
    prevent_initial_call=True,
)
def confirm_remove(clicks, whatif_sc, sc, classes, colors, temp, undo_stack, changelog, view_id):
    ctx = callback_context
    if not ctx.triggered or not any(v for v in clicks if v):
        raise PreventUpdate
    if not whatif_sc:
        raise PreventUpdate
    out = _do_commit(whatif_sc, sc, classes, colors, temp, undo_stack, changelog, view_id)
    return out[0], out[1], out[2], out[3], out[4], out[5], None


# ── 14. Cancel remove ────────────────────────────────────────────


@app.callback(
    Output("pending-remove-store", "data", allow_duplicate=True),
    Output("whatif-store",         "data", allow_duplicate=True),
    Input({"type": "cancel-rm-btn", "idx": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def cancel_remove(clicks):
    ctx = callback_context
    if not ctx.triggered or not any(v for v in clicks if v):
        raise PreventUpdate
    return None, None


# ── 15. Confirm add ──────────────────────────────────────────────


@app.callback(
    Output("sc-store",         "data", allow_duplicate=True),
    Output("master-undo-store", "data", allow_duplicate=True),
    Output("master-redo-store", "data", allow_duplicate=True),
    Output("whatif-store",     "data", allow_duplicate=True),
    Output("changelog-store",  "data", allow_duplicate=True),
    Output("master-view-id",      "data", allow_duplicate=True),
    Input("confirm-add-btn",   "n_clicks"),
    State("whatif-store",      "data"),
    State("sc-store",          "data"),
    State("active-classes-store", "data"),
    State("color-map-store",   "data"),
    State("temp-slider",       "value"),
    State("master-undo-store",  "data"),
    State("changelog-store",   "data"),
    State("master-view-id",       "data"),
    State("drawing-strokes-store", "data"),
    prevent_initial_call=True,
)
def confirm_add(n, whatif_sc, sc, classes, colors, temp, undo_stack, changelog, view_id, drawn_url):
    if not n or not whatif_sc:
        raise PreventUpdate
    return _do_commit(whatif_sc, sc, classes, colors, temp, undo_stack, changelog, view_id, drawn_url)


# ── 16. Cancel add ───────────────────────────────────────────────


@app.callback(
    Output("whatif-store", "data", allow_duplicate=True),
    Input("cancel-add-btn", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_add(n):
    if not n:
        raise PreventUpdate
    return None


# ── 17. Changelog display ────────────────────────────────────────


@app.callback(
    Output("changelog-display", "children"),
    Input("changelog-store", "data"),
)
def render_changelog(log):
    if not log:
        return []
    items = []
    for entry in log:
        delta = entry["delta"]
        delta_col = OK_COL if delta > 0.005 else (
            ERR_COL if delta < -0.005 else TEXT2)
        items.append(
            html.Div(
                [
                    html.Span(entry["label"], className="log-label"),
                    html.Span(
                        f"{delta:+.1%}",
                        className="log-delta",
                        style={"color": delta_col},
                    ),
                ],
                className="log-row",
            )
        )
    return [
        html.Div("HISTORY", className="log-header"),
        html.Div(items, className="log-list"),
    ]


# ── P4-1. Class config label (shows N/total, click to expand) ────


@app.callback(
    Output("class-config-label", "children"),
    Input("active-classes-store", "data"),
    Input("pending-classes-store", "data"),
)
def render_class_config_label(active, pending):
    active_set  = set(active  or engine.class_names)
    pending_set = set(pending or active or engine.class_names)
    n_active  = len(active_set)
    n_pending = len(pending_set)
    total = len(ALL_CLASSES)
    dirty = pending_set != active_set
    if dirty:
        return f"CLASSES  {n_active}\u2192{n_pending}/{total} \u25cf"
    return f"CLASSES  {n_active}/{total}"


# ── P4-2. Toggle class config panel open/closed ──────────────────


@app.callback(
    Output("class-config-panel", "style"),
    Input("class-config-label", "n_clicks"),
    State("class-config-panel", "style"),
    prevent_initial_call=True,
)
def toggle_class_config_panel(n, current_style):
    if not n:
        raise PreventUpdate
    hidden = current_style.get("display") == "none"
    return {"display": "block"} if hidden else {"display": "none"}


# ── P4-3. Render class config panel content ──────────────────────


COLOR_LABELS = ["red", "orange", "amber", "yel-grn", "lime", "green",
                "spring", "cyan", "sky", "blue", "indigo", "violet",
                "magenta", "pink", "rose"]


def _resolve_color(cname: str, active_list: list, color_map: dict) -> str:
    """Return the hex color for cname, respecting overrides in color_map."""
    if cname in color_map:
        return CLASS_COLORS[color_map[cname] % len(CLASS_COLORS)]
    if cname in active_list:
        idx = active_list.index(cname)
    else:
        idx = sorted(active_list).index(cname) if cname in active_list else 0
    return CLASS_COLORS[idx % len(CLASS_COLORS)]


@app.callback(
    Output("class-config-panel", "children"),
    Input("active-classes-store",  "data"),
    Input("pending-classes-store", "data"),
    Input("color-map-store",       "data"),
)
def render_class_config_panel(active, pending, color_map):
    active_list   = list(active  or engine.class_names)
    pending_list  = list(pending or active_list)
    active_set    = set(active_list)
    pending_set   = set(pending_list)
    color_map     = color_map or {}
    cap           = len(CLASS_COLORS)
    n_pending     = len(pending_set)
    is_dirty      = pending_set != active_set

    # ── Active chips (/currently in UMAP) ──────────────────────────
    active_chips = []
    for cname in ALL_CLASSES:
        if cname not in active_set:
            continue
        staged_rm = cname not in pending_set
        col    = _resolve_color(cname, active_list, color_map)
        r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
        color_idx = color_map.get(cname, active_list.index(cname) % cap)

        chip_cls = "class-chip class-chip--active"
        if staged_rm:
            chip_cls += " class-chip--staged-rm"

        # Color swatch cycle button
        swatch_btn = html.Button(
            "",
            id={"type": "color-swatch-btn", "name": cname},
            className="color-swatch-btn",
            n_clicks=0,
            title=f"Color: {COLOR_LABELS[color_idx % len(COLOR_LABELS)]} — click to cycle",
            style={
                "background": col,
                "borderColor": col,
            },
        )

        chip = html.Div(
            [
                html.Button(
                    ("✕ " if staged_rm else "● ") + cname,
                    id={"type": "class-chip", "name": cname},
                    className=chip_cls,
                    n_clicks=0,
                    title="Click to remove from selection" if not staged_rm else "Click to keep",
                    style={
                        "background": f"rgba({r},{g},{b},0.25)" if not staged_rm else "transparent",
                        "borderColor": col if not staged_rm else "#3a4235",
                        "color": "#ffffff" if not staged_rm else "#7a7860",
                        "borderWidth": "2px",
                        "textDecoration": "line-through" if staged_rm else "none",
                    },
                ),
                swatch_btn,
            ],
            className="chip-with-swatch",
        )
        active_chips.append(chip)

    # ── Available chips (not in UMAP yet) ──────────────────────────
    available_chips = []
    for cname in ALL_CLASSES:
        if cname in active_set:
            continue
        staged_add = cname in pending_set
        at_cap     = n_pending >= cap and not staged_add

        chip_cls = "class-chip"
        if staged_add:
            chip_cls += " class-chip--staged-add"
        elif at_cap:
            chip_cls += " class-chip--disabled"

        chip = html.Button(
            ("+ " if staged_add else "○ ") + cname,
            id={"type": "class-chip", "name": cname},
            className=chip_cls,
            n_clicks=0,
            disabled=at_cap,
            title="Staged to add — click to cancel" if staged_add else
                  ("Cap reached" if at_cap else "Click to stage for adding"),
            style={
                "borderColor": "#50a050" if staged_add else "#3a4235",
                "color": "#50a050" if staged_add else "#7a7860",
                "background": "rgba(80,160,80,0.12)" if staged_add else
                              "rgba(255,255,255,0.03)",
                "borderWidth": "1px",
            },
        )
        available_chips.append(chip)

    # ── APPLY button ───────────────────────────────────────────────
    apply_btn = html.Button(
        "APPLY CHANGES  →",
        id="apply-classes-btn",
        className="apply-classes-btn" + (" apply-classes-btn--active" if is_dirty else ""),
        n_clicks=0,
        disabled=not is_dirty,
        title="Rebuild UMAP with the staged class selection",
    )

    cap_note = html.Div(
        f"{n_pending}/{cap} staged — press APPLY to rebuild UMAP",
        className="hint-text",
        style={"marginTop": "4px"},
    ) if is_dirty else html.Div(
        f"{len(active_set)}/{cap} active — toggle classes then APPLY to update",
        className="hint-text",
        style={"marginTop": "4px"},
    )

    return [
        html.Div("ACTIVE", className="chip-section-label"),
        html.Div(active_chips, className="class-chip-grid"),
        html.Div("AVAILABLE", className="chip-section-label",
                 style={"marginTop": "8px"}),
        html.Div(available_chips, className="class-chip-grid class-chip-grid--available"),
        html.Div(
            [apply_btn],
            className="apply-row",
            style={"marginTop": "8px"},
        ),
        cap_note,
    ]


# ── P4-4. Draft toggle — update pending-classes-store only ────────
#   Chips now only stage changes; APPLY commits them to the engine.


@app.callback(
    Output("pending-classes-store", "data", allow_duplicate=True),
    Input({"type": "class-chip", "name": ALL}, "n_clicks"),
    State({"type": "class-chip", "name": ALL}, "id"),
    State("active-classes-store",  "data"),
    State("pending-classes-store", "data"),
    prevent_initial_call=True,
)
def toggle_chip_draft(clicks, ids, active, pending):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    try:
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        name = triggered["name"]
        trig_val = ctx.triggered[0].get("value")
        if not trig_val or trig_val < 1:
            raise PreventUpdate
    except (KeyError, TypeError):
        raise PreventUpdate

    active_list  = list(active  or engine.class_names)
    pending_list = list(pending or active_list)
    cap = len(CLASS_COLORS)

    if name in pending_list:
        if len(pending_list) <= 1:
            raise PreventUpdate
        pending_list.remove(name)
    else:
        if len(pending_list) >= cap:
            raise PreventUpdate
        pending_list.append(name)

    return pending_list


# ── P4-7. Commit pending → active (APPLY button) ─────────────────


@app.callback(
    Output("active-classes-store",  "data",  allow_duplicate=True),
    Output("pending-classes-store", "data",  allow_duplicate=True),
    Output("sc-store",              "data",  allow_duplicate=True),
    Output("class-selector",        "options", allow_duplicate=True),
    Output("class-selector",        "value",  allow_duplicate=True),
    Output("master-undo-store",     "data", allow_duplicate=True),
    Output("master-redo-store",     "data", allow_duplicate=True),
    Input("apply-classes-btn", "n_clicks"),
    State("active-classes-store",  "data"),
    State("pending-classes-store", "data"),
    State("sc-store",              "data"),
    State("color-map-store",       "data"),
    State("class-selector",        "value"),
    State("master-undo-store",     "data"),
    prevent_initial_call=True,
)
def apply_class_changes(n, active, pending, sc, colors, sel_class, undo_stack):
    if not n:
        raise PreventUpdate
    active_set  = set(active  or engine.class_names)
    pending_set = set(pending or active_set)
    if active_set == pending_set:
        raise PreventUpdate

    cap = len(CLASS_COLORS)
    old_sc  = dict(sc or {})
    sc = dict(sc or {})

    # Push to undo stack before executing changes
    undo_stack = list(undo_stack or [])
    undo_stack.append({
        "desc": "Modified Active Classes Map",
        "sc": old_sc, 
        "classes": active or list(engine.class_names), 
        "colors": colors or {},
        "excluded": list(getattr(engine, "excluded_indices", set()))
    })
    undo_stack = undo_stack[-_MAX_UNDO:]

    # Remove classes no longer wanted
    to_remove = active_set - pending_set
    if to_remove:
        idx_map = engine.remove_classes(list(to_remove))
        for cname in list(sc.keys()):
            updated = []
            for it in sc[cname]:
                if it["idx"] in idx_map:
                    updated.append({**it, "idx": idx_map[it["idx"]]})
            sc[cname] = updated

    # Rebuild sc after removals (indices are synced)
    n_imgs = len(engine.images)
    new_sc = {}
    for cname_r in engine.class_names:
        ci_r = engine.class_names.index(cname_r)
        old_items = sc.get(cname_r, [])
        valid = [it for it in old_items
                 if isinstance(it.get("idx"), int) and it["idx"] < n_imgs
                 and int(engine.labels[it["idx"]]) == ci_r]
        valid += [it for it in old_items if isinstance(it.get("idx"), str)]
        new_sc[cname_r] = valid or engine.default_support.get(cname_r, [])
    sc = new_sc

    # Add newly wanted classes
    to_add = pending_set - active_set
    for name in to_add:
        if len(engine.class_names) >= cap:
            break
        ok = engine.add_class(name)
        if ok:
            sc[name] = engine.default_support.get(name, [])
        else:
            print(f"add_class failed for {name!r}")

    new_active  = list(engine.class_names)
    new_pending = new_active[:]               
    options     = [{"label": c, "value": c} for c in new_active]
    new_sel     = sel_class if sel_class in new_active else (new_active[0] if new_active else None)
    return new_active, new_pending, sc, options, new_sel, undo_stack, []


# ── P4-8. Cycle class color ───────────────────────────────────────


@app.callback(
    Output("color-map-store", "data", allow_duplicate=True),
    Output("master-undo-store", "data", allow_duplicate=True),
    Output("master-redo-store", "data", allow_duplicate=True),
    Input({"type": "color-swatch-btn", "name": ALL}, "n_clicks"),
    State({"type": "color-swatch-btn", "name": ALL}, "id"),
    State("sc-store", "data"),
    State("active-classes-store", "data"),
    State("color-map-store",      "data"),
    State("master-undo-store", "data"),
    prevent_initial_call=True,
)
def cycle_class_color(clicks, ids, sc, active, color_map, undo_stack):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    try:
        triggered = ctx.triggered_id
        if not isinstance(triggered, dict):
            raise PreventUpdate
        name = triggered["name"]
        trig_val = ctx.triggered[0].get("value")
        if not trig_val or trig_val < 1:
            raise PreventUpdate
    except (KeyError, TypeError):
        raise PreventUpdate

    active_list = list(active or engine.class_names)
    old_color_map = dict(color_map or {})
    color_map   = dict(color_map or {})
    cap         = len(CLASS_COLORS)

    # Push to undo stack
    undo_stack = list(undo_stack or [])
    undo_stack.append({"sc": sc, "classes": active_list, "colors": old_color_map})
    undo_stack = undo_stack[-_MAX_UNDO:]

    # Current color index for this class
    cur_idx = color_map.get(
        name,
        active_list.index(name) % cap if name in active_list else 0
    )
    # Advance to next unused color (skip indices already claimed)
    used = {
        color_map.get(c, active_list.index(c) % cap if c in active_list else 0)
        for c in active_list if c != name
    }
    next_idx = (cur_idx + 1) % cap
    for _ in range(cap):
        if next_idx not in used:
            break
        next_idx = (next_idx + 1) % cap

    color_map[name] = next_idx
    return color_map, undo_stack, []


# ── P4-5. Manual save button ─────────────────────────────────────


@app.callback(
    Output("save-indicator", "children", allow_duplicate=True),
    Input("manual-save-btn", "n_clicks"),
    State("sc-store", "data"),
    State("active-classes-store", "data"),
    State("color-map-store", "data"),
    prevent_initial_call=True,
)
def manual_save(n, sc, classes, colors):
    if not n:
        raise PreventUpdate
    app_state = {
        "sc": sc or {},
        "classes": classes or list(engine.class_names),
        "colors": colors or {}
    }
    engine.save_session(_SESSION_PATH, app_state)
    return "● SAVED"


# ── P4-6. Sync class-selector options on startup ──────────────────


@app.callback(
    Output("class-selector", "options", allow_duplicate=True),
    Input("active-classes-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def sync_class_selector_options(active):
    return [{"label": c, "value": c} for c in (active or engine.class_names)]


# ── 12. Reset ─────────────────────────────────────────────────────


@app.callback(
    Output("sc-store",             "data", allow_duplicate=True),
    Output("proto-ov-store",       "data", allow_duplicate=True),
    Output("master-undo-store",     "data", allow_duplicate=True),
    Output("master-redo-store",     "data", allow_duplicate=True),
    Output("sel-query-store",      "data", allow_duplicate=True),
    Output("whatif-store",         "data", allow_duplicate=True),
    Output("changelog-store",      "data", allow_duplicate=True),
    Output("pending-remove-store", "data", allow_duplicate=True),
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_all(n):
    if not n:
        raise PreventUpdate
    return engine.default_support, {}, [], [], None, None, [], None


# ── Phase 5 Callbacks (Candidate Pool) ───────────────────────────

@app.callback(
    Output("sidebar-mode-store", "data", allow_duplicate=True),
    Output("whatif-store", "data", allow_duplicate=True),
    Output("pending-remove-store", "data", allow_duplicate=True),
    Input({"type": "sidebar-nav", "id": ALL}, "n_clicks"),
    State({"type": "sidebar-nav", "id": ALL}, "id"),
    prevent_initial_call=True,
)
def switch_sidebar_mode(clicks, ids):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered_id
    if not isinstance(triggered_id, dict):
        raise PreventUpdate
        
    btn_id = triggered_id.get("id")
    # Guard against initial mounts trigger for ALL pattern match
    if all(v in (None, 0) for v in (clicks or [])):
        raise PreventUpdate


    if btn_id == "candidates":
        return "candidates", None, None
    elif btn_id == "support":
        return "support", None, None
    elif btn_id == "draw":
        return "draw", None, None
    raise PreventUpdate


@app.callback(
    Output("candidate-scatter", "figure"),
    Input("class-selector", "value"),
    Input("sc-store", "data"),
    Input("candidate-delta-store", "data"),
    Input("whatif-store", "data"),
    State("temp-slider", "value"),
)
def render_candidate_scatter(sel_class, sc, delta_cache, whatif_sc, temp):
    cands = engine.class_images_pool(sel_class, sc)
    if not cands:
         fig = go.Figure()
         fig.update_layout(title="No images")
         return fig

    whatif_sc = whatif_sc or sc
    sc_idxs = {it["idx"] for it in sc.get(sel_class, [])}
    whatif_idxs = {it["idx"] for it in whatif_sc.get(sel_class, [])}

    x, y, idxs = [], [], []
    colors, symbols, sizes = [], [], []
    texts = []

    for c in cands:
        idx = c["idx"]
        in_sc = idx in sc_idxs
        in_wt = idx in whatif_idxs

        x.append(c["cos_sim"])
        y.append(c["competitor_dist"])
        idxs.append(idx)

        # Multi-state logic
        if in_sc and in_wt:
            colors.append("#e8a420")
            symbols.append("diamond")
            sizes.append(12)
            state_txt = "Support"
        elif in_sc and not in_wt:
            colors.append("#c84040") 
            symbols.append("diamond")
            sizes.append(12)
            state_txt = "Staged REMOVE"
        elif not in_sc and in_wt:
            colors.append("#50a050")
            symbols.append("circle")
            sizes.append(12)
            state_txt = "Staged ADD"
        else:
            colors.append("#7a7860")
            symbols.append("circle")
            sizes.append(9)
            state_txt = "Candidate"

        d = delta_cache.get(str(idx)) if delta_cache else None
        d_text = f"{d:+.1%}" if d is not None else "computing..."
        texts.append(f"#{idx}<br>State: {state_txt}<br>cos sim: {c['cos_sim']:.3f}<br>comp dist: {c['competitor_dist']:.1f}<br>delta: {d_text}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(size=sizes, color=colors, symbol=symbols, line=dict(width=1, color="white")),
        customdata=idxs,
        text=texts,
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{sel_class.upper()} Candidates",
        xaxis_title="cos sim to prototype →",
        yaxis_title="competitor dist →",
        dragmode="pan",
        margin=dict(l=50, r=20, t=40, b=50),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        uirevision=sel_class,
    )
    fig.update_xaxes(rangemode="normal", showgrid=True, gridcolor="#3a4235")
    fig.update_yaxes(rangemode="normal", showgrid=True, gridcolor="#3a4235")
    return fig


@app.callback(
    Output("candidate-delta-store", "data"),
    Output("delta-interval", "disabled"),
    Input("delta-interval", "n_intervals"),
    Input("sidebar-mode-store", "data"),
    Input("class-selector", "value"),
    State("sc-store", "data"),
    State("temp-slider", "value"),
    State("candidate-delta-store", "data"),
)
def candidate_interval_update(n, mode, sel_class, sc, temp, cache):
    ctx = callback_context
    triggered_ids = [t["prop_id"].split(".")[0] for t in ctx.triggered]

    if mode != "candidates":
        return {}, True

    if "sidebar-mode-store" in triggered_ids or "class-selector" in triggered_ids:
        cache = {}

    cands = engine.class_images_pool(sel_class, sc)
    if not cands:
        return {}, True

    # Only compute deltas for Non-Support images (candidates)
    cands = [c for c in cands if not c.get("is_support")]
    if not cands:
        return cache, True

    cache = dict(cache or {})
    uncomputed = [c for c in cands if str(c["idx"]) not in cache]
    
    if not uncomputed:
        return cache, True

    cand = uncomputed[0]
    idx = cand["idx"]
    delta = engine.candidate_add_delta(sel_class, idx, sc, temp)
    cache[str(idx)] = delta

    return cache, False


@app.callback(
    Output("whatif-store", "data", allow_duplicate=True),
    Input("candidate-scatter", "clickData"),
    State("class-selector", "value"),
    State("sc-store", "data"),
    State("whatif-store", "data"),
    prevent_initial_call=True,
)
def toggle_graph_item(click, sel_class, sc, whatif_sc):
    if not click or not click.get("points"):
        raise PreventUpdate
    pt = click["points"][0]
    idx = pt.get("customdata")
    if idx is None:
        raise PreventUpdate

    sc = sc or {}
    whatif_sc = whatif_sc or sc

    trial = {c: list(v) for c, v in whatif_sc.items()}
    class_items = trial.get(sel_class, [])
    item_idxs = {it["idx"] for it in class_items}

    if idx in item_idxs:
        trial[sel_class] = [it for it in class_items if it["idx"] != idx]
    else:
        trial[sel_class] = class_items + [{"idx": int(idx), "weight": 1.0}]

    return trial


@app.callback(
    Output("sc-store",             "data", allow_duplicate=True),
    Output("master-undo-store",     "data", allow_duplicate=True),
    Output("master-redo-store",     "data", allow_duplicate=True),
    Output("whatif-store",         "data", allow_duplicate=True),
    Output("changelog-store",      "data", allow_duplicate=True),
    Output("master-view-id",          "data", allow_duplicate=True),
    Input("apply-graph-btn", "n_clicks"),
    State("whatif-store", "data"),
    State("sc-store", "data"),
    State("active-classes-store", "data"),
    State("color-map-store", "data"),
    State("temp-slider", "value"),
    State("master-undo-store", "data"),
    State("changelog-store", "data"),
    State("master-view-id", "data"),
    prevent_initial_call=True,
)
def apply_graph_changes(n, whatif_sc, sc, classes, colors, temp, undo_stack, changelog, view_id):
    if not n or not whatif_sc:
        raise PreventUpdate
    return _do_commit(whatif_sc, sc, classes, colors, temp, undo_stack, changelog, view_id)


@app.callback(
    Output("sidebar-container", "className"),
    Input("sidebar-mode-store", "data")
)
def update_sidebar_classname(mode):
    if mode == "candidates" or mode == "draw":
        return "sidebar sidebar--expanded"
    return "sidebar"


app.clientside_callback(
    """
    function(mode) {
        if (mode !== 'draw') return window.dash_clientside.no_update;
        setTimeout(() => {
            const canvas = document.getElementById('draw-canvas');
            if (!canvas) return;

            const ctx = canvas.getContext('2d');
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 6;
            ctx.lineCap = 'round';

            let drawing = false;

            canvas.onmousedown = (e) => {
                drawing = true;
                ctx.beginPath();
                ctx.moveTo(e.offsetX, e.offsetY);
            };
            canvas.onmousemove = (e) => {
                if (drawing) {
                    ctx.lineTo(e.offsetX, e.offsetY);
                    ctx.stroke();
                }
            };
            let undoStack = [];
            let redoStack = [];

            function saveState() {
                undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
                if (undoStack.length > 25) undoStack.shift();
                redoStack = [];
            }

            saveState();

            canvas.onmouseup = () => { drawing = false; saveState(); };
            canvas.onmouseleave = () => { drawing = false; };
            
            const clearBtn = document.getElementById('clear-canvas-btn');
            if (clearBtn) {
                clearBtn.onclick = () => { 
                    ctx.clearRect(0, 0, canvas.width, canvas.height); 
                    saveState(); 
                };
            }

            const undoBtn = document.getElementById('undo-canvas-btn');
            if (undoBtn) {
                undoBtn.onclick = () => {
                    if (undoStack.length > 1) {
                        redoStack.push(undoStack.pop());
                        const prev = undoStack[undoStack.length - 1];
                        ctx.putImageData(prev, 0, 0);
                    }
                };
            }

            const redoBtn = document.getElementById('redo-canvas-btn');
            if (redoBtn) {
                redoBtn.onclick = () => {
                    if (redoStack.length > 0) {
                        const next = redoStack.pop();
                        undoStack.push(next);
                        ctx.putImageData(next, 0, 0);
                    }
                };
            }

            const brushBtn = document.getElementById('brush-tool-btn');
            const eraserBtn = document.getElementById('eraser-tool-btn');
            if (brushBtn && eraserBtn) {
                brushBtn.onclick = () => {
                    ctx.strokeStyle = 'white';
                    brushBtn.classList.add('btn-accent');
                    eraserBtn.classList.remove('btn-accent');
                };
                eraserBtn.onclick = () => {
                    ctx.strokeStyle = '#000000';
                    eraserBtn.classList.add('btn-accent');
                    brushBtn.classList.remove('btn-accent');
                };
            }

            const clearAfterClick = (btnId) => {
                const btn = document.getElementById(btnId);
                if (btn) {
                    btn.addEventListener('click', () => {
                        setTimeout(() => {
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            saveState();
                        }, 50);
                    });
                }
            };
            clearAfterClick('stage-draw-btn');
            clearAfterClick('add-query-btn');

        }, 100);
        return true;
    }
    """,
    Output("draw-interval-active", "data"),
    Input("sidebar-mode-store", "data"),
    prevent_initial_call=False
)

app.clientside_callback(
    """
    function(n) {
        const canvas = document.getElementById('draw-canvas');
        if (!canvas) return null;
        return canvas.toDataURL("image/png");
    }
    """,
    Output("drawing-strokes-store", "data"),
    Input("draw-interval", "n_intervals")
)

@app.callback(
    Output("draw-interval", "disabled"),
    Input("sidebar-mode-store", "data")
)
def toggle_draw_interval(mode):
    return mode != "draw"


import base64
from io import BytesIO

@app.callback(
    Output("drawing-ghost-store", "data"),
    Output("draw-live-confidence", "children"),
    Input("drawing-strokes-store", "data"),
    State("sc-store", "data"),
    State("temp-slider", "value"),
    State({"type": "assign-class-dropdown", "index": ALL}, "value"),
    prevent_initial_call=True
)
def process_draw_strokes(data_url, sc, temp, active_class_list):
    if not data_url or "base64," not in data_url:
        raise PreventUpdate
    
    active_class = active_class_list[0] if active_class_list else None
    try:
        from PIL import Image
        header, encoded = data_url.split("base64,")
        decoded = __import__('base64').b64decode(encoded)
        img = Image.open(__import__('io').BytesIO(decoded)).convert('L')
        img_64 = img.resize((64, 64), Image.BICUBIC)
        
        emb_hd, pos_2d = engine.encode_image(img_64)
        ghost = {
            "hd": emb_hd.tolist(),
            "pos2d": pos_2d.tolist()
        }

        conf_text = "Confidence: N/A"
        if sc and active_class:
            p2d, phd, order = engine.compute_prototypes(sc)
            if len(phd) > 0 and active_class in order:
                from scipy.spatial.distance import cdist
                import numpy as np
                dists = cdist(emb_hd.reshape(1, -1), phd, metric='euclidean')
                logits = -dists / max(temp or 1.0, 0.01)
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                idx_order = order.index(active_class)
                prob = probs[0, idx_order]
                conf_text = f"Confidence: {prob*100:.1f}% ({active_class.upper()})"
                
        return ghost, conf_text
    except Exception:
        raise PreventUpdate

@app.callback(
    Output("new-class-controls", "style"),
    Output("add-query-btn", "style"),
    Input({"type": "assign-class-dropdown", "index": "main"}, "value")
)
def toggle_new_class_controls(assign_class):
    if assign_class == "CREATE_NEW":
        return {"display": "block"}, {"display": "none", "width": "100%", "marginTop": "6px"}
    return {"display": "none"}, {"display": "block", "width": "100%", "marginTop": "6px"}




@app.callback(
    Output("new-class-name-store", "data"),
    Input("new-class-name", "value"),
    prevent_initial_call=True
)
def save_new_class_name(val):
    if val is None:
        raise PreventUpdate
    return val

@app.callback(
    Output("whatif-store", "data", allow_duplicate=True),
    Output("new-class-drawings-store", "data", allow_duplicate=True),
    Output("stage-draw-btn", "children"),
    Output("create-class-btn", "disabled"),
    Output("new-class-progress", "children"),
    Output("sidebar-mode-store", "data", allow_duplicate=True),
    Output("sc-store", "data", allow_duplicate=True),
    Input("stage-draw-btn", "n_clicks"),
    State("drawing-ghost-store", "data"),
    State({"type": "assign-class-dropdown", "index": ALL}, "value"),
    State("sc-store", "data"),
    State("drawing-strokes-store", "data"),
    State("whatif-store", "data"),
    State("new-class-drawings-store", "data"),
    prevent_initial_call=True
)
def stage_drawn_image(n, ghost_data, active_class_list, sc, data_url, whatif_sc, new_drawings):
    if not n or not ghost_data:
        raise PreventUpdate
    
    assign_class = active_class_list[0] if active_class_list else None
    if not assign_class:
        raise PreventUpdate
        
    if assign_class == "CREATE_NEW":
        new_drawings = new_drawings or []
        new_drawings.append(data_url)
        disabled = len(new_drawings) < 2
        progress = f"{len(new_drawings)} of 2 drawn"
        return no_update, new_drawings, "✓ STAGED", disabled, progress, no_update, no_update
        
    # Decode the drawing and register it with the engine immediately so it
    # gets a real integer index — avoids the string-idx crash in _render_support.
    try:
        import io as _io, numpy as _np
        from PIL import Image as _Image
        header, encoded = data_url.split("base64,")
        decoded = __import__('base64').b64decode(encoded)
        img = _Image.open(_io.BytesIO(decoded)).convert('L')
        img_arr = _np.array(img.resize((64, 64)))
        hd_emb = _np.array(ghost_data["hd"])
        pos_2d  = _np.array(ghost_data["pos2d"])
        new_idx = engine.add_custom_image(assign_class, img_arr, hd_emb, pos_2d)
    except Exception as e:
        print(f"Error registering drawn image: {e}")
        raise PreventUpdate

    if new_idx == -1:
        raise PreventUpdate

    sc = sc or {}
    new_sc = {c: list(v) for c, v in sc.items()}
    new_sc.setdefault(assign_class, []).append({
        "idx": new_idx,
        "weight": 1.0,
        "base64": data_url,
    })

    # Write straight to sc-store (no whatif staging needed) and keep sidebar
    # in draw mode without touching sidebar-mode-store so the canvas survives.
    return no_update, no_update, f"✓ ADDED TO {assign_class.upper()}", no_update, no_update, no_update, new_sc


@app.callback(
    Output("engine-version-store", "data", allow_duplicate=True),
    Output("add-query-btn", "children"),
    Input("add-query-btn", "n_clicks"),
    State("drawing-ghost-store", "data"),
    State({"type": "assign-class-dropdown", "index": ALL}, "value"),
    State("drawing-strokes-store", "data"),
    State("engine-version-store", "data"),
    prevent_initial_call=True,
)
def add_query_point(n, ghost_data, active_class_list, data_url, engine_ver):
    """Add a drawn sketch as a query/test point (not a support item).
    It lands on the scatter as a classifiable ○/✕ point for its class,
    without affecting prototypes or sc-store at all."""
    if not n or not ghost_data:
        raise PreventUpdate

    assign_class = active_class_list[0] if active_class_list else None
    if not assign_class or assign_class == "CREATE_NEW":
        raise PreventUpdate

    try:
        import io as _io
        import numpy as _np
        from PIL import Image as _Image
        header, encoded = data_url.split("base64,")
        decoded = __import__('base64').b64decode(encoded)
        img = _Image.open(_io.BytesIO(decoded)).convert('L')
        img_arr = _np.array(img.resize((64, 64)))
        hd_emb = _np.array(ghost_data["hd"])
        pos_2d  = _np.array(ghost_data["pos2d"])
        new_idx = engine.add_custom_image(assign_class, img_arr, hd_emb, pos_2d)
    except Exception as e:
        print(f"Error adding query point: {e}")
        raise PreventUpdate

    if new_idx == -1:
        raise PreventUpdate

    return (engine_ver or 0) + 1, f"✓ Query point added to {assign_class.upper()}"


@app.callback(
    Output("sc-store", "data", allow_duplicate=True),
    Output("new-class-drawings-store", "data", allow_duplicate=True),
    Output("sidebar-mode-store", "data", allow_duplicate=True),
    Output({"type": "assign-class-dropdown", "index": "main"}, "value"),
    Output("active-classes-store",  "data", allow_duplicate=True),
    Output("pending-classes-store", "data", allow_duplicate=True),
    Input("create-class-btn", "n_clicks"),
    State("new-class-name", "value"),
    State("new-class-drawings-store", "data"),
    State("sc-store", "data"),
    prevent_initial_call=True
)
def create_new_class(n_clicks, class_name, drawings, sc):
    if not n_clicks or not class_name or not drawings:
        raise PreventUpdate
        
    class_name = class_name.strip().lower()
    if not class_name:
        raise PreventUpdate
        
    from pathlib import Path
    _DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'quickdraw' / 'test'
    cls_dir = _DATA_DIR / class_name
    cls_dir.mkdir(parents=True, exist_ok=True)
    
    for i, data_url in enumerate(drawings):
        if "base64," not in data_url:
            continue
        header, encoded = data_url.split("base64,")
        decoded = __import__('base64').b64decode(encoded)
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(decoded)).convert('L')
        img.resize((64, 64)).save(cls_dir / f"drawn_{i}.png")
        
    success = engine.add_class(class_name)
    if not success:
        raise PreventUpdate
        
    new_sc = {c: list(v) for c, v in (sc or {}).items()}
    new_sc[class_name] = engine.default_support.get(class_name, [])

    # Notify all class-list consumers so the main dropdown, config label,
    # ACTIVE/AVAILABLE chips, and color swatches all reflect the new class.
    new_active = list(engine.class_names)
    return new_sc, [], "support", class_name, new_active, new_active[:]


# =====================================================================
# Entry
# =====================================================================
server = app.server

if __name__ == "__main__":
    app.run(debug=True, dev_tools_ui=False, port=8050)