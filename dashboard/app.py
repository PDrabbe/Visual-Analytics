"""
ProtoNet Visual Analytics Dashboard

Interactive Dash/Plotly dashboard for diagnosing and refining
a Prototypical Network few-shot sketch classifier.

Run:  python -m dashboard.app
"""

import json
import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, callback_context, MATCH, ALL, no_update
from dash.exceptions import PreventUpdate

from dashboard.engine import AnalyticsEngine, CLASS_COLORS, CLASS_NAMES

# =====================================================================
# Engine initialisation
# =====================================================================
print("Initialising analytics engine (generating demo data + UMAP)...")
engine = AnalyticsEngine().init_demo()
print(f"Ready: {len(engine.images)} sketches, {engine.n_classes} classes")

# =====================================================================
# Plotly template
# =====================================================================
DARK_BG = "#0d1117"
CARD_BG = "#161b22"
BORDER = "#21262d"
TEXT = "#c9d1d9"
TEXT2 = "#8b949e"
ACCENT = "#58a6ff"

PLOT_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=DARK_BG,
    font=dict(color=TEXT, size=11, family="Inter, system-ui, sans-serif"),
    margin=dict(l=40, r=20, t=36, b=36),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
)


def _base_fig(**kw) -> go.Figure:
    layout = {**PLOT_LAYOUT, **kw}
    return go.Figure(layout=layout)


# =====================================================================
# Layout
# =====================================================================
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    assets_folder="assets",
    title="ProtoNet Visual Analytics",
)

app.layout = html.Div(
    [
        # ---- header ----
        html.Div(
            [
                html.Div(
                    [
                        html.H1("ProtoNet Visual Analytics", className="header-title"),
                        html.Span("Few-Shot Sketch Classifier Explorer", className="header-sub"),
                    ],
                    className="header-left",
                ),
                html.Div(id="overall-stat", className="header-stat"),
            ],
            className="header",
        ),
        # ---- body ----
        html.Div(
            [
                # ---- sidebar ----
                html.Div(
                    [
                        # temperature
                        html.Div(
                            [
                                html.Label("Temperature", className="ctrl-label"),
                                dcc.Slider(
                                    id="temp-slider",
                                    min=0.1,
                                    max=5.0,
                                    step=0.1,
                                    value=1.0,
                                    marks={0.1: "0.1", 1: "1", 2: "2", 3: "3", 5: "5"},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                    className="temp-slider",
                                ),
                            ],
                            className="ctrl-block",
                        ),
                        # class selector
                        html.Div(
                            [
                                html.Label("Inspect Class", className="ctrl-label"),
                                dcc.Dropdown(
                                    id="class-selector",
                                    options=[{"label": c, "value": c} for c in CLASS_NAMES],
                                    value=CLASS_NAMES[0],
                                    clearable=False,
                                    className="class-dropdown",
                                ),
                            ],
                            className="ctrl-block",
                        ),
                        # prototype mode
                        html.Div(
                            [
                                dcc.Checklist(
                                    id="move-proto-toggle",
                                    options=[{"label": " Move Prototype Mode", "value": "on"}],
                                    value=[],
                                    className="toggle-check",
                                ),
                                html.Div(
                                    "Click the scatter to reposition the selected class prototype.",
                                    className="hint-text",
                                    id="proto-hint",
                                    style={"display": "none"},
                                ),
                            ],
                            className="ctrl-block",
                        ),
                        # support set
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Support Set", className="ctrl-label"),
                                        html.Button("Reset All", id="reset-btn", className="btn-small"),
                                    ],
                                    className="support-header",
                                ),
                                html.Div(id="support-panel", className="support-panel"),
                            ],
                            className="ctrl-block support-block",
                        ),
                        # sketch inspector
                        html.Div(
                            [
                                html.Label("Sketch Inspector", className="ctrl-label"),
                                html.Div(id="inspector-panel", className="inspector-panel"),
                            ],
                            className="ctrl-block",
                        ),
                    ],
                    className="sidebar",
                ),
                # ---- main grid ----
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [dcc.Graph(id="scatter", config={"scrollZoom": True}, style={"height": "100%"})],
                                    className="card card-large",
                                ),
                                html.Div(
                                    [dcc.Graph(id="confusion", style={"height": "100%"})],
                                    className="card",
                                ),
                            ],
                            className="grid-top",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [dcc.Graph(id="accuracy", style={"height": "100%"})],
                                    className="card",
                                ),
                                html.Div(
                                    [dcc.Graph(id="dist-breakdown", style={"height": "100%"})],
                                    className="card",
                                ),
                            ],
                            className="grid-bot",
                        ),
                    ],
                    className="main-grid",
                ),
            ],
            className="body-wrapper",
        ),
        # ---- stores ----
        dcc.Store(id="sc-store", data=engine.default_support),
        dcc.Store(id="proto-ov-store", data={}),
        dcc.Store(id="sel-query-store", data=None),
    ],
    className="root",
)

# =====================================================================
# Callbacks
# =====================================================================

# ---------- main visualisation update ----------


@app.callback(
    Output("scatter", "figure"),
    Output("confusion", "figure"),
    Output("accuracy", "figure"),
    Output("overall-stat", "children"),
    Input("sc-store", "data"),
    Input("temp-slider", "value"),
    Input("proto-ov-store", "data"),
    Input("sel-query-store", "data"),
)
def update_main(sc, temp, pov, sel_q):
    res = engine.classify(sc, temp, pov or None)
    order = res["order"]
    if not order:
        empty = _base_fig(title="No data")
        return empty, empty, empty, ""

    # ---- embedding scatter ----
    sfig = _base_fig(
        title="Embedding Space (UMAP)",
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, title="UMAP-1"),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, title="UMAP-2"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )

    # decision boundaries
    xx, ys, zz = engine.decision_mesh(sc, temp, pov or None, res=60)
    if len(xx) > 0:
        # build discrete colour scale
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
                x=xx[0],
                y=ys,
                z=zz,
                colorscale=cvals,
                opacity=0.10,
                showscale=False,
                hoverinfo="skip",
            )
        )

    # query points
    qm = engine.query_mask(sc)
    qidxs = np.where(qm)[0]
    name2oi = {n: i for i, n in enumerate(order)}
    for ci, cname in enumerate(CLASS_NAMES):
        if cname not in name2oi:
            continue
        oi = name2oi[cname]
        mask = engine.labels[qidxs] == ci
        if not mask.any():
            continue
        pts = qidxs[mask]
        correct = res["true_idx"][np.where(mask)[0]] == res["preds"][np.where(mask)[0]]
        symbols = np.where(correct, "circle", "x")
        col = CLASS_COLORS[ci % len(CLASS_COLORS)]
        sfig.add_trace(
            go.Scatter(
                x=engine.embeddings_2d[pts, 0],
                y=engine.embeddings_2d[pts, 1],
                mode="markers",
                marker=dict(
                    size=6,
                    color=col,
                    symbol=symbols.tolist(),
                    opacity=0.75,
                    line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
                ),
                name=cname,
                customdata=pts.tolist(),
                text=[
                    f"{cname} (id={idx})<br>{'correct' if c else 'WRONG'}"
                    for idx, c in zip(pts, correct)
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # support points
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
                marker=dict(size=9, color=col, symbol="diamond", line=dict(width=1.5, color="white")),
                name=f"{cname} support",
                showlegend=False,
                hoverinfo="text",
                text=[f"Support: {cname} (w={it['weight']:.1f})" for it in items],
            )
        )

    # prototypes
    p2d = res["p2d"]
    for i, cname in enumerate(order):
        ci = CLASS_NAMES.index(cname)
        col = CLASS_COLORS[ci % len(CLASS_COLORS)]
        sfig.add_trace(
            go.Scatter(
                x=[p2d[i, 0]],
                y=[p2d[i, 1]],
                mode="markers+text",
                marker=dict(size=16, color=col, symbol="star", line=dict(width=2, color="white")),
                text=[cname],
                textposition="top center",
                textfont=dict(size=9, color=col),
                showlegend=False,
                hoverinfo="text",
                hovertext=f"Prototype: {cname}",
            )
        )

    # highlight selected
    if sel_q is not None and sel_q < len(engine.labels):
        sfig.add_trace(
            go.Scatter(
                x=[engine.embeddings_2d[sel_q, 0]],
                y=[engine.embeddings_2d[sel_q, 1]],
                mode="markers",
                marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(width=2.5, color="#ff6b6b")),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    sfig.update_layout(legend=dict(x=1.02, y=1, xanchor="left"))

    # ---- confusion matrix ----
    cm = res["cm"]
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums
    cfig = _base_fig(title="Confusion Matrix (row-normalised)")
    cfig.add_trace(
        go.Heatmap(
            z=cm_norm,
            x=order,
            y=order,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{text}<br>Rate: %{z:.2f}<extra></extra>",
            showscale=False,
        )
    )
    cfig.update_layout(
        xaxis=dict(title="Predicted", side="bottom", tickangle=-45),
        yaxis=dict(title="True", autorange="reversed"),
    )

    # ---- accuracy bars ----
    accs = res["accs"]
    anames = list(accs.keys())
    avals = [accs[n] for n in anames]
    acols = [CLASS_COLORS[CLASS_NAMES.index(n) % len(CLASS_COLORS)] for n in anames]
    afig = _base_fig(title="Per-Class Accuracy")
    afig.add_trace(
        go.Bar(
            x=avals,
            y=anames,
            orientation="h",
            marker=dict(color=acols, line=dict(width=0)),
            text=[f"{v:.0%}" for v in avals],
            textposition="auto",
            textfont=dict(size=10),
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )
    afig.update_layout(
        xaxis=dict(range=[0, 1.05], title="Accuracy", tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
    )

    # ---- overall stat ----
    stat = html.Div(
        [
            html.Span(f"{res['overall']:.1%}", className="stat-value"),
            html.Span(" overall accuracy", className="stat-label"),
        ]
    )

    return sfig, cfig, afig, stat


# ---------- scatter click -> select query / move prototype ----------


@app.callback(
    Output("sel-query-store", "data", allow_duplicate=True),
    Output("proto-ov-store", "data", allow_duplicate=True),
    Input("scatter", "clickData"),
    State("move-proto-toggle", "value"),
    State("class-selector", "value"),
    State("proto-ov-store", "data"),
    State("sc-store", "data"),
    prevent_initial_call=True,
)
def scatter_click(click, move_mode, sel_class, pov, sc):
    if not click or not click.get("points"):
        raise PreventUpdate
    pt = click["points"][0]

    # move prototype mode
    if "on" in (move_mode or []):
        pov = pov or {}
        pov[sel_class] = [pt["x"], pt["y"]]
        return no_update, pov

    # select query
    cd = pt.get("customdata")
    if cd is not None:
        idx = int(cd) if not isinstance(cd, list) else int(cd[0]) if cd else None
        if idx is not None:
            return idx, no_update
    raise PreventUpdate


# ---------- distance breakdown ----------


@app.callback(
    Output("dist-breakdown", "figure"),
    Input("sel-query-store", "data"),
    State("sc-store", "data"),
    State("temp-slider", "value"),
)
def update_dist(sel_q, sc, temp):
    if sel_q is None:
        fig = _base_fig(title="Distance Breakdown")
        fig.add_annotation(
            text="Click a query point in the scatter to see distance details",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=TEXT2, size=12),
        )
        return fig

    info = engine.query_distances(int(sel_q), sc, temp)
    if not info["order"]:
        return _base_fig(title="No prototypes")

    names = info["order"]
    dists = info["dists"]
    probs = info["probs"]
    cols = [CLASS_COLORS[CLASS_NAMES.index(n) % len(CLASS_COLORS)] for n in names]

    fig = _base_fig(title=f"Distances — {info['true']} → pred: {info['pred']}")

    fig.add_trace(
        go.Bar(
            x=probs,
            y=names,
            orientation="h",
            marker=dict(color=cols),
            text=[f"{p:.1%} (d={d:.2f})" for p, d in zip(probs, dists)],
            textposition="auto",
            textfont=dict(size=10),
            hovertemplate="%{y}<br>Prob: %{x:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(title="Softmax Probability", range=[0, 1.05], tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ---------- inspector panel ----------


@app.callback(
    Output("inspector-panel", "children"),
    Input("sel-query-store", "data"),
    State("sc-store", "data"),
    State("temp-slider", "value"),
)
def update_inspector(sel_q, sc, temp):
    if sel_q is None:
        return html.Div("Click a query point to inspect", className="hint-text")

    idx = int(sel_q)
    true_class = engine.class_names[engine.labels[idx]]
    info = engine.query_distances(idx, sc, temp)
    is_support = idx in engine.support_indices(sc)

    children = [
        html.Img(src=engine.image_to_base64(idx, size=80), className="inspector-img"),
        html.Div(
            [
                html.Div(f"ID: {idx}", className="inspector-field"),
                html.Div(f"True: {true_class}", className="inspector-field"),
                html.Div(f"Pred: {info['pred']}", className="inspector-field"),
                html.Div(
                    "CORRECT" if info["true"] == info["pred"] else "WRONG",
                    className="inspector-tag " + ("tag-ok" if info["true"] == info["pred"] else "tag-err"),
                ),
            ],
            className="inspector-info",
        ),
    ]

    if not is_support:
        children.append(
            html.Button(
                f"Add to {true_class} support",
                id="add-support-btn",
                className="btn-accent",
                n_clicks=0,
            )
        )

    return html.Div(children, className="inspector-row")


# ---------- add to support from inspector ----------


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
    # check not already in support
    existing = {it["idx"] for it in sc.get(cname, [])}
    if idx in existing:
        raise PreventUpdate
    sc.setdefault(cname, []).append({"idx": idx, "weight": 1.0})
    return sc


# ---------- support panel render ----------


@app.callback(
    Output("support-panel", "children"),
    Input("class-selector", "value"),
    Input("sc-store", "data"),
)
def render_support(sel_class, sc):
    items = sc.get(sel_class, [])
    if not items:
        return html.Div("No support sketches for this class", className="hint-text")

    children = []
    for item in items:
        idx = item["idx"]
        w = item["weight"]
        children.append(
            html.Div(
                [
                    html.Img(src=engine.image_to_base64(idx), className="support-thumb"),
                    html.Div(
                        [
                            dcc.Input(
                                id={"type": "wt-input", "index": idx},
                                type="number",
                                min=0.1,
                                max=3.0,
                                step=0.1,
                                value=round(w, 1),
                                debounce=True,
                                className="wt-input",
                            ),
                        ],
                        className="wt-row",
                    ),
                    html.Button(
                        "\u00d7",
                        id={"type": "rm-btn", "index": idx},
                        className="rm-btn",
                        n_clicks=0,
                    ),
                ],
                className="support-item",
            )
        )

    return html.Div(children, className="support-list")


# ---------- weight change ----------


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


# ---------- remove from support ----------


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


# ---------- reset ----------


@app.callback(
    Output("sc-store", "data", allow_duplicate=True),
    Output("proto-ov-store", "data", allow_duplicate=True),
    Output("sel-query-store", "data", allow_duplicate=True),
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_all(n):
    if not n:
        raise PreventUpdate
    return engine.default_support, {}, None


# ---------- toggle hint ----------


@app.callback(
    Output("proto-hint", "style"),
    Input("move-proto-toggle", "value"),
)
def toggle_hint(v):
    if "on" in (v or []):
        return {"display": "block"}
    return {"display": "none"}


# =====================================================================
# Entry
# =====================================================================
server = app.server

if __name__ == "__main__":
    app.run(debug=True, port=8050)
