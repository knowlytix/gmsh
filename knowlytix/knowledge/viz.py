# SPDX-License-Identifier: Apache-2.0
"""Interactive 3-D visualization of a GMS knowledge-graph embedding.

Entity embeddings are reduced to three dimensions with PCA and projected onto
the unit sphere; each embedded ``(head, relation, tail)`` triple is drawn as a
geodesic (great-circle) arc colored by relation, with an arrowhead at the tail.

The geodesic — not a straight chord — is the geometrically faithful path: a
relation in a rotor embedding acts as a rotation ``R``, so the head→tail path is
the orbit of the one-parameter subgroup ``exp(s log R)``, which is the great
circle joining the two unit vectors (their spherical linear interpolation).

The scene is rendered with Plotly and is fully interactive (drag to rotate,
scroll to zoom, click a relation in the legend to toggle it). Plotly is an
optional dependency; install it with ``pip install knowlytix-knowledge[viz]``.

Typical use::

    from knowlytix.knowledge import GMSExpertStore
    from knowlytix.knowledge.viz import sphere_figure

    store = GMSExpertStore(config); store.load()
    sphere_figure(store, out_html="kg.html")     # live store object
    sphere_figure("/path/to/store_dir")          # saved store directory

Any embedding graph can be visualized without a store via :class:`KGData`::

    from knowlytix.knowledge.viz import KGData, sphere_figure
    sphere_figure(KGData(labels=[...], embeddings=arr, triples=[(h, r, t), ...]))
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

# ── Data container ─────────────────────────────────────────────────────────

@dataclass
class KGData:
    """A knowledge graph plus its entity embeddings.

    Parameters
    ----------
    labels:
        Entity names; index ``i`` names row ``i`` of ``embeddings``.
    embeddings:
        Float array of shape ``(n_entities, dim)``.
    triples:
        ``(head, relation, tail)`` name tuples. Triples referencing an entity
        absent from ``labels`` are dropped (only embedded nodes can be placed).
    relations:
        Optional ordered relation vocabulary for stable coloring; inferred from
        ``triples`` when omitted.
    """

    labels: list[str]
    embeddings: np.ndarray
    triples: list[tuple[str, str, str]]
    relations: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.embeddings = np.asarray(self.embeddings, dtype=float)
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be 2-D (n_entities, dim)")
        if self.embeddings.shape[0] != len(self.labels):
            raise ValueError(
                f"labels ({len(self.labels)}) and embedding rows "
                f"({self.embeddings.shape[0]}) disagree"
            )
        label_set = set(self.labels)
        self.triples = [(h, r, t) for (h, r, t) in self.triples
                        if h in label_set and t in label_set]
        if not self.relations:
            seen: dict[str, None] = {}
            for _, r, _ in self.triples:
                seen.setdefault(r, None)
            self.relations = list(seen)

    @property
    def index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.labels)}


# ── Loaders: from a live store, a store directory, or raw arrays ───────────

def _entity_embeddings(state: dict, source: str) -> tuple[np.ndarray, str]:
    """Pull the entity-embedding matrix from a model state_dict.

    Prefers ``dual_emb.{source}_embed.weight``; falls back to any 2-D
    ``*embed*.weight`` tensor. The row count defines how many entities are
    actually embedded (the adapter vocabulary may be larger).
    """
    key = f"dual_emb.{source}_embed.weight"
    if key not in state:
        cands = [k for k, v in state.items()
                 if getattr(v, "ndim", 0) == 2 and "embed" in k]
        if not cands:
            raise KeyError("no entity-embedding tensor found in model state; "
                           f"tried '{key}' and auto-detection")
        key = sorted(cands)[0]
    return state[key].detach().cpu().numpy(), key


def _labels_for(idx_to_entity, n: int) -> list[str]:
    """Ordered labels for indices 0..n-1 (keys may be int or str)."""
    def get(i):
        if i in idx_to_entity:
            return idx_to_entity[i]
        return idx_to_entity[str(i)]
    return [get(i) for i in range(n)]


def from_store(store) -> KGData:
    """Build :class:`KGData` from a live ``GMSExpertStore`` (already loaded).

    The embedding rows in ``model.pt`` and the index→entity map in
    ``adapter.json`` are written together by ``store.save()`` and are mutually
    consistent. The *in-memory* adapter, by contrast, is re-indexed on load
    (entities are re-sorted, so row order no longer matches), which would
    mislabel every embedded vector. So when the store has been persisted, read
    the authoritative mapping back from its files; fall back to the live
    objects only for an unsaved, same-session store.
    """
    if store.model is None or store.adapter is None:
        raise ValueError("store has no model/adapter; call store.load() first")

    sp = getattr(store, "store_path", None)
    if sp and os.path.isfile(os.path.join(sp, "adapter.json")) \
            and os.path.isfile(os.path.join(sp, "model.pt")):
        return from_store_dir(sp)

    embeddings, _ = _entity_embeddings(store.model.state_dict(), "v")
    labels = _labels_for(store.adapter.idx_to_entity, embeddings.shape[0])
    rel_vocab = list(getattr(store.adapter, "relation_to_idx", {}).keys())
    triples = [tuple(t) for t in (store.doc_graph.triples
                                  if store.doc_graph is not None else [])]
    return KGData(labels=labels, embeddings=embeddings, triples=triples,
                  relations=rel_vocab)


def from_store_dir(store_dir: str, source: str = "v") -> KGData:
    """Build :class:`KGData` from a saved store directory.

    Reads ``adapter.json`` (labels), ``model.pt`` (embeddings) and
    ``triples.json`` (edges). No live store or GMS session required.
    """
    import torch

    adapter = json.load(open(os.path.join(store_dir, "adapter.json")))
    idx_to_entity = adapter["idx_to_entity"]
    rel_vocab = list(adapter.get("relation_to_idx", {}).keys())

    state = torch.load(os.path.join(store_dir, "model.pt"),
                       map_location="cpu", weights_only=False)
    embeddings, _ = _entity_embeddings(state, source)
    labels = _labels_for(idx_to_entity, embeddings.shape[0])

    triples: list[tuple[str, str, str]] = []
    path = os.path.join(store_dir, "triples.json")
    if os.path.exists(path):
        triples = [tuple(t) for t in json.load(open(path))]

    return KGData(labels=labels, embeddings=embeddings, triples=triples,
                  relations=rel_vocab)


# ── Geometry ───────────────────────────────────────────────────────────────

def project_to_sphere(embeddings: np.ndarray, standardize: bool = False,
                      eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Reduce embeddings to 3-D with PCA and project onto the unit sphere.

    Returns the ``(n, 3)`` coordinate array (each row on the unit sphere) and
    the 3-vector of per-component explained-variance ratios.
    """
    from sklearn.decomposition import PCA

    X = np.asarray(embeddings, dtype=float)
    if standardize:
        std = X.std(axis=0)
        std[std == 0] = 1.0
        X = (X - X.mean(axis=0)) / std

    n_comp = min(3, X.shape[1])
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(X)
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))

    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    coords = coords / np.maximum(norms, eps)

    evr = np.zeros(3)
    evr[:len(pca.explained_variance_ratio_)] = pca.explained_variance_ratio_
    return coords, evr


def slerp_arc(h: np.ndarray, t: np.ndarray, n: int = 48) -> np.ndarray:
    """Great-circle (geodesic) arc between two unit vectors, ``n`` samples.

    The minimal geodesic joining two unit vectors is their spherical linear
    interpolation, which is the orbit of the rotation carrying ``h`` to ``t`` —
    the faithful path for a rotor / rotation-operator relation embedding.
    """
    h = h / max(np.linalg.norm(h), 1e-12)
    t = t / max(np.linalg.norm(t), 1e-12)
    dot = float(np.clip(h @ t, -1.0, 1.0))
    omega = np.arccos(dot)
    s = np.linspace(0.0, 1.0, n)
    if omega < 1e-6:                       # coincident
        return np.outer(np.ones_like(s), h)
    if np.pi - omega < 1e-6:               # antipodal: geodesic not unique
        perturb = np.eye(3)[int(np.argmin(np.abs(h)))]
        t = t + 1e-3 * perturb
        t = t / np.linalg.norm(t)
        dot = float(np.clip(h @ t, -1.0, 1.0))
        omega = np.arccos(dot)
    so = np.sin(omega)
    a = np.sin((1.0 - s) * omega) / so
    b = np.sin(s * omega) / so
    return np.outer(a, h) + np.outer(b, t)


# ── Rendering ──────────────────────────────────────────────────────────────

# Colorblind-safe qualitative palette (Paul Tol "muted", extended).
_PALETTE = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
    "#AA3377", "#EE8866", "#44BB99", "#99DDFF", "#FFAABB",
    "#AACC44", "#DDCC77",
]


def _require_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError as e:  # pragma: no cover - dependency guard
        raise ImportError(
            "knowlytix KG visualization needs Plotly. Install the optional "
            "extra:  pip install 'knowlytix-knowledge[viz]'"
        ) from e


def _graticule(color: str, n_lines: int = 9, n_pts: int = 90):
    go = _require_plotly()
    xs, ys, zs = [], [], []
    phi = np.linspace(0, 2 * np.pi, n_pts)
    for lat in np.linspace(-np.pi / 2, np.pi / 2, n_lines + 2)[1:-1]:
        r = np.cos(lat)
        xs += list(r * np.cos(phi)) + [None]
        ys += list(r * np.sin(phi)) + [None]
        zs += list(np.full(n_pts, np.sin(lat))) + [None]
    theta = np.linspace(0, np.pi, n_pts)
    for lon in np.linspace(0, 2 * np.pi, 2 * n_lines, endpoint=False):
        xs += list(np.sin(theta) * np.cos(lon)) + [None]
        ys += list(np.sin(theta) * np.sin(lon)) + [None]
        zs += list(np.cos(theta)) + [None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                        line={"color": color, "width": 1}, opacity=0.28,
                        hoverinfo="skip", showlegend=False, name="graticule")


def _add_highlight(fig, kg, coords, idx, highlight_triples, highlight_entities,
                   color, fg, arrow_size, arc_pts):
    """Overlay bright geodesic arcs for ``highlight_triples`` and bright markers
    for ``highlight_entities`` on top of the (dimmed) base graph."""
    go = _require_plotly()
    if highlight_triples:
        lx, ly, lz = [], [], []
        cx, cy, cz, cu, cv, cw = [], [], [], [], [], []
        for h, _r, t in highlight_triples:
            if h not in idx or t not in idx:
                continue
            arc = slerp_arc(coords[idx[h]], coords[idx[t]], n=arc_pts)
            lx += list(arc[:, 0]) + [None]
            ly += list(arc[:, 1]) + [None]
            lz += list(arc[:, 2]) + [None]
            tan = arc[-1] - arc[-4]
            tan = tan / max(np.linalg.norm(tan), 1e-12)
            cx.append(arc[-1][0])
            cy.append(arc[-1][1])
            cz.append(arc[-1][2])
            cu.append(tan[0] * arrow_size)
            cv.append(tan[1] * arrow_size)
            cw.append(tan[2] * arrow_size)
        if lx:
            fig.add_trace(go.Scatter3d(
                x=lx, y=ly, z=lz, mode="lines",
                line={"color": color, "width": 7}, opacity=1.0,
                name="retrieved", hoverinfo="skip"))
            fig.add_trace(go.Cone(
                x=cx, y=cy, z=cz, u=cu, v=cv, w=cw, sizeref=1.0, sizemode="raw",
                anchor="tip", showscale=False,
                colorscale=[[0, color], [1, color]], opacity=1.0,
                showlegend=False, hoverinfo="skip"))

    # Entities to spotlight: explicit list plus every endpoint of a highlight arc.
    marked: list[str] = list(highlight_entities or [])
    for h, _r, t in (highlight_triples or []):
        marked += [h, t]
    seen, keep = set(), []
    for e in marked:
        if e in idx and e not in seen:
            seen.add(e)
            keep.append(e)
    if keep:
        pts = coords[[idx[e] for e in keep]]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers+text", text=keep, textposition="top center",
            textfont={"color": color, "size": 13},
            marker={"size": 12, "color": color, "opacity": 1.0,
                    "line": {"color": "#1a1a1a", "width": 2},
                    "symbol": "diamond"},
            hovertemplate="<b>%{text}</b><extra>highlighted</extra>",
            name="query / retrieved nodes", showlegend=True))


def build_figure(kg: KGData, coords: np.ndarray, evr: np.ndarray,
                 title: str = "Knowledge-Graph Embedding",
                 relations: list[str] | None = None, dark: bool = True,
                 arrow_size: float = 0.08, arc_pts: int = 48,
                 highlight_triples: list[tuple[str, str, str]] | None = None,
                 highlight_entities: list[str] | None = None,
                 highlight_color: str = "#FFD166"):
    """Assemble the interactive Plotly figure with geodesic relation arcs.

    ``highlight_triples`` / ``highlight_entities`` draw a bright overlay on top of
    the base graph — used to show a query's bound head and its retrieved triplets
    against the whole embedding. When any highlight is given the base graph is
    dimmed so the overlay stands out. A highlighted triple whose head or tail is
    not an embedded node is skipped for the arc; its head is still marked if listed
    in ``highlight_entities``."""
    go = _require_plotly()

    idx = kg.index
    rels = relations or kg.relations
    rel_color = {r: _PALETTE[i % len(_PALETTE)] for i, r in enumerate(rels)}
    has_hl = bool(highlight_triples or highlight_entities)
    base_op = 0.25 if has_hl else 0.95   # dim the base graph under a highlight

    if dark:
        bg, fg, grat_c, node_c = "#080b11", "#e8eef6", "#5f7488", "#eaf2ff"
    else:
        bg, fg, grat_c, node_c = "#f7f9fc", "#141a22", "#9fb0c2", "#10233b"

    fig = go.Figure()
    fig.add_trace(_graticule(grat_c))

    deg = np.zeros(len(kg.labels))
    for h, _, t in kg.triples:
        deg[idx[h]] += 1
        deg[idx[t]] += 1
    sizes = 5 + 9 * (deg / deg.max() if deg.max() > 0 else deg)

    for r in rels:
        segs = [(h, t) for (h, rr, t) in kg.triples if rr == r]
        if not segs:
            continue
        color = rel_color[r]
        lx, ly, lz = [], [], []
        cx, cy, cz, cu, cv, cw = [], [], [], [], [], []
        for h, t in segs:
            arc = slerp_arc(coords[idx[h]], coords[idx[t]], n=arc_pts)
            lx += list(arc[:, 0]) + [None]
            ly += list(arc[:, 1]) + [None]
            lz += list(arc[:, 2]) + [None]
            tan = arc[-1] - arc[-4]
            tan = tan / max(np.linalg.norm(tan), 1e-12)
            cx.append(arc[-1][0])
            cy.append(arc[-1][1])
            cz.append(arc[-1][2])
            cu.append(tan[0] * arrow_size)
            cv.append(tan[1] * arrow_size)
            cw.append(tan[2] * arrow_size)
        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz, mode="lines",
            line={"color": color, "width": 4}, opacity=base_op,
            name=r, legendgroup=r, hoverinfo="skip"))
        fig.add_trace(go.Cone(
            x=cx, y=cy, z=cz, u=cu, v=cv, w=cw,
            sizeref=1.0, sizemode="raw", anchor="tip", showscale=False,
            colorscale=[[0, color], [1, color]], opacity=base_op,
            legendgroup=r, showlegend=False, hoverinfo="skip"))

    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers+text", text=kg.labels, textposition="top center",
        textfont={"color": fg, "size": 10},
        marker={"size": sizes, "color": node_c, "opacity": base_op,
                "line": {"color": bg, "width": 1.5}},
        customdata=deg,
        hovertemplate="<b>%{text}</b><br>degree %{customdata:.0f}<extra></extra>",
        name="entities", showlegend=False))

    _add_highlight(fig, kg, coords, idx, highlight_triples, highlight_entities,
                   highlight_color, fg, arrow_size, arc_pts)

    var_pct = 100 * evr.sum()
    n_rel = len({r for _, r, _ in kg.triples})
    subtitle = (f"{len(kg.labels)} entities · {len(kg.triples)} embedded "
                f"triples · {n_rel} relations · geodesic arcs · "
                f"PCA(3) captures {var_pct:.0f}% variance")
    ax = {"showbackground": False, "showgrid": False, "zeroline": False,
          "showticklabels": False, "title": "", "range": [-1.05, 1.05]}
    layout = {
        "paper_bgcolor": bg, "font": {"color": fg},
        "scene": {"xaxis": ax, "yaxis": ax, "zaxis": ax, "aspectmode": "cube",
                  "bgcolor": bg, "dragmode": "orbit",
                  "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.05}}},
        "legend": {"title": "relation (click to toggle)",
                   "itemsizing": "constant", "bgcolor": "rgba(0,0,0,0)",
                   "font": {"color": fg}},
        "margin": {"l": 0, "r": 0, "t": 70 if title else 0, "b": 0},
    }
    if title:
        layout["title"] = {"text": f"{title}<br><sub>{subtitle}</sub>",
                           "x": 0.5, "xanchor": "center",
                           "font": {"color": fg, "size": 20}}
    fig.update_layout(**layout)
    return fig


# ── Entry point ────────────────────────────────────────────────────────────

def sphere_figure(source, *, out_html: str | None = None, show: bool = False,
                  standardize: bool = False, source_embedding: str = "v",
                  **fig_kwargs):
    """Visualize a KG embedding on the unit sphere.

    ``source`` may be a live ``GMSExpertStore``, a store directory path, or a
    :class:`KGData`. Returns the Plotly figure; optionally writes a standalone
    interactive HTML file and/or shows it.
    """
    if isinstance(source, KGData):
        kg = source
    elif isinstance(source, (str, os.PathLike)):
        kg = from_store_dir(os.fspath(source), source=source_embedding)
    elif hasattr(source, "model") and hasattr(source, "adapter"):
        kg = from_store(source)
    else:
        raise TypeError(
            "source must be a GMSExpertStore, a store directory path, or KGData"
        )

    coords, evr = project_to_sphere(kg.embeddings, standardize=standardize)
    fig = build_figure(kg, coords, evr, **fig_kwargs)
    if out_html:
        fig.write_html(out_html, include_plotlyjs=True, full_html=True)
    if show:
        fig.show()
    return fig
