# ============================================================
# Dual Moderation Architecture Demo v2 — more visual
# ============================================================
!pip install -q plotly networkx scikit-learn pandas numpy

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------
# MOCK SCENARIO: 5 turns, benign -> drifting -> violation
# ------------------------------------------------------------
turns = [
    {"turn": 1, "text": "Hey, how's it going today?",                       "node": "Casual Greeting"},
    {"turn": 2, "text": "Do you know anyone who has a package for sale?",   "node": "Inquire Package"},
    {"turn": 3, "text": "I'd prefer cash only, no questions asked.",        "node": "Cash Only"},
    {"turn": 4, "text": "Can we meet somewhere private, like a side alley?","node": "Side Alley"},
    {"turn": 5, "text": "Confirmed — bring the restricted item, untraceable.","node": "Malicious Cluster"},
]
df = pd.DataFrame(turns)

# ============================================================
# APPROACH 1: Sequential / Recurrent Modeling
# Real RNNs carry a HIDDEN STATE VECTOR (not a scalar) forward.
# Same cell weights are reused at every timestep ("unrolled").
# We mock an 8-dim hidden state so you can SEE the vector evolve.
# ============================================================
np.random.seed(42)
HIDDEN_DIM = 8
turn_signal = np.array([0.05, 0.30, 0.55, 0.75, 0.95])  # mock suspicion input per turn

# Fixed "weights" so the same cell math applies every step (shared weights = key RNN idea)
W_h = np.random.uniform(0.1, 0.4, HIDDEN_DIM)
W_x = np.random.uniform(0.5, 1.0, HIDDEN_DIM)

def gru_style_step(h_prev, x_t):
    z = 1 / (1 + np.exp(-6 * (x_t - 0.4) * W_x))   # update gate (vector)
    h_cand = np.tanh(x_t * W_x + 0.1 * np.arange(HIDDEN_DIM))
    h_t = (1 - z) * (h_prev * W_h / W_h.mean()) + z * h_cand
    return h_t

h = np.zeros(HIDDEN_DIM)
hidden_matrix, risk_scores = [], []
for sig in turn_signal:
    h = gru_style_step(h, sig)
    hidden_matrix.append(h.copy())
    risk = 1 / (1 + np.exp(-4 * (h.mean() - 0.2)))   # readout layer: vector -> scalar risk
    risk_scores.append(risk)

hidden_matrix = np.array(hidden_matrix)   # shape (5 turns, 8 dims)
df["risk_score"] = risk_scores
RISK_THRESHOLD = 0.65

# ============================================================
# APPROACH 2: Graph-Based Semantic Traversal
# ============================================================
all_nodes = {
    "Casual Greeting":   (0, 4),
    "Weather Chat":      (-1.8, 4.6),
    "Inquire Package":   (1, 3),
    "Track Shipping":    (2.8, 3.6),
    "Cash Only":         (1.5, 2),
    "Side Alley":        (2, 1),
    "Malicious Cluster": (2.5, 0),
}
G = nx.DiGraph()
for n, p in all_nodes.items():
    G.add_node(n, pos=p)

path_nodes = df["node"].tolist()
path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
decoy_edges = [("Casual Greeting", "Weather Chat"), ("Inquire Package", "Track Shipping")]
G.add_edges_from(path_edges + decoy_edges)
pos = nx.get_node_attributes(G, "pos")

# DASHBOARD — 4 panels: RNN unroll diagram, hidden-state heatmap,
# risk timeline, semantic graph

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Panel A: Unrolled RNN Cell (same weights, reused each turn)",
        "Panel B: Hidden State Vector Evolving (8 dims x 5 turns)",
        "Panel C: Risk Score Readout Over Time",
        "Panel D: Semantic Graph Traversal",
    ),
    specs=[[{"type": "scatter"}, {"type": "heatmap"}],
           [{"type": "scatter"}, {"type": "scatter"}]],
    vertical_spacing=0.15, horizontal_spacing=0.12,
)

# --- Panel A: unrolled RNN diagram (boxes = cell, arrows = hidden state passed forward) ---
n_steps = 5
for i in range(n_steps):
    x = i * 2
    # cell box
    fig.add_shape(type="rect", x0=x, x1=x+1.2, y0=0, y1=1,
                  line=dict(color="navy"), fillcolor="lightblue", opacity=0.6,
                  row=1, col=1)
    fig.add_annotation(x=x+0.6, y=0.5, text=f"Cell t={i+1}<br>(shared W)",
                        showarrow=False, font=dict(size=9), row=1, col=1)
    # input arrow up
    fig.add_annotation(x=x+0.6, y=-0.6, ax=x+0.6, ay=0, xref="x1", yref="y1",
                        axref="x1", ayref="y1", text=f"x{i+1}", showarrow=True,
                        arrowhead=2, font=dict(size=9), row=1, col=1)
    # hidden state arrow to next cell
    if i < n_steps - 1:
        fig.add_annotation(x=x+2, y=0.5, ax=x+1.2, ay=0.5, xref="x1", yref="y1",
                            axref="x1", ayref="y1", text="h", showarrow=True,
                            arrowhead=3, arrowcolor="crimson", font=dict(size=9, color="crimson"),
                            row=1, col=1)
fig.add_trace(go.Scatter(x=[-1, n_steps*2], y=[-1, 1.5], mode="markers",
                          marker=dict(opacity=0), showlegend=False), row=1, col=1)
fig.update_xaxes(visible=False, row=1, col=1)
fig.update_yaxes(visible=False, row=1, col=1)

# --- Panel B: heatmap of hidden state vector across turns ---
fig.add_trace(go.Heatmap(
    z=hidden_matrix.T, x=[f"Turn {t}" for t in df["turn"]],
    y=[f"dim {d}" for d in range(HIDDEN_DIM)],
    colorscale="Reds", colorbar=dict(title="activation", x=1.02, len=0.4, y=0.85),
), row=1, col=2)

# --- Panel C: risk timeline ---
fig.add_trace(go.Scatter(
    x=df["turn"], y=df["risk_score"], mode="lines+markers+text",
    text=[f"T{t}" for t in df["turn"]], textposition="top center",
    line=dict(color="crimson", width=3), marker=dict(size=10),
    hovertext=df["text"], hoverinfo="text+y", showlegend=False,
), row=2, col=1)
fig.add_hline(y=RISK_THRESHOLD, line_dash="dash", line_color="orange",
              annotation_text="Violation Threshold", row=2, col=1)
fig.update_xaxes(title_text="Turn", dtick=1, row=2, col=1)
fig.update_yaxes(title_text="Risk (0-1)", range=[0, 1.05], row=2, col=1)

# --- Panel D: semantic graph, gradient path + arrows + risk-scaled nodes ---
for e in G.edges():
    x0, y0 = pos[e[0]]; x1, y1 = pos[e[1]]
    on_path = e in path_edges
    fig.add_trace(go.Scatter(
        x=[x0, x1], y=[y0, y1], mode="lines",
        line=dict(color="lightgray" if not on_path else "crimson",
                   width=1.5 if not on_path else 5),
        showlegend=False, hoverinfo="skip",
    ), row=2, col=2)
    if on_path:  # directional arrowhead
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x2", yref="y2",
                            axref="x2", ayref="y2", showarrow=True,
                            arrowhead=3, arrowsize=1.2, arrowcolor="crimson",
                            row=2, col=2)

node_list = list(G.nodes())
risk_by_node = {n: df.loc[df["node"] == n, "risk_score"].values[0] if n in path_nodes else 0.15
                for n in node_list}
fig.add_trace(go.Scatter(
    x=[pos[n][0] for n in node_list], y=[pos[n][1] for n in node_list],
    mode="markers+text", text=node_list, textposition="bottom center",
    marker=dict(
        size=[20 + 40*risk_by_node[n] for n in node_list],
        color=[risk_by_node[n] for n in node_list],
        colorscale="Reds", cmin=0, cmax=1, showscale=False,
        line=dict(width=1, color="black"),
    ),
    showlegend=False, hoverinfo="text",
), row=2, col=2)
fig.update_xaxes(visible=False, row=2, col=2)
fig.update_yaxes(visible=False, row=2, col=2)

fig.update_layout(
    title_text="Moderation Architecture Comparison: Sequential RNN vs Graph Traversal",
    height=850, width=1250, template="plotly_white",
)
fig.show()

print(df[["turn", "node", "risk_score"]].to_string(index=False))
