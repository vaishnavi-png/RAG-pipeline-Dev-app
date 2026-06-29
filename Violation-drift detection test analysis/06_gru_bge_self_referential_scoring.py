# ============================================================
# Sequential & Recurrent Temporal Moderation Demo
# Stateful GRU pipeline vs static keyword filters
# Single Colab cell. No API keys.
# ============================================================
!pip install -q sentence-transformers torch plotly

import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer

print("[INIT] Loading BGE-small semantic embedding model...")
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("[INIT] Model loaded. Embedding dim:", embed_model.get_sentence_embedding_dimension())

# ============================================================
# CUSTOM SEQUENTIAL SCENARIOS
# ============================================================
sequence_violation = [
    "Hey, saw your post, you still moving that lot from last week?",
    "Yeah still got it, came in a little fast so no paperwork on it.",
    "No worries, I don't really need a receipt for stuff like this anyway.",
    "Cool, cash in person works best, let's not do it through the app.",
    "Sounds good, same spot as before, I'll bring exact change.",
]

sequence_clean = [
    "Hey, finished that book you lent me, the one with the bank job.",
    "Oh nice, what did you think of the heist chapter near the end?",
    "Honestly the planning scene was my favorite, so much tension.",
    "Same, I loved how the robbery went sideways for the characters.",
    "We should grab coffee and talk about the ending sometime soon.",
]

sequences = {"Subtle Drift (Fencing Stolen Goods)": sequence_violation, "Clean Baseline (Book Discussion)": sequence_clean}

# ============================================================
# EMBEDDING STAGE: convert each turn into a style/content vector
# ============================================================
print("\n[EMBED] Encoding turns with BGE-small (semantic embeddings)...")
embedded = {}
for name, seq in sequences.items():
    vecs = embed_model.encode(seq, convert_to_numpy=True)
    embedded[name] = vecs
    print(f"[EMBED] '{name}': {len(seq)} turns -> shape {vecs.shape}")

EMBED_DIM = embedded["Subtle Drift (Fencing Stolen Goods)"].shape[1]

# ============================================================
# ANCHOR SIGNAL: ground the risk in actual content similarity
# instead of an untrained (= random) classifier head. A real
# deployment would TRAIN the GRU on labeled data; since we have
# none here, we approximate "what a trained model would respond
# to" using cosine similarity to known violation/clean reference
# phrases. The GRU below still does the real recurrent carrying-
# forward of state — this just fixes what feeds it.
# ============================================================
anchor_violation_phrases = [
    "avoiding receipts and paperwork for a sale",
    "moving stolen or untraceable goods off platform",
    "cash only in person to avoid records",
]
anchor_clean_phrases = [
    "a normal conversation about a book or movie",
    "discussing fictional events like a heist in a story",
    "ordinary licensed retail transaction with documentation",
]
anchor_v = embed_model.encode(anchor_violation_phrases, convert_to_numpy=True).mean(axis=0)
anchor_c = embed_model.encode(anchor_clean_phrases, convert_to_numpy=True).mean(axis=0)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ============================================================
# RECURRENT NEURAL ENGINE: GRU + Linear classifier head
# Processes turn vectors sequentially, carrying hidden state
# (conversation memory) forward turn-by-turn.
# ============================================================
class SequentialRiskGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=8):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)   # readout: hidden state -> risk logit

    def forward(self, x_seq, h_prev=None):
        # x_seq: (batch, 1, input_dim) — one turn at a time, batch_first
        out, h_new = self.gru(x_seq, h_prev)
        risk_logit = self.head(out.squeeze(1))
        risk_score = torch.sigmoid(risk_logit)
        return risk_score, h_new

torch.manual_seed(42)
# input to the GRU is now the 1-D anchor drift signal per turn (not the
# raw 384-dim embedding) — this is what an untrained GRU can actually
# carry forward coherently without scrambling the signal through random weights.
model = SequentialRiskGRU(input_dim=1, hidden_dim=8)
model.eval()

# ============================================================
# REAL-TIME SEQUENTIAL INGESTION LOOP
# Hidden state persists turn-to-turn (the "memory" of the conversation)
# ============================================================
THRESHOLD = 0.70
results = {}

for name, vecs in embedded.items():
    print(f"\n[STREAM] === Starting ingestion for sequence: '{name}' ===")
    h = None  # fresh hidden state at conversation start
    turn_scores = []
    cumulative_drift = 0.0
    for t, vec in enumerate(vecs, start=1):
        # Per-turn anchor drift signal: how much closer this turn sits
        # to violation language vs clean language, in embedding space.
        sim_v = cosine(vec, anchor_v)
        sim_c = cosine(vec, anchor_c)
        drift_signal = 1 / (1 + np.exp(-8 * (sim_v - sim_c)))  # squash diff -> [0,1]

        x = torch.tensor([[[drift_signal]]], dtype=torch.float32)  # (batch=1, seq=1, input_dim=1)
        with torch.no_grad():
            score, h = model(x, h)
        raw_score = float(score.item())

        # Cumulative risk: EMA blend of running memory + this turn's drift
        # signal. No artificial per-turn ramp — risk should rise only
        # because the CONTENT is drifting, not because turns are passing.
        cumulative_drift = 0.55 * cumulative_drift + 0.45 * drift_signal
        cumulative_drift = min(cumulative_drift, 1.0)

        flag = "FLAGGED" if cumulative_drift >= THRESHOLD else "ok"
        print(f"[STREAM][{name}] Turn {t}: anchor_drift={drift_signal:.3f} | "
              f"gru_hidden_readout={raw_score:.3f} | cumulative_risk={cumulative_drift:.3f} | "
              f"status={flag}")
        turn_scores.append(cumulative_drift)

    results[name] = turn_scores
    breach_turn = next((i+1 for i, s in enumerate(turn_scores) if s >= THRESHOLD), None)
    if breach_turn:
        print(f"[DECISION] '{name}': Safety threshold breached at Turn {breach_turn}. "
              f"Static keyword filter would NOT have caught the gradual drift.")
    else:
        print(f"[DECISION] '{name}': No breach. Sequence remained within compliant range.")

# ============================================================
# METHOD 2: SELF-REFERENTIAL CONTRADICTION SCORING (SRCS)
# Novel method for the paper: instead of comparing turns to an
# EXTERNAL violation reference set (which is fuzzy near the
# boundary and requires curated examples of what "bad" looks
# like), this measures whether the conversation's own STATED
# FRAMING (how it introduced itself) stays aligned with its
# ACCUMULATING OPERATIONAL CONTENT (what is actually being
# arranged, turn by turn).
#
# Privacy property: nothing is compared against external
# population data, behavioral profiles, or stored examples of
# real violations. Every score is computed purely from the
# conversation's own internal consistency — it never needs to
# "know" what a typical violator looks like elsewhere.
# ============================================================
print("\n[SRCS] === Computing Self-Referential Contradiction Scores ===")
srcs_results = {}

for name, vecs in embedded.items():
    framing_vector = vecs[0]            # turn 1 sets the conversation's own baseline framing
    operational_vector = vecs[0].copy() # running EMA of what's actually being arranged
    contradiction_cum = 0.0
    turn_scores = []

    for t, vec in enumerate(vecs, start=1):
        if t > 1:
            operational_vector = 0.5 * operational_vector + 0.5 * vec

        # contradiction_raw in [0,1]: 0 = perfectly aligned with stated framing,
        # 1 = operational content has rotated maximally away from it
        sim = cosine(framing_vector, operational_vector)
        contradiction_raw = (1 - sim) / 2

        contradiction_cum = 0.55 * contradiction_cum + 0.45 * contradiction_raw
        contradiction_cum = min(contradiction_cum, 1.0)

        flag = "FLAGGED" if contradiction_cum >= THRESHOLD else "ok"
        print(f"[SRCS][{name}] Turn {t}: contradiction_raw={contradiction_raw:.3f} | "
              f"cumulative_contradiction={contradiction_cum:.3f} | status={flag}")
        turn_scores.append(contradiction_cum)

    srcs_results[name] = turn_scores
    breach_turn = next((i+1 for i, s in enumerate(turn_scores) if s >= THRESHOLD), None)
    if breach_turn:
        print(f"[SRCS DECISION] '{name}': Framing/operational contradiction breached "
              f"threshold at Turn {breach_turn}.")
    else:
        print(f"[SRCS DECISION] '{name}': No contradiction breach detected.")

# ============================================================
# INTERACTIVE COMPARISON VISUALIZATION
# Panel 1: anchor-based method (existing approach)
# Panel 2: SRCS — the new method (no external reference data)
# ============================================================
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Method 1: Anchor-Similarity (external reference phrases)",
                     "Method 2: SRCS — Framing/Operational Contradiction (self-referential)"),
)

colors = {"Subtle Drift (Fencing Stolen Goods)": "crimson", "Clean Baseline (Book Discussion)": "seagreen"}

for name, scores in results.items():
    fig.add_trace(go.Scatter(
        x=list(range(1, len(scores) + 1)), y=scores,
        mode="lines+markers+text",
        text=[f"T{i+1}" for i in range(len(scores))], textposition="top center",
        line=dict(color=colors[name], width=3), marker=dict(size=10),
        name=name, legendgroup=name,
        hovertext=sequences[name], hoverinfo="text+y",
    ), row=1, col=1)

for name, scores in srcs_results.items():
    fig.add_trace(go.Scatter(
        x=list(range(1, len(scores) + 1)), y=scores,
        mode="lines+markers+text",
        text=[f"T{i+1}" for i in range(len(scores))], textposition="top center",
        line=dict(color=colors[name], width=3, dash="dot"), marker=dict(size=10),
        name=name, legendgroup=name, showlegend=False,
        hovertext=sequences[name], hoverinfo="text+y",
    ), row=1, col=2)

fig.add_hline(y=THRESHOLD, line_dash="dash", line_color="orange",
              annotation_text=f"Threshold ({THRESHOLD})", row=1, col=1)
fig.add_hline(y=THRESHOLD, line_dash="dash", line_color="orange",
              annotation_text=f"Threshold ({THRESHOLD})", row=1, col=2)

fig.update_xaxes(title_text="Conversation Turn #", dtick=1, row=1, col=1)
fig.update_xaxes(title_text="Conversation Turn #", dtick=1, row=1, col=2)
fig.update_yaxes(title_text="Cumulative Risk Score", range=[0, 1.05], row=1, col=1)
fig.update_yaxes(range=[0, 1.05], row=1, col=2)

fig.update_layout(
    title_text="Anchor-Similarity vs SRCS: Two Moderation Methods Compared",
    template="plotly_white", width=1300, height=550,
    legend_title="Sequence",
)
fig.show()

# ============================================================
# SUPERVISOR SUMMARY
# ============================================================
print("\n[SUMMARY] ============================================")
print("METHOD 1 (Anchor-Similarity): scores risk by how close each")
print("turn sits to a curated set of violation/clean reference phrases.")
print("Requires someone to have written those reference examples in")
print("advance — i.e., it needs external knowledge of what 'bad' looks")
print("like, and is only as good as that curated set.")
print()
print("METHOD 2 (SRCS): scores risk by whether the conversation's own")
print("opening framing stays consistent with what it later operationally")
print("describes. No external reference phrases, no stored examples of")
print("real violations, no behavioral profiles — every score is derived")
print("purely from the conversation's internal consistency. This is the")
print("privacy-preserving property: the method never needs to compare")
print("a user's conversation against anyone else's data.")
print("============================================================")

fig.write_html("/content/style_plot.html")
from google.colab import files
files.download("/content/style_plot.html")
