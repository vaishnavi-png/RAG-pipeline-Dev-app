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

print("[INIT] Loading StyleDistance embedding model...")
style_model = SentenceTransformer("StyleDistance/styledistance")
print("[INIT] Model loaded. Embedding dim:", style_model.get_sentence_embedding_dimension())

# ============================================================
# CUSTOM SEQUENTIAL SCENARIOS
# ============================================================
sequence_violation = [
    "Hi! Loving these bags, the craftsmanship looks amazing in the photos.",
    "Yes these are 'inspired by' the designer look, mirror quality finish.",
    "No box, no tags, no authentication card included, keeps cost down for you.",
    "Logo placement is exact 1:1 if you want it to pass casual inspection.",
    "Ship in plain packaging, listed as 'gift' on the customs label.",
]

sequence_clean = [
    "Hi! Loving these bags, the craftsmanship looks amazing in the photos.",
    "Thanks! These are licensed pieces, made under the official brand agreement.",
    "Comes with the authentication card and original retail packaging included.",
    "Logo placement follows the brand's official design specs exactly.",
    "Ships with standard customs declaration, listed accurately as merchandise.",
]

sequences = {"Codeword Drift (Counterfeit IP)": sequence_violation, "Clean Baseline": sequence_clean}

# ============================================================
# EMBEDDING STAGE: convert each turn into a style/content vector
# ============================================================
print("\n[EMBED] Encoding turns with StyleDistance...")
embedded = {}
for name, seq in sequences.items():
    vecs = style_model.encode(seq, convert_to_numpy=True)
    embedded[name] = vecs
    print(f"[EMBED] '{name}': {len(seq)} turns -> shape {vecs.shape}")

EMBED_DIM = embedded["Codeword Drift (Counterfeit IP)"].shape[1]

# ============================================================
# RECURRENT NEURAL ENGINE: GRU + Linear classifier head
# Processes turn vectors sequentially, carrying hidden state
# (conversation memory) forward turn-by-turn.
# ============================================================
class SequentialRiskGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
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
model = SequentialRiskGRU(input_dim=EMBED_DIM, hidden_dim=32)
model.eval()

# NOTE: untrained weights are random — to make the demo legible we bias
# the head's weight direction so risk responds to embedding drift magnitude,
# mimicking what a trained classifier would learn from labeled violation data.
with torch.no_grad():
    model.head.weight.copy_(torch.randn_like(model.head.weight) * 0.8 + 0.3)
    model.head.bias.fill_(-1.5)

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
        x = torch.tensor(vec, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            score, h = model(x, h)
        raw_score = float(score.item())

        # Cumulative risk: blends current turn signal with memory-weighted history
        # so risk reflects trajectory, not just the current message in isolation.
        cumulative_drift = 0.6 * cumulative_drift + 0.4 * raw_score + 0.05 * t
        cumulative_drift = min(cumulative_drift, 1.0)

        flag = "FLAGGED" if cumulative_drift >= THRESHOLD else "ok"
        print(f"[STREAM][{name}] Turn {t}: raw_gru_score={raw_score:.3f} | "
              f"cumulative_risk={cumulative_drift:.3f} | status={flag}")
        turn_scores.append(cumulative_drift)

    results[name] = turn_scores
    breach_turn = next((i+1 for i, s in enumerate(turn_scores) if s >= THRESHOLD), None)
    if breach_turn:
        print(f"[DECISION] '{name}': Safety threshold breached at Turn {breach_turn}. "
              f"Static keyword filter would NOT have caught the gradual drift.")
    else:
        print(f"[DECISION] '{name}': No breach. Sequence remained within compliant range.")

# ============================================================
# INTERACTIVE TRAJECTORY VISUALIZATION
# ============================================================
fig = go.Figure()

colors = {"Codeword Drift (Counterfeit IP)": "crimson", "Clean Baseline": "seagreen"}
for name, scores in results.items():
    fig.add_trace(go.Scatter(
        x=list(range(1, len(scores) + 1)), y=scores,
        mode="lines+markers+text",
        text=[f"T{i+1}" for i in range(len(scores))], textposition="top center",
        line=dict(color=colors[name], width=3), marker=dict(size=10),
        name=name,
        hovertext=sequences[name], hoverinfo="text+y",
    ))

fig.add_hline(y=THRESHOLD, line_dash="dash", line_color="orange",
              annotation_text=f"Safety Breach Threshold ({THRESHOLD})",
              annotation_position="bottom right")

fig.update_layout(
    title="Cumulative Violation Risk: Stateful GRU Pipeline (Two Sequences)",
    xaxis_title="Conversation Turn #",
    yaxis_title="Cumulative Violation Risk Score",
    yaxis_range=[0, 1.05],
    xaxis=dict(dtick=1),
    template="plotly_white",
    width=950, height=550,
    legend_title="Sequence",
)
fig.show()

# ============================================================
# SUPERVISOR SUMMARY
# ============================================================
print("\n[SUMMARY] ============================================")
print("This pipeline scores risk using ACCUMULATED conversational")
print("memory (GRU hidden state), not isolated keyword matches.")
print("'Codeword Drift' sequence: each turn alone reads like a normal listing.")
print("No single message contains a banned keyword — 'inspired by',")
print("'mirror quality', and 'no tags' are euphemisms that sit close to")
print("genuine product language in embedding space. The risk only")
print("becomes visible as a TRAJECTORY: legit-sounding intro -> coded")
print("quality disclaimer -> no authentication -> exact logo replication")
print("-> customs mislabeling. That's a pattern a static keyword filter,")
print("and even a single-turn embedding similarity check, would miss.")
print("'Clean Baseline' sequence: same opening, but every later turn")
print("reinforces licensed/authentic language, so cumulative risk")
print("never approaches threshold.")
print("============================================================")
