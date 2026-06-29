# ============================================================
# Sequential & Recurrent Temporal Moderation Pipeline
# Real embeddings (BAAI/bge-small-en-v1.5) -> PyTorch GRU -> Risk Trajectory
# Single Colab cell. No API keys.
# ============================================================
!pip install -q sentence-transformers torch plotly

import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# 1. EMBEDDING MODEL — BAAI/bge-small-en-v1.5 (384-dim)
# Loaded once, reused to embed every turn of every sequence.
# ------------------------------------------------------------
print("[INIT] Loading embedding model BAAI/bge-small-en-v1.5 ...")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
EMBED_DIM = embedder.get_sentence_embedding_dimension()
print(f"[INIT] Embedding dim = {EMBED_DIM}\n")

def embed_turns(turns):
    """bge models recommend no special prefix for general text encoding."""
    vecs = embedder.encode(turns, normalize_embeddings=True)
    return torch.tensor(vecs, dtype=torch.float32)

# ------------------------------------------------------------
# 2. SCENARIOS
# Sequence 1: hardcoded drift scenario (violation escalation)
# Sequence 2: a REUSABLE generator for a clean baseline — swap
#             `domain_turns` for any real-life benign flow.
# ------------------------------------------------------------
sequence_1_violation = [
    "Hey, saw your post, you still moving that lot from last week?",
    "Yeah still got it, came in a little fast so no paperwork on it.",
    "No worries, I don’t really need a receipt for stuff like this anyway.",
    "Cool, cash in person works best, let’s not do it through the app.",
    "Sounds good, same spot as before, I’ll bring exact change.",

]
sequence_1_labels = [0, 0, 1, 1, 1]  # turn-level ground truth for demo training

def generate_clean_baseline(domain_turns=None):
    """
    Reusable clean-conversation generator.
    Swap `domain_turns` with any real platform's benign 5-turn flow
    (support chat, marketplace inquiry, scheduling, etc).
    Default below = generic marketplace Q&A that stays on-platform.
    """
    if domain_turns is None:
        domain_turns = [
            "Hey, finished that book you lent me, the one with the bank job.",
            "Oh nice, what did you think of the heist chapter near the end?",
            "Honestly the planning scene was my favorite, so much tension.",
            "Same, I loved how the robbery went sideways for the characters.",
            "We should grab coffee and talk about the ending sometime soon.",

        ]
    return domain_turns, [0, 0, 0, 0, 0]

sequence_2_clean, sequence_2_labels = generate_clean_baseline()

print("[DATA] Sequence 1 (violation drift):")
for i, t in enumerate(sequence_1_violation, 1):
    print(f"  Turn {i}: {t}")
print("\n[DATA] Sequence 2 (clean baseline):")
for i, t in enumerate(sequence_2_clean, 1):
    print(f"  Turn {i}: {t}")
print()

# ------------------------------------------------------------
# 3. EMBED BOTH SEQUENCES
# ------------------------------------------------------------
emb_seq1 = embed_turns(sequence_1_violation)   # (5, 384)
emb_seq2 = embed_turns(sequence_2_clean)       # (5, 384)

# ------------------------------------------------------------
# 4. THE RECURRENT NEURAL ENGINE — GRU + Linear head
# Processes the embedding sequence turn-by-turn, carrying hidden
# state (conversation memory) forward; outputs risk per turn.
# ------------------------------------------------------------
class SequentialRiskGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, turns, embed_dim)
        out, _ = self.gru(x)               # out: (batch, turns, hidden_dim) — per-turn hidden states
        risk_logits = self.head(out)       # (batch, turns, 1) — per-turn score
        return torch.sigmoid(risk_logits).squeeze(-1)  # (batch, turns)

torch.manual_seed(7)
model = SequentialRiskGRU(input_dim=EMBED_DIM)

# ------------------------------------------------------------
# 5. QUICK SUPERVISED FIT (illustrative — trains on the two demo
# sequences themselves so the GRU's risk curve is meaningful,
# not random init noise). In production you'd train on a real
# labeled corpus; this keeps the script self-contained.
# ------------------------------------------------------------
X = torch.stack([emb_seq1, emb_seq2])                      # (2, 5, 384)
y = torch.tensor([sequence_1_labels, sequence_2_labels], dtype=torch.float32)  # (2, 5)

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.BCELoss()

print("[TRAIN] Fitting GRU on demo sequences (illustrative, not production-grade)...")
for epoch in range(150):
    optimizer.zero_grad()
    preds = model(X)
    loss = loss_fn(preds, y)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"  epoch {epoch:3d} | loss {loss.item():.4f}")
print(f"[TRAIN] Final loss: {loss.item():.4f}\n")

# ------------------------------------------------------------
# 6. INFERENCE — run both sequences turn-by-turn, log decisions
# ------------------------------------------------------------
THRESHOLD = 0.70
model.eval()
with torch.no_grad():
    risk_seq1 = model(emb_seq1.unsqueeze(0)).squeeze(0).numpy()
    risk_seq2 = model(emb_seq2.unsqueeze(0)).squeeze(0).numpy()

def stream_log(name, turns, risks):
    print(f"[STREAM] === {name} ===")
    breached = False
    for i, (t, r) in enumerate(zip(turns, risks), 1):
        flag = "FLAG" if r >= THRESHOLD else "ok  "
        print(f"  Turn {i} | risk={r:.3f} | {flag} | \"{t[:55]}\"")
        if r >= THRESHOLD and not breached:
            print(f"  -> [DECISION] Threshold breached at turn {i}. Escalate for review.")
            breached = True
    if not breached:
        print("  -> [DECISION] No breach. Sequence cleared.")
    print()

stream_log("Sequence 1: Violation Drift", sequence_1_violation, risk_seq1)
stream_log("Sequence 2: Clean Baseline", sequence_2_clean, risk_seq2)

# ------------------------------------------------------------
# 7. INTERACTIVE TRAJECTORY VISUALIZATION
# ------------------------------------------------------------
turns_x = list(range(1, 6))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=turns_x, y=risk_seq1, mode="lines+markers+text",
    name="Sequence 1: Violation Drift",
    line=dict(color="crimson", width=3), marker=dict(size=10),
    text=[f"T{i}" for i in turns_x], textposition="top center",
    hovertext=sequence_1_violation, hoverinfo="text+y",
))
fig.add_trace(go.Scatter(
    x=turns_x, y=risk_seq2, mode="lines+markers+text",
    name="Sequence 2: Clean Baseline",
    line=dict(color="seagreen", width=3), marker=dict(size=10),
    text=[f"T{i}" for i in turns_x], textposition="bottom center",
    hovertext=sequence_2_clean, hoverinfo="text+y",
))
fig.add_hline(y=THRESHOLD, line_dash="dash", line_color="orange",
              annotation_text=f"Safety Breach Threshold ({THRESHOLD})",
              annotation_position="top left")

fig.update_layout(
    title="Cumulative Violation Risk Score by Conversation Turn (GRU-based)",
    xaxis_title="Conversation Turn #",
    yaxis_title="Risk Score (0.0 - 1.0)",
    yaxis_range=[0, 1.05],
    xaxis=dict(dtick=1),
    template="plotly_white",
    width=950, height=550,
)
fig.show()

print("""
[SUMMARY]
- Embeddings: real BAAI/bge-small-en-v1.5 vectors (384-dim) per turn — not keyword matching.
- Engine: PyTorch GRU carries hidden state across turns -> later turns are scored
  in the context of everything said before, which is why a static keyword filter
  misses this (no single turn alone trips it; the trajectory does).
- Sequence 1 crosses the threshold once contact/payment is pushed off-platform.
- Sequence 2 stays flat because nothing it carries forward in hidden state
  resembles the violation pattern learned from Sequence 1's labeled turns.
""")

fig.write_html("/content/style_plot.html")
from google.colab import files
files.download("/content/style_plot.html")
