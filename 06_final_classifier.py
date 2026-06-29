# ============================================================================
# AI-vs-HUMAN DETECTION — Trained Classifier on StyleDistance Embeddings
# Dataset: AIGTxt (Human-Generated / ChatGPT-Generated / Mixed Text, scientific domains)
# Embedding model: StyleDistance/styledistance (content-independent style space)
# ============================================================================

# ---- Cell 1: Install ----
!pip install -q sentence-transformers scikit-learn pandas plotly openpyxl

# ---- Cell 2: Load + reshape data (assumes you already have AIGTxt.xlsx in /content) ----
import pandas as pd
import numpy as np

raw = pd.read_excel("/content/AIGTxt.xlsx")
raw = raw.drop(columns=[c for c in raw.columns if "Unnamed" in c])
raw["Domain"] = raw["Domain"].str.replace(
    "Materials science and engineering", "Materials Science and Engineering"
)

human_df = raw[["Human-Generated", "Domain"]].rename(columns={"Human-Generated": "text"})
human_df["label"] = "human"
ai_df = raw[["ChatGPT-Generated", "Domain"]].rename(columns={"ChatGPT-Generated": "text"})
ai_df["label"] = "ai"
mixed_df = raw[["Mixed Text", "Domain"]].rename(columns={"Mixed Text": "text"})
mixed_df["label"] = "mixed"

long_df = pd.concat([human_df, ai_df, mixed_df], ignore_index=True)
long_df = long_df.dropna(subset=["text"]).reset_index(drop=True)

# ---- Cell 3: BINARY split — human vs ai only. "mixed" held out separately as a
# confidence/ambiguity test set later (not used in training at all).
binary_df = long_df[long_df["label"].isin(["human", "ai"])].reset_index(drop=True)
mixed_only_df = long_df[long_df["label"] == "mixed"].reset_index(drop=True)

print(f"Binary training pool: {len(binary_df)} rows")
print(binary_df["label"].value_counts())
print(f"\nHeld-out mixed set: {len(mixed_only_df)} rows (not used in training)")

# ---- Cell 4: DOMAIN-HOLDOUT split ----
# This is the critical test for content-independence: if we trained and tested on
# the SAME domains, a classifier could cheat by learning "this text mentions neurons"
# instead of "this text sounds like an LLM." Holding out entire domains for testing
# checks whether the style signal generalizes to topics it has never seen.
all_domains = binary_df["Domain"].unique()
print(f"\nDomains available: {list(all_domains)}")

rng = np.random.RandomState(42)
held_out_domains = rng.choice(all_domains, size=max(2, len(all_domains)//5), replace=False)
print(f"\nHeld-out (test) domains: {list(held_out_domains)}")

train_df = binary_df[~binary_df["Domain"].isin(held_out_domains)].reset_index(drop=True)
test_df = binary_df[binary_df["Domain"].isin(held_out_domains)].reset_index(drop=True)

print(f"\nTrain rows: {len(train_df)} | Test rows (unseen domains): {len(test_df)}")
print(train_df["label"].value_counts())
print(test_df["label"].value_counts())

# ---- Cell 5: Embed with StyleDistance ----
from sentence_transformers import SentenceTransformer

print("\nLoading StyleDistance model...")
model = SentenceTransformer("StyleDistance/styledistance")

def embed_texts(texts, batch_size=64):
    return model.encode(list(texts), batch_size=batch_size, show_progress_bar=True,
                         normalize_embeddings=True)

print("Embedding train set...")
X_train = embed_texts(train_df["text"])
print("Embedding test set...")
X_test = embed_texts(test_df["text"])

y_train = (train_df["label"] == "ai").astype(int).values  # 1 = ai, 0 = human
y_test = (test_df["label"] == "ai").astype(int).values

# ---- Cell 6: Train classifiers ----
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

clf_logreg = LogisticRegression(max_iter=2000, C=1.0)
clf_logreg.fit(X_train, y_train)

clf_rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
clf_rf.fit(X_train, y_train)

def evaluate(clf, name, X, y):
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    cm = confusion_matrix(y, preds)
    print(f"\n--- {name} (evaluated on UNSEEN domains) ---")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}  (of texts flagged AI, how many really were)")
    print(f"Recall:    {rec:.3f}  (of actual AI texts, how many we caught)")
    print(f"F1:        {f1:.3f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]:\n{cm}")
    return {"name": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm, "preds": preds}

results_logreg = evaluate(clf_logreg, "Logistic Regression", X_test, y_test)
results_rf = evaluate(clf_rf, "Random Forest", X_test, y_test)

# ---- Cell 7: Visualize — PCA of test set colored by TRUE label vs PREDICTED label ----
from sklearn.decomposition import PCA
import plotly.express as px

pca = PCA(n_components=2, random_state=42)
test_coords = pca.fit_transform(X_test)

viz_df = test_df.copy()
viz_df["pca_x"] = test_coords[:, 0]
viz_df["pca_y"] = test_coords[:, 1]
viz_df["true_label"] = viz_df["label"]
viz_df["pred_logreg"] = ["ai" if p == 1 else "human" for p in results_logreg["preds"]]
viz_df["correct"] = viz_df["true_label"] == viz_df["pred_logreg"]
viz_df["hover_text"] = viz_df["text"].str.slice(0, 140) + "..."

fig = px.scatter(
    viz_df, x="pca_x", y="pca_y",
    color="true_label", symbol="correct",
    hover_data={"hover_text": True, "Domain": True, "pred_logreg": True,
                "pca_x": False, "pca_y": False},
    title="StyleDistance Embeddings — Held-Out Domains (Logistic Regression predictions)",
    color_discrete_map={"human": "#2ecc71", "ai": "#e74c3c"},
    labels={"pca_x": "PC1", "pca_y": "PC2", "correct": "Correctly classified"},
    width=950, height=650,
)
fig.update_traces(marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")))
fig.update_layout(template="plotly_white")
fig.show()

# ---- Cell 8: The interesting check — score the MIXED-text held-out set ----
# Mixed text was never used in training. If your classifier/embedding space is
# genuinely capturing a style continuum (not just a binary switch), mixed-text
# predicted probabilities should land BETWEEN human and ai, not confidently at
# either extreme. This is a strong qualitative check for your paper.
print("\n--- Scoring held-out MIXED text (never seen in training) ---")
X_mixed = embed_texts(mixed_only_df["text"])
mixed_probs = clf_logreg.predict_proba(X_mixed)[:, 1]  # P(ai)

human_probs = clf_logreg.predict_proba(X_test[y_test == 0])[:, 1]
ai_probs = clf_logreg.predict_proba(X_test[y_test == 1])[:, 1]

prob_df = pd.DataFrame({
    "group": (["Human (test)"] * len(human_probs) +
              ["AI (test)"] * len(ai_probs) +
              ["Mixed (held out)"] * len(mixed_probs)),
    "P(ai)": np.concatenate([human_probs, ai_probs, mixed_probs]),
})

fig_box = px.box(
    prob_df, x="group", y="P(ai)", color="group", points="all",
    title="Predicted P(AI) Distribution — Human vs AI vs Mixed (held-out, unseen by classifier)",
    color_discrete_map={"Human (test)": "#2ecc71", "AI (test)": "#e74c3c", "Mixed (held out)": "#f39c12"},
)
fig_box.update_layout(template="plotly_white", showlegend=False)
fig_box.show()

print(f"\nMean P(ai) — Human: {human_probs.mean():.3f} | AI: {ai_probs.mean():.3f} | Mixed: {mixed_probs.mean():.3f}")
print("If Mixed sits clearly between Human and AI means, that's evidence the style")
print("space captures a genuine continuum rather than an arbitrary binary cutoff —")
print("a good result to report in the paper.")

# ============================================================================
# NOTES FOR THE PAPER:
# 1. Domain-holdout accuracy is the number that matters, not in-domain accuracy.
#    If accuracy drops sharply on held-out domains, the model partially learned
#    topic, not style — worth reporting honestly either way.
# 2. Logistic Regression on top of StyleDistance embeddings is a strong, simple
#    baseline. Compare it against your earlier hand-engineered stylometric
#    features (TTR/burstiness/etc.) classifier as an ablation — does the learned
#    embedding actually beat the hand-crafted features, or is it comparable?
# 3. Mixed-text probability check is your best evidence for "style exists on a
#    continuum, not a switch" — a much more interesting paper claim than plain
#    binary accuracy.
# ============================================================================
