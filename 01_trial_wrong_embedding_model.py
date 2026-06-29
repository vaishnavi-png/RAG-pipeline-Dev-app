# ============================================================
# StyleDistance-based AI-Text Style Flagging Tool
# Paper: Patel et al., NAACL 2025 - "StyleDistance" (HF: StyleDistance/styledistance)
# Repurposed use case: style-distance heuristic for AI vs Human text flagging
# Run top to bottom in Google Colab. GPU optional (CPU works fine for inference).
# ============================================================

# ---- Cell 1: Install ----
!pip install -q sentence-transformers scikit-learn umap-learn plotly pandas

# ---- Cell 2: Imports & Model Load ----
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

print("Loading StyleDistance model (768-dim content-independent style embedder)...")
model = SentenceTransformer("StyleDistance/styledistance")
print("Loaded.")

# ---- Cell 3: Reference corpora ----
# Small seed sets. Swap in larger/your own samples for real use — more references = better centroid.
HUMAN_REFS = [
    "Honestly I just woke up and my coffee tastes weird today, not sure why.",
    "My cat knocked the plant off the shelf again, third time this week lol.",
    "Went for a run this morning, knees are killing me but I felt great after.",
    "I can't believe traffic was that bad, took me an hour to get to work.",
    "Tried this new ramen place downtown, pretty good but kinda pricey ngl.",
    "Spent the weekend fixing my bike, turns out it was just a loose chain.",
    "My neighbor's dog barks at literally everything, even leaves blowing by.",
    "Still haven't finished unpacking from the move, boxes everywhere honestly.",
    "Had a weird dream last night about losing my keys in an airport, woke up stressed.",
    "Finally got around to reading that book everyone recommended, it's decent so far.",
]

AI_REFS = [
    "It is important to note that effective time management can significantly enhance overall productivity.",
    "In conclusion, the aforementioned factors collectively contribute to a comprehensive understanding of the topic.",
    "This article will explore the multifaceted benefits of regular physical activity in detail.",
    "Furthermore, it is essential to consider the broader implications of this technological advancement.",
    "Overall, these findings underscore the significance of sustainable practices in modern society.",
    "It should be noted that proper hydration plays a crucial role in maintaining optimal health.",
    "Additionally, the integration of renewable energy sources presents numerous environmental advantages.",
    "To summarize, the key takeaways from this discussion highlight the value of strategic planning.",
    "Moreover, this approach offers a robust framework for addressing complex organizational challenges.",
    "In today's rapidly evolving landscape, businesses must adapt to remain competitive and relevant.",
]

# ---- Cell 4: Build style centroids ----
def embed(texts):
    return model.encode(texts, normalize_embeddings=True)

human_emb = embed(HUMAN_REFS)
ai_emb = embed(AI_REFS)

human_centroid = human_emb.mean(axis=0, keepdims=True)
ai_centroid = ai_emb.mean(axis=0, keepdims=True)

# ---- Cell 5: Scoring function ----
def style_distance_score(text):
    """Returns dict with similarity to human/ai centroid, verdict, and margin."""
    emb = embed([text])
    sim_human = cosine_similarity(emb, human_centroid)[0][0]
    sim_ai = cosine_similarity(emb, ai_centroid)[0][0]
    margin = sim_ai - sim_human
    verdict = "AI-leaning style" if margin > 0 else "Human-leaning style"
    confidence = abs(margin)
    return {
        "text": text,
        "sim_human": round(float(sim_human), 4),
        "sim_ai": round(float(sim_ai), 4),
        "margin": round(float(margin), 4),
        "verdict": verdict,
        "confidence": round(float(confidence), 4),
        "embedding": emb[0],
    }

# ---- Cell 6: Test bed (your case studies) ----
# Replace/extend with whatever text you want to test
TEST_CASES = [
    "It is worth noting that climate change poses significant risks to global ecosystems and biodiversity.",
    "climate change changes ",
    "Furthermore, the implementation of this strategy will yield substantial long-term benefits for stakeholders.",
    "can't believe i forgot my charger at home, gonna be at 2% battery all day",
    "This comprehensive analysis demonstrates the critical importance of data-driven decision making in contemporary business environments.",
    "lol my sister just texted me the most random meme, dying laughing rn",
]

results = [style_distance_score(t) for t in TEST_CASES]
df = pd.DataFrame(results).drop(columns=["embedding"])
df.insert(0, "id", range(1, len(df)+1))
print(df.to_string(index=False))

# ---- Cell 7: Visualization 1 — Bar chart of AI-lean margin per text ----
fig_bar = px.bar(
    df, x="id", y="margin", color="verdict",
    hover_data=["text", "sim_human", "sim_ai"],
    title="Style-Distance Margin per Text (positive = AI-leaning, negative = Human-leaning)",
    color_discrete_map={"AI-leaning style": "#e74c3c", "Human-leaning style": "#2ecc71"},
    labels={"margin": "AI-lean margin (cos_sim_ai - cos_sim_human)", "id": "Test case #"}
)
fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")
fig_bar.show()

# ---- Cell 8: Visualization 2 — PCA scatter: refs + test cases in style space ----
all_emb = np.vstack([human_emb, ai_emb, np.vstack([r["embedding"] for r in results])])
labels = (["Human ref"] * len(HUMAN_REFS) +
          ["AI ref"] * len(AI_REFS) +
          [f"Test #{i+1}: {r['verdict']}" for i, r in enumerate(results)])
groups = (["Human ref"] * len(HUMAN_REFS) +
          ["AI ref"] * len(AI_REFS) +
          [r["verdict"] for r in results])

pca = PCA(n_components=2)
coords = pca.fit_transform(all_emb)

plot_df = pd.DataFrame({
    "x": coords[:, 0],
    "y": coords[:, 1],
    "group": groups,
    "label": labels,
})

fig_scatter = px.scatter(
    plot_df, x="x", y="y", color="group", hover_name="label",
    title="StyleDistance Embedding Space (PCA projection): Human refs vs AI refs vs Test cases",
    color_discrete_map={
        "Human ref": "#2ecc71", "AI ref": "#e74c3c",
        "AI-leaning style": "#f39c12", "Human-leaning style": "#3498db"
    },
    symbol="group"
)
fig_scatter.update_traces(marker=dict(size=12, line=dict(width=1, color="DarkSlateGrey")))
fig_scatter.show()

# ---- Cell 9: Verdict table (clean flagging output) ----
def flag_table(df):
    out = df.copy()
    out["flag"] = out["margin"].apply(lambda m: "🚩 AI-leaning" if m > 0 else "✅ Human-leaning")
    return out[["id", "text", "sim_human", "sim_ai", "margin", "flag"]]

display_df = flag_table(df)
print("\n=== FINAL FLAGGING OUTPUT ===")
print(display_df.to_string(index=False))

# ============================================================
# NOTES (read before trusting this for anything real):
# 1. This is a heuristic repurposing of StyleDistance, NOT what the paper built it for.
#    Paper's actual use cases: authorship verification, style-transfer eval/steering.
# 2. Centroid quality depends entirely on your reference sets. 10 examples each = toy demo.
#    For anything serious, use 100s of verified human + verified-AI samples per "style"
#    you care about (e.g. essay style, chat style, news style) — styles drift by domain.
# 3. "Margin near 0" = ambiguous, don't over-trust hard verdicts near the boundary.
# 4. Style ≠ AI-origin. A formal human writer will score "AI-leaning" here. This catches
#    style register (e.g. generic LLM-formal-listicle voice), not authorship truth.
# ============================================================
