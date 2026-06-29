# =====================================================================================
# STYLEDISTANCE AI-FLAGGING — EXPANDED TESTBED
# Same pipeline structure as before. Changes: bigger sample size, 3 deliberate AI
# "stealth tiers", multiple test posts per tier, and accuracy/confusion-matrix metrics.
# =====================================================================================

!pip install -q sentence-transformers scikit-learn plotly

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

print("Loading StyleDistance model...")
model = SentenceTransformer("StyleDistance/styledistance")
print("Model loaded ✓")

# =====================================================================================
# REFERENCE SET — 12 human + 12 AI across 2 topics. AI references are Tier A:
# obvious, unprompted "assistant voice" — defines what the AI centroid represents.
# =====================================================================================

reference_data = [
    # ---- HUMAN: Concrete ----
    {"text": "So we tried cutting the cement content last week and honestly the cube strength barely budged, which kinda surprised me ngl.", "label": "Human", "topic": "Concrete"},
    {"text": "Cube results came back ok-ish, not great. We're probably over-designing the M30 mix tbh, nobody wants to touch it because of compliance audits, ugh.", "label": "Human", "topic": "Concrete"},
    {"text": "Honestly the CUSUM chart looked weird this month, spike around day 12 then it flattened out. Could be the new GGBS batch, hard to say without more data.", "label": "Human", "topic": "Concrete"},
    {"text": "We swapped the admixture supplier mid-month and slump went up like 30mm out of nowhere, took us forever to figure out why.", "label": "Human", "topic": "Concrete"},
    {"text": "ngl curing temps in summer are a nightmare here, strength numbers swing way more than they should given how tight the mix design is on paper.", "label": "Human", "topic": "Concrete"},
    {"text": "Fly ash batch 4 was clearly off-spec, you could tell just from how the mix flowed differently on site even before the cube tests came back.", "label": "Human", "topic": "Concrete"},
    # ---- HUMAN: Stock market ----
    {"text": "Honestly the RSI's been screaming overbought for like 3 days now but price just keeps grinding higher, so I'm not touching shorts yet.", "label": "Human", "topic": "Stock market"},
    {"text": "Volume's been kinda weak on this breakout ngl, which worries me a bit. Could be a fakeout, we've seen this fail twice this month already.", "label": "Human", "topic": "Stock market"},
    {"text": "Chart's messy rn tbh. MACD crossed bullish but the candles look indecisive, lots of wicks, I'd wait for confirmation honestly.", "label": "Human", "topic": "Stock market"},
    {"text": "I simply use them for sizing probable outcomes, no one could've predicted the timing of this pullback tbh.", "label": "Human", "topic": "Stock market"},
    {"text": "In my opinion you only need two levels of info, what's public and insider stuff, and most of us only ever get the first one.", "label": "Human", "topic": "Stock market"},
    {"text": "Bought the dip too early again lol, classic mistake, should've waited for the actual reversal candle instead of guessing.", "label": "Human", "topic": "Stock market"},

    # ---- AI Tier A (obvious assistant voice): Concrete ----
    {"text": "Optimizing concrete mix design requires a systematic approach. Furthermore, the water-cement ratio must be carefully controlled. Additionally, supplementary cementitious materials such as fly ash can enhance performance.", "label": "AI", "topic": "Concrete"},
    {"text": "Effective admixture optimization involves several key steps. Firstly, dosage rates should be calibrated based on ambient temperature. Additionally, compatibility testing is essential. Furthermore, monitoring slump retention ensures consistent quality.", "label": "AI", "topic": "Concrete"},
    {"text": "To achieve optimal concrete performance, engineers must consider multiple variables. These include: 1) Mix proportioning, 2) Material substitution, 3) Curing conditions. Therefore, a holistic approach yields the best outcomes.", "label": "AI", "topic": "Concrete"},
    {"text": "Concrete strength optimization involves several critical considerations. Firstly, cementitious content should be evaluated. Moreover, consistent monitoring ensures quality assurance standards are met.", "label": "AI", "topic": "Concrete"},
    {"text": "Several factors influence curing efficiency. Firstly, temperature control is essential. Additionally, moisture retention plays a significant role. Therefore, a structured curing protocol is recommended.", "label": "AI", "topic": "Concrete"},
    {"text": "When evaluating admixture performance, multiple metrics should be considered. Furthermore, dosage consistency across batches is critical. Therefore, rigorous quality control procedures are necessary.", "label": "AI", "topic": "Concrete"},
    # ---- AI Tier A: Stock market ----
    {"text": "Technical analysis of the stock reveals several key indicators. Firstly, the RSI suggests overbought conditions. Additionally, the MACD indicator shows bullish divergence. Moreover, volume analysis confirms trend strength.", "label": "AI", "topic": "Stock market"},
    {"text": "Stock market technical analysis involves evaluating multiple indicators. These include: 1) RSI, 2) MACD, 3) Volume trends. Therefore, a comprehensive evaluation provides actionable insights.", "label": "AI", "topic": "Stock market"},
    {"text": "When conducting technical analysis, traders should consider the following: support levels, momentum indicators, volume confirmation. Furthermore, combining these metrics enhances reliability.", "label": "AI", "topic": "Stock market"},
    {"text": "Effective portfolio management requires several considerations. Firstly, diversification across asset classes reduces risk. Additionally, periodic rebalancing maintains target allocations. Therefore, a disciplined strategy is essential.", "label": "AI", "topic": "Stock market"},
    {"text": "Several factors influence market timing decisions. Firstly, macroeconomic indicators provide context. Moreover, sentiment analysis offers additional insight. Therefore, a multi-factor approach is recommended.", "label": "AI", "topic": "Stock market"},
    {"text": "Risk management in trading involves multiple strategies. Firstly, position sizing limits exposure. Additionally, stop-loss orders mitigate downside risk. Therefore, a comprehensive risk framework is necessary.", "label": "AI", "topic": "Stock market"},
]
ref_df = pd.DataFrame(reference_data)

# =====================================================================================
# TEST SET — 3 stealth tiers x multiple posts each, with ground-truth labels so we can
# score accuracy automatically instead of eyeballing single examples.
# =====================================================================================

test_data = [
    # ---- Tier A: obvious AI voice (easy case) ----
    {"text": "Furthermore, optimizing the curing process requires careful temperature regulation. Additionally, humidity control plays a significant role in long-term durability.", "true_label": "AI", "tier": "A - Obvious AI voice"},
    {"text": "Several key factors influence trading performance. Firstly, discipline is essential. Moreover, risk management cannot be overlooked. Therefore, a structured plan is recommended.", "true_label": "AI", "tier": "A - Obvious AI voice"},
    {"text": "Honestly we had no idea why the slump dropped that much, took us half the day just messing with water ratios to figure it out.", "true_label": "Human", "tier": "A - Obvious AI voice"},
    {"text": "ngl this trade was pure luck, I got out way too early and missed half the move, classic me tbh.", "true_label": "Human", "tier": "A - Obvious AI voice"},

    # ---- Tier B: AI prompted to "sound casual" (middle ground) ----
    {"text": "Honestly, the slump test results were kind of surprising — we tweaked the water ratio a bit and strength held up better than expected, which was a nice change.", "true_label": "AI", "tier": "B - AI prompted casual"},
    {"text": "I gotta say, this trade caught me off guard a little. RSI was flashing overbought but price just kept climbing, so I ended up holding longer than planned.", "true_label": "AI", "tier": "B - AI prompted casual"},
    {"text": "we adjusted the fly ash ratio on a whim honestly and somehow it worked out better than the planned mix, still not sure why", "true_label": "Human", "tier": "B - AI prompted casual"},
    {"text": "kept second guessing the breakout the whole time, volume looked thin so I waited, turned out to be the right call for once", "true_label": "Human", "tier": "B - AI prompted casual"},

    # ---- Tier C: AI paraphrasing real human text sentence-by-sentence (hard case) ----
    {"text": "I mainly use them to size up probable outcomes rather than predict exact moves. Nobody could have called the timing or magnitude of this pullback.", "true_label": "AI", "tier": "C - AI paraphrase mimicry"},
    {"text": "In my view, there are really only two layers of information that matter: a general sense of what's happening externally, and insider knowledge.", "true_label": "AI", "tier": "C - AI paraphrase mimicry"},
    {"text": "I simply use them for sizing probable outcomes. No one could have predicted the timing and magnitude of this pullback.", "true_label": "Human", "tier": "C - AI paraphrase mimicry"},
    {"text": "In my opinion, you only need two levels of information: a broad understanding of what is going on outside and insider information.", "true_label": "Human", "tier": "C - AI paraphrase mimicry"},
]
test_df = pd.DataFrame(test_data)

print(f"\nReference set: {len(ref_df)} examples ({(ref_df['label']=='Human').sum()} human, {(ref_df['label']=='AI').sum()} AI)")
print(f"Test set: {len(test_df)} examples across 3 stealth tiers\n")

# =====================================================================================
# EMBEDDING — encode reference set once (forcing float32 to avoid the bfloat16
# TypeError seen earlier), build human/AI centroids
# =====================================================================================

ref_embeddings = model.encode(ref_df["text"].tolist(), convert_to_tensor=True).float()
human_mask = torch.tensor((ref_df["label"] == "Human").values)
ai_mask = torch.tensor((ref_df["label"] == "AI").values)

human_centroid = ref_embeddings[human_mask].mean(dim=0)
ai_centroid = ref_embeddings[ai_mask].mean(dim=0)

print(f"Built centroids from {human_mask.sum().item()} human refs and {ai_mask.sum().item()} AI refs ✓\n")

# =====================================================================================
# RUN EVERY TEST POST AGAINST BOTH CENTROIDS, RECORD PREDICTION
# =====================================================================================

results = []
for _, row in test_df.iterrows():
    emb = model.encode(row["text"], convert_to_tensor=True).float()
    sim_human = cos_sim(emb, human_centroid).item()
    sim_ai = cos_sim(emb, ai_centroid).item()
    predicted = "AI" if sim_ai > sim_human else "Human"
    results.append({
        "tier": row["tier"],
        "true_label": row["true_label"],
        "predicted_label": predicted,
        "sim_human_centroid": round(sim_human, 3),
        "sim_ai_centroid": round(sim_ai, 3),
        "margin": round(sim_ai - sim_human, 3),
        "correct": predicted == row["true_label"],
        "snippet": row["text"][:70] + "...",
    })

results_df = pd.DataFrame(results)

# =====================================================================================
# METRICS — overall + per-tier accuracy, plus a confusion matrix
# =====================================================================================

print("=" * 100)
print("PER-EXAMPLE RESULTS")
print("=" * 100)
print(results_df[["tier", "true_label", "predicted_label", "sim_human_centroid", "sim_ai_centroid", "margin", "correct"]].to_string(index=False))

print("\n" + "=" * 100)
print("ACCURACY BY TIER")
print("=" * 100)
tier_accuracy = results_df.groupby("tier")["correct"].agg(["sum", "count"])
tier_accuracy["accuracy"] = (tier_accuracy["sum"] / tier_accuracy["count"] * 100).round(1)
tier_accuracy.columns = ["correct", "total", "accuracy_%"]
print(tier_accuracy.to_string())

overall_acc = results_df["correct"].mean() * 100
print(f"\nOVERALL ACCURACY: {overall_acc:.1f}% ({results_df['correct'].sum()}/{len(results_df)})")

print("\n" + "=" * 100)
print("CONFUSION MATRIX (rows = true label, columns = predicted label)")
print("=" * 100)
confusion = pd.crosstab(results_df["true_label"], results_df["predicted_label"], rownames=["True"], colnames=["Predicted"])
print(confusion.to_string())

print("\n" + "=" * 100)
print("INTERPRETATION NOTE")
print("=" * 100)
print("Tier A (obvious AI voice)      -> expect high accuracy, large similarity margins")
print("Tier B (AI prompted casual)    -> expect moderate accuracy, smaller margins")
print("Tier C (AI paraphrase mimicry) -> expect accuracy near chance level (~50%),")
print("                                  margins close to zero -- this is the known")
print("                                  limit of style-based detection: when AI text")
print("                                  closely mimics human sentence structure and")
print("                                  rhythm, style signal alone is insufficient.")
