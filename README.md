# RAG-pipeline-Dev-app

Two ongoing experiments in embedding-based content moderation and authenticity detection.

## What this repo is

This explores whether sentence-embedding models can be used to (1) tell AI-generated text
apart from human-written text by *style* rather than topic, and (2) detect a conversation
quietly drifting from legal to illegal intent across multiple turns, rather than flagging
any single message in isolation.

Full write-up: `AI_Detection_and_Legal-Illegal_Conversational_Drift.docx`

---

## Part I — AI Text Detection (`Ai detection analysis code/`)

Standard semantic embedding models (OpenAI, BGE, etc.) group text by topic, not by who
wrote it — a human and an AI both writing about "cement admixtures" land near each other
just because they share vocabulary. This part instead uses **StyleDistance**, a model
trained to embed by writing style, independent of subject matter.

**Findings:**
- Against obviously formulaic AI text (heavy "Furthermore," "Moreover," "Therefore"
  structure), human vs. AI clusters separate cleanly — cosine similarity gap of roughly
  0.81+ (human) vs. below 0.50 (AI).
- Against AI text that's been paraphrased to closely mimic a specific human sample's
  structure and rhythm, separation drops sharply and gets ambiguous (~0.45–0.57 range,
  overlapping).
- **Takeaway:** style-based detection works well against AI's *default* voice, but breaks
  down once AI is explicitly steered to imitate a particular human's structural fingerprint.

## Part II — Legal/Illegal Conversational Drift (`Violation-drift detection test analysis/`)

Tests whether a 5-stage pipeline (embed each turn → score against anchor centroids →
GRU for sequential state-carrying → cumulative running risk score → threshold flag) can
catch a conversation drifting toward a policy violation over several turns, even when no
single message looks suspicious alone.

Two scoring methods were compared:

1. **Anchor-similarity scoring** — compares each turn against hand-built "violation" and
   "clean" reference centroids. Worked reliably once a calibration bug (a turn-count-based
   increment that grew regardless of content) was found and removed. Main limitation:
   needs hand-authored reference phrases per violation category.
2. **Self-Referential Contradiction Scoring (SRCS)** — no external references; instead
   scores how far a conversation's later turns drift from its own opening framing.
   Privacy-preserving by construction, but in its current form it confuses *ordinary topic
   drift* with *adversarial intent* — it flagged a fictional book discussion about a heist
   as riskier than an actual circumvention conversation, because the book discussion moved
   across more topics.

**Takeaway:** the two methods are complementary, not competing. Anchor-similarity is more
deployable today; SRCS points at a real, underused signal (framing vs. substance
contradiction) but needs attribute-level tracking (payment method, channel, documentation
status) rather than whole-sentence comparison to be usable.

---

## Limitations

- All conversational test sequences are hand-constructed for demonstration, not sampled
  from real data — proof of concept, not a validated system.
- The GRU's weights are untrained; the actual risk signal comes from anchor similarity fed
  into it, demonstrating the mechanism rather than learned behavior.
- Both scoring methods were tested on a small number of constructed examples; broader
  evaluation is needed before generalizability claims hold up.

## Future Work

- Train the GRU on real or realistically simulated labeled conversations.
- Extend SRCS to track specific operational attributes instead of whole-turn embeddings.
- Test anchor-similarity scoring across multiple violation categories using policy-derived
  anchor sets.
- Re-run Part I's experiments with a larger, more diverse human/AI text sample.

## References

- StyleDistance model: https://huggingface.co/StyleDistance/styledistance
- BAAI BGE embeddings: https://huggingface.co/BAAI/bge-small-en-v1.5
- Wegmann, Schraagen & Nguyen, *"Same Author or Just Same Topic?"* (NAACL 2022) — https://aclanthology.org/2022.repl4nlp-1.26/
- Patel et al., *"StyleDistance: Stronger Content-Independent Style Embeddings with Synthetic Parallel Examples"* (NAACL 2025)
