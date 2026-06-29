# Violation Detection — Sequential / Recurrent Moderation

Testing whether a stateful model (GRU carrying conversation memory turn-by-turn) can catch
policy violations that drift in slowly across a conversation — the kind static keyword
filters miss because no single message looks bad on its own.

Numbered in the order I actually built them. Earlier files are trials/dead ends kept on
purpose, not cleaned into a fake straight line.

## Files

- **01_dual_moderation_demo.py** — first pass, GRU + vector embeddings, comparing against
  a static-filter baseline.
- **02_trial_wrong_embedding_model.py** — used the wrong embedding model here, kept for
  reference + the html export step.
- **03_gru_styledistance_v1.py** — first real attempt: StyleDistance embeddings into a GRU,
  scenario = counterfeit-goods conversation drifting from "clean" to "violation."
- **04_gru_styledistance_v2_anchored.py** — same architecture, swapped scenario to
  "fencing stolen goods" vs an unrelated clean baseline, and added an anchor-similarity
  signal so the GRU isn't just reacting to a randomly-initialized head (since there's no
  labeled training data here).
- **05_gru_bge_embedding_v1.py** — switched the embedding model to BAAI/bge-small-en-v1.5
  (semantic, not stylistic) to see if content embeddings work better for this than style
  embeddings did.
- **06_gru_bge_self_referential_scoring.py** — adds a second scoring method: instead of
  comparing each turn to an external "what violations look like" reference set, it checks
  whether the conversation's own stated framing (turn 1) stays consistent with what's
  actually being arranged as the conversation goes on. Doesn't need external violation
  examples, which is the privacy upside — plus the html export.

## Notes

- hidden_dim and a few hyperparameters change between versions (32 → 8) — not finalized,
  just experimentation.
- No labeled training data anywhere here — the GRU head is untrained, this is closer to
  a proof-of-concept for the architecture than a working classifier.

## Running

Built for Colab. Needs `sentence-transformers`, `torch`, `plotly`. No API keys required.
