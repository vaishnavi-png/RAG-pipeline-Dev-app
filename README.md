# AI Detection — StyleDistance experiments

Trying to flag AI-written text vs human text using style embeddings (StyleDistance model)
instead of content. Idea: AI text has a "voice" that's somewhat separable from human voice,
regardless of topic.

Files are numbered in the order I actually ran them, not cleaned up to look linear —
left in the dead ends on purpose so it's clear what didn't work and why.

## Files

- **01_trial_wrong_embedding_model.py** — first attempt, small hand-picked human/AI
  reference sets, cosine similarity to centroids. Used the wrong embedding model setup here,
  kept for reference.
- **02_trial_expanded_testbed.py** — same idea, bigger test set, added 3 "stealth tiers"
  (obvious AI → AI prompted to sound casual → AI paraphrasing real human text). Tier C
  basically breaks the approach, which is the interesting finding — style-only detection
  caps out when AI mimics human sentence structure closely.
- **03_kaggle_dataset_download.py** — pulling in the AIGTxt dataset (human/ChatGPT/mixed,
  scientific domains) from Kaggle instead of relying on hand-written examples.
- **04_dataset_inspection.py** — sanity-checking the dataset (nulls, duplicate columns,
  length distributions, whether "Mixed Text" is actually different from ChatGPT text).
- **05_dataset_reshape.py** — reshaping wide → long format for training.
- **06_final_classifier.py** — actual trained classifier (Logistic Regression + Random
  Forest) on StyleDistance embeddings, with domain-holdout split so it can't cheat by
  learning topic instead of style. Also scores the held-out "mixed" text to see if
  predictions land between human and AI (continuum check).
- **07_export_plot.py** — exports the final plot as html.

## Known limitations (noted in code comments too)

- Reference sets in the early trials are small (10-12 examples) — toy-demo scale.
- Style ≠ authorship. A formal human writer can score "AI-leaning."
- Detection accuracy drops hard on Tier C (AI closely mimicking human phrasing) —
  expected, and arguably the most useful result here.

## Running

Built for Google Colab. Needs `sentence-transformers`, `scikit-learn`, `pandas`, `plotly`,
and (for 03) a Kaggle API token. `06_final_classifier.py` expects `AIGTxt.xlsx` in `/content`.
