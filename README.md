# ScamShield AI — ML Prediction Model + Streamlit App

This upgrades your original **ScamShield AI** prototype (which used only
hardcoded regex rules in the browser) into a real **trained machine-learning
classifier** wrapped in a Streamlit app.

## What changed vs. the HTML prototype

| | Original prototype | This version |
|---|---|---|
| Detection | 6 hardcoded regex rules, fixed point weights | TF-IDF (word + char n-grams) + engineered signal features → Logistic Regression, learns from data |
| Score | Sum of rule weights, capped at 100 | Calibrated `P(scam)` from the model, shown as 0–100 |
| Explainability | Which regex matched | Which regex-style signals matched **plus** the text patterns the model actually learned |
| Runs in | Browser JS only | Python / Streamlit (can run locally or be deployed) |

The six original red-flag categories (upfront fee, urgency language, unrealistic
pay, sensitive-info requests, unofficial-only contact channel, no-interview
process) are kept as **engineered features** feeding into the model, so the
category breakdown and highlighted-text explanation you had before still work
— but they now sit alongside a proper statistical text classifier instead of
being the entire system.

## Project structure

```
scamshield/
├── app.py                     # Streamlit app (run this)
├── train_model.py             # Trains and saves the model
├── generate_dataset.py        # Builds the synthetic training dataset
├── scamshield_features.py     # Shared feature-engineering code
├── requirements.txt
├── data/
│   └── job_offers.csv         # ~700 synthetic labeled examples
└── model/
    └── scamshield_model.joblib # Trained pipeline (already included)
```

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The trained model is already included in `model/`, so it runs immediately.
To regenerate the dataset and retrain from scratch:

```bash
python generate_dataset.py   # writes data/job_offers.csv
python train_model.py        # writes model/scamshield_model.joblib
```

`train_model.py` prints cross-validated ROC-AUC and a held-out test report so
you can see the model's performance before shipping it.

## Deploy to Streamlit Community Cloud

1. Push this folder to a **public GitHub repo** (include `data/` and `model/`,
   or add a build step that runs `generate_dataset.py` + `train_model.py`).
2. Go to **share.streamlit.io** → "New app" → connect your repo.
3. Set the main file path to `app.py`.
4. Deploy. Streamlit Cloud will install everything from `requirements.txt`
   automatically.

## About the training data — please read

The `data/job_offers.csv` dataset is **synthetically generated** (see
`generate_dataset.py`) by combining realistic sentence templates for scam vs.
genuine job offers in many random combinations (~700 rows). This lets the
model learn generalizable *patterns* (phrasing around fees, urgency, pay
claims, personal info requests, etc.) rather than memorizing 6 fixed regexes.

It is **not** real-world labeled data scraped from actual scam reports, so
before treating this as production-ready you should:

- Swap in or add real labeled examples if you have access to any (e.g. from
  a placement cell's reported scam messages, with personal details removed).
- Re-run `train_model.py` and check the classification report on a held-out
  set of *real* messages, not just the synthetic test split.
- Keep treating the output as a **risk assessment aid**, not a final verdict
  — the app's own footer says this, and it should stay true regardless of
  how good the model's offline metrics look.

## Extending it

- **More data = better generalization.** The biggest lever here is adding
  more diverse real or synthetic examples to `data/job_offers.csv`.
- **Try a different classifier**: swap `LogisticRegression` in
  `train_model.py` for `RandomForestClassifier` or a small gradient-boosted
  model (e.g. `xgboost`) — the rest of the pipeline (TF-IDF + signal
  features) stays the same.
- **Multilingual offers**: if you expect Hindi/Hinglish scam text, add
  translated/transliterated examples to the training data; TF-IDF char
  n-grams already help somewhat with mixed-script text.
