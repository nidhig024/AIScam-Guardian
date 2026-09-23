"""
train_model.py
---------------
Trains the ScamShield job-offer risk classifier and saves it to
model/scamshield_model.joblib for the Streamlit app to load.

Pipeline:
  1. TF-IDF (word 1-2 grams + char n-grams) over the raw offer text.
  2. A small set of hand-engineered "signal" features (fee mentions,
     urgency phrasing, unrealistic pay, sensitive-info requests,
     unofficial-only contact channel, no-interview language, caps/!!! ),
     mirroring the six categories from the original rule-based prototype.
  3. Both feature sets are combined and fed into a Logistic Regression
     classifier, giving a P(scam) that is used as the 0-100 risk score
     in the app.

Run:  python train_model.py
"""

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from scamshield_features import SignalFeaturizer

RANDOM_STATE = 42


def build_pipeline():
    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=0.9
    )

    features = FeatureUnion([
        ("word_tfidf", word_tfidf),
        ("char_tfidf", char_tfidf),
        ("signals", SignalFeaturizer()),
    ])

    clf = LogisticRegression(
        C=3.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    )

    pipe = Pipeline([
        ("features", features),
        ("clf", clf),
    ])
    return pipe


def main():
    df = pd.read_csv("data/job_offers.csv")
    X = df["text"].astype(str).values
    y = df["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipe = build_pipeline()

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"5-fold CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print("\nTest set classification report:")
    print(classification_report(y_test, y_pred, target_names=["genuine", "scam"]))
    print("Confusion matrix [rows=true, cols=pred] (order: genuine, scam):")
    print(confusion_matrix(y_test, y_pred))
    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

    # refit on ALL data for the shipped model
    pipe.fit(X, y)
    joblib.dump(pipe, "model/scamshield_model.joblib")

    print("\nSaved trained pipeline -> model/scamshield_model.joblib")


if __name__ == "__main__":
    main()
