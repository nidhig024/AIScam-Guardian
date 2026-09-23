"""
scamshield_features.py
-----------------------
Shared feature-engineering code used by BOTH train_model.py and app.py.
Kept in its own module (rather than inside train_model.py) so the
saved joblib pipeline can be unpickled correctly wherever it's loaded.

Defines the six red-flag signal categories (same ones the original
rule-based prototype used) as regex patterns, a function to turn raw
text into a numeric signal-feature matrix, and an sklearn-compatible
transformer wrapping that function.
"""

import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

SIGNAL_PATTERNS = {
    "fee": r"(registration|processing|security|training|verification|onboarding)\s*fee|refundable deposit|pay(?:ment)?\s*(?:of\s*)?(?:₹|rs\.?|inr)\s*\d|via\s*(upi|paytm|gpay)",
    "urgency": r"urgent(ly)?\s*requirement|act now|limited (slots|seats|time)|hurry|only\s*\d+\s*(slots|seats)|apply within\s*\d+\s*hours?|immediate joining",
    "pay": r"(₹|rs\.?|inr)\s?\d{3,6}\s?(per day|\/day|daily)|earn (up ?to)\s*(₹|rs\.?)?\s*\d{4,}.{0,15}(day|daily)|no experience.{0,20}(high salary|lakh)",
    "info": r"bank (account|details)|otp|aadhaar|aadhar|pan card\b|cvv|debit card|credit card number",
    "channel": r"whatsapp\s*only|telegram\s*only|contact.{0,12}(whatsapp|telegram)\s*(only)?|personal\s*(gmail|email)",
    "process": r"no interview|without\s*(any\s*)?interview|selected without|instant selection|guaranteed job",
}
COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in SIGNAL_PATTERNS.items()}

SIGNAL_LABELS = {
    "fee": "Upfront payment or fee request",
    "urgency": "Pressure / urgency language",
    "pay": "Unrealistic pay for the effort",
    "info": "Requests for sensitive personal/financial info",
    "channel": "Unofficial-only contact channel",
    "process": "No interview / instant selection",
}

FEATURE_NAMES = list(SIGNAL_PATTERNS.keys()) + ["exclamations", "caps_words", "length"]


def extract_signal_features(texts):
    """Raw text list -> (n_samples, n_features) numeric matrix."""
    rows = []
    for t in texts:
        feats = [1.0 if COMPILED[k].search(t) else 0.0 for k in SIGNAL_PATTERNS]
        exclam = len(re.findall(r"!", t))
        caps_words = len(re.findall(r"\b[A-Z]{4,}\b", t))
        feats.append(min(exclam, 10) / 10.0)
        feats.append(min(caps_words, 5) / 5.0)
        feats.append(min(len(t), 1000) / 1000.0)
        rows.append(feats)
    return np.array(rows)


def matched_signals(text):
    """Returns list of (key, label, matched_substrings) for signals found in text."""
    out = []
    for key, pattern in COMPILED.items():
        matches = pattern.findall(text)
        if matches:
            found = pattern.search(text)
            out.append((key, SIGNAL_LABELS[key], found.group(0)))
    return out


class SignalFeaturizer(BaseEstimator, TransformerMixin):
    """sklearn-compatible transformer wrapping extract_signal_features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return extract_signal_features(X)
