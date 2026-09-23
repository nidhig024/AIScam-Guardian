"""
ScamShield AI — Streamlit app
------------------------------
Paste a job/internship offer and get a machine-learning-based risk
score (not just hardcoded regex rules), a category breakdown, the
specific phrases that drove the score, and next-step guidance.

Run locally:   streamlit run app.py
Deploy:        push this folder to GitHub, then deploy on
                https://share.streamlit.io  (Streamlit Community Cloud)
"""

import re
import html
import datetime
import joblib
import streamlit as st

from scamshield_features import matched_signals, SIGNAL_LABELS

MODEL_PATH = "model/scamshield_model.joblib"

st.set_page_config(page_title="ScamShield AI", page_icon="🛡️", layout="centered")

# ---------------------------------------------------------------- helpers

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def extract_company(text):
    m = re.search(
        r"\bat\s+([A-Z][A-Za-z&.,'-]*(?:\s+[A-Z][A-Za-z&.,'-]*){0,4}\s?"
        r"(?:Pvt\.?\s?Ltd\.?|Private Limited|LLP|Technologies|Solutions|Ltd\.?|Inc\.?))",
        text,
    )
    return m.group(1).strip() if m else ""


def build_verify_links(company):
    from urllib.parse import quote_plus
    q = quote_plus(company)
    return [
        ("Search for the official company website", f"https://www.google.com/search?q={q}+official+website"),
        ("Find the company on LinkedIn", f"https://www.linkedin.com/search/results/companies/?keywords={q}"),
        ("Check employee reviews (Glassdoor / AmbitionBox)", f"https://www.google.com/search?q={q}+reviews+glassdoor+OR+ambitionbox"),
        ("Check company registration (MCA India)", f"https://www.google.com/search?q={q}+company+registration+mca.gov.in"),
    ]


NEXT_STEPS = {
    "low": [
        "Confirm the sender's email domain matches the company's official website (not a free @gmail/@yahoo address).",
        "Look up the recruiter on LinkedIn and confirm they actually work at that company.",
        "Never pay any fee — even a 'refundable' one — before you have a signed offer letter.",
        "Keep a copy of every message until your joining is fully confirmed.",
    ],
    "medium": [
        "Ask for the offer in writing, on official company letterhead or from a verified company email.",
        "Request a proper interview (call or video) before sharing any further details.",
        "Search the company name plus 'reviews' or 'scam' before responding further.",
        "Do not share ID numbers, bank details, or pay any amount at this stage.",
        "Run it past a placement officer, mentor, or trusted senior before proceeding.",
    ],
    "high": [
        "Stop responding immediately — do not pay, click links, or share any information.",
        "Take screenshots of the offer, sender details, and any payment request as evidence.",
        "Block the sender's number, email, or account.",
        "Report it at cybercrime.gov.in (India's National Cyber Crime Reporting Portal) or call the helpline 1930.",
        "Warn classmates or your placement cell so others don't fall for the same offer.",
    ],
}

EXAMPLES = {
    "🔴 Suspicious example": (
        "URGENT REQUIREMENT!! Work-from-home data entry job. Earn ₹5000 per day, no experience "
        "required. Only 10 seats left, apply within 24 hours! To confirm your seat, pay a refundable "
        "registration fee of ₹1500 via UPI. You are selected without interview. Contact only on "
        "WhatsApp: +91 90000 00000. Share your Aadhaar number and bank account details for verification."
    ),
    "🟠 Borderline example": (
        "We're hiring for a Content Writing Intern, work from home, flexible hours. Stipend "
        "₹8,000/month. Please reply on WhatsApp for quick processing and share your resume. A small "
        "₹200 verification fee applies for onboarding kit."
    ),
    "🟢 Genuine example": (
        "We are pleased to offer you the position of Software Engineering Intern at Acme Technologies "
        "Pvt Ltd. This 6-month internship follows your interview on 14th September. Stipend: "
        "₹25,000/month. Please find the offer letter attached and reply to hr@acmetech.com to confirm. "
        "No fees are required at any stage of our hiring process."
    ),
}


def highlight_text(text, hits):
    escaped = html.escape(text)
    for _, _, snippet in hits:
        esc_snip = re.escape(html.escape(snippet))
        escaped = re.sub(esc_snip, lambda m: f"<mark>{m.group(0)}</mark>", escaped, flags=re.IGNORECASE)
    return escaped


def risk_bucket(score):
    if score < 25:
        return "low", "🟢 Low risk", "#2f7a4e"
    if score < 55:
        return "medium", "🟠 Medium risk", "#8a5f08"
    return "high", "🔴 High risk", "#a1362f"


# ---------------------------------------------------------------- state

if "history" not in st.session_state:
    st.session_state.history = []
if "offer_text" not in st.session_state:
    st.session_state.offer_text = ""

# ---------------------------------------------------------------- header

st.title("🛡️ ScamShield AI")
st.caption(
    "Paste a job or internship message. A trained ML model scores it against learned "
    "scam patterns and explains exactly what it found — nothing is uploaded or stored."
)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found at `model/scamshield_model.joblib`. "
        "Run `python train_model.py` first to train and save the model."
    )
    st.stop()

# ---------------------------------------------------------------- input

offer_text = st.text_area(
    "Job / internship offer text",
    key="offer_text",
    height=180,
    placeholder="Paste the message, email, or job posting here...",
)

company_input = st.text_input(
    "Company name (optional — sharpens the verification links below)",
    placeholder="e.g. Acme Technologies Pvt Ltd",
)

cols = st.columns(3)
for i, (label, sample) in enumerate(EXAMPLES.items()):
    if cols[i].button(label, use_container_width=True):
        st.session_state.offer_text = sample
        st.rerun()

analyze = st.button("Analyze offer", type="primary")

# ---------------------------------------------------------------- analysis

if analyze:
    text = st.session_state.offer_text.strip()
    if not text:
        st.warning("Please paste some offer text first.")
    else:
        with st.spinner("Scanning for red flags…"):
            proba = model.predict_proba([text])[0, 1]
        score = int(round(proba * 100))
        bucket, level_label, color = risk_bucket(score)
        hits = matched_signals(text)
        company = company_input.strip() or extract_company(text)

        result = {
            "text": text, "score": score, "bucket": bucket, "level_label": level_label,
            "hits": hits, "company": company, "timestamp": datetime.datetime.now(),
        }
        st.session_state.history.insert(0, result)
        st.session_state.history = st.session_state.history[:6]
        st.session_state["last_result"] = result

# ---------------------------------------------------------------- results

result = st.session_state.get("last_result")
if result:
    st.divider()
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Risk score", f"{result['score']} / 100")
    with r2:
        st.markdown(f"### {result['level_label']}")
        st.progress(result["score"] / 100)
    st.caption(f"Based on {len(result['hits'])} signal(s) detected by the model.")

    advice = {
        "low": "No strong scam indicators detected — but still verify the company's official domain and never pay any fee before you're formally hired.",
        "medium": "Some suspicious signals found. Verify the company independently (official website, LinkedIn, employee reviews) before sharing information or money.",
        "high": "Multiple strong scam indicators found. Do not pay any fee, share OTPs/bank/ID details, or proceed further. Report and block this offer.",
    }[result["bucket"]]
    st.info(advice)

    st.markdown("#### Signal category breakdown")
    for key, label in SIGNAL_LABELS.items():
        matched = any(h[0] == key for h in result["hits"])
        st.progress(1.0 if matched else 0.0, text=f"{label} — {'flagged' if matched else 'not detected'}")

    st.markdown("#### Flagged patterns")
    if result["hits"]:
        for key, label, snippet in result["hits"]:
            st.markdown(f"- **{label}** — matched: _\"{snippet}\"_")
    else:
        st.markdown("- No predefined red-flag patterns matched this text (score is driven by overall wording).")

    st.markdown("#### Highlighted offer text")
    st.markdown(
        f'<div style="white-space:pre-wrap;background:#f0e6d2;border:1px solid #e2d3b3;'
        f'border-radius:10px;padding:14px;font-size:0.9rem;">{highlight_text(result["text"], result["hits"])}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### What to do next")
    for i, step in enumerate(NEXT_STEPS[result["bucket"]], 1):
        st.markdown(f"{i}. {step}")

    st.markdown("#### Verify the company")
    if result["company"]:
        st.caption(f'Quick checks for "{result["company"]}":')
        for label, url in build_verify_links(result["company"]):
            st.markdown(f"- [{label}]({url})")
    else:
        st.caption("No company name detected. Enter one above and re-analyze for direct verification links.")

    report = (
        f"ScamShield AI — Risk Report\n\nRisk score: {result['score']}/100 ({result['level_label']})\n\n"
        f"Detected signals:\n" + ("\n".join(f"- {l}: matched \"{s}\"" for _, l, s in result["hits"]) or "- None")
        + f"\n\nRecommendation:\n{advice}\n\nWhat to do next:\n"
        + "\n".join(f"{i}. {s}" for i, s in enumerate(NEXT_STEPS[result['bucket']], 1))
        + (f"\n\nCompany to verify: {result['company']}" if result["company"] else "")
        + f"\n\nOffer text analyzed:\n{result['text']}"
    )
    st.download_button("⬇️ Download report (.txt)", report, file_name="scamshield-report.txt")

# ---------------------------------------------------------------- history

if st.session_state.history:
    st.divider()
    st.markdown("#### Session history")
    for i, h in enumerate(st.session_state.history):
        snippet = h["text"][:60].replace("\n", " ")
        if st.button(f"{h['level_label']} ({h['score']}) — {snippet}…", key=f"hist_{i}"):
            st.session_state["last_result"] = h
            st.session_state.offer_text = h["text"]
            st.rerun()

st.divider()
st.caption(
    "ML-based prototype (TF-IDF + Logistic Regression). Output is a risk assessment, not a final verdict. "
    "Always independently verify any job offer before sharing information or paying money."
)
