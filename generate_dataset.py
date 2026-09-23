"""
generate_dataset.py
--------------------
Builds a synthetic, labeled dataset of job/internship offer messages
(scam vs genuine) by combining sentence-level building blocks in many
random combinations. This gives the ML model hundreds of realistic,
varied examples instead of the ~6 hardcoded regex rules the old
prototype used.

Run:  python generate_dataset.py
Output: data/job_offers.csv  (columns: text, label)   label=1 -> scam, 0 -> genuine
"""

import random
import csv
import itertools

random.seed(42)

COMPANIES = [
    "Acme Technologies Pvt Ltd", "Nova Softwares", "Brightpath Solutions",
    "Zenith Infotech", "Blue Orbit Digital", "Meridian Systems Pvt Ltd",
    "Skyline Consulting LLP", "Vertex Analytics", "Pinnacle Retail Group",
    "Quantum Labs India", "Evergreen Logistics", "Silverline Media",
    "TrueNorth Consulting", "Crestview Technologies", "Orion Data Systems",
]

ROLES = [
    "Software Engineering Intern", "Data Entry Executive", "Content Writing Intern",
    "Customer Support Associate", "Digital Marketing Intern", "HR Trainee",
    "Business Development Associate", "Graphic Design Intern", "Sales Executive",
    "Backend Developer", "Social Media Intern", "Operations Trainee",
]

CITIES = ["Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai", "remote"]

# ---------- Reusable scam-signal fragments ----------
FEE_LINES = [
    "To confirm your seat, pay a refundable registration fee of ₹{amt} via UPI.",
    "A one-time processing fee of ₹{amt} is required before we issue your ID.",
    "Please pay ₹{amt} as a security deposit, fully refundable after 1 month.",
    "An onboarding kit fee of ₹{amt} must be paid via Paytm/GPay to activate your offer.",
    "Training material charges of ₹{amt} apply and must be paid upfront.",
]
URGENCY_LINES = [
    "URGENT REQUIREMENT!! Apply within 24 hours, only {n} seats left!",
    "Hurry! Limited slots available, act now to secure your position.",
    "This offer is valid only for the next {n} hours — apply immediately.",
    "Only {n} seats remaining, immediate joining required.",
]
PAY_LINES = [
    "Earn up to ₹{amt}/day, no experience required.",
    "Guaranteed income of ₹{amt} per day, work just 2 hours!",
    "No experience needed — earn ₹{amt} daily starting day one.",
]
INFO_LINES = [
    "Share your Aadhaar number and bank account details for verification.",
    "Send your OTP to confirm your registration.",
    "Provide your PAN card number and debit card details to proceed.",
    "We need your CVV and bank account number to process your first payment.",
]
CHANNEL_LINES = [
    "Contact only on WhatsApp: +91 9000000000.",
    "For quick processing, reply only on Telegram.",
    "Reach out on our personal Gmail id for further steps.",
    "All further communication will happen on WhatsApp only.",
]
PROCESS_LINES = [
    "You are selected without any interview.",
    "Instant selection — no interview required.",
    "Guaranteed job, no interview or test needed.",
]

# ---------- Reusable genuine-offer fragments ----------
GENUINE_OPENERS = [
    "We are pleased to offer you the position of {role} at {company}.",
    "Congratulations! Following your interview on {date}, we are happy to offer you the role of {role} at {company}.",
    "This letter confirms your selection as {role} at {company}, based in {city}.",
]
GENUINE_BODY = [
    "This is a {dur}-month {kind} position with a stipend of ₹{amt}/month.",
    "Your compensation will be ₹{amt} per month, paid on the last working day of each month.",
    "The role is {kind} and comes with a stipend of ₹{amt}, along with mentorship from our senior engineers.",
]
GENUINE_CLOSERS = [
    "Please find the offer letter attached and reply to hr@{domain} to confirm.",
    "Kindly sign and return the attached offer letter to hr@{domain} within 5 working days.",
    "You can reach our HR team at hr@{domain} for any questions about onboarding.",
]
GENUINE_NOFEE = [
    "No fees are required at any stage of our hiring process.",
    "There is no cost to you at any point during recruitment or onboarding.",
    "We never ask candidates for payment as part of our hiring process.",
]
GENUINE_EXTRA = [
    "Your onboarding documents will be shared by our HR team next week.",
    "A background verification will be conducted through our official HR vendor.",
    "You will receive login credentials to our HR portal after joining.",
    "",
]


def rand_amt(low, high, step=50):
    return random.choice(range(low, high, step))


def domain_from_company(company):
    first = company.split()[0].lower()
    return f"{first}.com"


def make_scam_example():
    n_signals = random.randint(2, 5)
    pool = ["fee", "urgency", "pay", "info", "channel", "process"]
    chosen = random.sample(pool, n_signals)
    parts = []

    role = random.choice(ROLES)
    parts.append(f"Work-from-home {role} position available. " if random.random() < 0.6 else f"{role} opening. ")

    if "urgency" in chosen:
        parts.append(random.choice(URGENCY_LINES).format(n=random.randint(3, 15)))
    if "pay" in chosen:
        parts.append(random.choice(PAY_LINES).format(amt=rand_amt(2000, 9000)))
    if "fee" in chosen:
        parts.append(random.choice(FEE_LINES).format(amt=rand_amt(500, 3000)))
    if "process" in chosen:
        parts.append(random.choice(PROCESS_LINES))
    if "channel" in chosen:
        parts.append(random.choice(CHANNEL_LINES))
    if "info" in chosen:
        parts.append(random.choice(INFO_LINES))

    random.shuffle(parts)
    text = " ".join(parts)
    return text


def make_borderline_scam_example():
    # Only 1 weak signal -> still risky but subtler, keep label 1 (scam-ish/unsafe)
    role = random.choice(ROLES)
    company = random.choice(COMPANIES)
    parts = [f"We're hiring for a {role}, work from home, flexible hours."]
    parts.append(f"Stipend ₹{rand_amt(4000, 12000)}/month.")
    choice = random.choice(["fee", "channel", "process"])
    if choice == "fee":
        parts.append(f"A small ₹{rand_amt(100,500)} verification fee applies for the onboarding kit.")
    elif choice == "channel":
        parts.append("Please reply on WhatsApp for quick processing and share your resume.")
    else:
        parts.append("Fast-track selection, minimal interview process.")
    return " ".join(parts)


def make_genuine_example():
    company = random.choice(COMPANIES)
    role = random.choice(ROLES)
    city = random.choice(CITIES)
    domain = domain_from_company(company)
    date = f"{random.randint(1,28)} September"
    dur = random.choice([3, 6, 12])
    kind = random.choice(["full-time", "internship", "internship (pre-placement offer track)"])
    amt = rand_amt(12000, 60000, 1000)

    parts = [
        random.choice(GENUINE_OPENERS).format(role=role, company=company, date=date, city=city),
        random.choice(GENUINE_BODY).format(dur=dur, kind=kind, amt=amt),
        random.choice(GENUINE_CLOSERS).format(domain=domain),
        random.choice(GENUINE_NOFEE),
        random.choice(GENUINE_EXTRA),
    ]
    return " ".join(p for p in parts if p)


def build_dataset(n_scam=260, n_borderline=140, n_genuine=300):
    rows = []
    for _ in range(n_scam):
        rows.append((make_scam_example(), 1))
    for _ in range(n_borderline):
        rows.append((make_borderline_scam_example(), 1))
    for _ in range(n_genuine):
        rows.append((make_genuine_example(), 0))
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = build_dataset()
    # de-duplicate while preserving variety
    seen = set()
    unique_rows = []
    for text, label in rows:
        if text not in seen:
            seen.add(text)
            unique_rows.append((text, label))

    with open("data/job_offers.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(unique_rows)

    n_scam = sum(1 for _, l in unique_rows if l == 1)
    n_gen = sum(1 for _, l in unique_rows if l == 0)
    print(f"Wrote {len(unique_rows)} rows -> data/job_offers.csv  (scam={n_scam}, genuine={n_gen})")
