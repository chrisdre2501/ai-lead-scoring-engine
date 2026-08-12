"""
Synthetic lead/customer data generator for Halcyon Collectibles,
a fictional premium collectibles e-commerce brand (AOV ~$385).

Generates ~1,000 leads drawn from six realistic personas. The personas are
the "hidden patterns" for the scoring and segmentation engine to surface:

  1. Whale Collector   — high LTV, frequent purchases, deep engagement
  2. At-Risk VIP       — high LTV historically, but activity has gone quiet
  3. Engaged Prospect  — no purchase yet, but strong buying signals
                          (pricing-page visits, cart abandons, wishlist)
  4. Window Shopper    — browses and opens emails, near-zero intent signals
  5. Lapsed Buyer      — purchased before, inactive for months
  6. New Signup        — joined recently, thin data, unproven

Each lead gets realistic, persona-consistent behavioral signals with noise,
so segment boundaries are fuzzy — like real CRM data.

Usage:
    python generate_leads.py
Output:
    data/leads.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(7)

TODAY = pd.Timestamp("2026-08-01")

FIRST = ["Alex", "Jordan", "Sam", "Casey", "Morgan", "Riley", "Taylor", "Devon",
         "Quinn", "Avery", "Hayden", "Rowan", "Skyler", "Reese", "Emerson",
         "Marcus", "Elena", "Priya", "Diego", "Naomi", "Kenji", "Sofia", "Omar",
         "Ingrid", "Andre", "Lucia", "Felix", "Amara", "Victor", "Zara"]
LAST = ["Reyes", "Chen", "Okafor", "Novak", "Silva", "Kim", "Haddad", "Ito",
        "Kowalski", "Mbeki", "Larsen", "Torres", "Nguyen", "Petrov", "Rossi",
        "Anders", "Fuentes", "Osei", "Lindqvist", "Barros", "Tanaka", "Meyer"]

SOURCES = ["Organic Search", "Paid Social", "Email Referral", "Convention Booth",
           "Influencer", "Direct"]
SOURCE_P = [0.28, 0.24, 0.14, 0.12, 0.12, 0.10]

PERSONAS = {
    "Whale Collector": 0.07,
    "At-Risk VIP": 0.06,
    "Engaged Prospect": 0.16,
    "Window Shopper": 0.30,
    "Lapsed Buyer": 0.18,
    "New Signup": 0.23,
}


def clip(v, lo, hi):
    return float(np.clip(v, lo, hi))


def make_lead(i: int, persona: str) -> dict:
    """Draw persona-consistent signals with noise."""
    name = f"{RNG.choice(FIRST)} {RNG.choice(LAST)}"

    if persona == "Whale Collector":
        tenure = RNG.integers(400, 1400)
        purchases = int(RNG.integers(6, 22))
        ltv = purchases * RNG.normal(520, 90)          # whales buy premium
        recency = int(RNG.integers(1, 21))
        opens = int(clip(RNG.normal(14, 4), 4, 30))
        clicks = int(clip(RNG.normal(6, 2), 1, 15))
        visits = int(clip(RNG.normal(12, 4), 3, 30))
        pricing = int(clip(RNG.normal(4, 2), 0, 12))
        carts = int(clip(RNG.normal(1.2, 1), 0, 5))
        wishlist = int(clip(RNG.normal(7, 3), 1, 20))

    elif persona == "At-Risk VIP":
        tenure = RNG.integers(500, 1500)
        purchases = int(RNG.integers(5, 15))
        ltv = purchases * RNG.normal(480, 80)
        recency = int(RNG.integers(45, 120))            # gone quiet
        opens = int(clip(RNG.normal(2, 1.5), 0, 6))
        clicks = int(clip(RNG.normal(0.5, 0.7), 0, 3))
        visits = int(clip(RNG.normal(1, 1), 0, 4))
        pricing = 0
        carts = 0
        wishlist = int(clip(RNG.normal(5, 3), 0, 15))   # old wishlist lingers

    elif persona == "Engaged Prospect":
        tenure = RNG.integers(20, 200)
        purchases = 0
        ltv = 0.0
        recency = int(RNG.integers(0, 10))
        opens = int(clip(RNG.normal(10, 3), 3, 25))
        clicks = int(clip(RNG.normal(5, 2), 1, 12))
        visits = int(clip(RNG.normal(9, 3), 3, 25))
        pricing = int(clip(RNG.normal(3.5, 1.5), 1, 10))
        carts = int(clip(RNG.normal(1.8, 1), 0, 5))
        wishlist = int(clip(RNG.normal(4, 2), 0, 12))

    elif persona == "Window Shopper":
        tenure = RNG.integers(30, 700)
        purchases = 0
        ltv = 0.0
        recency = int(RNG.integers(3, 45))
        opens = int(clip(RNG.normal(6, 3), 0, 18))
        clicks = int(clip(RNG.normal(1.5, 1.2), 0, 6))
        visits = int(clip(RNG.normal(3, 2), 0, 10))
        pricing = int(clip(RNG.normal(0.3, 0.6), 0, 2))
        carts = 0
        wishlist = int(clip(RNG.normal(1, 1.2), 0, 5))

    elif persona == "Lapsed Buyer":
        tenure = RNG.integers(300, 1200)
        purchases = int(RNG.integers(1, 4))
        ltv = purchases * RNG.normal(360, 70)
        recency = int(RNG.integers(90, 360))
        opens = int(clip(RNG.normal(1, 1), 0, 4))
        clicks = 0
        visits = int(clip(RNG.normal(0.4, 0.7), 0, 2))
        pricing = 0
        carts = 0
        wishlist = int(clip(RNG.normal(1.5, 1.5), 0, 6))

    else:  # New Signup
        tenure = RNG.integers(1, 30)
        purchases = int(RNG.random() < 0.08)            # a few convert fast
        ltv = purchases * RNG.normal(340, 60)
        recency = int(RNG.integers(0, 14))
        opens = int(clip(RNG.normal(3, 2), 0, 10))
        clicks = int(clip(RNG.normal(1, 1), 0, 5))
        visits = int(clip(RNG.normal(3, 2), 0, 10))
        pricing = int(clip(RNG.normal(0.8, 1), 0, 4))
        carts = int(RNG.random() < 0.15)
        wishlist = int(clip(RNG.normal(1, 1.2), 0, 5))

    return {
        "lead_id": f"HC-{i:04d}",
        "name": name,
        "signup_date": (TODAY - pd.Timedelta(days=int(tenure))).date().isoformat(),
        "source": RNG.choice(SOURCES, p=SOURCE_P),
        "email_opens_30d": opens,
        "email_clicks_30d": clicks,
        "site_visits_30d": visits,
        "pricing_page_visits_30d": pricing,
        "cart_abandons_90d": carts,
        "wishlist_items": wishlist,
        "past_purchases": purchases,
        "lifetime_value": round(max(0.0, ltv), 2),
        "days_since_last_activity": recency,
        "_true_persona": persona,   # ground truth, kept for validation
    }


N = 1000
personas = RNG.choice(list(PERSONAS), size=N, p=list(PERSONAS.values()))
df = pd.DataFrame(make_lead(i, p) for i, p in enumerate(personas, start=1))

out = Path(__file__).parent / "data" / "leads.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)

print(f"Wrote {len(df):,} leads -> {out}")
print(df._true_persona.value_counts().to_string())
print(f"Total LTV: ${df.lifetime_value.sum():,.0f}")
