"""
AI Lead Scoring & Segmentation Engine — Halcyon Collectibles (fictional brand)

Architecture note (the important design decision):
    Scores are computed with transparent, deterministic Python — weighted
    behavioral signals with weights documented below. Claude never does the
    arithmetic. The AI layer's job is interpretation: explaining scores in
    plain English and drafting segment-specific outreach. Deterministic
    scoring + AI explanation = a system marketers can audit and trust.

Run:
    streamlit run app.py
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="AI Lead Scoring — Halcyon Collectibles",
    page_icon="🎯",
    layout="wide",
)

CLAUDE_MODEL = "claude-sonnet-4-6"

SEGMENT_COLORS = {
    "VIP Collector": "#6C5CE7",
    "At-Risk VIP": "#D63031",
    "Hot Prospect": "#00B894",
    "Nurture": "#74B9FF",
    "Win-Back": "#E17055",
    "New & Unproven": "#B2BEC3",
}

# ---------------------------------------------------------------------------
# Scoring engine (deterministic, documented, auditable)
# ---------------------------------------------------------------------------

# Each signal is normalized to 0-1 against a "saturation" ceiling, then
# weighted. Weights sum to 100 = max score. Recency is inverted (recent
# activity scores high).
WEIGHTS = {
    #  signal                      weight  ceiling  note
    "email_opens_30d":            (8,     15),   # engagement
    "email_clicks_30d":           (10,     8),   # engagement (stronger)
    "site_visits_30d":            (10,    12),   # engagement
    "recency":                    (14,    60),   # inverted days-since-activity
    "pricing_page_visits_30d":    (14,     5),   # intent
    "cart_abandons_90d":          (10,     3),   # intent (they almost bought)
    "wishlist_items":             (6,     10),   # intent
    "past_purchases":             (14,     8),   # value
    "lifetime_value":             (14,  3000),   # value
}


def score_leads(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    def norm(col, ceiling):
        return (d[col].clip(upper=ceiling) / ceiling)

    d["_recency_score"] = 1 - (d["days_since_last_activity"].clip(upper=WEIGHTS["recency"][1])
                               / WEIGHTS["recency"][1])

    score = pd.Series(0.0, index=d.index)
    contributions = {}
    for signal, (weight, ceiling) in WEIGHTS.items():
        if signal == "recency":
            contrib = d["_recency_score"] * weight
        else:
            contrib = norm(signal, ceiling) * weight
        contributions[signal] = contrib.round(1)
        score += contrib

    d["score"] = score.round(0).astype(int)
    for signal, contrib in contributions.items():
        d[f"pts_{signal}"] = contrib
    return d


def assign_segment(row) -> str:
    """Lifecycle segment rules — order matters, first match wins."""
    tenure_days = (pd.Timestamp("2026-08-01") - pd.Timestamp(row.signup_date)).days
    buyer = row.past_purchases >= 1
    vip = row.past_purchases >= 5 or row.lifetime_value >= 2000
    quiet = row.days_since_last_activity > 30
    high_intent = (row.pricing_page_visits_30d >= 2 or row.cart_abandons_90d >= 1)

    if vip and not quiet:
        return "VIP Collector"
    if vip and quiet:
        return "At-Risk VIP"
    if buyer and row.days_since_last_activity > 60:
        return "Win-Back"
    if not buyer and high_intent and row.days_since_last_activity <= 14:
        return "Hot Prospect"
    if tenure_days <= 30:
        return "New & Unproven"
    return "Nurture"


SEGMENT_PLAYS = {
    "VIP Collector": "White-glove treatment: early access, exclusives, concierge touches.",
    "At-Risk VIP": "Highest-priority save: personal re-engagement before they churn.",
    "Hot Prospect": "Strike now: remove friction, nudge the almost-purchase over the line.",
    "Nurture": "Slow build: editorial content and brand storytelling, no hard sell.",
    "Win-Back": "Reactivation: remind them why they bought, give a reason to return.",
    "New & Unproven": "Onboard: welcome series, learn their interests, first-purchase nudge.",
}

# ---------------------------------------------------------------------------
# AI layer
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior lifecycle marketing strategist at Halcyon
Collectibles, a premium collectibles e-commerce brand (AOV ~$385, products
are statues and premium format figures for serious collectors).

You are given lead-scoring data. Scores (0-100) come from a deterministic
weighted model — you did not compute them; your job is to interpret them.

Ground rules:
- Cite the specific signal values behind every claim.
- Speak like a marketer, not a data scientist. No jargon.
- Where you infer motivation or likelihood, label it as an inference.
- Be concise and actionable. Format responses in clean Markdown."""


def ask_claude(api_key: str, prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def lead_context(row: pd.Series, seg_stats: pd.DataFrame) -> str:
    lead = {
        k: (round(v, 1) if isinstance(v, (int, float, np.floating)) else v)
        for k, v in row.items()
        if not k.startswith("_") and k != "name"
    }
    return json.dumps({
        "lead": lead,
        "segment_averages": seg_stats.round(1).to_dict(),
        "scoring_weights": {k: v[0] for k, v in WEIGHTS.items()},
    }, default=str)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_default() -> pd.DataFrame:
    from pathlib import Path
    return pd.read_csv(Path(__file__).parent / "data" / "leads.csv")


with st.sidebar:
    st.title("🎯 Halcyon Leads")
    st.caption("AI-powered lead scoring & segmentation (demo data)")

    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Needed only for AI explanations and outreach drafts.",
    )
    st.divider()

    uploaded = st.file_uploader(
        "Or score your own CSV",
        type="csv",
        help="Must include the same behavioral columns as the demo data.",
    )

raw = pd.read_csv(uploaded) if uploaded else load_default()

REQUIRED = ["email_opens_30d", "email_clicks_30d", "site_visits_30d",
            "pricing_page_visits_30d", "cart_abandons_90d", "wishlist_items",
            "past_purchases", "lifetime_value", "days_since_last_activity",
            "signup_date"]
missing = [c for c in REQUIRED if c not in raw.columns]
if missing:
    st.error(f"Uploaded CSV is missing columns: {missing}")
    st.stop()

df = score_leads(raw)
df["segment"] = df.apply(assign_segment, axis=1)

seg_profile = df.groupby("segment")[
    ["score", "email_opens_30d", "email_clicks_30d", "site_visits_30d",
     "pricing_page_visits_30d", "cart_abandons_90d", "wishlist_items",
     "past_purchases", "lifetime_value", "days_since_last_activity"]
].mean()

# ---------------------------------------------------------------------------
# Header & KPI row
# ---------------------------------------------------------------------------

st.title("Lead Scoring & Segmentation — Halcyon Collectibles")

c = st.columns(5)
c[0].metric("Leads", f"{len(df):,}")
c[1].metric("Avg score", f"{df.score.mean():.0f}")
c[2].metric("Hot Prospects", f"{(df.segment == 'Hot Prospect').sum():,}")
at_risk_ltv = df.loc[df.segment == "At-Risk VIP", "lifetime_value"].sum()
c[3].metric("LTV at risk", f"${at_risk_ltv:,.0f}",
            help="Combined lifetime value of At-Risk VIPs")
c[4].metric("Total LTV", f"${df.lifetime_value.sum():,.0f}")

tab_over, tab_leads, tab_segments, tab_method = st.tabs(
    ["📊 Overview", "🔍 Lead Explorer", "👥 Segments", "⚙️ Methodology"]
)

# ---------------------------------------------------------------------------
with tab_over:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="score", color="segment", nbins=25,
                           color_discrete_map=SEGMENT_COLORS,
                           title="Score distribution by segment")
        fig.update_layout(height=400, legend_title=None)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        seg_counts = df.segment.value_counts().reset_index()
        fig = px.pie(seg_counts, names="segment", values="count", hole=0.45,
                     color="segment", color_discrete_map=SEGMENT_COLORS,
                     title="Segment breakdown")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    if api_key and st.button("🤖 Generate audience summary", type="primary"):
        digest = json.dumps({
            "segment_counts": df.segment.value_counts().to_dict(),
            "segment_profiles": seg_profile.round(1).to_dict(orient="index"),
            "segment_playbooks": SEGMENT_PLAYS,
        })
        with st.spinner("Analyzing audience…"):
            try:
                st.markdown(ask_claude(
                    api_key,
                    "Audience data:\n" + digest +
                    "\n\nTask: Write an executive audience summary: the 3 most "
                    "important things about this lead base, where the biggest "
                    "revenue opportunity sits, and where the biggest risk sits. "
                    "Cite counts and averages."
                ))
            except Exception as e:
                st.error(f"API call failed: {e}")
    elif not api_key:
        st.info("Add an API key in the sidebar to enable the AI audience summary.")

# ---------------------------------------------------------------------------
with tab_leads:
    seg_filter = st.multiselect(
        "Filter segments", sorted(df.segment.unique()),
        default=sorted(df.segment.unique()),
    )
    show = df[df.segment.isin(seg_filter)].sort_values("score", ascending=False)

    display_cols = ["lead_id", "name", "segment", "score", "past_purchases",
                    "lifetime_value", "days_since_last_activity",
                    "pricing_page_visits_30d", "cart_abandons_90d", "source"]
    st.dataframe(show[display_cols].set_index("lead_id"),
                 use_container_width=True, height=380)

    st.subheader("Explain a lead")
    pick = st.selectbox(
        "Lead", show.lead_id,
        format_func=lambda x: f"{x} — {df.set_index('lead_id').loc[x, 'name']} "
                              f"(score {df.set_index('lead_id').loc[x, 'score']})",
    )
    row = df[df.lead_id == pick].iloc[0]

    b1, b2, b3 = st.columns(3)
    b1.metric("Score", row.score)
    b2.metric("Segment", row.segment)
    b3.metric("LTV", f"${row.lifetime_value:,.0f}")

    if api_key:
        if st.button("🤖 Explain this score", type="primary"):
            ctx = lead_context(row, seg_profile.loc[row.segment])
            with st.spinner("Explaining…"):
                try:
                    st.markdown(ask_claude(
                        api_key,
                        "Lead data with segment averages and scoring weights:\n"
                        + ctx +
                        "\n\nTask: Explain this lead's score in plain English "
                        "for a marketer: the 2-3 signals driving it up or down "
                        "(with values, compared to segment averages), what the "
                        "lead's behavior suggests (labeled as inference), and "
                        "the single best next action."
                    ))
                except Exception as e:
                    st.error(f"API call failed: {e}")
    else:
        st.info("Add an API key in the sidebar to enable AI explanations.")

# ---------------------------------------------------------------------------
with tab_segments:
    seg = st.selectbox("Segment", sorted(df.segment.unique()))
    seg_df = df[df.segment == seg]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Leads", f"{len(seg_df):,}")
    c2.metric("Avg score", f"{seg_df.score.mean():.0f}")
    c3.metric("Total LTV", f"${seg_df.lifetime_value.sum():,.0f}")
    c4.metric("Avg days inactive", f"{seg_df.days_since_last_activity.mean():.0f}")

    st.markdown(f"**Playbook:** {SEGMENT_PLAYS[seg]}")
    st.dataframe(seg_profile.loc[[seg]].round(1), use_container_width=True)

    if api_key:
        if st.button("🤖 Draft outreach for this segment", type="primary"):
            digest = json.dumps({
                "segment": seg,
                "size": len(seg_df),
                "profile": seg_profile.loc[seg].round(1).to_dict(),
                "playbook": SEGMENT_PLAYS[seg],
                "brand": "Halcyon Collectibles — premium statues and premium "
                         "format figures for serious collectors, AOV ~$385",
            })
            with st.spinner("Drafting…"):
                try:
                    st.markdown(ask_claude(
                        api_key,
                        "Segment data:\n" + digest +
                        "\n\nTask: Draft outreach for this segment: (1) the "
                        "messaging angle and why it fits this behavioral "
                        "profile, (2) one subject line A/B pair, (3) a short "
                        "email draft (under 120 words) in a premium-but-warm "
                        "collector-to-collector voice. No discounts unless the "
                        "behavioral profile justifies one."
                    ))
                except Exception as e:
                    st.error(f"API call failed: {e}")
    else:
        st.info("Add an API key in the sidebar to enable AI outreach drafts.")

# ---------------------------------------------------------------------------
with tab_method:
    st.subheader("How scoring works")
    st.markdown(
        "Scores are **deterministic and auditable** — a weighted sum of nine "
        "behavioral signals, each normalized against a saturation ceiling. "
        "The AI never computes scores; it only interprets them."
    )
    wt = pd.DataFrame(
        [(k, v[0], v[1]) for k, v in WEIGHTS.items()],
        columns=["Signal", "Max points", "Saturation ceiling"],
    )
    st.dataframe(wt.set_index("Signal"), use_container_width=True)
    st.markdown(
        "**Segment rules** (first match wins): VIP Collector → At-Risk VIP → "
        "Win-Back → Hot Prospect → New & Unproven → Nurture. "
        "See `assign_segment()` in the source for exact thresholds."
    )

st.divider()
st.caption(
    "Demo project · synthetic data · built by Chris Dreiling · "
    "[GitHub](https://github.com/chrisdre2501)"
)
