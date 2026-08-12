# AI Lead Scoring & Segmentation Engine

**Deterministic scores. AI explanations. A lead engine marketers can actually trust.**

A Streamlit app for a fictional premium collectibles e-commerce brand
(**Halcyon Collectibles**) that scores 1,000 leads on nine behavioral signals,
assigns them to lifecycle segments, and uses **Claude to do what LLMs are
actually good at**: explaining every score in plain English and drafting
segment-specific outreach — never doing the arithmetic.

> 📸 *Screenshots coming — 2-minute local setup below, or deploy free on
> Streamlit Community Cloud.*

Companion project to my [AI Marketing Analytics Dashboard](https://github.com/chrisdre2501) —
same fictional brand: that project analyzes the campaigns, this one analyzes
the audience those campaigns feed.

---

## The design decision that matters

Most "AI lead scoring" demos hand the whole problem to an LLM and get back
unauditable black-box numbers. Marketers won't trust that, and they shouldn't.

This project splits the work the way a production system should:

| Layer | Owner | Why |
|---|---|---|
| **Scoring** (0–100) | Deterministic Python — 9 weighted signals, weights documented in code and in the app's Methodology tab | Auditable, reproducible, cheap, instant |
| **Segmentation** | Rule-based lifecycle logic (first match wins) | Every assignment is explainable in one sentence |
| **Interpretation** | Claude | Explains *why* a lead scored what it did, infers intent (labeled as inference), drafts the outreach |

The system prompt explicitly tells the model: *you did not compute these
scores; your job is to interpret them.*

## What it does

- **Overview** — score distribution by segment, segment breakdown, and an
  AI-generated executive audience summary (biggest opportunity, biggest risk,
  with counts and averages cited)
- **Lead Explorer** — sortable, filterable 1,000-lead table; pick any lead and
  get a plain-English explanation of its score: which signals drove it, how the
  lead compares to segment averages, and the single best next action
- **Segments** — six lifecycle segments (VIP Collector, At-Risk VIP, Hot
  Prospect, Nurture, Win-Back, New & Unproven), each with a profile, a
  playbook, and AI-drafted outreach: messaging angle, an A/B subject-line
  pair, and a sub-120-word email in the brand voice
- **Methodology** — full weights table and segmentation rules, in the app
  itself, because a scoring system you can't inspect is a scoring system
  nobody adopts
- **Bring your own data** — upload any CSV with the same behavioral columns
  and the engine scores it

## The synthetic data has ground truth built in

`generate_leads.py` draws 1,000 leads from six realistic personas (whale
collectors, at-risk VIPs, engaged prospects, window shoppers, lapsed buyers,
new signups) with noisy, persona-consistent signals — then keeps the true
persona as a hidden column.

That makes the system **testable**: the behavior-based segments recover the
ground-truth personas with near-perfect agreement, and the one systematic
"error" is the system working as intended — new signups who show strong intent
get promoted to Hot Prospect, because behavior should outrank tenure.

Another deliberate wrinkle: **At-Risk VIPs average a score of just ~32** while
sitting on the highest lifetime value in the base. Score measures *current
heat*; segmentation preserves *value context*. A score-only system would
quietly deprioritize your most valuable save-able customers — which is exactly
why this app has both.

## Quick start

```bash
git clone https://github.com/chrisdre2501/ai-lead-scoring-engine.git
cd ai-lead-scoring-engine
pip install -r requirements.txt

# regenerate leads (optional — CSV included)
python generate_leads.py

streamlit run app.py
```

Enter an [Anthropic API key](https://console.anthropic.com/) in the sidebar
for the AI features. Scoring, segmentation, and all charts work without one.

## Stack

Python · Streamlit · Plotly · pandas/NumPy · Anthropic API (Claude Sonnet 4.6)

## Project structure

```
├── app.py               # Scoring engine, segmentation rules, UI, AI layer
├── generate_leads.py    # Persona-based synthetic lead generator
├── data/
│   └── leads.csv
├── requirements.txt
└── README.md
```

## Limitations & next steps

- Weights are hand-tuned; the natural upgrade is fitting them to conversion
  outcomes (logistic regression) while keeping the same explainable structure
- Rule-based segments could be complemented with clustering to *discover*
  segments rather than assert them
- A CRM connector (HubSpot/Salesforce export) would replace the CSV upload

---

*Part of my AI marketing portfolio — more projects at
[github.com/chrisdre2501](https://github.com/chrisdre2501).*
