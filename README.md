# PRAEVIA — AI Decision Intelligence for Sales

PRAEVIA is an AI-assisted sales decision engine that helps sales teams identify which leads should be prioritized, understand the evidence behind each decision, and determine the next best action.

Instead of relying only on an AI-generated answer, PRAEVIA combines machine-learning predictions with historical business-data evidence and keeps a human approval step before the recommendation is accepted.

## Problem

Sales teams may have thousands of leads but limited time and resources.

The challenge is to answer:

- Which leads should be contacted first?
- Why is a lead considered high priority?
- What action should the sales team take?
- Can the decision be traced back to the underlying business data?

## Solution

PRAEVIA follows this decision pipeline:

**Business Data → ML Prediction → Lead Ranking → Evidence → AI Explanation → Next Best Action → Human Approval**

The system:

1. Processes historical lead data.
2. Predicts conversion probability using machine learning.
3. Combines the prediction with historical business-data signals.
4. Produces a decision score and priority.
5. Provides evidence supporting the decision.
6. Generates a grounded natural-language explanation.
7. Recommends a next-best action.
8. Allows a human to Approve, Review, or Reject the recommendation.
9. Provides a decision trace from source data to human decision.

## Key Features

- ML-based lead prioritization
- Conversion probability prediction
- Evidence-backed decision making
- Historical origin and landing-page signals
- Next-best-action recommendations
- LLM-based grounded explanations
- Human approval gate
- Decision traceability
- Temporal held-out evaluation
- Baseline vs ML vs PRAEVIA comparison

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- OpenRouter API
- OpenAI Python SDK
- Logistic Regression
- One-hot encoding
- Feature engineering
- Historical evidence analysis

## Dataset

PRAEVIA uses the **Olist Marketing Funnel** dataset.

The dataset contains marketing-qualified leads and historical closed-deal information.

The project creates a conversion outcome by matching marketing leads with closed deals.

Post-outcome fields are excluded from the decision-time model to reduce data leakage.

## Machine Learning

PRAEVIA uses a Logistic Regression model to estimate lead conversion probability.

The model uses decision-time features such as:

- First contact date
- Landing page
- Lead origin

A chronological train/test split is used so that evaluation is performed on later, unseen leads.

The current held-out evaluation contains:

- 1,600 evaluation leads
- 156 conversions
- ROC-AUC: 0.6928
- Top-10 conversion rate: 17.50%
- Top-10 lift: 1.79×

## Decision Engine

The PRAEVIA decision score combines:

- Machine-learning conversion probability
- Historical conversion rate by lead origin
- Historical conversion rate by landing page

The resulting score is used to prioritize leads as:

- HIGH
- MEDIUM
- LOW

The LLM does **not** calculate or determine the decision score.

The LLM is used to explain the structured decision and evidence in natural language.

## Human Approval

PRAEVIA keeps a human in control of the final decision.

For each recommendation, the user can:

- **Approve** — accept the recommendation
- **Review** — send the recommendation for further human inspection
- **Reject** — decline the recommendation

The current prototype demonstrates the human approval state inside the application. A production implementation could connect approved actions to an existing CRM workflow.

## Project Structure

```text
praevia-ai-decision-engine/
│
├── app/
│   └── app.py
│
├── data/
│   ├── olist_marketing_qualified_leads_dataset.csv
│   ├── olist_closed_deals_dataset.csv
│   ├── praevia_decision_engine.csv
│   ├── praevia_evaluation.csv
│   └── praevia_strategy_comparison.csv
│
├── src/
│   ├── decision_engine.py
│   ├── evaluate_praevia.py
│   ├── compare_strategies.py
│   ├── feature_audit.py
│   ├── build_decisions.py
│   ├── train_model.py
│   └── llm_explainer.py
│
├── .gitignore
└── README.md