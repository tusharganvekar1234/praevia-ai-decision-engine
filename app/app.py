import sys
from pathlib import Path
from html import escape

import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data"

sys.path.insert(0, str(SRC))

from llm_explainer import generate_explanation


DECISION_FILE = DATA / "praevia_decision_engine.csv"
EVALUATION_FILE = DATA / "praevia_evaluation.csv"
STRATEGY_FILE = DATA / "praevia_strategy_comparison.csv"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PRAEVIA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.65;
        margin-bottom: 25px;
    }

    .evidence-box {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 10px;
        padding: 20px;
        min-height: 180px;
    }

    .trace-box {
        border: 1px solid rgba(128,128,128,0.20);
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .trace-chain {
        font-size: 16px;
        font-weight: 700;
    }

    .section-note {
        opacity: 0.70;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_decisions():
    if not DECISION_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(DECISION_FILE)


@st.cache_data
def load_evaluation():
    if not EVALUATION_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(EVALUATION_FILE)


@st.cache_data
def load_strategy():
    if not STRATEGY_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(STRATEGY_FILE)


df = load_decisions()
evaluation_df = load_evaluation()
strategy_df = load_strategy()


# ============================================================
# DATA CHECK
# ============================================================

if df.empty:
    st.error("PRAEVIA decision data was not found.")
    st.code("python src/decision_engine.py")
    st.stop()


# ============================================================
# HELPERS
# ============================================================

def value(row, column, default=""):
    if column not in row.index:
        return default

    result = row[column]

    if pd.isna(result):
        return default

    return result


def number(row, column, default=0.0):
    try:
        result = value(row, column, default)
        return float(result)
    except (TypeError, ValueError):
        return default


def exact_action(row):
    """PRAEVIA's single source of truth for the recommended action."""
    action = value(row, "next_best_action", "")

    if not action:
        action = value(row, "recommended_action", "")

    if not action:
        action = "Review and follow up."

    return str(action)


def clean_display_df(dataframe):
    result = dataframe.copy()

    if "conversion_probability" in result.columns:
        result["conversion_probability"] = (
            result["conversion_probability"] * 100
        ).round(1)

        result = result.rename(
            columns={"conversion_probability": "Probability %"}
        )

    if "decision_score" in result.columns:
        result["decision_score"] = result["decision_score"].round(3)

    return result


def priority_order(dataframe):
    """Keep the product UI consistently ordered HIGH -> MEDIUM -> LOW."""
    result = dataframe.copy()

    if "priority" in result.columns:
        result["priority"] = pd.Categorical(
            result["priority"],
            categories=["HIGH", "MEDIUM", "LOW"],
            ordered=True,
        )
        result = result.sort_values("priority")

    return result


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        font-size:27px;
        font-weight:800;
    ">
        PRAEVIA
    </div>

    <div style="
        opacity:0.65;
        margin-bottom:20px;
    ">
        AI Decision Intelligence
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()
st.sidebar.caption("Navigate")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Lead Decisions",
        "Evaluation",
        "Decision Trace",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **DATA → DECISION → EVIDENCE → ACTION → HUMAN**
    """
)

st.sidebar.divider()
st.sidebar.caption("PRAEVIA v1.0 • PS-04")

st.sidebar.divider()
st.sidebar.caption("PRAEVIA • AI Decision Engine for Business Data")


# ============================================================
# GLOBAL METRICS
# ============================================================

total_leads = len(df)

high_count = int((df["priority"] == "HIGH").sum())
medium_count = int((df["priority"] == "MEDIUM").sum())
low_count = int((df["priority"] == "LOW").sum())

average_probability = (
    df["conversion_probability"].mean()
    if "conversion_probability" in df.columns
    else 0
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">PRAEVIA</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
        AI Decision Intelligence for Sales
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "PRAEVIA converts business lead data into ranked, "
        "evidence-backed sales decisions."
    )

    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Decision Leads", f"{total_leads:,}")
    c2.metric("HIGH", f"{high_count:,}")
    c3.metric("MEDIUM", f"{medium_count:,}")
    c4.metric("LOW", f"{low_count:,}")
    c5.metric("Avg Probability", f"{average_probability * 100:.1f}%")

    st.divider()

    st.subheader("Priority Distribution")

    priority_chart = pd.DataFrame(
        {
            "Leads": [
                high_count,
                medium_count,
                low_count,
            ]
        },
        index=["HIGH", "MEDIUM", "LOW"],
    )

    st.bar_chart(priority_chart)

    st.divider()

    st.subheader("Top Ranked Leads")

    top_columns = [
        "rank",
        "mql_id",
        "origin",
        "landing_page_id",
        "conversion_probability",
        "decision_score",
        "priority",
    ]

    available = [
        column for column in top_columns
        if column in df.columns
    ]

    top_leads = (
        df[available]
        .sort_values("rank", ascending=True)
        .head(10)
    )

    st.dataframe(
        clean_display_df(top_leads),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.caption(
        "PRAEVIA ranking is evaluated on a held-out temporal test set."
    )


# ============================================================
# LEAD DECISIONS
# ============================================================

elif page == "Lead Decisions":

    st.header("Lead Decision Engine")

    st.caption(
        "Inspect the complete decision for an individual lead."
    )

    lead_ids = df["mql_id"].astype(str).tolist()

    selected_id = st.selectbox(
        "Select a lead",
        lead_ids,
    )

    selected = df[
        df["mql_id"].astype(str) == selected_id
    ]

    if selected.empty:
        st.error("Lead not found.")
        st.stop()

    lead = selected.iloc[0]

    st.divider()

    priority = str(value(lead, "priority", "LOW"))
    rank = int(number(lead, "rank", 0))
    probability = number(lead, "conversion_probability", 0)
    decision_score = number(lead, "decision_score", 0)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Priority", priority)
    c2.metric("Rank", f"#{rank}")
    c3.metric(
        "Conversion Probability",
        f"{probability * 100:.1f}%",
    )
    c4.metric("Decision Score", f"{decision_score:.3f}")

    st.divider()

    st.subheader("Evidence Behind Decision")

    origin = str(value(lead, "origin", "Unknown"))
    origin_rate = number(lead, "origin_rate", 0)
    origin_signal = str(value(lead, "origin_signal", "N/A"))

    landing_page = str(
        value(lead, "landing_page_id", "Unknown")
    )
    landing_rate = number(lead, "landing_rate", 0)
    landing_signal = str(
        value(lead, "landing_signal", "N/A")
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="evidence-box">
            <b>Lead Origin</b>
            <br><br>
            {escape(origin)}
            <br><br>
            Historical conversion rate:
            <b>{origin_rate * 100:.2f}%</b>
            <br><br>
            Signal:
            <b>{escape(origin_signal)}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="evidence-box">
            <b>Landing Page</b>
            <br><br>
            {escape(landing_page)}
            <br><br>
            Historical conversion rate:
            <b>{landing_rate * 100:.2f}%</b>
            <br><br>
            Signal:
            <b>{escape(landing_signal)}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("Decision Reason")

    reason = str(
        value(
            lead,
            "decision_reason",
            "Decision generated from conversion probability "
            "and historical evidence.",
        )
    )

    st.info(reason)

    action = exact_action(lead)

    st.subheader("🤖 PRAEVIA AI Explanation")

    if st.button(
        "Generate AI Explanation",
        use_container_width=True,
    ):
        lead_for_llm = lead.copy()

        # Force the LLM to use the exact same action shown by PRAEVIA.
        lead_for_llm["next_best_action"] = action
        lead_for_llm["recommended_action"] = action

        with st.spinner("Generating grounded explanation..."):
            try:
                explanation = generate_explanation(lead_for_llm)

                st.markdown(
                    f"""
                    <div class="evidence-box">
                    {explanation}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as error:
                st.error(f"AI explanation error: {error}")

    st.subheader("Next Best Action")
    st.success(action)

    st.divider()

    st.subheader("Human Approval Gate")

    st.caption(
        "PRAEVIA recommends. A human makes the final decision."
    )

    approve, review, reject = st.columns(3)

    if approve.button("✅ APPROVE", use_container_width=True):
        st.session_state["human_decision"] = {
            "lead_id": selected_id,
            "status": "APPROVED",
            "message": "PRAEVIA recommendation accepted by human reviewer.",
        }

    if review.button("🔍 REVIEW", use_container_width=True):
        st.session_state["human_decision"] = {
            "lead_id": selected_id,
            "status": "REVIEW",
            "message": "PRAEVIA recommendation sent for human review.",
        }

    if reject.button("❌ REJECT", use_container_width=True):
        st.session_state["human_decision"] = {
            "lead_id": selected_id,
            "status": "REJECTED",
            "message": "PRAEVIA recommendation rejected by human reviewer.",
        }

    decision_state = st.session_state.get("human_decision")

    if decision_state and decision_state["lead_id"] == selected_id:
        st.divider()
        st.subheader("Human Decision")

        status = decision_state["status"]

        if status == "APPROVED":
            st.success(
                f"✅ APPROVED — {decision_state['message']}"
            )
        elif status == "REVIEW":
            st.warning(
                f"🔍 REVIEW — {decision_state['message']}"
            )
        else:
            st.error(
                f"❌ REJECTED — {decision_state['message']}"
            )


# ============================================================
# EVALUATION
# ============================================================

elif page == "Evaluation":

    st.header("Evaluation & Proof")

    st.caption(
        "Held-out temporal test set — 1,600 leads"
    )

    roc_auc = 0.6928
    top10_rate = 0.1750
    top10_lift = 1.79

    c1, c2, c3 = st.columns(3)

    c1.metric("ROC-AUC", f"{roc_auc:.4f}")
    c2.metric(
        "Top-10% Conversion",
        f"{top10_rate * 100:.2f}%",
    )
    c3.metric("Top-10% Lift", f"{top10_lift:.2f}x")

    st.divider()

    st.subheader("Ranking Strategy Comparison")

    strategy_display = pd.DataFrame(
        {
            "Strategy": [
                "Random Baseline",
                "ML Probability",
                "PRAEVIA Decision Score",
            ],
            "Top-10 Conversion": [
                "10.06%",
                "16.88%",
                "17.50%",
            ],
            "Lift": [
                "1.03x",
                "1.73x",
                "1.79x",
            ],
            "ROC-AUC": [
                "N/A",
                "0.6904",
                "0.6928",
            ],
        }
    )

    # Use the stored strategy file as the source when available,
    # while normalizing its presentation for the UI.
    if not strategy_df.empty:
        raw = strategy_df.copy()

        if len(raw) >= 3:
            rows = []
            for _, row in raw.iterrows():
                strategy_name = str(
                    row.get("strategy", row.get("Strategy", ""))
                )

                rate = row.get(
                    "top_10_conversion_rate",
                    row.get("Top-10 Conversion", None),
                )
                lift = row.get(
                    "lift",
                    row.get("Lift", None),
                )
                auc = row.get(
                    "roc_auc",
                    row.get("ROC-AUC", None),
                )

                try:
                    rate_text = (
                        f"{float(rate) * 100:.2f}%"
                        if pd.notna(rate)
                        else "N/A"
                    )
                except (TypeError, ValueError):
                    rate_text = str(rate) if rate is not None else "N/A"

                try:
                    lift_text = (
                        f"{float(lift):.2f}x"
                        if pd.notna(lift)
                        else "N/A"
                    )
                except (TypeError, ValueError):
                    lift_text = str(lift) if lift is not None else "N/A"

                try:
                    auc_text = (
                        f"{float(auc):.4f}"
                        if pd.notna(auc)
                        else "N/A"
                    )
                except (TypeError, ValueError):
                    auc_text = (
                        "N/A"
                        if str(auc).lower() in {"none", "nan", ""}
                        else str(auc)
                    )

                rows.append(
                    {
                        "Strategy": strategy_name,
                        "Top-10 Conversion": rate_text,
                        "Lift": lift_text,
                        "ROC-AUC": auc_text,
                    }
                )

            candidate = pd.DataFrame(rows)

            if not candidate.empty:
                strategy_display = candidate

    st.dataframe(
        strategy_display,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Random baseline = mean top-10% conversion rate across "
        "100 random selections."
    )

    st.divider()

    st.subheader("Observed Conversion by Priority")

    if "converted" in df.columns:

        performance = (
            df.groupby("priority")
            .agg(
                leads=("mql_id", "count"),
                conversions=("converted", "sum"),
            )
            .reset_index()
        )

        performance["conversion_rate"] = (
            performance["conversions"] / performance["leads"]
        )

        performance["priority"] = pd.Categorical(
            performance["priority"],
            categories=["HIGH", "MEDIUM", "LOW"],
            ordered=True,
        )

        performance = performance.sort_values("priority")

        display_performance = performance.copy()

        display_performance["Conversion Rate %"] = (
            display_performance["conversion_rate"] * 100
        ).round(2)

        display_performance = display_performance[
            ["priority", "leads", "conversions", "Conversion Rate %"]
        ].rename(
            columns={
                "priority": "Priority",
                "leads": "Leads",
                "conversions": "Conversions",
            }
        )

        st.dataframe(
            display_performance,
            use_container_width=True,
            hide_index=True,
        )

        chart_data = performance.set_index("priority")[
            ["conversion_rate"]
        ].copy()

        chart_data["conversion_rate"] = (
            chart_data["conversion_rate"] * 100
        )

        chart_data = chart_data.rename(
            columns={"conversion_rate": "Conversion Rate %"}
        )

        st.bar_chart(chart_data)

    else:
        st.info(
            "Conversion labels are not available in the decision file."
        )

    st.divider()

    st.subheader("Held-Out Test Results")

    st.markdown(
        """
        **Evaluation leads:** 1,600

        **Conversions in test set:** 156

        **Overall test conversion rate:** 9.75%

        **Top 10% conversion rate:** 17.50%

        **Top 20% conversion rate:** 19.38%

        **Top 30% conversion rate:** 18.12%
        """
    )

    st.warning(
        "These are held-out evaluation results. They demonstrate "
        "ranking performance on this dataset and are not guarantees "
        "about future leads."
    )


# ============================================================
# DECISION TRACE
# ============================================================

elif page == "Decision Trace":

    st.header("Decision Trace")

    st.caption(
        "Trace a recommendation from source data to human action."
    )

    lead_ids = df["mql_id"].astype(str).tolist()

    selected_id = st.selectbox(
        "Select a lead",
        lead_ids,
        key="trace_selector",
    )

    selected = df[
        df["mql_id"].astype(str) == selected_id
    ]

    if selected.empty:
        st.error("Lead not found.")
        st.stop()

    lead = selected.iloc[0]
    action = exact_action(lead)

    # --------------------------------------------------------
    # TRACE 1 — SOURCE DATA
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="trace-box">
        <b>1. SOURCE DATA</b>
        <br><br>
        Lead ID:
        <b>{escape(str(value(lead, "mql_id", "Unknown")))}</b>
        <br><br>
        Origin:
        <b>{escape(str(value(lead, "origin", "Unknown")))}</b>
        <br><br>
        Landing Page:
        <b>{escape(str(value(lead, "landing_page_id", "Unknown")))}</b>
        <br><br>
        First Contact:
        <b>{escape(str(value(lead, "first_contact_date", "Unknown")))}</b>
        <br><br>
        <span class="section-note">
        Source: Olist Marketing Funnel dataset<br>
        Decision-time fields: origin, landing page, first contact date
        </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # TRACE 2 — MODEL DECISION
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="trace-box">
        <b>2. MODEL DECISION</b>
        <br><br>
        Conversion probability:
        <b>{number(lead, "conversion_probability") * 100:.1f}%</b>
        <br><br>
        Decision score:
        <b>{number(lead, "decision_score"):.3f}</b>
        <br><br>
        Priority:
        <b>{escape(str(value(lead, "priority", "LOW")))}</b>
        <br><br>
        Rank:
        <b>#{int(number(lead, "rank", 0))}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # TRACE 3 — EVIDENCE
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="trace-box">
        <b>3. EVIDENCE</b>
        <br><br>
        Origin rate:
        <b>{number(lead, "origin_rate") * 100:.2f}%</b>
        <br><br>
        Origin signal:
        <b>{escape(str(value(lead, "origin_signal", "N/A")))}</b>
        <br><br>
        Landing page rate:
        <b>{number(lead, "landing_rate") * 100:.2f}%</b>
        <br><br>
        Landing signal:
        <b>{escape(str(value(lead, "landing_signal", "N/A")))}</b>
        <br><br>
        <span class="section-note">
        Evidence is computed from training-period data only;
        post-outcome fields are excluded from the decision.
        </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # TRACE 4 — NEXT BEST ACTION
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="trace-box">
        <b>4. NEXT BEST ACTION</b>
        <br><br>
        {escape(action)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # TRACE 5 — HUMAN APPROVAL
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="trace-box">
        <b>5. HUMAN APPROVAL</b>
        <br><br>
        Final action requires human approval.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.success(
        "DATA → DECISION → EVIDENCE → ACTION → HUMAN"
    )

    st.caption(
        "The LLM explains the existing PRAEVIA decision. "
        "It does not calculate or override the decision."
    )

    # Human approval is also available directly in the trace,
    # making the complete decision path demonstrable in one page.
    st.subheader("Human Approval Gate")

    approve, review, reject = st.columns(3)

    if approve.button(
        "✅ APPROVE",
        use_container_width=True,
        key="trace_approve",
    ):
        st.session_state["trace_human_decision"] = {
            "lead_id": selected_id,
            "status": "APPROVED",
        }

    if review.button(
        "🔍 REVIEW",
        use_container_width=True,
        key="trace_review",
    ):
        st.session_state["trace_human_decision"] = {
            "lead_id": selected_id,
            "status": "REVIEW",
        }

    if reject.button(
        "❌ REJECT",
        use_container_width=True,
        key="trace_reject",
    ):
        st.session_state["trace_human_decision"] = {
            "lead_id": selected_id,
            "status": "REJECTED",
        }

    trace_decision = st.session_state.get("trace_human_decision")

    if trace_decision and trace_decision["lead_id"] == selected_id:
        status = trace_decision["status"]

        if status == "APPROVED":
            st.success(
                "✅ APPROVED — PRAEVIA recommendation accepted by human reviewer."
            )
        elif status == "REVIEW":
            st.warning(
                "🔍 REVIEW — PRAEVIA recommendation sent for human review."
            )
        else:
            st.error(
                "❌ REJECTED — PRAEVIA recommendation rejected by human reviewer."
            )
