import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. LOAD DATA
# ============================================================

leads = pd.read_csv(
    "data/olist_marketing_qualified_leads_dataset.csv"
)

deals = pd.read_csv(
    "data/olist_closed_deals_dataset.csv"
)

leads["converted"] = leads["mql_id"].isin(
    deals["mql_id"]
).astype(int)

leads["first_contact_date"] = pd.to_datetime(
    leads["first_contact_date"]
)


# ============================================================
# 2. SORT BY TIME
# ============================================================

leads = leads.sort_values(
    "first_contact_date"
).reset_index(drop=True)


# ============================================================
# 3. TEMPORAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(leads) * 0.80)

train = leads.iloc[:split_index].copy()
test = leads.iloc[split_index:].copy()


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

for df in [train, test]:

    df["contact_year"] = (
        df["first_contact_date"].dt.year
    )

    df["contact_month"] = (
        df["first_contact_date"].dt.month
    )

    df["contact_dayofweek"] = (
        df["first_contact_date"].dt.dayofweek
    )


features = [
    "landing_page_id",
    "origin",
    "contact_year",
    "contact_month",
    "contact_dayofweek"
]


categorical_features = [
    "landing_page_id",
    "origin"
]

numeric_features = [
    "contact_year",
    "contact_month",
    "contact_dayofweek"
]


# ============================================================
# 5. BUILD ML PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        )
    ]
)


model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# 6. TRAIN MODEL
# ============================================================

pipeline.fit(
    train[features],
    train["converted"]
)


# ============================================================
# 7. PREDICT FUTURE LEADS
# ============================================================

test["conversion_probability"] = (
    pipeline.predict_proba(
        test[features]
    )[:, 1]
)


# ============================================================
# 8. HISTORICAL BASELINE
# ============================================================

overall_rate = (
    train["converted"].mean()
)


# ============================================================
# 9. BUILD HISTORICAL EVIDENCE
# ============================================================

origin_stats = (
    train.groupby(
        "origin",
        dropna=False
    )["converted"]
    .agg(
        leads="count",
        conversions="sum"
    )
    .reset_index()
)


landing_stats = (
    train.groupby(
        "landing_page_id"
    )["converted"]
    .agg(
        leads="count",
        conversions="sum"
    )
    .reset_index()
)


# ============================================================
# 10. SMOOTHED CONVERSION RATE
# ============================================================

def smoothed_rate(
    conversions,
    leads,
    baseline,
    prior_strength=20
):

    return (
        conversions +
        prior_strength * baseline
    ) / (
        leads +
        prior_strength
    )


origin_stats["origin_rate"] = origin_stats.apply(
    lambda row: smoothed_rate(
        row["conversions"],
        row["leads"],
        overall_rate
    ),
    axis=1
)


landing_stats["landing_rate"] = landing_stats.apply(
    lambda row: smoothed_rate(
        row["conversions"],
        row["leads"],
        overall_rate
    ),
    axis=1
)


# ============================================================
# 11. MERGE EVIDENCE INTO TEST LEADS
# ============================================================

test = test.merge(
    origin_stats[
        [
            "origin",
            "leads",
            "conversions",
            "origin_rate"
        ]
    ],
    on="origin",
    how="left"
)

test = test.rename(
    columns={
        "leads": "origin_sample_size",
        "conversions": "origin_conversions"
    }
)


test = test.merge(
    landing_stats[
        [
            "landing_page_id",
            "leads",
            "conversions",
            "landing_rate"
        ]
    ],
    on="landing_page_id",
    how="left"
)

test = test.rename(
    columns={
        "leads": "landing_sample_size",
        "conversions": "landing_conversions"
    }
)


# ============================================================
# 12. HANDLE UNSEEN CATEGORIES
# ============================================================

test["origin_rate"] = (
    test["origin_rate"]
    .fillna(overall_rate)
)

test["landing_rate"] = (
    test["landing_rate"]
    .fillna(overall_rate)
)

test["origin_sample_size"] = (
    test["origin_sample_size"]
    .fillna(0)
)

test["landing_sample_size"] = (
    test["landing_sample_size"]
    .fillna(0)
)


# ============================================================
# 13. EVIDENCE SIGNAL
# ============================================================

def signal(
    rate,
    sample_size,
    baseline
):

    if sample_size < 10:
        return "LIMITED"

    difference = rate - baseline

    if difference >= 0.03:
        return "STRONG"

    if difference >= 0:
        return "POSITIVE"

    if difference <= -0.03:
        return "NEGATIVE"

    return "NEUTRAL"


test["origin_signal"] = test.apply(
    lambda row: signal(
        row["origin_rate"],
        row["origin_sample_size"],
        overall_rate
    ),
    axis=1
)


test["landing_signal"] = test.apply(
    lambda row: signal(
        row["landing_rate"],
        row["landing_sample_size"],
        overall_rate
    ),
    axis=1
)


# ============================================================
# 14. COMBINE MODEL + EVIDENCE
# ============================================================

test["decision_score"] = (
    0.60 * test["conversion_probability"]
    +
    0.20 * test["origin_rate"]
    +
    0.20 * test["landing_rate"]
)


# ============================================================
# 15. PRIORITY
# ============================================================

def priority(score):

    

    if score >= 0.50:
        return "HIGH"

    if score >= 0.40:
        return "MEDIUM"

    return "LOW"


test["priority"] = (
    test["decision_score"]
    .apply(priority)
)


# ============================================================
# 16. NEXT BEST ACTION
# ============================================================

def next_action(row):

    if row["priority"] == "HIGH":

        if (
            row["origin_signal"] == "STRONG"
            or
            row["landing_signal"] == "STRONG"
        ):
            return (
                "Prioritize immediate sales follow-up"
            )

        return (
            "Prioritize for sales follow-up"
        )

    if row["priority"] == "MEDIUM":

        return (
            "Review and follow up based on capacity"
        )

    return (
        "Keep in nurture queue and monitor"
    )


test["recommended_action"] = test.apply(
    next_action,
    axis=1
)


# ============================================================
# 17. DECISION REASON
# ============================================================

def build_reason(row):

    reasons = []

    if row["origin_signal"] == "STRONG":
        reasons.append(
            f"origin historically converts above baseline"
        )

    elif row["origin_signal"] == "POSITIVE":
        reasons.append(
            f"origin shows a positive conversion signal"
        )

    elif row["origin_signal"] == "NEGATIVE":
        reasons.append(
            f"origin historically converts below baseline"
        )

    if row["landing_signal"] == "STRONG":
        reasons.append(
            "landing page shows a strong historical signal"
        )

    elif row["landing_signal"] == "POSITIVE":
        reasons.append(
            "landing page shows a positive historical signal"
        )

    elif row["landing_signal"] == "NEGATIVE":
        reasons.append(
            "landing page shows a negative historical signal"
        )

    if not reasons:
        reasons.append(
            "limited historical evidence available"
        )

    return "; ".join(reasons)


test["decision_reason"] = test.apply(
    build_reason,
    axis=1
)


# ============================================================
# 18. RANK LEADS
# ============================================================

test = test.sort_values(
    "decision_score",
    ascending=False
).reset_index(drop=True)


test["rank"] = (
    test.index + 1
)


# ============================================================
# 19. DISPLAY
# ============================================================

print("\n========== PRAEVIA DECISION ENGINE ==========")

print(
    f"Historical baseline: "
    f"{overall_rate * 100:.2f}%"
)

print(
    f"Decision leads: {len(test)}"
)


print("\n========== TOP 15 PRAEVIA DECISIONS ==========")

display_columns = [
    "rank",
    "mql_id",
    "origin",
    "landing_page_id",
    "conversion_probability",
    "origin_rate",
    "landing_rate",
    "origin_signal",
    "landing_signal",
    "decision_score",
    "priority",
    "recommended_action",
    "decision_reason"
]

print(
    test[
        display_columns
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 20. SAVE
# ============================================================

output_path = (
    "data/praevia_decision_engine.csv"
)

test.to_csv(
    output_path,
    index=False
)

print(
    f"\nSaved to: {output_path}"
)