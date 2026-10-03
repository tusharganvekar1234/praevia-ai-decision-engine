import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)


# ============================================================
# 1. LOAD DATA
# ============================================================

leads = pd.read_csv(
    "data/olist_marketing_qualified_leads_dataset.csv"
)

deals = pd.read_csv(
    "data/olist_closed_deals_dataset.csv"
)


# ============================================================
# 2. CREATE CONVERSION TARGET
# ============================================================

leads["converted"] = leads["mql_id"].isin(
    deals["mql_id"]
).astype(int)


# ============================================================
# 3. DATE PROCESSING
# ============================================================

leads["first_contact_date"] = pd.to_datetime(
    leads["first_contact_date"]
)

leads["contact_year"] = leads["first_contact_date"].dt.year
leads["contact_month"] = leads["first_contact_date"].dt.month
leads["contact_dayofweek"] = (
    leads["first_contact_date"].dt.dayofweek
)


# ============================================================
# 4. SORT CHRONOLOGICALLY
# ============================================================

leads = leads.sort_values(
    "first_contact_date"
).reset_index(drop=True)


# ============================================================
# 5. FEATURES
# ============================================================

features = [
    "landing_page_id",
    "origin",
    "contact_year",
    "contact_month",
    "contact_dayofweek"
]

X = leads[features]
y = leads["converted"]


# ============================================================
# 6. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(leads) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n========== TEMPORAL SPLIT ==========")

print(
    f"Training period: "
    f"{leads['first_contact_date'].iloc[0].date()} "
    f"→ "
    f"{leads['first_contact_date'].iloc[split_index - 1].date()}"
)

print(
    f"Testing period: "
    f"{leads['first_contact_date'].iloc[split_index].date()} "
    f"→ "
    f"{leads['first_contact_date'].iloc[-1].date()}"
)

print(f"Training leads: {len(X_train)}")
print(f"Testing leads: {len(X_test)}")

print(
    f"Training conversion rate: "
    f"{y_train.mean() * 100:.2f}%"
)

print(
    f"Testing conversion rate: "
    f"{y_test.mean() * 100:.2f}%"
)


# ============================================================
# 7. PREPROCESSING
# ============================================================

categorical_features = [
    "landing_page_id",
    "origin"
]

numeric_features = [
    "contact_year",
    "contact_month",
    "contact_dayofweek"
]


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


# ============================================================
# 8. MODEL
# ============================================================

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
# 9. TRAIN
# ============================================================

print("\n========== TRAINING PRAEVIA ==========")

pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# 10. PREDICT
# ============================================================

probabilities = pipeline.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= 0.50
).astype(int)


# ============================================================
# 11. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


print("\n========== PRAEVIA RESULTS ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# 12. TOP LEADS
# ============================================================

results = X_test.copy()

results["actual"] = y_test.values
results["conversion_probability"] = probabilities
results["predicted"] = predictions

results = results.sort_values(
    "conversion_probability",
    ascending=False
)


print("\n========== TOP PRIORITIZED LEADS ==========")

print(
    results.head(10).to_string(
        index=False
    )
)