import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score


# ============================================================
# 1. LOAD PRAEVIA DECISIONS
# ============================================================

data = pd.read_csv(
    "data/praevia_decision_engine.csv"
)


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

total_leads = len(data)
total_conversions = data["converted"].sum()

overall_rate = (
    data["converted"].mean()
)


print("\n========== PRAEVIA EVALUATION ==========")

print(f"Evaluation leads: {total_leads}")
print(f"Conversions: {total_conversions}")

print(
    f"Overall conversion rate: "
    f"{overall_rate * 100:.2f}%"
)


# ============================================================
# 3. ROC-AUC
# ============================================================

auc = roc_auc_score(
    data["converted"],
    data["decision_score"]
)

print(
    f"\nROC-AUC: {auc:.4f}"
)


# ============================================================
# 4. TOP-K EVALUATION
# ============================================================

def evaluate_top_k(
    df,
    fraction
):

    k = max(
        1,
        int(len(df) * fraction)
    )

    top = df.head(k)

    conversions = top["converted"].sum()

    conversion_rate = (
        top["converted"].mean()
    )

    lift = (
        conversion_rate /
        overall_rate
    )

    return {
        "leads": k,
        "conversions": int(conversions),
        "conversion_rate": conversion_rate,
        "lift": lift
    }


# ============================================================
# 5. TOP 10%
# ============================================================

top_10 = evaluate_top_k(
    data,
    0.10
)


# ============================================================
# 6. TOP 20%
# ============================================================

top_20 = evaluate_top_k(
    data,
    0.20
)


# ============================================================
# 7. TOP 30%
# ============================================================

top_30 = evaluate_top_k(
    data,
    0.30
)


# ============================================================
# 8. PRINT RESULTS
# ============================================================

print("\n========== TOP-K RESULTS ==========")

print(
    "\nTOP 10%"
)

print(
    f"Leads: {top_10['leads']}"
)

print(
    f"Conversions: {top_10['conversions']}"
)

print(
    f"Conversion rate: "
    f"{top_10['conversion_rate'] * 100:.2f}%"
)

print(
    f"Lift: "
    f"{top_10['lift']:.2f}x"
)


print(
    "\nTOP 20%"
)

print(
    f"Leads: {top_20['leads']}"
)

print(
    f"Conversions: {top_20['conversions']}"
)

print(
    f"Conversion rate: "
    f"{top_20['conversion_rate'] * 100:.2f}%"
)

print(
    f"Lift: "
    f"{top_20['lift']:.2f}x"
)


print(
    "\nTOP 30%"
)

print(
    f"Leads: {top_30['leads']}"
)

print(
    f"Conversions: {top_30['conversions']}"
)

print(
    f"Conversion rate: "
    f"{top_30['conversion_rate'] * 100:.2f}%"
)

print(
    f"Lift: "
    f"{top_30['lift']:.2f}x"
)


# ============================================================
# 9. PRIORITY DISTRIBUTION
# ============================================================

print("\n========== PRIORITY DISTRIBUTION ==========")

priority_counts = (
    data["priority"]
    .value_counts()
)

print(priority_counts)


# ============================================================
# 10. CONVERSION BY PRIORITY
# ============================================================

print(
    "\n========== CONVERSION BY PRIORITY =========="
)

priority_analysis = (
    data.groupby("priority")["converted"]
    .agg(
        leads="count",
        conversions="sum",
        conversion_rate="mean"
    )
)

priority_analysis["conversion_rate"] *= 100

print(priority_analysis)


# ============================================================
# 11. RANDOM BASELINE
# ============================================================

np.random.seed(42)

random_data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


random_top_10 = random_data.head(
    top_10["leads"]
)

random_rate = (
    random_top_10["converted"].mean()
)

praevia_rate = (
    top_10["conversion_rate"]
)

print(
    "\n========== BASELINE COMPARISON =========="
)

print(
    f"Random top-10% conversion rate: "
    f"{random_rate * 100:.2f}%"
)

print(
    f"PRAEVIA top-10% conversion rate: "
    f"{praevia_rate * 100:.2f}%"
)

print(
    f"PRAEVIA improvement: "
    f"{(praevia_rate - random_rate) * 100:.2f} percentage points"
)


# ============================================================
# 12. SAVE EVALUATION
# ============================================================

evaluation = pd.DataFrame(
    [
        {
            "strategy": "PRAEVIA",
            "segment": "Top 10%",
            "leads": top_10["leads"],
            "conversions": top_10["conversions"],
            "conversion_rate": top_10["conversion_rate"],
            "lift": top_10["lift"]
        },
        {
            "strategy": "PRAEVIA",
            "segment": "Top 20%",
            "leads": top_20["leads"],
            "conversions": top_20["conversions"],
            "conversion_rate": top_20["conversion_rate"],
            "lift": top_20["lift"]
        },
        {
            "strategy": "PRAEVIA",
            "segment": "Top 30%",
            "leads": top_30["leads"],
            "conversions": top_30["conversions"],
            "conversion_rate": top_30["conversion_rate"],
            "lift": top_30["lift"]
        }
    ]
)

evaluation.to_csv(
    "data/praevia_evaluation.csv",
    index=False
)

print(
    "\nEvaluation saved to: "
    "data/praevia_evaluation.csv" 
)