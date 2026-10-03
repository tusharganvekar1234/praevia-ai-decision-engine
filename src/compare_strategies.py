import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/praevia_decision_engine.csv"
)

overall_rate = df["converted"].mean()

print("\n========== PRAEVIA STRATEGY COMPARISON ==========")

print(f"Evaluation leads: {len(df)}")
print(f"Overall conversion rate: {overall_rate * 100:.2f}%")


# ============================================================
# TOP-K FUNCTION
# ============================================================

def evaluate_ranking(data, score_column, name):

    ranked = data.sort_values(
        score_column,
        ascending=False
    ).reset_index(drop=True)

    k = max(1, int(len(ranked) * 0.10))

    top = ranked.head(k)

    conversion_rate = top["converted"].mean()

    lift = conversion_rate / overall_rate

    auc = roc_auc_score(
        ranked["converted"],
        ranked[score_column]
    )

    return {
        "strategy": name,
        "top_10_leads": k,
        "top_10_conversions": int(top["converted"].sum()),
        "top_10_conversion_rate": conversion_rate,
        "lift": lift,
        "roc_auc": auc
    }


# ============================================================
# 1. ML-ONLY
# ============================================================

ml_result = evaluate_ranking(
    df,
    "conversion_probability",
    "ML Probability"
)


# ============================================================
# 2. PRAEVIA
# ============================================================

praevia_result = evaluate_ranking(
    df,
    "decision_score",
    "PRAEVIA Decision Score"
)


# ============================================================
# 3. RANDOM BASELINE
# ============================================================

random_results = []

for seed in range(100):

    shuffled = df.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    k = max(
        1,
        int(len(shuffled) * 0.10)
    )

    top = shuffled.head(k)

    random_results.append(
        top["converted"].mean()
    )


random_rate = np.mean(
    random_results
)

random_lift = (
    random_rate / overall_rate
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========== TOP 10% COMPARISON ==========")

for result in [
    ml_result,
    praevia_result
]:

    print(
        f"\n{result['strategy']}"
    )

    print(
        f"Top 10% conversion rate: "
        f"{result['top_10_conversion_rate'] * 100:.2f}%"
    )

    print(
        f"Lift: "
        f"{result['lift']:.2f}x"
    )

    print(
        f"ROC-AUC: "
        f"{result['roc_auc']:.4f}"
    )


print("\nRandom Baseline")

print(
    f"Average top 10% conversion rate: "
    f"{random_rate * 100:.2f}%"
)

print(
    f"Lift: "
    f"{random_lift:.2f}x"
)


# ============================================================
# PRAEVIA VS ML
# ============================================================

improvement = (
    praevia_result["top_10_conversion_rate"]
    -
    ml_result["top_10_conversion_rate"]
)

print(
    "\n========== PRAEVIA VS ML =========="
)

print(
    f"Conversion-rate improvement: "
    f"{improvement * 100:.2f} percentage points"
)


# ============================================================
# SAVE RESULTS
# ============================================================

comparison = pd.DataFrame(
    [
        {
            "strategy": "Random Baseline",
            "top_10_conversion_rate": random_rate,
            "lift": random_lift,
            "roc_auc": np.nan
        },
        {
            "strategy": "ML Probability",
            "top_10_conversion_rate":
                ml_result["top_10_conversion_rate"],
            "lift": ml_result["lift"],
            "roc_auc": ml_result["roc_auc"]
        },
        {
            "strategy": "PRAEVIA Decision Score",
            "top_10_conversion_rate":
                praevia_result["top_10_conversion_rate"],
            "lift": praevia_result["lift"],
            "roc_auc": praevia_result["roc_auc"]
        }
    ]
)

comparison.to_csv(
    "data/praevia_strategy_comparison.csv",
    index=False
)

print(
    "\nSaved to: "
    "data/praevia_strategy_comparison.csv"
)