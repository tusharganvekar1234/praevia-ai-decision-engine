import pandas as pd

# Load datasets
leads = pd.read_csv("data/olist_marketing_qualified_leads_dataset.csv")
deals = pd.read_csv("data/olist_closed_deals_dataset.csv")

# Create conversion label
leads["converted"] = leads["mql_id"].isin(deals["mql_id"]).astype(int)

print("\n========== FEATURE AUDIT ==========")

print("\n--- MARKETING LEAD DATA ---")
print("Columns available for every lead:")
for col in leads.columns:
    print(f"- {col}")

print("\n--- CLOSED DEAL DATA ---")
print("Columns available only in the closed-deal table:")
for col in deals.columns:
    print(f"- {col}")

print("\n--- AVAILABILITY CHECK ---")

# Check how many leads have a matching closed-deal record
merged = leads.merge(
    deals,
    on="mql_id",
    how="left",
    suffixes=("_lead", "_deal")
)

print(f"Total leads: {len(leads)}")
print(f"Leads with closed-deal record: {merged['seller_id'].notna().sum()}")
print(f"Leads without closed-deal record: {merged['seller_id'].isna().sum()}")

print("\n--- POTENTIAL POST-OUTCOME FIELDS ---")

post_outcome_fields = [
    "seller_id",
    "sdr_id",
    "sr_id",
    "won_date"
]

for col in post_outcome_fields:
    print(f"{col}: DO NOT USE FOR PREDICTION")

print("\n--- DECISION-TIME CANDIDATE FEATURES ---")

candidate_features = [
    "first_contact_date",
    "landing_page_id",
    "origin"
]

for col in candidate_features:
    missing = leads[col].isna().sum()
    unique = leads[col].nunique(dropna=True)

    print(
        f"{col}: "
        f"missing={missing}, "
        f"unique_values={unique}"
    )

print("\n--- CONVERSION BY CANDIDATE FEATURE ---")

for col in candidate_features[1:]:
    print(f"\n### {col}")

    analysis = (
        leads.groupby(col, dropna=False)["converted"]
        .agg(["count", "sum", "mean"])
        .sort_values("mean", ascending=False)
    )

    analysis["mean"] = analysis["mean"] * 100

    print(analysis.head(15))

print("\n========== AUDIT COMPLETE ==========")