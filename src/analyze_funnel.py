import pandas as pd

leads = pd.read_csv("data/olist_marketing_qualified_leads_dataset.csv")
deals = pd.read_csv("data/olist_closed_deals_dataset.csv")

# Identify which marketing leads eventually became closed deals
leads["converted"] = leads["mql_id"].isin(deals["mql_id"]).astype(int)

print("\n========== FUNNEL OVERVIEW ==========")

print(f"Total marketing leads: {len(leads)}")
print(f"Closed deals: {len(deals)}")
print(f"Converted leads: {leads['converted'].sum()}")
print(f"Non-converted leads: {(leads['converted'] == 0).sum()}")

conversion_rate = leads["converted"].mean() * 100
print(f"Overall conversion rate: {conversion_rate:.2f}%")

print("\n========== CONVERSION BY ORIGIN ==========")

origin_analysis = (
    leads.groupby("origin", dropna=False)["converted"]
    .agg(["count", "sum", "mean"])
    .sort_values("mean", ascending=False)
)

origin_analysis["mean"] *= 100

print(origin_analysis)

print("\n========== CLOSED DEAL PROFILE ==========")

print(deals["business_segment"].value_counts(dropna=False).head(10))

print("\nLead type:")
print(deals["lead_type"].value_counts(dropna=False))

print("\nBusiness type:")
print(deals["business_type"].value_counts(dropna=False))