import pandas as pd

leads_path = "data/olist_marketing_qualified_leads_dataset.csv"
deals_path = "data/olist_closed_deals_dataset.csv"

leads = pd.read_csv(leads_path)
deals = pd.read_csv(deals_path)

print("\n========== MARKETING LEADS ==========")
print("Rows:", len(leads))
print("Columns:", len(leads.columns))
print("\nColumn names:")
print(leads.columns.tolist())

print("\nFirst 5 rows:")
print(leads.head())

print("\nMissing values:")
print(leads.isnull().sum())


print("\n========== CLOSED DEALS ==========")
print("Rows:", len(deals))
print("Columns:", len(deals.columns))
print("\nColumn names:")
print(deals.columns.tolist())

print("\nFirst 5 rows:")
print(deals.head())

print("\nMissing values:")
print(deals.isnull().sum())