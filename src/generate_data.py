import numpy as np
import pandas as pd

np.random.seed(42)

N = 1000

industries = [
    "SaaS", "FinTech", "Healthcare", "Retail",
    "Manufacturing", "EdTech", "Logistics", "Cybersecurity"
]

lead_sources = [
    "Website", "LinkedIn", "Referral",
    "Webinar", "Email Campaign", "Advertisement"
]

companies = [
    "Apex", "Nova", "Vertex", "Quantum", "BluePeak",
    "Nexora", "Innova", "Orion", "Zenith", "Vantage",
    "BrightCore", "CloudAxis", "TechVista", "DataForge",
    "PrimeWorks", "NextGen", "CoreLabs", "Skyline",
    "Fusion", "Elevate"
]

data = {
    "lead_id": [f"L{str(i).zfill(4)}" for i in range(1, N + 1)],
    "company": [
        f"{np.random.choice(companies)} {i}"
        for i in range(1, N + 1)
    ],
    "industry": np.random.choice(industries, N),
    "company_size": np.random.randint(10, 5001, N),
    "lead_source": np.random.choice(lead_sources, N),
    "website_visits": np.random.poisson(8, N),
    "emails_opened": np.random.poisson(5, N),
    "demo_requested": np.random.choice([0, 1], N, p=[0.78, 0.22]),
    "engagement_score": np.random.randint(10, 101, N),
    "deal_value": np.random.randint(10000, 1000000, N),
    "last_activity_days": np.random.randint(0, 61, N),
    "decision_maker": np.random.choice([0, 1], N, p=[0.65, 0.35]),
}

df = pd.DataFrame(data)

# Add realistic missing values
missing_columns = [
    "company_size",
    "website_visits",
    "emails_opened",
    "deal_value"
]

for column in missing_columns:
    indices = np.random.choice(
        df.index,
        size=int(N * 0.03),
        replace=False
    )
    df.loc[indices, column] = np.nan

# Create a hidden business outcome for evaluation.
# PRAEVIA will NOT use this directly when making decisions.
df["converted"] = (
    (
        (df["engagement_score"].fillna(0) >= 70)
        & (df["demo_requested"] == 1)
    )
    | (
        (df["decision_maker"] == 1)
        & (df["engagement_score"].fillna(0) >= 80)
        & (df["last_activity_days"] <= 14)
    )
).astype(int)

# Save dataset
df.to_csv("data/leads.csv", index=False)

print("PRAEVIA dataset created successfully.")
print(f"Total leads: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Converted leads: {df['converted'].sum()}")
print("\nFirst 5 rows:")
print(df.head())