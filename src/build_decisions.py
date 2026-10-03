import pandas as pd


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
# 2. CREATE CONVERSION LABEL
# ============================================================

leads["converted"] = leads["mql_id"].isin(
    deals["mql_id"]
).astype(int)


# ============================================================
# 3. DATE FEATURES
# ============================================================

leads["first_contact_date"] = pd.to_datetime(
    leads["first_contact_date"]
)

leads["contact_year"] = (
    leads["first_contact_date"].dt.year
)

leads["contact_month"] = (
    leads["first_contact_date"].dt.month
)

leads["contact_dayofweek"] = (
    leads["first_contact_date"].dt.dayofweek
)


# ============================================================
# 4. CREATE HISTORICAL EVIDENCE
# ============================================================

# Overall conversion rate
overall_rate = leads["converted"].mean() * 100


# Origin-level evidence
origin_stats = (
    leads.groupby("origin", dropna=False)
    .agg(
        leads=("converted", "count"),
        conversions=("converted", "sum"),
        conversion_rate=("converted", "mean")
    )
    .reset_index()
)

origin_stats["conversion_rate"] *= 100


# Landing-page evidence
landing_stats = (
    leads.groupby("landing_page_id")
    .agg(
        leads=("converted", "count"),
        conversions=("converted", "sum"),
        conversion_rate=("converted", "mean")
    )
    .reset_index()
)

landing_stats["conversion_rate"] *= 100


# ============================================================
# 5. ADD EVIDENCE TO EVERY LEAD
# ============================================================

decisions = leads.merge(
    origin_stats[
        [
            "origin",
            "leads",
            "conversions",
            "conversion_rate"
        ]
    ],
    on="origin",
    how="left",
    suffixes=("", "_origin")
)

decisions = decisions.rename(
    columns={
        "leads": "origin_leads",
        "conversions": "origin_conversions",
        "conversion_rate": "origin_conversion_rate"
    }
)


decisions = decisions.merge(
    landing_stats[
        [
            "landing_page_id",
            "leads",
            "conversions",
            "conversion_rate"
        ]
    ],
    on="landing_page_id",
    how="left"
)

decisions = decisions.rename(
    columns={
        "leads": "landing_page_leads",
        "conversions": "landing_page_conversions",
        "conversion_rate": "landing_page_conversion_rate"
    }
)


# ============================================================
# 6. CREATE EVIDENCE SIGNALS
# ============================================================

def evidence_strength(rate, sample_size, overall):
    """
    Prevent tiny groups from looking extremely strong.

    Small groups receive weaker evidence labels.
    """

    if sample_size < 10:
        return "LIMITED"

    if rate >= overall + 3:
        return "STRONG"

    if rate >= overall:
        return "POSITIVE"

    if rate <= overall - 3:
        return "NEGATIVE"

    return "NEUTRAL"


decisions["origin_signal"] = decisions.apply(
    lambda row: evidence_strength(
        row["origin_conversion_rate"],
        row["origin_leads"],
        overall_rate
    ),
    axis=1
)


decisions["landing_signal"] = decisions.apply(
    lambda row: evidence_strength(
        row["landing_page_conversion_rate"],
        row["landing_page_leads"],
        overall_rate
    ),
    axis=1
)


# ============================================================
# 7. DISPLAY SAMPLE
# ============================================================

columns_to_show = [
    "mql_id",
    "first_contact_date",
    "origin",
    "landing_page_id",
    "converted",
    "origin_leads",
    "origin_conversion_rate",
    "origin_signal",
    "landing_page_leads",
    "landing_page_conversion_rate",
    "landing_signal"
]

print("\n========== PRAEVIA EVIDENCE ENGINE ==========")

print(
    f"Overall historical conversion rate: "
    f"{overall_rate:.2f}%"
)

print("\n========== SAMPLE DECISIONS ==========")

print(
    decisions[
        columns_to_show
    ].head(20).to_string(index=False)
)


# ============================================================
# 8. SAVE DECISION DATA
# ============================================================

output_path = "data/praevia_decisions.csv"

decisions.to_csv(
    output_path,
    index=False
)

print(
    f"\nDecision dataset saved to: {output_path}"
)