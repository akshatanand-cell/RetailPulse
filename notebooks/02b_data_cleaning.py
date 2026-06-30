"""
02b_data_cleaning.py — Cleaning + Feature Engineering
Day 2 task: clean raw data and build features needed for modeling.
Main output: data/processed/customer_features.csv (RFM + behavioural features)
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

RAW = "data/raw"
OUT = "data/processed"
os.makedirs(OUT, exist_ok=True)

print("Loading raw data...")
sales     = pd.read_csv(f"{RAW}/sales.csv", parse_dates=["date"])
customers = pd.read_csv(f"{RAW}/customers.csv", parse_dates=["join_date"])
products  = pd.read_csv(f"{RAW}/products.csv")
inventory = pd.read_csv(f"{RAW}/inventory.csv")

print("Cleaning...")
before = len(sales)
sales = sales[(sales["revenue"] > 0) & (sales["quantity"] > 0)]
print(f"  Removed {before - len(sales)} bad sales rows")

price_cap = sales["unit_price"].quantile(0.995)
sales["unit_price"] = sales["unit_price"].clip(upper=price_cap)

# RFM
ref_date = sales["date"].max() + pd.Timedelta(days=1)
rfm = (
    sales.groupby("customer_id")
    .agg(recency=("date", lambda x: (ref_date - x.max()).days),
         frequency=("transaction_id", "count"),
         monetary=("revenue", "sum"))
    .reset_index()
)

def score_series(col, ascending=True):
    labels = [1, 2, 3, 4, 5] if not ascending else [5, 4, 3, 2, 1]
    return pd.qcut(col, q=5, labels=labels, duplicates="drop").astype(int)

rfm["R"] = score_series(rfm["recency"], ascending=True)
rfm["F"] = score_series(rfm["frequency"], ascending=False)
rfm["M"] = score_series(rfm["monetary"], ascending=False)
rfm["rfm_score"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
rfm["rfm_total"] = rfm["R"] + rfm["F"] + rfm["M"]

agg2 = sales.groupby("customer_id").agg(
    avg_order_value=("revenue", "mean"),
    total_orders=("transaction_id", "count"),
    online_ratio=("channel", lambda x: (x == "Online").mean()),
    avg_discount_used=("discount", "mean"),
    last_order_date=("date", "max"),
    first_order_date=("date", "min"),
).reset_index()
agg2["customer_lifetime_days"] = (agg2["last_order_date"] - agg2["first_order_date"]).dt.days

cat_merge = sales.merge(products[["product_id", "category"]], on="product_id")
cat_div = cat_merge.groupby("customer_id")["category"].nunique().reset_index()
cat_div.columns = ["customer_id", "category_diversity"]

features = (
    rfm
    .merge(customers[["customer_id", "age", "gender", "region", "loyalty_tier"]], on="customer_id")
    .merge(agg2.drop(columns=["last_order_date", "first_order_date"]), on="customer_id")
    .merge(cat_div, on="customer_id")
)

features["gender_enc"] = features["gender"].map({"M": 0, "F": 1, "Other": 2})
features["region_enc"] = features["region"].astype("category").cat.codes
features["loyalty_enc"] = features["loyalty_tier"].map({"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3})

daily_sales = sales.groupby("date")["revenue"].sum().reset_index().sort_values("date")
daily_sales["rolling_7d"] = daily_sales["revenue"].rolling(7).mean()
daily_sales["rolling_30d"] = daily_sales["revenue"].rolling(30).mean()
daily_sales["month"] = daily_sales["date"].dt.month
daily_sales["dayofweek"] = daily_sales["date"].dt.dayofweek
daily_sales["is_weekend"] = (daily_sales["dayofweek"] >= 5).astype(int)

features.to_csv(f"{OUT}/customer_features.csv", index=False)
daily_sales.to_csv(f"{OUT}/daily_sales.csv", index=False)
sales.to_csv(f"{OUT}/sales_clean.csv", index=False)

print(f"\nDone.")
print(f"  customer_features : {features.shape}")
print(f"  daily_sales       : {daily_sales.shape}")
print(f"  sales_clean       : {sales.shape}")
print(f"\nSample RFM scores:")
print(rfm[["customer_id", "recency", "frequency", "monetary", "R", "F", "M", "rfm_total"]].head(5).to_string(index=False))
