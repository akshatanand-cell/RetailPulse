"""
01_eda.py — Exploratory Data Analysis
Day 1 task: understand the data before touching any model.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os, warnings
warnings.filterwarnings("ignore")

REPORTS = "reports/eda"
os.makedirs(REPORTS, exist_ok=True)

print("Loading data...")
sales     = pd.read_csv("data/raw/sales.csv", parse_dates=["date"])
customers = pd.read_csv("data/raw/customers.csv", parse_dates=["join_date"])
products  = pd.read_csv("data/raw/products.csv")
inventory = pd.read_csv("data/raw/inventory.csv")

print("\n=== DATASET SUMMARY ===")
print(f"Sales        : {len(sales):,} rows | {sales['date'].min().date()} -> {sales['date'].max().date()}")
print(f"Customers    : {len(customers):,} unique")
print(f"Products     : {len(products):,} unique | {products['category'].nunique()} categories")
print(f"Missing vals : {sales.isnull().sum().sum()} (sales), {customers.isnull().sum().sum()} (customers)")

# 1. Revenue over time
monthly = sales.set_index("date").resample("ME")["revenue"].sum().reset_index()
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(monthly["date"], monthly["revenue"] / 1e6, color="#2563eb", linewidth=2)
ax.fill_between(monthly["date"], monthly["revenue"] / 1e6, alpha=0.15, color="#2563eb")
ax.set_title("Monthly Revenue (2022-2024)", fontsize=14, fontweight="bold")
ax.set_ylabel("Revenue (Rs. Millions)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{REPORTS}/revenue_over_time.png", dpi=150)
plt.close()
print("Saved: revenue_over_time.png")

# 2. Revenue by category
merged = sales.merge(products[["product_id", "category"]], on="product_id")
cat_rev = merged.groupby("category")["revenue"].sum().sort_values()
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(cat_rev.index, cat_rev.values / 1e6, color="#16a34a")
ax.set_xlabel("Total Revenue (Rs. Millions)")
ax.set_title("Revenue by Product Category", fontsize=13, fontweight="bold")
for bar, val in zip(bars, cat_rev.values):
    ax.text(val / 1e6 + 0.1, bar.get_y() + bar.get_height() / 2, f"Rs.{val/1e6:.1f}M", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{REPORTS}/revenue_by_category.png", dpi=150)
plt.close()
print("Saved: revenue_by_category.png")

# 3. Customer demographics
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(customers["age"], bins=25, color="#7c3aed", edgecolor="white", linewidth=0.5)
axes[0].set_title("Customer Age Distribution", fontweight="bold")
axes[0].set_xlabel("Age"); axes[0].set_ylabel("Count")
loyalty_counts = customers["loyalty_tier"].value_counts()
axes[1].pie(loyalty_counts, labels=loyalty_counts.index, autopct="%1.1f%%", startangle=90,
            colors=["#cd7f32", "#c0c0c0", "#ffd700", "#e5e4e2"])
axes[1].set_title("Loyalty Tier Split", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS}/customer_demographics.png", dpi=150)
plt.close()
print("Saved: customer_demographics.png")

# 4. Discount impact
fig, ax = plt.subplots(figsize=(8, 4))
discount_bins = pd.cut(sales["discount"], bins=[0, 0.01, 0.06, 0.11, 0.16, 0.21],
                        labels=["0%", "1-5%", "6-10%", "11-15%", "16-20%"], include_lowest=True)
discount_rev = sales.groupby(discount_bins)["revenue"].mean()
ax.bar(discount_rev.index.astype(str), discount_rev.values, color="#dc2626", width=0.5)
ax.set_title("Avg Transaction Revenue by Discount Level", fontweight="bold")
ax.set_xlabel("Discount Range"); ax.set_ylabel("Avg Revenue (Rs.)")
plt.tight_layout()
plt.savefig(f"{REPORTS}/discount_analysis.png", dpi=150)
plt.close()
print("Saved: discount_analysis.png")

# 5. Correlation heatmap
num_cols = ["quantity", "unit_price", "discount", "revenue"]
corr = sales[num_cols].corr()
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=30)
ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=10)
ax.set_title("Correlation Heatmap - Sales Features", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS}/correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: correlation_heatmap.png")

# 6. Channel split over time
channel_q = sales.groupby([pd.Grouper(key="date", freq="QE"), "channel"])["revenue"].sum().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, 4))
channel_q.plot(kind="bar", ax=ax, color=["#f59e0b", "#3b82f6"], width=0.7)
ax.set_title("Quarterly Revenue: Online vs In-Store", fontweight="bold")
ax.set_ylabel("Revenue (Rs.)")
ax.set_xticklabels([str(d.date()) for d in channel_q.index], rotation=35)
plt.tight_layout()
plt.savefig(f"{REPORTS}/channel_split.png", dpi=150)
plt.close()
print("Saved: channel_split.png")

summary = {
    "total_revenue": round(sales["revenue"].sum(), 2),
    "avg_order_value": round(sales["revenue"].mean(), 2),
    "total_transactions": len(sales),
    "unique_customers": sales["customer_id"].nunique(),
    "top_category": cat_rev.idxmax(),
    "avg_discount": round(sales["discount"].mean() * 100, 1),
}
pd.Series(summary).to_csv(f"{REPORTS}/summary_stats.csv")
print("\n=== KEY METRICS ===")
for k, v in summary.items():
    print(f"  {k:<22}: {v}")
print(f"\nAll EDA outputs saved to {REPORTS}/")
