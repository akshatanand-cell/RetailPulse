"""
06_inventory_optimization.py — Inventory Recommendations
Day 10: use historical demand to compute EOQ and safety stock per product.

EOQ = sqrt(2 * D * S / H)
  D = annual demand units, S = ordering cost per order, H = holding cost/unit/year
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings("ignore")

OUT = "reports/inventory"
os.makedirs(OUT, exist_ok=True)

print("Loading data...")
sales = pd.read_csv("data/processed/sales_clean.csv", parse_dates=["date"])
products = pd.read_csv("data/raw/products.csv")
inventory = pd.read_csv("data/raw/inventory.csv")

last_12m = sales[sales["date"] >= sales["date"].max() - pd.Timedelta(days=365)]
demand = last_12m.groupby("product_id")["quantity"].sum().reset_index()
demand.columns = ["product_id", "annual_demand_units"]

inv = inventory.merge(products[["product_id", "product_name", "category", "cost"]], on="product_id")
inv = inv.merge(demand, on="product_id", how="left").fillna({"annual_demand_units": 0})
inv["daily_demand"] = inv["annual_demand_units"] / 365

S = 500       # assumed ordering cost per order (Rs.)
H_rate = 0.2  # 20% of item cost as annual holding cost
inv["holding_cost"] = inv["cost"] * H_rate
inv["EOQ"] = np.sqrt((2 * inv["annual_demand_units"] * S) / inv["holding_cost"].clip(lower=1)).round(0)

daily_std = last_12m.groupby("product_id")["quantity"].std().reset_index()
daily_std.columns = ["product_id", "demand_std"]
inv = inv.merge(daily_std, on="product_id", how="left").fillna({"demand_std": 0})

Z = 1.65  # 95% service level
inv["safety_stock"] = (Z * inv["demand_std"] * np.sqrt(inv["lead_time_days"])).round(0)
inv["reorder_point_calc"] = (inv["daily_demand"] * inv["lead_time_days"] + inv["safety_stock"]).round(0)

def classify_stock(row):
    if row["stock_on_hand"] <= row["safety_stock"]:
        return "Critical - Order Now"
    elif row["stock_on_hand"] <= row["reorder_point_calc"]:
        return "Reorder Soon"
    elif row["stock_on_hand"] > row["EOQ"] * 1.5:
        return "Overstocked"
    return "Healthy"

inv["stock_status"] = inv.apply(classify_stock, axis=1)
inv["days_of_stock"] = (inv["stock_on_hand"] / inv["daily_demand"].clip(lower=0.01)).round(0)
inv["recommended_order"] = np.where(inv["stock_on_hand"] <= inv["reorder_point_calc"], inv["EOQ"], 0).astype(int)

print("\n=== INVENTORY STATUS SUMMARY ===")
status_counts = inv["stock_status"].value_counts()
for status, count in status_counts.items():
    print(f"  {status}: {count} products")

critical = inv[inv["stock_status"].str.contains("Critical")]
print(f"\nProducts needing immediate reorder: {len(critical)}")
if len(critical) > 0:
    print(critical[["product_name", "category", "stock_on_hand", "reorder_point_calc", "recommended_order"]].head(8).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ["#dc2626", "#f59e0b", "#2563eb", "#16a34a"]
axes[0].pie(status_counts, labels=status_counts.index, autopct="%1.1f%%", colors=colors[:len(status_counts)], startangle=90)
axes[0].set_title("Inventory Health Overview", fontweight="bold")
cat_days = inv.groupby("category")["days_of_stock"].median().sort_values()
axes[1].barh(cat_days.index, cat_days.values, color="#7c3aed")
axes[1].axvline(30, color="red", linestyle="--", linewidth=1.5, label="30-day minimum")
axes[1].set_title("Median Days of Stock by Category", fontweight="bold")
axes[1].set_xlabel("Days of Stock Remaining"); axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/inventory_overview.png", dpi=150)
plt.close()
print("Saved: inventory_overview.png")

output_cols = ["product_id", "product_name", "category", "stock_on_hand", "daily_demand",
               "reorder_point_calc", "safety_stock", "EOQ", "recommended_order", "days_of_stock", "stock_status"]
inv[output_cols].to_csv("data/processed/inventory_recommendations.csv", index=False)
inv[output_cols].to_csv(f"{OUT}/inventory_recommendations.csv", index=False)
print(f"\nRecommendations saved -> data/processed/inventory_recommendations.csv")
print(f"Total recommended order value (Rs.): {(inv['recommended_order'] * inv['cost']).sum():,.0f}")
