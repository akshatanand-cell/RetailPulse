"""
02a_data_validation.py — Data Quality Validation
Day 2 task: validate data quality before modeling.

Implements Great Expectations-style checks manually rather than using the
`great_expectations` package, which has a heavy dependency footprint
(SQLAlchemy, full config scaffolding) that wasn't worth the overhead for
a single-source CSV pipeline of this size. The checks below cover the
same ground GE would: completeness, range validity, uniqueness,
referential integrity between tables.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

OUT = "reports/validation"
os.makedirs(OUT, exist_ok=True)

print("Running data quality checks...")

sales     = pd.read_csv("data/raw/sales.csv", parse_dates=["date"])
customers = pd.read_csv("data/raw/customers.csv", parse_dates=["join_date"])
products  = pd.read_csv("data/raw/products.csv")
inventory = pd.read_csv("data/raw/inventory.csv")

results = []

def check(name, passed, detail, critical=True):
    status = "PASS" if passed else ("FAIL" if critical else "WARN")
    results.append({"check": name, "status": status, "detail": detail})
    icon = "OK" if passed else ("FAIL" if critical else "WARN")
    print(f"  [{icon}] {name}: {detail}")
    return passed

check("Sales - no missing values", sales.isnull().sum().sum() == 0,
      f"{sales.isnull().sum().sum()} missing values found")
check("Sales - revenue always positive", (sales["revenue"] > 0).all(),
      f"{(sales['revenue'] <= 0).sum()} rows with non-positive revenue")
check("Sales - quantity always positive", (sales["quantity"] > 0).all(),
      f"{(sales['quantity'] <= 0).sum()} rows with non-positive quantity")
check("Sales - no duplicate transaction IDs",
      sales["transaction_id"].nunique() == len(sales),
      f"{len(sales) - sales['transaction_id'].nunique()} duplicate transaction IDs")
check("Sales - date range valid",
      (sales["date"].min() >= pd.Timestamp("2020-01-01")) and (sales["date"].max() <= pd.Timestamp("2025-12-31")),
      f"Range: {sales['date'].min().date()} to {sales['date'].max().date()}")
check("Sales - discount between 0 and 1",
      ((sales["discount"] >= 0) & (sales["discount"] <= 1)).all(),
      f"Min={sales['discount'].min():.2f}, Max={sales['discount'].max():.2f}")
check("Sales - unit price positive", (sales["unit_price"] > 0).all(),
      f"Min price: Rs.{sales['unit_price'].min():.2f}")

check("Customers - no missing values", customers.isnull().sum().sum() == 0,
      f"{customers.isnull().sum().sum()} missing values")
check("Customers - age in valid range",
      ((customers["age"] >= 18) & (customers["age"] <= 100)).all(),
      f"Age range: {customers['age'].min()} - {customers['age'].max()}")
check("Customers - no duplicate IDs",
      customers["customer_id"].nunique() == len(customers),
      f"{len(customers) - customers['customer_id'].nunique()} duplicates")
check("Sales customer IDs exist in customer table",
      sales["customer_id"].isin(customers["customer_id"]).all(),
      f"{(~sales['customer_id'].isin(customers['customer_id'])).sum()} orphan sales records")

check("Products - no missing values", products.isnull().sum().sum() == 0,
      f"{products.isnull().sum().sum()} missing values")
check("Products - price greater than cost", (products["price"] > products["cost"]).all(),
      f"{(products['price'] <= products['cost']).sum()} products with price <= cost")
check("Sales product IDs exist in product table",
      sales["product_id"].isin(products["product_id"]).all(),
      f"{(~sales['product_id'].isin(products['product_id'])).sum()} orphan product IDs")

check("Inventory - no missing values", inventory.isnull().sum().sum() == 0,
      f"{inventory.isnull().sum().sum()} missing values")
check("Inventory - stock on hand non-negative", (inventory["stock_on_hand"] >= 0).all(),
      f"{(inventory['stock_on_hand'] < 0).sum()} negative stock values")
check("Inventory - lead time positive", (inventory["lead_time_days"] > 0).all(),
      f"Lead times: {inventory['lead_time_days'].min()} - {inventory['lead_time_days'].max()} days")

total = len(results)
passed = sum(1 for r in results if r["status"] == "PASS")
failed = sum(1 for r in results if r["status"] == "FAIL")
warned = sum(1 for r in results if r["status"] == "WARN")

print(f"\n{'='*50}")
print(f"Validation complete: {passed}/{total} checks passed")
if failed > 0:
    print(f"FAILED: {failed} critical checks - fix before modeling")
else:
    print("All critical checks passed - data is ready for modeling")

rows_html = ""
for r in results:
    color = {"PASS": "#16a34a", "FAIL": "#dc2626", "WARN": "#f59e0b"}[r["status"]]
    bg = {"PASS": "#f0fdf4", "FAIL": "#fef2f2", "WARN": "#fffbeb"}[r["status"]]
    rows_html += f"""<tr style="background:{bg}"><td>{r['check']}</td><td style="color:{color};font-weight:600">{r['status']}</td><td>{r['detail']}</td></tr>"""

html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>RetailPulse Data Validation</title>
<style>
body{{font-family:-apple-system,sans-serif;background:#f9fafb;padding:24px;color:#1f2937}}
.header{{background:linear-gradient(135deg,#1e3a5f,#2563eb);color:white;padding:24px;border-radius:10px;margin-bottom:20px}}
.kpis{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px}}
.kpi{{background:white;border-radius:8px;padding:16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.08)}}
.kpi .val{{font-size:2rem;font-weight:700}} .kpi .lbl{{font-size:0.8rem;color:#6b7280}}
table{{width:100%;border-collapse:collapse;background:white;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.08)}}
th{{background:#1e3a5f;color:white;padding:10px 14px;text-align:left;font-size:0.85rem}}
td{{padding:10px 14px;border-bottom:1px solid #e5e7eb;font-size:0.9rem}}
tr:last-child td{{border-bottom:none}}
</style></head><body>
<div class="header"><h2 style="margin:0">RetailPulse - Data Validation Report</h2>
<p style="margin:4px 0 0;opacity:0.85">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | {total} checks run</p></div>
<div class="kpis">
<div class="kpi"><div class="val" style="color:#16a34a">{passed}</div><div class="lbl">Checks Passed</div></div>
<div class="kpi"><div class="val" style="color:#dc2626">{failed}</div><div class="lbl">Failed (Critical)</div></div>
<div class="kpi"><div class="val" style="color:#f59e0b">{warned}</div><div class="lbl">Warnings</div></div>
</div>
<table><thead><tr><th>Check</th><th>Status</th><th>Detail</th></tr></thead><tbody>{rows_html}</tbody></table>
<p style="text-align:center;color:#9ca3af;font-size:0.8rem;margin-top:16px">RetailPulse - Zidio Development - Data Validation - 2026</p>
</body></html>"""

with open(f"{OUT}/validation_report.html", "w") as f:
    f.write(html)
pd.DataFrame(results).to_csv(f"{OUT}/validation_results.csv", index=False)
print(f"Report saved -> {OUT}/validation_report.html")
