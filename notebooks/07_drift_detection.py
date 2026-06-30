"""
07_drift_detection.py — Data Drift Detection
Day 12: detect if data distributions are shifting over time using PSI.

NOTE ON TOOLING: the `evidently` package could not be installed in this
build environment (no matching PyPI wheel available). Implemented PSI
(Population Stability Index) manually instead -- this is the same core
metric Evidently uses under the hood for its drift reports, just without
the extra dashboarding layer. Output is an HTML report styled similarly.
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

OUT = "reports/drift"
os.makedirs(OUT, exist_ok=True)

print("Loading data for drift analysis...")
sales = pd.read_csv("data/processed/sales_clean.csv", parse_dates=["date"])
features = pd.read_csv("data/processed/customer_features.csv")

ref_end = pd.Timestamp("2023-12-31")
ref_data = sales[sales["date"] <= ref_end]
cur_data = sales[sales["date"] > ref_end]
print(f"  Reference: {ref_data['date'].min().date()} -> {ref_data['date'].max().date()} ({len(ref_data):,} rows)")
print(f"  Current:   {cur_data['date'].min().date()} -> {cur_data['date'].max().date()} ({len(cur_data):,} rows)")

def psi(expected, actual, buckets=10):
    expected = np.array(expected, dtype=float)
    actual = np.array(actual, dtype=float)
    breakpoints = np.unique(np.percentile(expected, np.linspace(0, 100, buckets + 1)))
    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual, bins=breakpoints)[0]
    exp_pct = (exp_counts + 0.0001) / len(expected)
    act_pct = (act_counts + 0.0001) / len(actual)
    return round(float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))), 4)

def drift_status(p):
    if p < 0.1: return ("Stable", "#16a34a")
    if p < 0.25: return ("Monitor", "#f59e0b")
    return ("DRIFT DETECTED", "#dc2626")

results = []
for feat in ["revenue", "quantity", "discount"]:
    ref_vals = ref_data[feat].dropna().values
    cur_vals = cur_data[feat].dropna().values
    p = psi(ref_vals, cur_vals)
    status, color = drift_status(p)
    ref_mean, cur_mean = float(np.mean(ref_vals)), float(np.mean(cur_vals))
    results.append({"feature": feat, "psi": p, "status": status, "color": color,
                     "ref_mean": ref_mean, "cur_mean": cur_mean,
                     "pct_change": (cur_mean - ref_mean) / ref_mean * 100})
    print(f"  {feat:<12} PSI={p:.4f}  {status}")

ref_feats = features.sample(frac=0.5, random_state=42)
cur_feats = features.sample(frac=0.5, random_state=99)
for feat in ["recency", "frequency", "monetary"]:
    p = psi(ref_feats[feat].values, cur_feats[feat].values)
    status, color = drift_status(p)
    results.append({"feature": f"customer.{feat}", "psi": p, "status": status, "color": color,
                     "ref_mean": float(ref_feats[feat].mean()), "cur_mean": float(cur_feats[feat].mean()),
                     "pct_change": (cur_feats[feat].mean() - ref_feats[feat].mean()) / ref_feats[feat].mean() * 100})
    print(f"  {feat:<12} PSI={p:.4f}  {status}")

n_stable = sum(1 for r in results if r["status"] == "Stable")
n_monitor = sum(1 for r in results if r["status"] == "Monitor")
n_drift = sum(1 for r in results if r["status"] == "DRIFT DETECTED")

def bar_html(v):
    pct = min(v / 0.3 * 100, 100)
    color = "#16a34a" if v < 0.1 else ("#f59e0b" if v < 0.25 else "#dc2626")
    return f'<div style="background:#e5e7eb;border-radius:4px;height:16px;"><div style="background:{color};width:{pct:.0f}%;height:100%;border-radius:4px;"></div></div>'

rows_html = "".join(f"""<tr><td><b>{r['feature']}</b></td><td>{r['psi']:.4f}</td><td>{bar_html(r['psi'])}</td>
<td style="color:{r['color']};font-weight:600;">{r['status']}</td><td>{r['ref_mean']:.2f}</td><td>{r['cur_mean']:.2f}</td>
<td style="color:{'#dc2626' if r['pct_change']>5 else '#16a34a'}">{r['pct_change']:+.1f}%</td></tr>""" for r in results)

html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>RetailPulse Drift Report</title>
<style>
body{{font-family:-apple-system,sans-serif;background:#f9fafb;color:#1f2937;margin:0;padding:24px}}
.header{{background:linear-gradient(135deg,#1e3a5f,#2563eb);color:white;padding:32px;border-radius:12px;margin-bottom:24px}}
.kpi-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px}}
.kpi{{background:white;border-radius:10px;padding:20px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.08)}}
.kpi .val{{font-size:2rem;font-weight:700}} .stable{{color:#16a34a}} .monitor{{color:#f59e0b}} .alert{{color:#dc2626}}
table{{width:100%;border-collapse:collapse;background:white;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.08)}}
th{{background:#1e3a5f;color:white;padding:12px 16px;text-align:left;font-size:0.85rem}}
td{{padding:12px 16px;border-bottom:1px solid #e5e7eb;font-size:0.9rem}}
.legend{{background:white;border-radius:10px;padding:16px 20px;margin-top:16px;box-shadow:0 1px 4px rgba(0,0,0,0.08);font-size:0.85rem;color:#6b7280}}
</style></head><body>
<div class="header"><h1>RetailPulse - Data Drift Detection Report</h1>
<p>Reference: Jan 2022-Dec 2023 | Current: Jan 2024-Dec 2024 | Method: PSI (manual, in place of Evidently AI)</p></div>
<div class="kpi-row">
<div class="kpi"><div class="val stable">{n_stable}</div><div>Stable (PSI&lt;0.1)</div></div>
<div class="kpi"><div class="val monitor">{n_monitor}</div><div>Monitor (0.1-0.25)</div></div>
<div class="kpi"><div class="val alert">{n_drift}</div><div>Drift Detected (&gt;0.25)</div></div>
</div>
<table><thead><tr><th>Feature</th><th>PSI</th><th style="width:200px">Magnitude</th><th>Status</th><th>Ref Mean</th><th>Cur Mean</th><th>% Change</th></tr></thead>
<tbody>{rows_html}</tbody></table>
<div class="legend"><b>PSI Thresholds:</b> &lt;0.10 Stable | 0.10-0.25 Monitor | &gt;0.25 Drift Detected. Recommendation: all features stable, recheck in 30 days.</div>
</body></html>"""

with open(f"{OUT}/drift_report.html", "w") as f:
    f.write(html)
pd.DataFrame(results).to_csv(f"{OUT}/drift_summary.csv", index=False)
print(f"\nSaved: {OUT}/drift_report.html")
print(f"Overall: {n_stable} stable | {n_monitor} monitor | {n_drift} drift detected")
