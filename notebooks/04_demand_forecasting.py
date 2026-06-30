"""
04_demand_forecasting.py — Demand Forecasting
Days 4-8: stationarity check, baseline model, then the model that
actually hits the MAPE target.

HONEST NOTE ON METHOD CHOICE:
The original plan (see Zidio brief) called for Prophet + LSTM. I tried
Prophet first; in this sandboxed build environment the `prophet` /
`pystan` toolchain could not be installed (compiler + cmdstan
dependency issues), and a full LSTM (PyTorch) likewise wasn't available
to install here. Rather than block on tooling, I built a Ridge
regression model with the same inputs Prophet would use conceptually
(trend via lag features, seasonality via Fourier terms) and the same
evaluation discipline (60-day held-out test, MAPE target <=12%).
Earlier attempts and why they failed are in experiments/forecast_v1_baseline.py
and experiments/forecast_v2_gbr.py.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import os, warnings
warnings.filterwarnings("ignore")

OUT = "reports/forecasting"
os.makedirs(OUT, exist_ok=True)

print("Loading daily sales data...")
df = pd.read_csv("data/processed/daily_sales.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"  {len(df)} days: {df['date'].min().date()} -> {df['date'].max().date()}")

# Stationarity check via rolling mean/std
rolling_mean = df["revenue"].rolling(30).mean()
rolling_std = df["revenue"].rolling(30).std()
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(df["date"], df["revenue"], label="Daily Revenue", alpha=0.7, color="#2563eb")
axes[0].plot(df["date"], rolling_mean, label="30d Rolling Mean", color="red", linewidth=2)
axes[0].set_title("Revenue + Rolling Mean", fontweight="bold"); axes[0].legend()
axes[1].plot(df["date"], rolling_std, color="orange", linewidth=2)
axes[1].set_title("30-Day Rolling Std Dev (stability check)", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/stationarity_check.png", dpi=150)
plt.close()
print("Saved: stationarity_check.png")

df["log_rev"] = np.log1p(df["revenue"])

def build_features(d_in):
    d = d_in.copy()
    d["month"] = d["date"].dt.month
    d["dow"] = d["date"].dt.dayofweek
    d["is_wknd"] = (d["dow"] >= 5).astype(int)
    d["is_q4"] = d["month"].isin([10, 11, 12]).astype(int)
    d["m_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["m_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["d_sin"] = np.sin(2 * np.pi * d["dow"] / 7)
    d["d_cos"] = np.cos(2 * np.pi * d["dow"] / 7)
    for lag in [1, 7, 14, 21, 28, 30]:
        d[f"lag_{lag}"] = d["log_rev"].shift(lag)
    for w in [7, 14, 30]:
        d[f"rm_{w}"] = d["log_rev"].shift(1).rolling(w).mean()
        d[f"rs_{w}"] = d["log_rev"].shift(1).rolling(w).std()
    return d

df_feat = build_features(df).dropna()
EXCLUDE = ["date", "revenue", "log_rev", "rolling_7d", "rolling_30d", "month", "dow", "dayofweek", "is_weekend"]
FEAT = [c for c in df_feat.columns if c not in EXCLUDE]

split = df_feat["date"].max() - pd.Timedelta(days=60)
train = df_feat[df_feat["date"] <= split]
test = df_feat[df_feat["date"] > split]

sc = StandardScaler()
model = Ridge(alpha=0.01)
model.fit(sc.fit_transform(train[FEAT]), train["log_rev"])

log_preds = model.predict(sc.transform(test[FEAT]))
preds = np.expm1(log_preds)
mape = mean_absolute_percentage_error(test["revenue"], preds) * 100
rmse = float(np.sqrt(np.mean((test["revenue"].values - preds) ** 2)))
mae = float(np.mean(np.abs(test["revenue"].values - preds)))
print(f"\nMAPE on 60-day hold-out: {mape:.2f}%  (target <=12%: {'PASS' if mape <= 12 else 'FAIL'})")

history_log = df["log_rev"].tolist()
future_dates = pd.date_range(df["date"].max() + pd.Timedelta(days=1), periods=30)
future_preds = []
for fdate in future_dates:
    row = {
        "is_wknd": int(fdate.dayofweek >= 5),
        "is_q4": int(fdate.month in [10, 11, 12]),
        "m_sin": np.sin(2 * np.pi * fdate.month / 12), "m_cos": np.cos(2 * np.pi * fdate.month / 12),
        "d_sin": np.sin(2 * np.pi * fdate.dayofweek / 7), "d_cos": np.cos(2 * np.pi * fdate.dayofweek / 7),
    }
    for lag in [1, 7, 14, 21, 28, 30]:
        row[f"lag_{lag}"] = history_log[-lag] if len(history_log) >= lag else np.mean(history_log[-7:])
    for w in [7, 14, 30]:
        sl = history_log[-w:] if len(history_log) >= w else history_log
        row[f"rm_{w}"] = float(np.mean(sl)); row[f"rs_{w}"] = float(np.std(sl))
    x = pd.DataFrame([row])[FEAT]
    log_pred = float(model.predict(sc.transform(x))[0])
    rev_pred = max(float(np.expm1(log_pred)), 0)
    future_preds.append(rev_pred)
    history_log.append(log_pred)

forecast_df = pd.DataFrame({"date": future_dates, "forecast_revenue": future_preds})

fig, ax = plt.subplots(figsize=(14, 5))
recent = df[df["date"] >= df["date"].max() - pd.Timedelta(days=90)]
ax.plot(recent["date"], recent["revenue"], label="Actual", color="#2563eb", linewidth=1.5)
ax.plot(test["date"], preds, label=f"Predictions (MAPE={mape:.1f}%)", color="#16a34a", linestyle="--", linewidth=1.5)
ax.plot(forecast_df["date"], forecast_df["forecast_revenue"], label="30-Day Forecast", color="#dc2626", linewidth=2)
ax.fill_between(forecast_df["date"], forecast_df["forecast_revenue"] * 0.9, forecast_df["forecast_revenue"] * 1.1,
                alpha=0.2, color="#dc2626", label="+/-10% Band")
ax.axvline(x=df["date"].max(), color="gray", linestyle=":", linewidth=1.5)
ax.set_title("RetailPulse - 30-Day Demand Forecast", fontsize=14, fontweight="bold")
ax.set_ylabel("Daily Revenue (Rs.)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/demand_forecast.png", dpi=150)
plt.close()
print("Saved: demand_forecast.png")

forecast_df.to_csv(f"{OUT}/30day_forecast.csv", index=False)
pd.DataFrame({
    "metric": ["MAPE", "RMSE", "MAE", "forecast_period_start", "forecast_period_end"],
    "value": [round(mape, 2), round(rmse, 2), round(mae, 2), str(future_dates[0].date()), str(future_dates[-1].date())]
}).to_csv(f"{OUT}/model_metrics.csv", index=False)

print(f"\n30-day forecast preview:")
print(forecast_df.head(7).to_string(index=False))
print(f"\nOutputs saved to {OUT}/")
