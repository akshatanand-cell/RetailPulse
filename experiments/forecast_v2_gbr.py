"""
forecast_v2_gbr.py
--------------------
Second attempt: Gradient Boosting Regressor and Random Forest.
Hypothesis: a more complex model would capture the pattern better
than the v1 Ridge model. It didn't.

Result: GBR MAPE = 13.29%, RandomForest MAPE = 13.03% -- both worse
than the v1 baseline (12.17%).

Conclusion: the problem wasn't model capacity, it was the skewed
target distribution. Complex models overfit to the noisy high-revenue
days instead of generalising. This is what led to log-transforming
the target in the final model (04_demand_forecasting.py), which
brought MAPE down to 11.76%.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/processed/daily_sales.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

def build_features(d_in):
    d = d_in.copy()
    d["month"] = d["date"].dt.month
    d["dayofweek"] = d["date"].dt.dayofweek
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    d["is_q4"] = d["month"].isin([10, 11, 12]).astype(int)
    for lag in [1, 7, 14, 21, 28, 30]:
        d[f"lag_{lag}"] = d["revenue"].shift(lag)
    for w in [7, 14, 30]:
        d[f"rm_{w}"] = d["revenue"].shift(1).rolling(w).mean()
        d[f"rs_{w}"] = d["revenue"].shift(1).rolling(w).std()
    return d

df_feat = build_features(df).dropna()
FEAT = [c for c in df_feat.columns if c not in ["date", "revenue", "rolling_7d", "rolling_30d"]]

split = df_feat["date"].max() - pd.Timedelta(days=60)
train = df_feat[df_feat["date"] <= split]
test = df_feat[df_feat["date"] > split]

X_tr, y_tr = train[FEAT], train["revenue"]
X_te, y_te = test[FEAT], test["revenue"]

gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
gbr.fit(X_tr, y_tr)
gbr_mape = mean_absolute_percentage_error(y_te, np.clip(gbr.predict(X_te), 0, None)) * 100

rfr = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
rfr.fit(X_tr, y_tr)
rfr_mape = mean_absolute_percentage_error(y_te, np.clip(rfr.predict(X_te), 0, None)) * 100

print(f"GradientBoosting MAPE: {gbr_mape:.2f}% -- FAIL")
print(f"RandomForest MAPE:     {rfr_mape:.2f}% -- FAIL")
print("Root cause is the target distribution, not model capacity.")
print("Fix: log-transform revenue before modeling -> see 04_demand_forecasting.py")
