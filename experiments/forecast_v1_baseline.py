"""
forecast_v1_baseline.py
------------------------
First attempt at demand forecasting. Raw revenue, basic lag features,
no log transform. Kept here to show the actual trial-and-error path.

Result: MAPE = 12.17% -- just above the 12% target. Not good enough.

What I noticed: revenue is right-skewed. High-sales days (weekends,
Q4) create large absolute errors that inflate MAPE. That observation
is what led to forecast_v2 and eventually the log-transform fix used
in the final 04_demand_forecasting.py.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/processed/daily_sales.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

def basic_features(d_in):
    d = d_in.copy()
    d["month"] = d["date"].dt.month
    d["dayofweek"] = d["date"].dt.dayofweek
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    d["is_q4"] = d["month"].isin([10, 11, 12]).astype(int)
    for lag in [1, 7, 14, 21, 30]:
        d[f"lag_{lag}"] = d["revenue"].shift(lag)
    for w in [7, 14, 30]:
        d[f"rm_{w}"] = d["revenue"].shift(1).rolling(w).mean()
    return d

df_feat = basic_features(df).dropna()
FEAT = [c for c in df_feat.columns if c not in ["date", "revenue", "rolling_7d", "rolling_30d"]]

split = df_feat["date"].max() - pd.Timedelta(days=60)
train = df_feat[df_feat["date"] <= split]
test = df_feat[df_feat["date"] > split]

sc = StandardScaler()
m = Ridge(alpha=1.0)
m.fit(sc.fit_transform(train[FEAT]), train["revenue"])
preds = np.clip(m.predict(sc.transform(test[FEAT])), 0, None)
mape = mean_absolute_percentage_error(test["revenue"], preds) * 100

print(f"V1 (raw revenue, basic lags): MAPE = {mape:.2f}%")
print("Result: FAIL -- just above target")
print("Next step: try a more complex model -> see forecast_v2_gbr.py")
