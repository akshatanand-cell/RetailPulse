"""
churn_v1_leakage.py
---------------------
First churn model attempt. Included recency and RFM scores as features.
AUC came out as 1.000 -- suspiciously perfect.

The problem: recency directly encodes the churn label. Churn is defined
as "no purchase in the last 90 days", which IS high recency. Including
recency as a model feature means the model can read the answer off the
label instead of learning a real pattern -- classic data leakage.

Fixed in the final model (05_churn_prediction.py) by removing recency,
R, F, M and rfm_total from the feature set. Final AUC: 0.9476, using
only behavioural and demographic features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

features = pd.read_csv("data/processed/customer_features.csv")
sales = pd.read_csv("data/processed/sales_clean.csv", parse_dates=["date"])

ref_date = sales["date"].max()
cutoff_mid = ref_date - pd.Timedelta(days=90)
cutoff_start = ref_date - pd.Timedelta(days=180)

obs = set(sales[(sales["date"] >= cutoff_start) & (sales["date"] < cutoff_mid)]["customer_id"])
ret = set(sales[sales["date"] >= cutoff_mid]["customer_id"])

df = features[features["customer_id"].isin(obs)].copy()
df["churned"] = df["customer_id"].isin(obs - ret).astype(int)

# V1 feature set -- includes recency, which leaks the label
V1_FEATURES = ["recency", "frequency", "monetary", "avg_order_value", "online_ratio",
               "R", "F", "M", "rfm_total", "customer_lifetime_days", "age", "loyalty_enc"]
V1_FEATURES = [f for f in V1_FEATURES if f in df.columns]

X = df[V1_FEATURES].fillna(0)
y = df["churned"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
sw = np.where(y_tr == 1, neg / pos, 1.0)

m = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
m.fit(X_tr, y_tr, sample_weight=sw)
auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])

print(f"V1 AUC (with recency + RFM scores): {auc:.4f}")
print("This looks perfect but it's data leakage -- recency = churn by definition")
print("See 05_churn_prediction.py for the fixed version without leaky features")
