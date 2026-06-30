"""
05_churn_prediction.py — Customer Churn Prediction
Day 9: predict which customers are likely to stop buying.

HONEST NOTE ON METHOD CHOICE:
The brief calls for XGBoost. The `xgboost` package could not be
installed in this build environment (no matching distribution
available). I used scikit-learn's GradientBoostingClassifier instead
-- same gradient-boosted-trees family, same bias/variance behaviour,
same need for class-imbalance handling. I used `sample_weight` to
achieve the same effect XGBoost's `scale_pos_weight` gives.

SHAP was also unavailable to install, so feature explainability uses
scikit-learn's built-in `permutation_importance` plus the model's
native `feature_importances_` -- same goal (which features drive the
prediction) via a different tool.

Churn definition: active in the 90-180 day window before the
reference date, but no purchase in the most recent 90 days.

IMPORTANT FIX: an early version of this model included `recency` and
the RFM `R` score as features and got AUC = 1.00. That is a giveaway
of data leakage, not a good model -- recency directly encodes whether
someone churned (churned customers have high recency by definition).
That version is kept in experiments/churn_v1_leakage.py. This final
version excludes recency and the RFM R score; AUC settles at a more
realistic value.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import os, warnings
warnings.filterwarnings("ignore")

OUT = "reports/churn"
os.makedirs(OUT, exist_ok=True)

print("Loading data...")
sales = pd.read_csv("data/processed/sales_clean.csv", parse_dates=["date"])
features = pd.read_csv("data/processed/customer_features.csv")

ref_date = sales["date"].max()
cutoff_mid = ref_date - pd.Timedelta(days=90)
cutoff_start = ref_date - pd.Timedelta(days=180)

obs_window = sales[(sales["date"] >= cutoff_start) & (sales["date"] < cutoff_mid)]
active_in_obs = set(obs_window["customer_id"].unique())
retained = set(sales[sales["date"] >= cutoff_mid]["customer_id"].unique())
churned = active_in_obs - retained

print(f"Observation window: {cutoff_start.date()} to {cutoff_mid.date()}")
print(f"Outcome window:     {cutoff_mid.date()} to {ref_date.date()}")
print(f"Active customers in obs: {len(active_in_obs)}")
print(f"Churned: {len(churned)} ({len(churned)/len(active_in_obs)*100:.1f}%)")

df = features[features["customer_id"].isin(active_in_obs)].copy()
df["churned"] = df["customer_id"].isin(churned).astype(int)

MODEL_FEATURES = [
    "frequency", "monetary", "avg_order_value", "online_ratio",
    "avg_discount_used", "customer_lifetime_days", "category_diversity",
    "F", "M", "age", "gender_enc", "region_enc", "loyalty_enc",
]
MODEL_FEATURES = [f for f in MODEL_FEATURES if f in df.columns]

X = df[MODEL_FEATURES].fillna(0)
y = df["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
sample_weight = np.where(y_train == 1, neg / pos, 1.0)

model = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, random_state=42)
model.fit(X_train, y_train, sample_weight=sample_weight)

proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
preds_binary = (proba >= 0.5).astype(int)

print(f"\nAUC-ROC: {auc:.4f}  (target >=0.88: {'PASS' if auc >= 0.88 else 'FAIL'})")
print("\nClassification Report:")
print(classification_report(y_test, preds_binary, target_names=["Retained", "Churned"]))

n_top20 = max(int(0.2 * len(y_test)), 1)
top20_idx = np.argsort(proba)[::-1][:n_top20]
prec_top20 = float(y_test.iloc[top20_idx].mean())
print(f"Precision @ top 20%: {prec_top20:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fpr, tpr, _ = roc_curve(y_test, proba)
axes[0].plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"AUC = {auc:.3f}")
axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[0].fill_between(fpr, tpr, alpha=0.1, color="#2563eb")
axes[0].set_title("ROC Curve", fontweight="bold"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].legend()

axes[1].hist(proba[y_test == 0], bins=30, alpha=0.6, color="#16a34a", label="Retained")
axes[1].hist(proba[y_test == 1], bins=30, alpha=0.6, color="#dc2626", label="Churned")
axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Threshold=0.5")
axes[1].set_title("Predicted Churn Probability", fontweight="bold")
axes[1].set_xlabel("Churn Probability Score"); axes[1].set_ylabel("Count"); axes[1].legend()

perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc")
fi = pd.Series(perm.importances_mean, index=MODEL_FEATURES).sort_values(ascending=True).tail(12)
axes[2].barh(fi.index, fi.values, color="#7c3aed")
axes[2].set_title("Permutation Importance (Top 12)", fontweight="bold")
axes[2].set_xlabel("Mean AUC Drop When Shuffled")

plt.suptitle("Churn Prediction Model - RetailPulse", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/churn_model_report.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: churn_model_report.png")

all_X = features[MODEL_FEATURES].fillna(0)
features["churn_probability"] = model.predict_proba(all_X)[:, 1]
features["churn_risk"] = pd.cut(features["churn_probability"], bins=[0, 0.3, 0.6, 1.0],
                                 labels=["Low Risk", "Medium Risk", "High Risk"])

at_risk = features[features["churn_risk"] == "High Risk"].sort_values("churn_probability", ascending=False)
print(f"\nHigh-risk customers (intervention needed): {len(at_risk)}")
print(at_risk[["customer_id", "churn_probability", "recency", "monetary"]].head(10).to_string(index=False))

features[["customer_id", "churn_probability", "churn_risk"]].to_csv("data/processed/churn_scores.csv", index=False)
pd.DataFrame({
    "metric": ["AUC_ROC", "Precision_Top20pct", "Churn_Rate_pct", "High_Risk_Customers"],
    "value": [round(auc, 4), round(prec_top20, 4), round(df["churned"].mean() * 100, 1), len(at_risk)]
}).to_csv(f"{OUT}/churn_metrics.csv", index=False)

print(f"\nChurn scores saved -> data/processed/churn_scores.csv")
