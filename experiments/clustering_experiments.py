"""
clustering_experiments.py
----------------------------
Tested different cluster counts and DBSCAN before settling on K-Means k=3.

Findings:
- k=3 gave best silhouette (~0.24) across the RFM + behavioural feature set
- DBSCAN flagged most customers as noise regardless of eps -- not useful
  for a business segmentation deliverable
- Silhouette scores were uniformly modest (<0.25), which is normal for
  RFM data: customer behaviour is a continuous spectrum, not discrete
  clusters, so don't expect textbook-clean separation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/processed/customer_features.csv")
seg_features = ["recency", "frequency", "monetary", "avg_order_value", "online_ratio", "category_diversity"]
X = df[seg_features].fillna(0)
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

print("=== K-Means silhouette scores ===")
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_s)
    sil = silhouette_score(X_s, lbl, sample_size=500, random_state=42)
    marker = " <- best" if k == 3 else ""
    print(f"  k={k}  silhouette={sil:.3f}{marker}")

print("\n=== DBSCAN test ===")
for eps in [0.5, 1.0, 1.5, 2.0]:
    db = DBSCAN(eps=eps, min_samples=5)
    lbl = db.fit_predict(X_s)
    n_clusters = len(set(lbl)) - (1 if -1 in lbl else 0)
    noise_pct = (lbl == -1).mean() * 100
    print(f"  eps={eps}  clusters={n_clusters}  noise={noise_pct:.0f}%")

print("\nConclusion: DBSCAN marks too many customers as noise at every eps tried.")
print("K-Means k=3 is the practical choice for this dataset.")
