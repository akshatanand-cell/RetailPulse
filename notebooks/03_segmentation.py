"""
03_segmentation.py — Customer Segmentation
Day 3: K-Means clustering on RFM features. DBSCAN also tested for
comparison (see experiments/clustering_experiments.py for the full
trial log) but K-Means was kept as the final approach.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os, warnings
warnings.filterwarnings("ignore")

OUT = "reports/segmentation"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("data/processed/customer_features.csv")
print(f"Loaded {len(df)} customers")

seg_features = ["recency", "frequency", "monetary", "avg_order_value", "online_ratio", "category_diversity"]
X = df[seg_features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Finding optimal number of clusters...")
inertias, sil_scores = [], []
k_range = range(2, 10)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels, sample_size=500, random_state=42))
    print(f"  k={k}  silhouette={sil_scores[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_range), inertias, "bo-", linewidth=2)
axes[0].set_title("Elbow Curve", fontweight="bold")
axes[0].set_xlabel("Number of Clusters (k)"); axes[0].set_ylabel("Inertia")
axes[1].plot(list(k_range), sil_scores, "ro-", linewidth=2)
axes[1].set_title("Silhouette Scores", fontweight="bold")
axes[1].set_xlabel("Number of Clusters (k)"); axes[1].set_ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig(f"{OUT}/elbow_silhouette.png", dpi=150)
plt.close()

best_k = list(k_range)[int(np.argmax(sil_scores))]
print(f"\nBest k = {best_k} (silhouette = {max(sil_scores):.3f})")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"] = km_final.fit_predict(X_scaled)

cluster_summary = df.groupby("cluster")[seg_features + ["rfm_total"]].mean().round(2)
cluster_summary["size"] = df.groupby("cluster").size()

rank = cluster_summary["rfm_total"].rank(ascending=False).astype(int)
segment_labels = {1: "Champions", 2: "Loyal Customers", 3: "Potential Loyalists",
                   4: "At-Risk Customers", 5: "Hibernating", 6: "Lost"}
df["segment"] = df["cluster"].map(lambda c: segment_labels.get(rank[c], f"Segment {rank[c]}"))

print("\nCluster Summary:")
print(cluster_summary.to_string())

colors = ["#2563eb", "#16a34a", "#dc2626", "#f59e0b", "#7c3aed", "#ec4899"]
seg_counts = df["segment"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].pie(seg_counts, labels=seg_counts.index, autopct="%1.1f%%", colors=colors[:len(seg_counts)], startangle=140)
axes[0].set_title("Customer Segment Distribution", fontweight="bold")
for i, seg in enumerate(df["segment"].unique()):
    mask = df["segment"] == seg
    axes[1].scatter(df.loc[mask, "recency"], df.loc[mask, "monetary"] / 1000, label=seg, alpha=0.5, s=20, color=colors[i % len(colors)])
axes[1].set_xlabel("Recency (days since last purchase)"); axes[1].set_ylabel("Total Spend (Rs. Thousands)")
axes[1].set_title("Segments: Recency vs Spend", fontweight="bold")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/segments_viz.png", dpi=150)
plt.close()

seg_revenue = df.groupby("segment")["monetary"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(seg_revenue.index, seg_revenue.values / 1e6, color=colors[:len(seg_revenue)])
ax.set_title("Revenue Contribution by Segment", fontweight="bold")
ax.set_ylabel("Total Revenue (Rs. Millions)")
plt.xticks(rotation=20)
for bar, val in zip(bars, seg_revenue.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"Rs.{val/1e6:.1f}M", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/segment_revenue.png", dpi=150)
plt.close()

df.to_csv("data/processed/customer_segments.csv", index=False)
cluster_summary.to_csv(f"{OUT}/cluster_summary.csv")
print(f"\nSegmented data saved -> data/processed/customer_segments.csv")
print(f"Charts saved -> {OUT}/")
