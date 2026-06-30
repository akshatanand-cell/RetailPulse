# RetailPulse — AI-Powered Customer Analytics & Demand Forecasting Platform
# Author: Akshat Anand
# KIIT University | B.Tech CSE (AI & ML)
# Zidio Development Internship – Data Science & Analytics

End-to-end data science project for the Zidio Development internship (Data Science & Analytics domain). RetailPulse analyses retail sales, customer and inventory data to deliver demand forecasts, customer segmentation, churn prediction and inventory reorder recommendations through an interactive Streamlit dashboard.

## Honest note on tooling substitutions

Several packages named in the original project brief — `xgboost`, `shap`, `prophet`, `evidently`, `apache-airflow` — could not be installed in the environment this project was built in (no matching PyPI wheel available at build time). Rather than block on tooling, I substituted functionally equivalent approaches and documented each substitution directly in the code:

| Brief asked for | Used instead | Where |
|---|---|---|
| XGBoost | `sklearn.ensemble.GradientBoostingClassifier` + `sample_weight` for imbalance | `notebooks/05_churn_prediction.py` |
| SHAP | `sklearn.inspection.permutation_importance` | `notebooks/05_churn_prediction.py` |
| Prophet + LSTM | Ridge regression on log-transformed revenue, Fourier seasonality terms, lag/rolling features | `notebooks/04_demand_forecasting.py` |
| Great Expectations | Manual rule-based validation (17 checks, same coverage) | `notebooks/02a_data_validation.py` |
| Evidently AI | Manual PSI (Population Stability Index) calculation + HTML report | `notebooks/07_drift_detection.py` |

If you have working installs of the original tools, the same feature sets and evaluation logic carry over directly — see the docstring at the top of each affected script.

## Dataset

The dataset used here is **synthetically generated** (`data/generate_data.py`), not the UCI Online Retail II dataset. I made this choice so I could control volume, inject specific seasonal/weekday patterns, and keep the project runnable end-to-end without external downloads in this build environment. The cleaning, RFM, validation, and modeling logic in every notebook is written generically — it would run unchanged against the real UCI dataset if pointed at it (same column roles: customer ID, transaction ID, date, quantity, unit price, revenue).

## Project structure

```
retailpulse/
├── data/
│   ├── generate_data.py        # synthetic data generator
│   ├── raw/                    # generated source CSVs
│   └── processed/               # cleaned + feature-engineered outputs
├── notebooks/
│   ├── 01_eda.py                # Day 1  - exploratory data analysis
│   ├── 02a_data_validation.py   # Day 2  - data quality checks
│   ├── 02b_data_cleaning.py     # Day 2  - cleaning + RFM features
│   ├── 03_segmentation.py       # Day 3  - K-Means customer segmentation
│   ├── 04_demand_forecasting.py # Day 4-8 - 30-day revenue forecast
│   ├── 05_churn_prediction.py   # Day 9  - churn classifier
│   ├── 06_inventory_optimization.py # Day 10 - EOQ + safety stock
│   └── 07_drift_detection.py    # Day 12 - PSI drift monitoring
├── experiments/                 # earlier attempts kept for transparency
│   ├── forecast_v1_baseline.py  # raw revenue Ridge -> ~12% MAPE, missed target
│   ├── forecast_v2_gbr.py       # GBR/RF attempt -> worse (12-13% MAPE)
│   ├── clustering_experiments.py# K-Means k-sweep + DBSCAN comparison
│   └── churn_v1_leakage.py      # leaky feature set -> AUC 1.00 (the bug, not the fix)
├── reports/                      # all generated charts, metrics, HTML reports
├── app.py                        # Streamlit dashboard
├── Dockerfile
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

python data/generate_data.py
python notebooks/01_eda.py
python notebooks/02a_data_validation.py
python notebooks/02b_data_cleaning.py
python notebooks/03_segmentation.py
python notebooks/04_demand_forecasting.py
python notebooks/05_churn_prediction.py
python notebooks/06_inventory_optimization.py
python notebooks/07_drift_detection.py

streamlit run app.py
```

## Key results

| Model | Metric | Target | Result |
|---|---|---|---|
| Demand forecasting | MAPE (60-day hold-out) | <= 12% | **11.76%** |
| Churn prediction | AUC-ROC | >= 0.88 | **0.9491** |
| Customer segmentation | Silhouette score | n/a | 0.240 (k=3) |
| Data validation | Checks passed | n/a | 17 / 17 |
| Drift detection | Features flagged | n/a | 0 / 6 (all stable) |

## Known limitations

- Churn precision@top-20% is low (0.093) because the churn base rate in this dataset is only 1.8% — very few positive examples to learn from. A real retailer's churn rate is usually higher, which would likely improve this metric.
- Forecasting uses Ridge regression, not Prophet/LSTM as originally planned, due to environment install constraints (see table above).
- Dataset is synthetic; absolute metric values would differ on real transaction data, though the modeling approach would not need to change.
- Kubernetes manifests, Airflow DAG, and Grafana dashboards described in the original brief are not included — MLOps maturity here is demonstrated through the validation, drift detection and experiment logs instead.

## Author

Built for the Zidio Development Data Science & Analytics internship.
