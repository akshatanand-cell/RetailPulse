"""
app.py — RetailPulse Streamlit Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="RetailPulse Analytics", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
div[data-testid="metric-container"] { background:#f0f4ff; border:1px solid #d1d9f0; border-radius:8px; padding:0.5rem 1rem; }
.stTabs [data-baseweb="tab"] { font-size:0.95rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_all():
    base = os.path.dirname(__file__)
    def rd(rel, dates=None):
        path = os.path.join(base, rel)
        try:
            return pd.read_csv(path, parse_dates=dates or False)
        except Exception:
            return pd.DataFrame()
    return {
        "sales": rd("data/processed/sales_clean.csv", ["date"]),
        "customers": rd("data/processed/customer_segments.csv"),
        "forecast": rd("reports/forecasting/30day_forecast.csv", ["date"]),
        "churn": rd("data/processed/churn_scores.csv"),
        "inventory": rd("data/processed/inventory_recommendations.csv"),
        "daily": rd("data/processed/daily_sales.csv", ["date"]),
        "products": rd("data/raw/products.csv"),
    }

D = load_all()
sales, customers, forecast = D["sales"], D["customers"], D["forecast"]
churn, inventory, daily, products = D["churn"], D["inventory"], D["daily"], D["products"]

def fmt(v):
    if v >= 1e7: return f"Rs.{v/1e7:.2f}Cr"
    if v >= 1e5: return f"Rs.{v/1e5:.1f}L"
    return f"Rs.{v:,.0f}"

def csv_download(df, label, fname):
    st.download_button(label, df.to_csv(index=False).encode(), fname, "text/csv", use_container_width=True)

with st.sidebar:
    st.markdown("### 📊 RetailPulse")
    st.caption("AI-Powered Retail Analytics")
    st.divider()
    page = st.radio("Navigate", ["📊 Overview", "📈 Demand Forecast", "👥 Customer Segments",
                                  "⚠️ Churn Risk", "📦 Inventory", "🔬 What-If Analysis"])
    st.divider()
    st.caption("Zidio Development Internship | 2026")

if page == "📊 Overview":
    st.title("📊 Executive Overview")
    st.caption("Key performance indicators across all retail operations")
    if not sales.empty:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Revenue", fmt(sales["revenue"].sum()))
        c2.metric("Transactions", f"{len(sales):,}")
        c3.metric("Avg Order Value", f"Rs.{sales['revenue'].mean():.0f}")
        c4.metric("Active Customers", f"{sales['customer_id'].nunique():,}")
        hr = len(churn[churn["churn_risk"] == "High Risk"]) if not churn.empty else 0
        c5.metric("High Churn Risk", str(hr), delta=f"{hr} need action", delta_color="inverse")
        st.divider()
        cl, cr = st.columns([2, 1])
        with cl:
            st.subheader("Monthly Revenue Trend")
            monthly = sales.set_index("date").resample("ME")["revenue"].sum().reset_index()
            fig = px.area(monthly, x="date", y="revenue", color_discrete_sequence=["#2563eb"],
                          labels={"revenue": "Revenue (Rs.)", "date": ""})
            fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with cr:
            st.subheader("Channel Split")
            ch = sales.groupby("channel")["revenue"].sum().reset_index()
            fig = px.pie(ch, values="revenue", names="channel", hole=0.45, color_discrete_sequence=["#2563eb", "#16a34a"])
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        c3a, c3b = st.columns(2)
        with c3a:
            st.subheader("Revenue by Category")
            if not products.empty:
                merged = sales.merge(products[["product_id", "category"]], on="product_id")
                cat_rev = merged.groupby("category")["revenue"].sum().sort_values(ascending=False).reset_index()
                fig = px.bar(cat_rev, x="category", y="revenue", color="revenue", color_continuous_scale="Blues",
                             labels={"revenue": "Revenue (Rs.)", "category": ""})
                fig.update_layout(height=300, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)
        with c3b:
            st.subheader("Daily Revenue Distribution")
            fig = px.histogram(daily, x="revenue", nbins=40, color_discrete_sequence=["#7c3aed"],
                               labels={"revenue": "Daily Revenue (Rs.)"})
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        st.divider()
        st.subheader("Export Overview Data")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_download(sales, "Download Sales Data (CSV)", "sales_data.csv")
        with col_dl2:
            monthly_export = sales.set_index("date").resample("ME")["revenue"].sum().reset_index()
            csv_download(monthly_export, "Download Monthly Revenue (CSV)", "monthly_revenue.csv")
    else:
        st.warning("No data found. Run the notebooks in order first (see README).")

elif page == "📈 Demand Forecast":
    st.title("📈 Demand Forecasting")
    st.caption("30-day ahead revenue predictions | Ridge regression + log transform | MAPE: ~11.8% (target <=12%)")
    if not forecast.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("30-Day Forecast Total", fmt(forecast["forecast_revenue"].sum()))
        c2.metric("Avg Daily Forecast", f"Rs.{forecast['forecast_revenue'].mean():,.0f}")
        c3.metric("Peak Day Forecast", f"Rs.{forecast['forecast_revenue'].max():,.0f}")
        st.divider()
        if not daily.empty:
            recent = daily[daily["date"] >= daily["date"].max() - pd.Timedelta(days=90)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recent["date"], y=recent["revenue"], name="Actual Revenue", line=dict(color="#2563eb", width=2)))
            fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["forecast_revenue"], name="30-Day Forecast",
                                     line=dict(color="#dc2626", width=2.5, dash="dot"), fill="tozeroy", fillcolor="rgba(220,38,38,0.07)"))
            fig.add_vrect(x0=str(daily["date"].max()), x1=str(forecast["date"].max()), fillcolor="rgba(220,38,38,0.05)",
                         line_width=0, annotation_text="Forecast Period", annotation_position="top left")
            fig.update_layout(title="Revenue Forecast - Next 30 Days", height=400, hovermode="x unified",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02), yaxis_title="Daily Revenue (Rs.)",
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Forecast Table")
        disp = forecast.copy()
        disp["date"] = disp["date"].astype(str)
        disp["forecast_revenue"] = disp["forecast_revenue"].apply(lambda x: f"Rs.{x:,.0f}")
        st.dataframe(disp.rename(columns={"date": "Date", "forecast_revenue": "Forecast Revenue"}), use_container_width=True, height=260)
        st.divider()
        csv_download(forecast, "Download Forecast Data (CSV)", "30day_forecast.csv")
    else:
        st.warning("No forecast data found. Run notebooks/04_demand_forecasting.py first.")

elif page == "👥 Customer Segments":
    st.title("👥 Customer Segmentation")
    st.caption("K-Means clustering on RFM + behavioural features | k=3 selected by silhouette score (~0.24)")
    if not customers.empty and "segment" in customers.columns:
        seg_counts = customers["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        seg_revenue = customers.groupby("segment")["monetary"].sum().reset_index()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Segments", customers["segment"].nunique())
        c2.metric("Champions", len(customers[customers["segment"] == "Champions"]))
        c3.metric("Avg Customer Spend", f"Rs.{customers['monetary'].mean():,.0f}")
        st.divider()
        cl, cr = st.columns(2)
        with cl:
            fig = px.pie(seg_counts, values="count", names="segment", title="Segment Distribution",
                         color_discrete_sequence=px.colors.qualitative.Set2, hole=0.3)
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)
        with cr:
            fig = px.bar(seg_revenue.sort_values("monetary", ascending=False), x="segment", y="monetary",
                         title="Revenue by Segment", color="monetary", color_continuous_scale="Viridis",
                         labels={"monetary": "Total Revenue (Rs.)", "segment": ""})
            fig.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("RFM Scatter - Recency vs Total Spend")
        fig = px.scatter(customers, x="recency", y="monetary", color="segment", size="frequency", opacity=0.7,
                         hover_data=["customer_id", "R", "F", "M"],
                         labels={"recency": "Days Since Last Purchase", "monetary": "Total Spend (Rs.)"}, height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Segment Summary")
        seg_tbl = customers.groupby("segment").agg(Customers=("customer_id", "count"), Avg_Recency=("recency", "mean"),
                                                     Avg_Frequency=("frequency", "mean"), Avg_Spend=("monetary", "mean"),
                                                     Avg_RFM=("rfm_total", "mean")).round(1).reset_index()
        st.dataframe(seg_tbl, use_container_width=True)
        st.divider()
        csv_download(customers[["customer_id", "segment", "recency", "frequency", "monetary", "rfm_total"]],
                     "Download Segments (CSV)", "customer_segments.csv")
    else:
        st.warning("No segment data found. Run notebooks/03_segmentation.py first.")

elif page == "⚠️ Churn Risk":
    st.title("⚠️ Churn Risk Analysis")
    st.caption("GradientBoostingClassifier (XGBoost substitute - see README) | AUC-ROC: ~0.95")
    if not churn.empty:
        risk_counts = churn["churn_risk"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("High Risk", int(risk_counts.get("High Risk", 0)), delta="Needs Action", delta_color="inverse")
        c2.metric("Medium Risk", int(risk_counts.get("Medium Risk", 0)))
        c3.metric("Low Risk", int(risk_counts.get("Low Risk", 0)))
        st.divider()
        cl, cr = st.columns(2)
        with cl:
            fig = px.pie(risk_counts.reset_index(), values="count", names="churn_risk", title="Customer Risk Distribution",
                         color_discrete_map={"High Risk": "#dc2626", "Medium Risk": "#f59e0b", "Low Risk": "#16a34a"})
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)
        with cr:
            fig = px.histogram(churn, x="churn_probability", nbins=40, color_discrete_sequence=["#7c3aed"],
                               title="Churn Probability Distribution", labels={"churn_probability": "Churn Score"})
            fig.add_vline(x=0.6, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)
        img_path = "reports/churn/churn_model_report.png"
        if os.path.exists(img_path):
            st.subheader("Model Diagnostics (ROC, score distribution, permutation importance)")
            st.image(img_path, use_container_width=True)
        st.subheader("High-Risk Customers - Intervention List")
        if not customers.empty:
            hr_df = churn[churn["churn_risk"] == "High Risk"].merge(
                customers[["customer_id", "recency", "frequency", "monetary", "loyalty_tier", "region"]],
                on="customer_id", how="left").sort_values("churn_probability", ascending=False)
            hr_display = hr_df.copy()
            hr_display["churn_probability"] = hr_display["churn_probability"].apply(lambda x: f"{x:.1%}")
            hr_display["monetary"] = hr_display["monetary"].apply(lambda x: f"Rs.{x:,.0f}" if pd.notna(x) else "-")
            st.dataframe(hr_display[["customer_id", "churn_probability", "churn_risk", "recency", "frequency",
                                      "monetary", "loyalty_tier", "region"]].head(20).reset_index(drop=True), use_container_width=True)
        st.divider()
        csv_download(churn, "Download Churn Scores (CSV)", "churn_scores.csv")
    else:
        st.warning("No churn data found. Run notebooks/05_churn_prediction.py first.")

elif page == "📦 Inventory":
    st.title("📦 Inventory Optimization")
    st.caption("EOQ formula + Safety Stock at 95% service level")
    if not inventory.empty:
        status_counts = inventory["stock_status"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Critical", int(status_counts.get("Critical - Order Now", 0)))
        c2.metric("Reorder Soon", int(status_counts.get("Reorder Soon", 0)))
        c3.metric("Overstocked", int(status_counts.get("Overstocked", 0)))
        c4.metric("Healthy", int(status_counts.get("Healthy", 0)))
        st.divider()
        cl, cr = st.columns(2)
        with cl:
            fig = px.pie(status_counts.reset_index(), values="count", names="stock_status", title="Inventory Health Overview",
                         color_discrete_sequence=["#dc2626", "#f59e0b", "#2563eb", "#16a34a"])
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)
        with cr:
            cat_days = inventory.groupby("category")["days_of_stock"].median().sort_values().reset_index()
            fig = px.bar(cat_days, x="days_of_stock", y="category", orientation="h", title="Median Days of Stock by Category",
                         color="days_of_stock", color_continuous_scale="RdYlGn", labels={"days_of_stock": "Days of Stock", "category": ""})
            fig.add_vline(x=30, line_dash="dash", line_color="red", annotation_text="30-day min")
            fig.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.info("Reorder Logic: EOQ = sqrt(2 x Annual Demand x Ordering Cost / Holding Cost) | "
                "Safety Stock = 1.65 x Demand StdDev x sqrt(Lead Time) | Reorder Point = (Daily Demand x Lead Time) + Safety Stock")
        st.subheader("Products Needing Reorder")
        reorder = inventory[inventory["recommended_order"] > 0].sort_values("days_of_stock")
        if len(reorder) > 0:
            st.dataframe(reorder[["product_name", "category", "stock_on_hand", "daily_demand", "reorder_point_calc",
                                   "safety_stock", "EOQ", "recommended_order", "days_of_stock", "stock_status"]].reset_index(drop=True),
                        use_container_width=True)
        else:
            st.success("All products are above their reorder points.")
        st.subheader("Full Inventory Table")
        fcat = st.selectbox("Filter by Category", ["All"] + list(inventory["category"].unique()))
        show = inventory if fcat == "All" else inventory[inventory["category"] == fcat]
        st.dataframe(show[["product_name", "category", "stock_on_hand", "daily_demand", "days_of_stock", "safety_stock",
                            "EOQ", "recommended_order", "stock_status"]].reset_index(drop=True), use_container_width=True, height=320)
        st.divider()
        csv_download(inventory, "Download Inventory Recommendations (CSV)", "inventory_recommendations.csv")
    else:
        st.warning("No inventory data found. Run notebooks/06_inventory_optimization.py first.")

elif page == "🔬 What-If Analysis":
    st.title("🔬 What-If Analysis")
    st.caption("Adjust business parameters and see how forecast and inventory change")
    st.subheader("Forecast Sensitivity")
    col1, col2 = st.columns(2)
    with col1:
        demand_change = st.slider("Demand Change (%)", -30, 50, 0, step=5)
        discount_impact = st.slider("Discount Campaign Impact (%)", 0, 30, 0, step=5)
    with col2:
        seasonality = st.selectbox("Season Multiplier", ["Normal (1.0x)", "Festive/Q4 (1.3x)", "Off-Season (0.8x)", "Sale Event (1.5x)"])
        season_mult = {"Normal (1.0x)": 1.0, "Festive/Q4 (1.3x)": 1.3, "Off-Season (0.8x)": 0.8, "Sale Event (1.5x)": 1.5}[seasonality]
    if not forecast.empty:
        adj_factor = (1 + demand_change / 100) * (1 + discount_impact / 100) * season_mult
        fc_adj = forecast.copy()
        fc_adj["original"] = fc_adj["forecast_revenue"]
        fc_adj["adjusted"] = fc_adj["forecast_revenue"] * adj_factor
        orig_total, adj_total = fc_adj["original"].sum(), fc_adj["adjusted"].sum()
        diff = adj_total - orig_total
        diff_pct = (diff / orig_total) * 100
        m1, m2, m3 = st.columns(3)
        m1.metric("Original Forecast (30d)", fmt(orig_total))
        m2.metric("Adjusted Forecast (30d)", fmt(adj_total), delta=f"{diff_pct:+.1f}%")
        m3.metric("Revenue Impact", fmt(abs(diff)), delta="gain" if diff >= 0 else "loss")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fc_adj["date"], y=fc_adj["original"], name="Original", line=dict(color="#2563eb", width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=fc_adj["date"], y=fc_adj["adjusted"], name="Adjusted", line=dict(color="#16a34a", width=2.5),
                                 fill="tonexty", fillcolor="rgba(22,163,74,0.1)"))
        fig.update_layout(title="Original vs Adjusted 30-Day Forecast", height=380, hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02), yaxis_title="Daily Revenue (Rs.)", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
        csv_download(fc_adj[["date", "original", "adjusted"]], "Download Adjusted Forecast (CSV)", "adjusted_forecast.csv")
    st.divider()
    st.subheader("Inventory What-If")
    col3, col4 = st.columns(2)
    with col3:
        demand_mult = st.slider("Demand Surge (%)", -20, 100, 0, step=10)
    with col4:
        lead_time_add = st.slider("Extra Lead Time (days)", 0, 14, 0)
    if not inventory.empty:
        inv_adj = inventory.copy()

        inv_adj["adj_daily_demand"] = (
            inv_adj["daily_demand"] * (1 + demand_mult / 100)
        )

        inv_adj["adj_days_of_stock"] = (
            inv_adj["stock_on_hand"] /
            inv_adj["adj_daily_demand"].clip(lower=0.01)
        ).round(0)

        # Safe lead time handling - falls back to a default if the column
        # isn't named exactly "lead_time_days" in inventory_recommendations.csv
        if "lead_time_days" in inv_adj.columns:
            lead_time = inv_adj["lead_time_days"]
        elif "lead_time" in inv_adj.columns:
            lead_time = inv_adj["lead_time"]
        else:
            lead_time = 7

        inv_adj["adj_reorder_point"] = (
            inv_adj["adj_daily_demand"] *
            (lead_time + lead_time_add) +
            inv_adj["safety_stock"]
        ).round(0)

        inv_adj["adj_status"] = inv_adj.apply(
            lambda r: "NOW Critical"
            if r["stock_on_hand"] <= r["adj_reorder_point"]
            else "Still OK",
            axis=1,
        )

        newly_critical = inv_adj[inv_adj["adj_status"] == "NOW Critical"]

        st.warning(
            f"Under these conditions: {len(newly_critical)} products would need immediate reorder"
        )

        c5, c6 = st.columns(2)
        c5.metric("Products Newly at Risk", len(newly_critical))
        c6.metric(
            "Avg Days of Stock (adjusted)",
            f"{inv_adj['adj_days_of_stock'].median():.0f} days"
        )

        st.dataframe(
            inv_adj[
                [
                    "product_name",
                    "category",
                    "stock_on_hand",
                    "adj_daily_demand",
                    "adj_days_of_stock",
                    "adj_reorder_point",
                    "adj_status",
                ]
            ]
            .sort_values("adj_days_of_stock")
            .head(15)
            .reset_index(drop=True),
            use_container_width=True,
        )

        csv_download(
            inv_adj[
                [
                    "product_name",
                    "category",
                    "stock_on_hand",
                    "adj_daily_demand",
                    "adj_days_of_stock",
                    "adj_status",
                ]
            ],
            "Download What-If Inventory (CSV)",
            "whatif_inventory.csv",
        )