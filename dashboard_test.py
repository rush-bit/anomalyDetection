import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Anomaly Dashboard", layout="wide")

# --- Data Loader ---
@st.cache_data
def load_data(path):
    df = pd.read_csv(
        path,
        parse_dates=[c for c in ['timestamp','date','time'] if c in pd.read_csv(path, nrows=0).columns],
        low_memory=False
    )
    # create generic datetime if none exists
    if 'timestamp' not in df.columns:
        if 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        else:
            df['timestamp'] = pd.RangeIndex(len(df), name='t')
    return df

# --- Sidebar Filters ---
st.sidebar.header("Filters")
data_path = st.sidebar.text_input("Scored CSV path", "new_data_scored.csv")
df = load_data(data_path)

clusters = sorted(df['cluster'].dropna().unique().tolist()) if 'cluster' in df.columns else []
c_sel = st.sidebar.multiselect("Clusters", clusters, default=clusters if clusters else [])
if c_sel:
    df = df[df['cluster'].isin(c_sel)]

devs = sorted(df['deviceID'].dropna().unique().tolist()) if 'deviceID' in df.columns else []
d_sel = st.sidebar.multiselect("Devices", devs, default=devs[:10] if devs else [])
if d_sel:
    df = df[df['deviceID'].isin(d_sel)]

if 'timestamp' in df.columns and np.issubdtype(df['timestamp'].dtype, np.datetime64):
    min_d, max_d = df['timestamp'].min(), df['timestamp'].max()
    start, end = st.sidebar.date_input("Date range", value=[min_d.date(), max_d.date()])
    df = df[(df['timestamp'] >= pd.to_datetime(start)) & (df['timestamp'] <= pd.to_datetime(end))]

st.title("Fleet Anomaly Dashboard")

# --- Compute Isolation Forest if not already present ---
feature_cols = [c for c in df.columns if c not in ['deviceID','tripID','cluster','label','anomaly_score_ebm','alert_ebm','y_pred','residual']]
if feature_cols:
    try:
        if 'anomaly_score_iforest' not in df.columns:
            iforest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
            iforest.fit(df[feature_cols])
            df['anomaly_score_iforest'] = iforest.decision_function(df[feature_cols]) * -1
            df['alert_iforest'] = iforest.predict(df[feature_cols]) == -1
    except Exception as e:
        st.warning(f"Isolation Forest could not run: {e}")

# --- Tabs for Better Layout ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["KPIs & Trends", "Distributions", "Top Anomalies", "Explanations", "Model Comparison"])


# --- Tab 1: KPIs & Time Series ---
with tab1:
    col1, col2, col3 = st.columns(3)
    total_rows = len(df)
    n_alert_ebm = int(df.get('alert_ebm', pd.Series(False, index=df.index)).sum())
    alert_rate = (n_alert_ebm / total_rows * 100) if total_rows > 0 else 0
    col1.metric("Total Rows", f"{total_rows:,}")
    col2.metric("EBM Alerts", f"{n_alert_ebm:,}")
    col3.metric("Alert Rate", f"{alert_rate:.1f}%" if total_rows > 0 else "N/A")

    # Gauge Chart for Alert Rate
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=alert_rate,
        title={'text': "Alert Rate %"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "red"}}
    ))
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Time-series with alert overlay
    def ts_plot(score_col, title):
        if score_col not in df.columns:
            return
        y = score_col
        fig = px.line(df.sort_values('timestamp'), x='timestamp', y=y,
                      color='cluster' if 'cluster' in df.columns else None,
                      title=title)
        if 'alert_ebm' in df.columns:
            pts = df[df['alert_ebm'] == True]
            if not pts.empty:
                fig.add_scatter(
                    x=pts['timestamp'], y=pts[y],
                    mode='markers', marker=dict(color='red', size=7, symbol='x'),
                    name="EBM Alert"
                )
        st.plotly_chart(fig, use_container_width=True)

    ts_plot('anomaly_score_ebm', "EBM Anomaly Score Over Time")

# --- Tab 2: Distributions ---
with tab2:
    st.subheader("Alert Distribution by Cluster & Device")
    col1, col2 = st.columns(2)

    if 'cluster' in df.columns:
        cluster_agg = df.groupby('cluster')['alert_ebm'].sum().reset_index()
        fig1 = px.pie(cluster_agg, names='cluster', values='alert_ebm',
                      title="Alerts by Cluster")
        col1.plotly_chart(fig1, use_container_width=True)

    if 'deviceID' in df.columns:
        dev_agg = df.groupby('deviceID')['alert_ebm'].sum().reset_index()
        top_devs = dev_agg.sort_values('alert_ebm', ascending=False).head(15)
        fig2 = px.bar(top_devs, x='deviceID', y='alert_ebm', title="Top 15 Devices by Alerts")
        col2.plotly_chart(fig2, use_container_width=True)

# --- Tab 3: Top Anomalies ---
with tab3:
    st.subheader("Top Anomalies")
    cols = ['deviceID','tripID','cluster','timestamp','anomaly_score_ebm','alert_ebm','y_pred','residual']
    present_cols = [c for c in cols if c in df.columns]

    sample_flag = st.checkbox("Randomly sample anomalies", value=False)
    if sample_flag:
        top = df[df['alert_ebm']].sample(min(200, len(df)), random_state=None)[present_cols]
    else:
        top = df.sort_values(['anomaly_score_ebm'], ascending=False).head(200)[present_cols]

    st.dataframe(top, use_container_width=True)

    # Download option for filtered data
    st.download_button(
        "Download Filtered Alerts (CSV)",
        top.to_csv(index=False).encode('utf-8'),
        file_name="filtered_alerts.csv",
        mime="text/csv"
    )

# --- Tab 4: Explanation Viewer ---
with tab4:
    st.subheader("Per-row Explanation Viewer")
    exp_dir = st.text_input("Explanations folder", "explanations_new/per_row")
    if os.path.isdir(exp_dir):
        files = sorted([f for f in os.listdir(exp_dir) if f.endswith(('.png','.html'))])
        pick = st.selectbox("Choose a saved explanation file", files)
        if pick:
            pth = os.path.join(exp_dir, pick)
            if pth.endswith('.png'):
                st.image(pth, use_column_width=True)
            else:
                html_content = open(pth, 'r', encoding='utf-8').read()
                styled_html = f"""
                <div style="background-color:white; color:black; padding:15px; border-radius:10px;">
                {html_content}
                </div>
                """
                st.components.v1.html(styled_html, height=500, scrolling=True)
    else:
        st.info("Provide a directory containing saved force plots (PNG/HTML).")

# --- Tab 5: Model Comparison ---
with tab5:
    st.subheader("EBM vs Isolation Forest Comparison")
    if 'alert_ebm' in df.columns and 'alert_iforest' in df.columns:
        # Agreement calculation
        agreement = (df['alert_ebm'] == df['alert_iforest']).mean() * 100
        both_true = ((df['alert_ebm']) & (df['alert_iforest'])).sum()
        ebm_only = ((df['alert_ebm']) & (~df['alert_iforest'])).sum()
        iforest_only = ((~df['alert_ebm']) & (df['alert_iforest'])).sum()
        neither = ((~df['alert_ebm']) & (~df['alert_iforest'])).sum()

        col1, col2 = st.columns(2)
        col1.metric("Agreement Rate", f"{agreement:.2f}%")
        overlap_df = pd.DataFrame({
            "Category": ["Both True", "EBM Only", "IForest Only", "Neither"],
            "Count": [both_true, ebm_only, iforest_only, neither]
        })
        fig_overlap = px.bar(overlap_df, x="Category", y="Count", text="Count",
                             title="Alert Overlap Breakdown", color="Category")
        col2.plotly_chart(fig_overlap, use_container_width=True)

        # Histogram comparison
        if 'anomaly_score_iforest' in df.columns:
            fig_hist = go.Figure()
            fig_hist.add_histogram(x=df['anomaly_score_ebm'], nbinsx=30, name="EBM", opacity=0.5)
            fig_hist.add_histogram(x=df['anomaly_score_iforest'], nbinsx=30, name="IForest", opacity=0.5)
            fig_hist.update_layout(barmode="overlay", title="Anomaly Score Distribution")
            fig_hist.update_traces(opacity=0.6)
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Isolation Forest scores not computed yet or EBM alerts missing.")