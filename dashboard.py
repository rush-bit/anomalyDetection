import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Anomaly Dashboard", layout="wide")

@st.cache_data
def load_data(path):
    df = pd.read_csv(
        path,
        parse_dates=[c for c in ['timestamp','date','time'] if c in pd.read_csv(path, nrows=0).columns],
        low_memory=False
    )
    # create a generic datetime if none exists (optional)
    if 'timestamp' not in df.columns:
        if 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        else:
            df['timestamp'] = pd.RangeIndex(len(df), name='t')
    return df

# 1) Data
data_path = st.sidebar.text_input("Scored CSV path", "new_data_scored.csv")
df = load_data(data_path)

# 2) Filters
clusters = sorted(df['cluster'].dropna().unique().tolist()) if 'cluster' in df.columns else []
c_sel = st.sidebar.multiselect("Clusters", clusters, default=clusters if clusters else [])
if c_sel:
    df = df[df['cluster'].isin(c_sel)]

devs = sorted(df['deviceID'].dropna().unique().tolist()) if 'deviceID' in df.columns else []
d_sel = st.sidebar.multiselect("Devices", devs, default=devs[:10] if devs else [])
if d_sel:
    df = df[df['deviceID'].isin(d_sel)]

# date range filter
if 'timestamp' in df.columns and np.issubdtype(df['timestamp'].dtype, np.datetime64):
    min_d, max_d = df['timestamp'].min(), df['timestamp'].max()
    start, end = st.sidebar.date_input("Date range", value=[min_d.date(), max_d.date()])
    df = df[(df['timestamp'] >= pd.to_datetime(start)) & (df['timestamp'] <= pd.to_datetime(end))]

st.title("Fleet Anomaly Dashboard (EBM only)")

# 3) KPI cards
col1, col2, col3 = st.columns(3)
total_rows = len(df)
n_alert_ebm = int(df.get('alert_ebm', pd.Series(False, index=df.index)).sum())
col1.metric("Rows", f"{total_rows:,}")
col2.metric("EBM alerts", f"{n_alert_ebm:,}")
col3.metric("Alert rate", f"{(n_alert_ebm/total_rows*100):.1f}%" if total_rows > 0 else "N/A")

# 4) Time-series with alert overlays
def ts_plot(score_col, title):
    if score_col not in df.columns:
        return
    y = score_col
    fig = px.line(
        df.sort_values('timestamp'),
        x='timestamp', y=y,
        color='cluster' if 'cluster' in df.columns else None,
        title=title, markers=False
    )
    # overlay alerts as red markers
    alert_col = 'alert_ebm'
    if alert_col in df.columns:
        pts = df[df[alert_col] == True]
        if not pts.empty:
            fig.add_scatter(
                x=pts['timestamp'], y=pts[y],
                mode='markers', marker=dict(color='red', size=7, symbol='x'),
                name="EBM Alert"
            )
    st.plotly_chart(fig, use_container_width=True)

ts_plot('anomaly_score_ebm', "EBM anomaly score over time")

# 5) Cluster/Device distribution plots
st.subheader("Alert distribution")
if 'cluster' in df.columns:
    agg = df.groupby('cluster')['alert_ebm'].sum().reset_index()
    fig = px.bar(agg, x='cluster', y='alert_ebm',
                 title="EBM Alerts per Cluster")
    st.plotly_chart(fig, use_container_width=True)

if 'deviceID' in df.columns:
    agg = df.groupby('deviceID')['alert_ebm'].sum().reset_index()
    top_devs = agg.sort_values('alert_ebm', ascending=False).head(20)
    fig = px.bar(top_devs, x='deviceID', y='alert_ebm',
                 title="Top 20 Devices by Alerts")
    st.plotly_chart(fig, use_container_width=True)

# 6) Top anomalies tables
st.subheader("Top anomalies")
cols = ['deviceID','tripID','cluster','timestamp','anomaly_score_ebm','alert_ebm','y_pred','residual']
present_cols = [c for c in cols if c in df.columns]
top = df.sort_values(['anomaly_score_ebm'], ascending=False).head(200)[present_cols]
st.dataframe(top, use_container_width=True)

# 7) Per-row explanation viewer
st.subheader("Per-row explanation")
exp_dir = st.text_input("Explanations folder", "explanations_new/per_row")
if os.path.isdir(exp_dir):
    files = sorted([f for f in os.listdir(exp_dir) if f.endswith(('.png','.html'))])
    pick = st.selectbox("Choose a saved explanation file", files)
    if pick:
        pth = os.path.join(exp_dir, pick)
        if pth.endswith('.png'):
            st.image(pth, use_column_width=True)
        else:
            st.components.v1.html(open(pth,'r', encoding='utf-8').read(), height=400, scrolling=True)
else:
    st.info("Provide a directory containing saved force plots (PNG/HTML).")

# 8) Download review shortlist
dl_cols = [c for c in ['deviceID','tripID','cluster','timestamp','anomaly_score_ebm','alert_ebm','y_pred','residual'] if c in df.columns]
shortlist = df.sort_values('anomaly_score_ebm', ascending=False)[dl_cols].head(500)
st.download_button(
    "Download review shortlist (CSV)",
    shortlist.to_csv(index=False).encode('utf-8'),
    file_name="anomaly_review_shortlist.csv",
    mime="text/csv"
)
