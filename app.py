"""UAC System Capacity & Care Load Analytics - Streamlit dashboard.
Run:  streamlit run app.py   (after running uac_analysis.py / the notebook once)
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="UAC Care Load Analytics", layout="wide")
HERE = Path(__file__).resolve().parent      # works from any working directory
PROC = HERE / "data" / "processed"
DATA = PROC / "uac_clean_reported_days.csv"


if not DATA.exists():
    st.error(f"Processed data not found at {DATA}. Run the notebook / uac_analysis.py first "
             "(it creates data/processed/), then reload this page.")
    st.stop()


@st.cache_data
def load():
    return pd.read_csv(DATA, parse_dates=["date"])


df_all = load()

# ---------------- Sidebar controls ----------------
st.sidebar.header("Controls")
dmin, dmax = df_all.date.min().date(), df_all.date.max().date()
rng = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if not isinstance(rng, (tuple, list)) or len(rng) != 2:
    st.info("Select both a start and an end date.")
    st.stop()
gran = st.sidebar.radio("Time granularity", ["Daily", "Weekly", "Monthly"], horizontal=True)
METRICS = {"Total system load": "total_load", "HHS care": "hhs_care", "CBP custody": "cbp_custody",
           "Apprehended (intake)": "apprehended", "Transferred to HHS": "transferred",
           "Discharged from HHS": "discharged", "Net daily intake": "net_intake"}
chosen = st.sidebar.multiselect("Metrics to show", list(METRICS), default=["Total system load", "HHS care", "CBP custody"])
show_ma = st.sidebar.multiselect("Rolling averages (daily view)", ["7-day", "14-day"], default=["7-day"])
show_windows = st.sidebar.checkbox("Shade strain / relief windows", True)
st.sidebar.caption("Statistics use reported days only. Fridays/Saturdays are essentially never reported in the source; "
                   "no values are imputed.")

d = df_all[(df_all.date.dt.date >= rng[0]) & (df_all.date.dt.date <= rng[1])].copy()
if d.empty:
    st.warning("No reported days in this range.")
    st.stop()


def resample(d, gran):
    if gran == "Daily":
        return d
    rule = "W" if gran == "Weekly" else "MS"
    cols = [c for c in METRICS.values() if c in d.columns] + ["cum_net_intake", "load_ma7", "load_ma14", "net_ma7"]
    return d.set_index("date")[cols].resample(rule).mean().dropna(subset=["total_load"]).reset_index()


g = resample(d, gran)


def kpis(d):
    tr, di = d.transferred.sum(), d.discharged.sum()
    bs = d[d.backlog_sustained]
    return dict(latest=int(d.total_load.iloc[-1]), mean=d.total_load.mean(), peak=int(d.total_load.max()),
                nip=d.net_intake.mean(), vol=d.local_cv14.mean(),
                bar=bs.net_intake.mean() if len(bs) else 0.0, share=len(bs) / len(d) * 100,
                dor=di / tr if tr else float("nan"))


def shade(fig, d, col, color, row=None):
    runs = (d[col] != d[col].shift()).cumsum()
    for _, s in d.groupby(runs):
        if s[col].iloc[0]:
            kw = dict(row=row, col=1) if row else {}
            fig.add_vrect(x0=s.date.iloc[0], x1=s.date.iloc[-1], fillcolor=color, opacity=.15, line_width=0, **kw)


st.title("System Capacity & Care Load Analytics: Unaccompanied Children Program")
st.caption(f"{rng[0]:%d %b %Y} to {rng[1]:%d %b %Y} | {len(d)} reported days | granularity: {gran}")

# ---------------- KPI cards ----------------
k = kpis(d)
c = st.columns(5)
c[0].metric("Total children under care (latest)", f"{k['latest']:,}", help=f"Window mean {k['mean']:,.0f}; peak {k['peak']:,}. Total = CBP custody + HHS care.")
c[1].metric("Net intake pressure (avg/day)", f"{k['nip']:+.1f}", help="Mean of (transfers into HHS - discharges). Positive = inflow exceeds outflow.")
c[2].metric("Care load volatility index", f"{k['vol']:.2f}%", help="Mean 14-day rolling coefficient of variation of total load.")
c[3].metric("Backlog accumulation rate", f"{k['bar']:+.1f}/day", help=f"Mean net intake during sustained backlog spells ({k['share']:.1f}% of reported days in window).")
c[4].metric("Discharge offset ratio", f"{k['dor']:.2f}", help="Total discharges / total transfers into HHS. >1 = discharges outpace new inflow.")

tab1, tab2, tab3, tab4 = st.tabs(["System Load Overview", "CBP vs HHS", "Net Intake & Backlog", "Strain / Relief & Data"])

with tab1:
    fig = go.Figure()
    for name in chosen:
        fig.add_trace(go.Scatter(x=g.date, y=g[METRICS[name]], name=name, mode="lines"))
    if gran == "Daily":
        if "7-day" in show_ma:
            fig.add_trace(go.Scatter(x=g.date, y=g.load_ma7, name="Total load 7-day avg", line=dict(dash="dot")))
        if "14-day" in show_ma:
            fig.add_trace(go.Scatter(x=g.date, y=g.load_ma14, name="Total load 14-day avg", line=dict(dash="dash")))
    if show_windows:
        shade(fig, d, "strain", "red")
        shade(fig, d, "relief", "green")
    fig.update_layout(height=470, title=f"Selected metrics ({gran.lower()})", yaxis_title="Children", legend=dict(orientation="h", y=-.2))
    st.plotly_chart(fig, width="stretch")
    st.caption("Red shading = strain (total load at or above the 75th percentile of the full period for 14+ consecutive reported days). "
               "Green = relief (at or below the 25th percentile).")

with tab2:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=g.date, y=g.hhs_care, name="HHS care", line=dict(color="#1f6f8b")), secondary_y=False)
    fig.add_trace(go.Scatter(x=g.date, y=g.cbp_custody, name="CBP custody", line=dict(color="#e07a1f")), secondary_y=True)
    fig.update_yaxes(title_text="HHS care", secondary_y=False)
    fig.update_yaxes(title_text="CBP custody", secondary_y=True)
    fig.update_layout(height=430, title="CBP vs HHS load (separate axes: HHS is ~97% of total load)")
    st.plotly_chart(fig, width="stretch")
    share = d.hhs_care.sum() / d.total_load.sum() * 100
    a, b, cc = st.columns(3)
    a.metric("Mean HHS care", f"{d.hhs_care.mean():,.0f}")
    b.metric("Mean CBP custody", f"{d.cbp_custody.mean():,.0f}")
    cc.metric("HHS share of total load", f"{share:.1f}%")

with tab3:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.08,
                        subplot_titles=("Net daily intake (transfers - discharges)", "Cumulative net intake (reported days)"))
    fig.add_trace(go.Scatter(x=g.date, y=g.net_intake, name="Net intake", line=dict(color="#aaa")), row=1, col=1)
    if gran == "Daily":
        fig.add_trace(go.Scatter(x=g.date, y=g.net_ma7, name="7-day avg", line=dict(color="#1f6f8b")), row=1, col=1)
    fig.add_hline(y=0, line_color="black", row=1, col=1)
    ci = d.net_intake.cumsum()
    fig.add_trace(go.Scatter(x=d.date, y=ci, name="Cumulative (window)", line=dict(color="#222")), row=2, col=1)
    if show_windows:
        shade(fig, d, "backlog_sustained", "red", row=1)
    fig.update_layout(height=560, legend=dict(orientation="h", y=-.12))
    st.plotly_chart(fig, width="stretch")
    st.caption("Red shading = sustained backlog (7-day average net intake positive for 14+ consecutive reported days). "
               "Caution: net intake is a weak proxy for the change in HHS care (see Data tab).")

with tab4:
    st.subheader("Prolonged strain and relief windows (full period)")
    a, b = st.columns(2)
    a.markdown("**Strain windows**")
    a.dataframe(pd.read_csv(PROC / "strain_windows.csv"), hide_index=True)
    b.markdown("**Relief windows**")
    b.dataframe(pd.read_csv(PROC / "relief_windows.csv"), hide_index=True)
    st.subheader("Data quality flags in selected range")
    q = st.columns(3)
    q[0].metric("Transfers > CBP custody (days)", int(d.flag_transfer_gt_custody.sum()))
    q[1].metric("Discharges > HHS care (days)", int(d.flag_discharge_gt_hhs.sum()))
    q[2].metric("Days following a reporting gap > 3 days", int(d.flag_gap_before.sum()))
    with st.expander("Download the filtered data"):
        st.download_button("CSV", d.to_csv(index=False).encode(), "uac_filtered.csv", "text/csv")
        st.dataframe(d.tail(30), hide_index=True)
