"""
app.py
------
Streamlit interactive dashboard for the Congestion Pricing Impact Analyzer.

Sections:
  1. Overview  — headline numbers, interactive metric cards
  2. Event Study — interactive event-study chart (Plotly)
  3. CATE Map  — folium choropleth of zone-level CATEs
  4. DML Results — multi-outcome coefficient plot
  5. Dose-Response — interactive dose-response curve
  6. Heterogeneity — CATE by borough / peak / trip type
  7. Robustness  — forest plot of all checks

Run:
  streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from loguru import logger

RESULTS_DIR = Path("artifacts/results")
FIGURES_DIR = Path("artifacts/figures")
PROC_DIR    = Path("artifacts/processed")

st.set_page_config(
    page_title="NYC Congestion Pricing Impact",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #F8FAFC; border-radius: 10px;
    padding: 16px 20px; margin-bottom: 10px;
    border-left: 4px solid #2563EB;
  }
  .metric-value { font-size: 28px; font-weight: 700; color: #2563EB; }
  .metric-label { font-size: 13px; color: #6B7280; margin-top: 4px; }
  h1 { color: #111827; }
  .stTabs [data-baseweb="tab"] { font-size: 15px; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data
def load_panel():
    p = PROC_DIR / "zone_day_panel_enriched.parquet"
    if not p.exists():
        p = PROC_DIR / "zone_day_panel.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data
def load_result(name: str):
    import pickle
    p = RESULTS_DIR / f"{name}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    p2 = RESULTS_DIR / f"{name}.parquet"
    if p2.exists():
        return pd.read_parquet(p2)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/NYC_DOT_Logo.svg/200px-NYC_DOT_Logo.svg.png",
                     width=80)
    st.sidebar.title("⚙️ Controls")
    outcome = st.sidebar.selectbox(
        "Primary Outcome",
        ["log_trip_count", "avg_distance", "avg_duration", "avg_fare"],
        index=0,
    )
    trip_type = st.sidebar.multiselect(
        "Trip Types", ["yellow", "green", "fhvhv"], default=["yellow", "green", "fhvhv"]
    )
    window = st.sidebar.slider("Event Study Window (weeks)", 4, 24, 16, step=4)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Data**: NYC TLC Trip Records  \n"
        "**Shock**: 2025-01-05  \n"
        "**Methods**: DiD, CS-DiD, SDID, DML, Causal Forest"
    )
    return outcome, trip_type, window


# ── Section 1: Overview ────────────────────────────────────────────────────────

def section_overview(panel):
    st.markdown("## 🏙️ Overview")

    if panel is None:
        st.warning("Panel data not found. Run `python src/run_pipeline.py` first.")
        return

    SHOCK = pd.Timestamp("2025-01-05")
    pre  = panel[panel["date"] < SHOCK]
    post = panel[panel["date"] >= SHOCK]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pre_trips  = pre["trip_count"].sum()
        post_trips = post["trip_count"].sum()
        pct_change = (post_trips / pre_trips - 1) * 100 if pre_trips else 0
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{pct_change:+.1f}%</div>
          <div class="metric-label">Total trip change (post vs pre)</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        cbd = panel[panel["treat_binary"] == 1]
        pre_cbd  = cbd[cbd["date"] < SHOCK]["trip_count"].mean()
        post_cbd = cbd[cbd["date"] >= SHOCK]["trip_count"].mean()
        cbd_chg  = (post_cbd / pre_cbd - 1) * 100 if pre_cbd else 0
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{cbd_chg:+.1f}%</div>
          <div class="metric-label">CBD zone trip change</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        fee_total = post.get("cbd_fee_total", pd.Series([0])).sum()
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">${fee_total/1e6:.1f}M</div>
          <div class="metric-label">Total congestion fees collected</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        n_zones = panel["zone_id"].nunique()
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{n_zones}</div>
          <div class="metric-label">Zones in analysis</div>
        </div>""", unsafe_allow_html=True)

    # Daily trip trend
    st.markdown("### Daily Trip Volume: CBD vs Non-CBD")
    daily = (
        panel.groupby(["date", "treat_binary"])["trip_count"]
        .sum().reset_index()
        .rename(columns={"treat_binary": "CBD Zone"})
    )
    daily["CBD Zone"] = daily["CBD Zone"].map({1: "CBD (treated)", 0: "Non-CBD (control)"})

    fig = px.line(daily, x="date", y="trip_count", color="CBD Zone",
                  color_discrete_map={
                      "CBD (treated)":      "#2563EB",
                      "Non-CBD (control)":  "#9CA3AF",
                  })
    fig.add_vline(x="2025-01-05", line_dash="dash", line_color="red",
                  annotation_text="Congestion Pricing", annotation_position="top")
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Daily Trips",
        legend_title="", height=380,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Section 2: Event Study ─────────────────────────────────────────────────────

def section_event_study(window: int):
    st.markdown("## 📈 Event Study")
    st.markdown(
        "Dynamic treatment effects via relative-week dummies. "
        "Flat pre-period confirms parallel trends assumption."
    )

    es_df = load_result("event_study")
    if es_df is None:
        st.info("Run the pipeline to generate event study results.")
        return

    es_df = es_df[es_df["rel_week"].between(-window, window)]

    fig = go.Figure()
    pre  = es_df[es_df["rel_week"] < 0]
    post = es_df[es_df["rel_week"] >= 0]

    for df_part, color, name in [
        (pre,  "#9CA3AF", "Pre-period"),
        (post, "#2563EB", "Post-period"),
    ]:
        fig.add_trace(go.Scatter(
            x=pd.concat([df_part["rel_week"], df_part["rel_week"].iloc[::-1]]),
            y=pd.concat([df_part["ci_high"], df_part["ci_low"].iloc[::-1]]),
            fill="toself", fillcolor=color,
            line=dict(color="rgba(0,0,0,0)"),
            opacity=0.2, showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=df_part["rel_week"], y=df_part["coef"],
            mode="lines+markers", name=name,
            line=dict(color=color, width=2),
            marker=dict(size=5),
        ))

    fig.add_hline(y=0, line_color="black", line_width=0.8)
    fig.add_vline(x=0, line_dash="dash", line_color="red",
                  annotation_text="Shock: 2025-01-05")

    fig.update_layout(
        xaxis_title="Weeks relative to congestion pricing",
        yaxis_title="Estimated effect on log(trip count)",
        height=420,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    ptp = es_df.attrs.get("pre_trend_pvalue", None)
    if ptp:
        color = "green" if ptp > 0.05 else "red"
        st.markdown(
            f"**Pre-trend joint F-test p-value: :{color}[{ptp:.4f}]**  "
            f"{'✓ Parallel trends hold' if ptp > 0.05 else '⚠ Possible pre-trend violation'}"
        )


# ── Section 3: DML Results ─────────────────────────────────────────────────────

def section_dml():
    st.markdown("## 🤖 Double ML Results")
    st.markdown(
        "DML partials out high-dimensional confounders via ML, "
        "then estimates an unbiased ATE on residuals."
    )

    dml_df = load_result("dml_results")
    if dml_df is None:
        st.info("Run the pipeline to generate DML results.")
        return

    if isinstance(dml_df, dict):
        dml_df = pd.DataFrame(dml_df)

    colors = ["#2563EB" if p < 0.05 else "#9CA3AF" for p in dml_df["p_value"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dml_df["theta"], y=dml_df["outcome"],
        orientation="h",
        error_x=dict(type="data", array=1.96 * dml_df["se"], visible=True),
        marker_color=colors,
        text=[f"{t:+.4f}{s}" for t, s in zip(dml_df["theta"], dml_df["sig"])],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="black", line_width=0.8)
    fig.update_layout(
        xaxis_title="DML-ATE (θ̂)",
        height=max(300, len(dml_df) * 55),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        dml_df[["outcome", "theta", "se", "p_value", "ci_low", "ci_high", "sig"]]
        .style.format({"theta": "{:+.4f}", "se": "{:.4f}", "p_value": "{:.4f}",
                       "ci_low": "{:.4f}", "ci_high": "{:.4f}"}),
        use_container_width=True,
    )


# ── Section 4: CATE Heterogeneity ─────────────────────────────────────────────

def section_cate():
    st.markdown("## 🗺️ Causal Forest: Heterogeneous Treatment Effects")

    zone_cate = load_result("zone_cate")
    feat_imp  = load_result("feature_importance")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### CATE Distribution")
        if zone_cate is not None:
            fig = px.histogram(zone_cate, x="cate", nbins=40,
                               color_discrete_sequence=["#2563EB"])
            fig.add_vline(x=zone_cate["cate"].mean(), line_dash="dash",
                          line_color="red", annotation_text="Mean CATE")
            fig.add_vline(x=0, line_color="black", line_width=0.8)
            fig.update_layout(height=300, plot_bgcolor="white",
                              paper_bgcolor="white",
                              xaxis_title="CATE τ̂(x)", yaxis_title="Zones")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("CATE results not found.")

    with col2:
        st.markdown("### Feature Importance (CATE drivers)")
        if feat_imp is not None:
            if isinstance(feat_imp, pd.Series):
                feat_df = feat_imp.head(12).reset_index()
                feat_df.columns = ["feature", "importance"]
            else:
                feat_df = feat_imp.head(12)

            fig = px.bar(feat_df, x="importance", y="feature",
                         orientation="h",
                         color_discrete_sequence=["#7C3AED"])
            fig.update_layout(height=300, plot_bgcolor="white",
                              paper_bgcolor="white",
                              yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not found.")


# ── Section 5: Robustness ──────────────────────────────────────────────────────

def section_robustness():
    st.markdown("## 🔍 Robustness Checks")

    rob_df = load_result("robustness")
    l1     = load_result("l1_did")

    if rob_df is None:
        st.info("Run pipeline to generate robustness results.")
        return

    main_est = l1.get("ate", 0) if isinstance(l1, dict) else 0
    main_se  = l1.get("se",  0) if isinstance(l1, dict) else 0

    fig = go.Figure()
    colors = ["#2563EB" if p < 0.05 else "#9CA3AF" for p in rob_df["p"]]
    fig.add_trace(go.Scatter(
        x=rob_df["theta"], y=rob_df["check"],
        mode="markers",
        error_x=dict(type="data", array=1.96 * rob_df["se"], visible=True),
        marker=dict(color=colors, size=8),
    ))
    fig.add_vline(x=main_est, line_dash="dash", line_color="#2563EB",
                  annotation_text=f"Main estimate ({main_est:.4f})")
    fig.add_vrect(
        x0=main_est - 1.96 * main_se,
        x1=main_est + 1.96 * main_se,
        fillcolor="#2563EB", opacity=0.08, layer="below",
    )
    fig.add_vline(x=0, line_color="black", line_width=0.8)
    fig.update_layout(
        xaxis_title="Estimated ATT / θ̂",
        height=max(400, len(rob_df) * 28),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        rob_df[["group", "check", "theta", "se", "p", "sig"]]
        .style.format({"theta": "{:+.4f}", "se": "{:.4f}", "p": "{:.4f}"}),
        use_container_width=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("🚕 NYC Congestion Pricing Impact Analyzer")
    st.markdown(
        "**Causal analysis of the 2025-01-05 MTA congestion pricing shock "
        "on urban trip behavior.**  "
        "Methods: L1 DiD → L2 CS-DiD → L3 Continuous DiD → L4 SDID → "
        "L5a DML → L5b Causal Forest"
    )

    outcome, trip_type, window = render_sidebar()
    panel = load_panel()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Event Study",
        "🤖 Double ML",
        "🗺️ Causal Forest",
        "🔍 Robustness",
    ])

    with tab1:
        section_overview(panel)
    with tab2:
        section_event_study(window)
    with tab3:
        section_dml()
    with tab4:
        section_cate()
    with tab5:
        section_robustness()


if __name__ == "__main__":
    main()
