"""
dashboard/app.py
----------------
Streamlit dashboard for the Garud Constellation Optimizer.

Run with:
    streamlit run dashboard/app.py

Make sure you have run the pipeline first:
    python run_pipeline.py
"""

import streamlit as st
import numpy as np
import json
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# ── Path setup ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Garud Constellation Optimizer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0a0a1a; }
  .block-container { padding-top: 1.5rem; }
  .metric-card {
      background: linear-gradient(135deg, #1a1a2e, #16213e);
      border: 1px solid #0f3460;
      border-radius: 12px;
      padding: 1.2rem 1.5rem;
      text-align: center;
  }
  .metric-value { font-size: 2.2rem; font-weight: 700; color: #00d4ff; }
  .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 0.2rem; }
  .metric-delta-good { font-size: 1rem; color: #00ff88; font-weight: 600; }
  .metric-delta-bad  { font-size: 1rem; color: #ff4444; font-weight: 600; }
  .section-header {
      font-size: 1.3rem; font-weight: 700;
      color: #00d4ff; margin: 1.5rem 0 0.8rem 0;
      border-bottom: 1px solid #0f3460; padding-bottom: 0.4rem;
  }
  div[data-testid="stSidebar"] { background-color: #0d0d1f; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        results = json.load(open(os.path.join(DATA, "pipeline_results.json")))
        grid_lats = np.load(os.path.join(DATA, "grid_lats.npy"))
        grid_lons = np.load(os.path.join(DATA, "grid_lons.npy"))
        naive_revisit = np.load(os.path.join(DATA, "naive_revisit.npy"))
        opt_revisit   = np.load(os.path.join(DATA, "optimized_revisit.npy"))
        naive_cov     = np.load(os.path.join(DATA, "naive_coverage.npy"))
        opt_cov       = np.load(os.path.join(DATA, "optimized_coverage.npy"))
        naive_tracks  = json.load(open(os.path.join(DATA, "naive_tracks.json")))
        opt_tracks    = json.load(open(os.path.join(DATA, "optimized_tracks.json")))
        return results, grid_lats, grid_lons, naive_revisit, opt_revisit, \
               naive_cov, opt_cov, naive_tracks, opt_tracks, True
    except FileNotFoundError:
        return {}, None, None, None, None, None, None, None, None, False


def india_mask(grid_lats, grid_lons):
    lat_m = (grid_lats >= 8)  & (grid_lats <= 37)
    lon_m = (grid_lons >= 68) & (grid_lons <= 98)
    return lat_m, lon_m


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/240px-Emblem_of_India.svg.png", width=60)
    st.markdown("## 🛰️ Garud Optimizer")
    st.markdown("*Constellation Design Tool*")
    st.markdown("---")

    st.markdown("### About")
    st.markdown("""
    This tool finds the **optimal orbital configuration** for a LEO satellite constellation
    to maximize coverage over India.

    **Pipeline:**
    1. SGP4 orbit propagation
    2. Coverage gap analysis
    3. Genetic Algorithm optimization
    4. ML surrogate acceleration
    """)
    st.markdown("---")
    st.markdown("### Mission Parameters")
    st.markdown("🌍 **Target Region:** India")
    st.markdown("🛸 **Orbit Type:** LEO (Low Earth Orbit)")
    st.markdown("📡 **Propagator:** SGP4 (NORAD standard)")
    st.markdown("---")
    st.caption("Built for Dhruva Space — Project Garud")


# ── Load data ─────────────────────────────────────────────────────────
results, grid_lats, grid_lons, naive_revisit, opt_revisit, \
naive_cov, opt_cov, naive_tracks, opt_tracks, data_ok = load_data()

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#00d4ff; margin-bottom:0;'>🛰️ Garud Constellation Optimizer</h1>
<p style='color:#aaa; margin-top:0.2rem;'>
Surrogate-Assisted ML Optimization for LEO Satellite Constellation Design &nbsp;|&nbsp;
<b style='color:#00ff88;'>Dhruva Space — Project Garud</b>
</p>
""", unsafe_allow_html=True)
st.markdown("---")

if not data_ok:
    st.error("⚠️ Data not found. Please run the pipeline first:")
    st.code("python run_pipeline.py", language="bash")
    st.info("After the pipeline completes (~30-60 min), refresh this page.")
    st.stop()

# ── Extract key numbers ───────────────────────────────────────────────
n_sats     = results.get("n_sats", 6)
alt_km     = results.get("altitude_km", 550)
score_naive = results["naive"]["score"]
score_ga    = results["ga_optimized"]["best_score_min"]
score_sga   = results.get("surrogate_ga", {}).get("verified_score_min", score_ga)
best_score  = min(score_ga, score_sga)
improvement = (score_naive - best_score) / score_naive * 100
r2          = results.get("surrogate_metrics", {}).get("r2_score", "N/A")
mae         = results.get("surrogate_metrics", {}).get("mae_minutes", "N/A")

lat_m, lon_m = india_mask(grid_lats, grid_lons)
india_naive_avg = naive_revisit[np.ix_(lat_m, lon_m)][naive_revisit[np.ix_(lat_m, lon_m)] > 0].mean()
india_opt_avg   = opt_revisit[np.ix_(lat_m, lon_m)][opt_revisit[np.ix_(lat_m, lon_m)] > 0].mean()

# ── KPI Cards ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Key Results</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-value">{n_sats}</div>
      <div class="metric-label">Satellites</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-value">{alt_km:.0f} km</div>
      <div class="metric-label">Orbital Altitude</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-value">{score_naive:.0f} min</div>
      <div class="metric-label">Naive Revisit (95th pct)</div>
      <div class="metric-delta-bad">▲ Baseline</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-value">{best_score:.0f} min</div>
      <div class="metric-label">Optimized Revisit</div>
      <div class="metric-delta-good">▼ {improvement:.1f}% better</div>
    </div>""", unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-value">{r2}</div>
      <div class="metric-label">ML Surrogate R²</div>
      <div class="metric-delta-good">Neural Network</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Ground Tracks",
    "🔥 Coverage Heatmap",
    "📈 Optimization Progress",
    "🤖 ML Surrogate"
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Ground Tracks Globe
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">🌍 Satellite Ground Tracks</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    COLORS = ["#00d4ff", "#00ff88", "#ff6b35", "#ff0080", "#ffd700", "#a855f7"]

    def make_globe(tracks, title, color_list):
        fig = go.Figure()

        # Earth sphere base
        fig.add_trace(go.Scattergeo(
            lat=[], lon=[], mode='markers',
            marker=dict(size=0), showlegend=False
        ))

        for i, t in enumerate(tracks):
            lats = t["lats"]
            lons = t["lons"]
            color = color_list[i % len(color_list)]

            # Break track at dateline crossings
            fig.add_trace(go.Scattergeo(
                lat=lats, lon=lons,
                mode='lines',
                line=dict(width=1.5, color=color),
                name=f'Sat {t["sat_id"]}',
                opacity=0.85,
            ))
            # Current position dot
            if lats:
                fig.add_trace(go.Scattergeo(
                    lat=[lats[0]], lon=[lons[0]],
                    mode='markers',
                    marker=dict(size=8, color=color,
                                line=dict(width=1, color='white')),
                    showlegend=False,
                ))

        # India highlight box
        fig.add_trace(go.Scattergeo(
            lat=[8, 8, 37, 37, 8],
            lon=[68, 98, 98, 68, 68],
            mode='lines',
            line=dict(color='#ffd700', width=2, dash='dot'),
            name='India Region',
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(color='#00d4ff', size=15)),
            geo=dict(
                showland=True, landcolor='#1a2a1a',
                showocean=True, oceancolor='#0a0a2a',
                showcountries=True, countrycolor='#334',
                showcoastlines=True, coastlinecolor='#445',
                bgcolor='#0a0a1a',
                projection_type='natural earth',
            ),
            paper_bgcolor='#0a0a1a',
            plot_bgcolor='#0a0a1a',
            legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.5)'),
            margin=dict(l=0, r=0, t=40, b=0),
            height=420,
        )
        return fig

    with col_left:
        st.plotly_chart(
            make_globe(naive_tracks, f"Naive Config — {n_sats} Satellites (Equal RAAN spacing)", COLORS),
            use_container_width=True
        )

    with col_right:
        st.plotly_chart(
            make_globe(opt_tracks, f"Optimized Config — {n_sats} Satellites (GA + ML)", COLORS),
            use_container_width=True
        )

    # Config details
    st.markdown('<div class="section-header">⚙️ Configuration Details</div>', unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)

    with dc1:
        st.markdown("**Naive Configuration**")
        inc_n  = results["naive"]["inclination"]
        raans_n = results["naive"]["raans"]
        df_naive = pd.DataFrame({
            "Satellite": [f"Sat {i+1}" for i in range(len(raans_n))],
            "RAAN (°)": [round(r, 1) for r in raans_n],
            "Inclination (°)": [inc_n] * len(raans_n),
        })
        st.dataframe(df_naive, use_container_width=True, hide_index=True)

    with dc2:
        st.markdown("**Optimized Configuration**")
        ga = results["ga_optimized"]
        inc_o  = ga["inclination"]
        raans_o = ga["raan_spacings"]
        df_opt = pd.DataFrame({
            "Satellite": [f"Sat {i+1}" for i in range(len(raans_o))],
            "RAAN (°)": [round(r, 1) for r in raans_o],
            "Inclination (°)": [round(inc_o, 2)] * len(raans_o),
        })
        st.dataframe(df_opt, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Coverage Heatmap
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🔥 Revisit Time Heatmap — Naive vs Optimized</div>', unsafe_allow_html=True)
    st.caption("Darker = longer gap between satellite passes (worse). Brighter = more frequent coverage (better).")

    view = st.radio("View", ["Side by Side", "Difference Map"], horizontal=True)

    def make_heatmap(data, title, colorscale='RdYlGn_r', zmax=None):
        masked = np.where(data > 0, data, np.nan)
        if zmax is None:
            zmax = np.nanpercentile(masked, 95)
        fig = go.Figure(go.Heatmap(
            z=masked,
            x=grid_lons,
            y=grid_lats,
            colorscale=colorscale,
            zmin=0, zmax=zmax,
            colorbar=dict(title=dict(text="Minutes", font=dict(color="white")), tickfont=dict(color="white")),
            hoverongaps=False,
        ))
        # India box
        fig.add_shape(type="rect",
            x0=68, y0=8, x1=98, y1=37,
            line=dict(color="#ffd700", width=2, dash="dot"))
        fig.add_annotation(x=83, y=38.5, text="India",
            font=dict(color="#ffd700", size=12), showarrow=False)

        fig.update_layout(
            title=dict(text=title, font=dict(color="#fdfdfd", size=14)),
            paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
            font=dict(color='white'),
            xaxis=dict(title="Longitude (°)", gridcolor='#222'),
            yaxis=dict(title="Latitude (°)", gridcolor='#222'),
            margin=dict(l=50, r=20, t=50, b=50),
            height=400,
        )
        return fig

    if view == "Side by Side":
        hc1, hc2 = st.columns(2)
        zmax = max(np.nanpercentile(naive_revisit[naive_revisit>0], 95),
                   np.nanpercentile(opt_revisit[opt_revisit>0], 95))
        with hc1:
            st.plotly_chart(make_heatmap(naive_revisit,
                f"Naive — India avg {india_naive_avg:.0f} min revisit", zmax=zmax),
                use_container_width=True)
        with hc2:
            st.plotly_chart(make_heatmap(opt_revisit,
                f"Optimized — India avg {india_opt_avg:.0f} min revisit", zmax=zmax),
                use_container_width=True)

    else:
        diff = naive_revisit - opt_revisit
        fig_diff = go.Figure(go.Heatmap(
            z=diff, x=grid_lons, y=grid_lats,
            colorscale='RdYlGn',
            colorbar=dict(title=dict(text="Minutes saved", font=dict(color="white")), tickfont=dict(color="white")),
        ))
        fig_diff.add_shape(type="rect", x0=68, y0=8, x1=98, y1=37,
            line=dict(color="#ffd700", width=2, dash="dot"))
        fig_diff.add_annotation(x=83, y=38.5, text="India",
            font=dict(color="#ffd700", size=12), showarrow=False)
        fig_diff.update_layout(
            title=dict(text="Improvement Map (green = optimized config saves more time)",
                       font=dict(color="#f8fafb", size=14)),
            paper_bgcolor='#0a0a1a', plot_bgcolor='#0a0a1a',
            font=dict(color='white'),
            xaxis=dict(title="Longitude (°)", gridcolor='#222'),
            yaxis=dict(title="Latitude (°)", gridcolor='#222'),
            height=450,
        )
        st.plotly_chart(fig_diff, use_container_width=True)

    # India stats
    st.markdown('<div class="section-header">📍 India Region Statistics</div>', unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    india_naive = naive_revisit[np.ix_(lat_m, lon_m)]
    india_opt   = opt_revisit[np.ix_(lat_m, lon_m)]
    india_naive = india_naive[india_naive > 0]
    india_opt   = india_opt[india_opt > 0]

    metrics = [
        ("Avg Revisit — Naive",     f"{india_naive.mean():.1f} min", ""),
        ("Avg Revisit — Optimized", f"{india_opt.mean():.1f} min",
         f"▼ {(india_naive.mean()-india_opt.mean())/india_naive.mean()*100:.1f}% better"),
        ("Max Gap — Naive",         f"{india_naive.max():.0f} min", ""),
        ("Max Gap — Optimized",     f"{india_opt.max():.0f} min",
         f"▼ {(india_naive.max()-india_opt.max())/india_naive.max()*100:.1f}% better"),
    ]
    for col, (label, val, delta) in zip([sc1, sc2, sc3, sc4], metrics):
        col_cls = "metric-delta-good" if "▼" in delta else ""
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="font-size:1.6rem">{val}</div>
              <div class="metric-label">{label}</div>
              <div class="{col_cls}">{delta}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Optimization Progress
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📈 Genetic Algorithm Convergence</div>', unsafe_allow_html=True)

    ga_history = results["ga_optimized"].get("score_history", [])
    sga_history = results.get("surrogate_ga", {}).get("score_history", [])

    fig_conv = go.Figure()

    if ga_history:
        fig_conv.add_trace(go.Scatter(
            x=list(range(1, len(ga_history)+1)), y=ga_history,
            mode='lines+markers', name='GA (Full Simulator)',
            line=dict(color="#f9fbfb", width=2),
            marker=dict(size=7),
        ))

    if sga_history:
        fig_conv.add_trace(go.Scatter(
            x=list(range(1, len(sga_history)+1)), y=sga_history,
            mode='lines+markers', name='Surrogate GA (ML-Accelerated)',
            line=dict(color='#00ff88', width=2, dash='dash'),
            marker=dict(size=5),
        ))

    # Naive baseline line
    fig_conv.add_hline(y=score_naive, line_dash="dot",
        line_color="#ff4444",
        annotation_text=f"Naive Baseline: {score_naive:.0f} min",
        annotation_font_color="#ff4444")

    fig_conv.update_layout(
        paper_bgcolor='#0a0a1a', plot_bgcolor='#111128',
        font=dict(color='white'),
        xaxis=dict(title="Generation", gridcolor='#222', color='white'),
        yaxis=dict(title="Best Score (min) — lower is better",
                   gridcolor='#222', color='white'),
        legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.5)'),
        height=420,
        margin=dict(l=60, r=20, t=20, b=60),
    )
    st.plotly_chart(fig_conv, use_container_width=True)

    # Comparison bar chart
    st.markdown('<div class="section-header">📊 Method Comparison</div>', unsafe_allow_html=True)

    methods = ["Naive Baseline", "Genetic Algorithm", "Surrogate GA (verified)"]
    scores  = [score_naive, score_ga, results.get("surrogate_ga", {}).get("verified_score_min", score_ga)]
    colors  = ["#ff4444", "#eff5f6", "#00ff88"]

    fig_bar = go.Figure(go.Bar(
        x=methods, y=scores,
        marker_color=colors,
        text=[f"{s:.1f} min" for s in scores],
        textposition='outside',
        textfont=dict(color='white', size=13),
    ))
    fig_bar.update_layout(
        paper_bgcolor='#0a0a1a', plot_bgcolor='#111128',
        font=dict(color='white'),
        yaxis=dict(title="95th Pct Revisit Time (min) over India",
                   gridcolor='#222', color='white'),
        xaxis=dict(color='white'),
        height=380,
        margin=dict(l=60, r=20, t=20, b=60),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # How GA works explainer
    with st.expander("📖 How the Genetic Algorithm Works"):
        st.markdown("""
        The **Genetic Algorithm** mimics natural evolution to find the best constellation:

        1. **Initialize** — Create 20 random constellation configurations (random inclination + RAANs)
        2. **Evaluate** — Run the full SGP4 physics simulation for each, compute India revisit time
        3. **Select** — Keep the best 50% (survival of the fittest)
        4. **Crossover** — Combine two parent configs to create a child
           - e.g., Parent 1: `[55°, 0°, 60°, 120°, 180°, 240°, 300°]`
           - Parent 2:    `[72°, 10°, 85°, 155°, 200°, 270°, 340°]`
           - Child:       `[55°, 0°, 60°, 155°, 200°, 270°, 340°]`
        5. **Mutate** — Randomly tweak some values (prevents getting stuck)
        6. **Repeat** — Run for N generations until convergence
        """)


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — ML Surrogate
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🤖 ML Surrogate Model — Neural Network Coverage Predictor</div>',
                unsafe_allow_html=True)

    sm = results.get("surrogate_metrics", {})
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, (label, val) in zip([mc1, mc2, mc3, mc4], [
        ("R² Score",        str(sm.get("r2_score", "N/A"))),
        ("MAE",             f"{sm.get('mae_minutes', 'N/A')} min"),
        ("Training Samples",str(sm.get("n_train", "N/A"))),
        ("Test Samples",    str(sm.get("n_test", "N/A"))),
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style="font-size:1.6rem">{val}</div>
              <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture diagram
    with st.expander("🧠 Neural Network Architecture", expanded=True):
        st.markdown("""
        ```
        Input Layer       Hidden Layer 1    Hidden Layer 2    Hidden Layer 3    Output
        ─────────────     ──────────────    ──────────────    ──────────────    ──────
        inclination  ──►                                                      
        RAAN_1       ──►                                                      
        RAAN_2       ──►   128 neurons  ──►  64 neurons   ──►  32 neurons  ──►  Score
        ...          ──►   (ReLU)           (ReLU)            (ReLU)            (minutes)
        RAAN_n       ──►                                                      
        gap_min      ──►  ← feature engineering adds geometry-aware inputs
        gap_max      ──►
        gap_std      ──►
        sin(inc)     ──►
        cos(inc)     ──►
        ```

        **Feature Engineering** adds domain knowledge:
        - `sin(inclination)`, `cos(inclination)` — circular encoding (0° = 360°)
        - RAAN gap statistics (min, max, std) — captures how evenly spaced satellites are
        - These help the network learn the *geometry* of coverage, not just raw numbers
        """)

    # Why surrogate explanation
    st.markdown('<div class="section-header">⚡ Why the Surrogate Matters</div>', unsafe_allow_html=True)

    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("""
        **Without Surrogate (Standard GA):**
        - Each evaluation = full 6-hour physics simulation
        - ~30 seconds per config
        - 20 configs × 12 generations = 240 evaluations
        - Total: **~2 hours**
        - Population size limited by time
        """)
    with tc2:
        st.markdown("""
        **With Surrogate (Surrogate-Assisted GA):**
        - Each evaluation = neural network forward pass
        - ~0.001 seconds per config
        - 200 configs × 40 generations = 8,000 evaluations
        - Total: **~8 seconds** (+ 1 final verification)
        - **~1000x faster** → much larger search space explored
        """)

    st.info("""
    💡 **This is real aerospace engineering practice.**
    NASA, ESA, and Airbus use surrogate models (also called "metamodels") for
    computationally expensive simulations — aerodynamic CFD, thermal analysis,
    structural FEA. The technique is called **Surrogate-Assisted Optimization (SAO)**.
    """)

    # Try it live
    st.markdown('<div class="section-header"> Try the Surrogate Live</div>', unsafe_allow_html=True)
    st.markdown("Adjust parameters and instantly predict the coverage score:")

    try:
        import pickle
        model_path = os.path.join(DATA, "surrogate_model.pkl")
        with open(model_path, 'rb') as f:
            sm_data = pickle.load(f)

        # Rebuild surrogate
        surrogate_model = sm_data['model']
        scaler_X = sm_data['scaler_X']
        scaler_y = sm_data['scaler_y']

        sys.path.insert(0, SRC)
        from surrogate_model import SurrogateModel
        surr = SurrogateModel()
        surr.model    = surrogate_model
        surr.scaler_X = scaler_X
        surr.scaler_y = scaler_y
        surr.is_trained = True

        inc_live = st.slider("Inclination (°)", 30.0, 90.0, 55.0, 0.5)
        st.markdown(f"**RAAN spacings** — drag to rearrange {n_sats} satellites:")
        raan_cols = st.columns(n_sats)
        raans_live = []
        for i, col in enumerate(raan_cols):
            default = (i * 360 / n_sats) % 360
            r = col.number_input(f"Sat {i+1}", 0.0, 360.0, float(round(default, 1)), 5.0,
                                 key=f"raan_{i}")
            raans_live.append(r)

        config_vec = np.array([inc_live] + raans_live)
        pred_score = surr.predict(config_vec)[0]

        delta_pct = (score_naive - pred_score) / score_naive * 100
        color = "#00ff88" if pred_score < score_naive else "#ff4444"
        arrow = "▼" if pred_score < score_naive else "▲"

        st.markdown(f"""
        <div class="metric-card" style="margin-top:1rem">
          <div style="font-size:0.9rem; color:#aaa;">Predicted 95th Pct Revisit Time (India)</div>
          <div style="font-size:2.5rem; font-weight:700; color:{color};">{pred_score:.1f} min</div>
          <div style="color:{color}; font-size:1.1rem;">{arrow} {abs(delta_pct):.1f}% vs naive baseline</div>
          <div style="color:#888; font-size:0.8rem; margin-top:0.3rem;">Predicted instantly by neural network (no simulation needed)</div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Live predictor unavailable: {e}")

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#555; font-size:0.8rem;'>
  Built for <b style='color:#F4F6F6;'>Dhruva Space — Project Garud</b> Internship Application &nbsp;|&nbsp;
  SGP4 Propagation · Genetic Algorithm · ML Surrogate · Streamlit
</div>
""", unsafe_allow_html=True)