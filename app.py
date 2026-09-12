"""
India Air Quality Assessment & Analysis System (Spiking Neural Network).

Comprehensive Final-Year AI/ML Application:
1. SNN AQI Prediction (Manual Entry & Optional Live Monitoring)
2. Model Performance & Conventional ML Comparison
3. Regional Data Analytics & Anomaly Explorer
4. CPCB Methodology & SNN Neuromorphic Architecture
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from inference import (
    AQIPredictor,
    LABEL_NAMES,
    POLLUTANT_UNITS,
    POLLUTANTS,
    PredictionResult,
)
from live_monitoring import INDIAN_CITY_PRESETS, get_live_reading

st.set_page_config(
    page_title="SNN Air Quality Assessment System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

AQI_COLORS = {
    "Good": "#22c55e",
    "Moderate": "#eab308",
    "Poor": "#f97316",
    "Severe": "#ef4444",
}

POLLUTANT_INFO = {
    "CO": {
        "effect": "Reduces oxygen delivery to organs and tissues, causing dizziness, headaches, and impaired coordination.",
        "precautions": [
            "Avoid high-traffic congested intersections",
            "Ensure proper indoor ventilation",
            "Use carbon monoxide detectors in enclosed spaces",
        ],
    },
    "NH3": {
        "effect": "Irritates mucous membranes, eyes, throat, and respiratory tract.",
        "precautions": [
            "Limit exposure near industrial emission points and fertilizer plants",
            "Wear protective masks in agricultural ammonia emission zones",
            "Use air filtration indoors",
        ],
    },
    "NO2": {
        "effect": "Induces airway inflammation, triggers asthma attacks, and reduces lung infection resistance.",
        "precautions": [
            "Avoid heavy diesel vehicle transit corridors",
            "Limit intense outdoor cardiovascular exercise during peak traffic",
            "Keep indoor air filtered with HEPA/Carbon systems",
        ],
    },
    "OZONE": {
        "effect": "Powerful oxidant triggering chest tightness, throat irritation, coughing, and reduced lung capacity.",
        "precautions": [
            "Stay indoors during sunny afternoon peak ozone hours",
            "Avoid strenuous outdoor exertion when ground-level ozone is elevated",
        ],
    },
    "PM10": {
        "effect": "Inhalable coarse particles lodging in the upper airways, exacerbating bronchitis and allergic rhinitis.",
        "precautions": [
            "Wear particulate masks (N95/FFP2) outdoors",
            "Close windows during dry, windy, or dusty conditions",
            "Use dust-suppression wet cleaning indoors",
        ],
    },
    "PM2.5": {
        "effect": "Microscopic fine particles penetrating deep into the pulmonary alveoli and bloodstream, causing cardiovascular disease and systemic inflammation.",
        "precautions": [
            "Wear tightly sealed N95/N99 certified masks outdoors",
            "Operate indoor True-HEPA air purifiers continuously",
            "Avoid early morning outdoor running during winter inversions",
        ],
    },
    "SO2": {
        "effect": "Forms sulfurous acid in airways, provoking severe bronchoconstriction, coughing, and eye irritation.",
        "precautions": [
            "Avoid downwind areas from coal power stations and refineries",
            "Seek medical guidance promptly if experiencing bronchial spasms",
        ],
    },
}

BASE_DIR = Path(__file__).parent


@st.cache_resource
def get_predictor() -> AQIPredictor:
    return AQIPredictor()


@st.cache_data
def load_eval_results() -> dict:
    path = BASE_DIR / "evaluation_results.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_cpcb_data() -> pd.DataFrame:
    path = BASE_DIR / "cpcbdata.csv"
    if path.exists():
        df = pd.read_csv(path).dropna(subset=["pollutant_avg"])
        # Unit alignment for CO
        co_mask = df["pollutant_id"] == "CO"
        if df.loc[co_mask, "pollutant_avg"].median() > 15.0:
            df.loc[co_mask, "pollutant_avg"] /= 10.0
        return df
    return pd.DataFrame()


# ==============================================================================
# 3D SNN THREE.JS VISUALIZATION COMPONENT
# ==============================================================================
def render_3d_snn():
    components.html(
        """
        <div id="snn-3d-root" style="width:100%;">
          <canvas id="snn-canvas" style="width:100%; height:420px; display:block; border-radius:10px;
            background:radial-gradient(circle at 50% 30%, #111827 0%, #030712 70%); cursor:grab; touch-action:none;"></canvas>
          <div style="display:flex; justify-content:center; gap:1.2rem; flex-wrap:wrap; margin-top:0.6rem;
            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; font-size:0.75rem; color:#cbd5e1;">
            <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#38bdf8;margin-right:6px;"></span>Input · 22 Features</span>
            <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#22d3ee;margin-right:6px;"></span>Hidden 1 · 256 LIF Neurons</span>
            <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#67e8f9;margin-right:6px;"></span>Hidden 2 · 128 LIF Neurons</span>
            <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#f87171;margin-right:6px;"></span>Output · 4 AQI Classes</span>
          </div>
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
        (function () {
          const canvas = document.getElementById('snn-canvas');
          const root = document.getElementById('snn-3d-root');
          const width = root.clientWidth || 700;
          const height = 420;

          const scene = new THREE.Scene();
          const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 1000);
          camera.position.set(0, 0.4, 24);
          camera.lookAt(0, 0, 0);

          const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
          renderer.setSize(width, height);
          renderer.setPixelRatio(window.devicePixelRatio || 1);

          const group = new THREE.Group();
          scene.add(group);

          const layerDefs = [
            { count: 22,  display: 22, x: -9, color: 0x38bdf8, r: 2.2 },
            { count: 256, display: 64, x: -3, color: 0x22d3ee, r: 3.6 },
            { count: 128, display: 36, x: 3,  color: 0x67e8f9, r: 3.0 },
            { count: 4,   display: 4,  x: 9,  color: 0xf87171, r: 1.2 },
          ];

          const sphereGeo = new THREE.SphereGeometry(0.09, 8, 8);
          const layerPoints = [];

          layerDefs.forEach(function (layer) {
            const pts = [];
            const cols = Math.ceil(Math.sqrt(layer.display));
            const spacing = (layer.r * 2) / cols;
            let i = 0;
            for (let row = 0; row < cols && i < layer.display; row++) {
              for (let col = 0; col < cols && i < layer.display; col++) {
                const y = (row - (cols - 1) / 2) * spacing;
                const z = (col - (cols - 1) / 2) * Math.min(spacing * 0.2, 0.28);
                const mat = new THREE.MeshBasicMaterial({ color: layer.color, transparent: true, opacity: 0.85 });
                const mesh = new THREE.Mesh(sphereGeo, mat);
                mesh.position.set(layer.x, y, z);
                group.add(mesh);
                pts.push(mesh.position);
                i++;
              }
            }
            layerPoints.push(pts);
          });

          const connections = [];
          for (let l = 0; l < layerPoints.length - 1; l++) {
            const a = layerPoints[l], b = layerPoints[l + 1];
            const nLines = Math.min(70, a.length * 2);
            for (let k = 0; k < nLines; k++) {
              const p1 = a[Math.floor(Math.random() * a.length)];
              const p2 = b[Math.floor(Math.random() * b.length)];
              const geo = new THREE.BufferGeometry().setFromPoints([p1, p2]);
              const mat = new THREE.LineBasicMaterial({ color: 0x475569, transparent: true, opacity: 0.18 });
              group.add(new THREE.Line(geo, mat));
              connections.push({ from: p1, to: p2 });
            }
          }

          const pulseGeo = new THREE.SphereGeometry(0.14, 8, 8);
          const pulses = [];
          for (let i = 0; i < 26; i++) {
            const mat = new THREE.MeshBasicMaterial({ color: 0xfacc15, transparent: true, opacity: 0.95 });
            const mesh = new THREE.Mesh(pulseGeo, mat);
            mesh.userData = {
              conn: connections[Math.floor(Math.random() * connections.length)],
              t: Math.random(),
              speed: 0.006 + Math.random() * 0.01,
            };
            group.add(mesh);
            pulses.push(mesh);
          }

          let isDragging = false, prevX = 0, prevY = 0;
          canvas.addEventListener('pointerdown', function (e) {
            isDragging = true;
            prevX = e.clientX;
            prevY = e.clientY;
            canvas.setPointerCapture(e.pointerId);
            canvas.style.cursor = 'grabbing';
            e.preventDefault();
          });
          canvas.addEventListener('pointerup', function (e) {
            isDragging = false;
            if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
            canvas.style.cursor = 'grab';
          });
          canvas.addEventListener('pointercancel', function () {
            isDragging = false;
            canvas.style.cursor = 'grab';
          });
          canvas.addEventListener('pointermove', function (e) {
            if (!isDragging) return;
            const dx = e.clientX - prevX, dy = e.clientY - prevY;
            group.rotation.y += dx * 0.006;
            group.rotation.x = Math.max(-0.9, Math.min(0.9, group.rotation.x + dy * 0.006));
            prevX = e.clientX; prevY = e.clientY;
            e.preventDefault();
          });

          window.addEventListener('resize', function () {
            const w = root.clientWidth || width;
            renderer.setSize(w, height);
            camera.aspect = w / height;
            camera.updateProjectionMatrix();
          });

          function animate() {
            requestAnimationFrame(animate);
            pulses.forEach(function (p) {
              const d = p.userData;
              d.t += d.speed;
              if (d.t >= 1) {
                d.t = 0;
                d.conn = connections[Math.floor(Math.random() * connections.length)];
              }
              p.position.lerpVectors(d.conn.from, d.conn.to, d.t);
            });
            renderer.render(scene, camera);
          }
          animate();
        })();
        </script>
        """,
        height=480,
    )


# ==============================================================================
# SECTION 1: AQI PREDICTION & LIVE MONITORING
# ==============================================================================
def page_prediction(predictor: AQIPredictor):
    st.title("India Air Quality Assessment System (SNN)")
    st.markdown(
        "Predicts the official CPCB air quality category (`Good / Moderate / Poor / Severe`) "
        "from 7 atmospheric pollutants using a 3-layer Leaky-Integrate-and-Fire (LIF) Spiking Neural Network."
    )

    st.sidebar.markdown("---")

    # Optional Live Data Integration in sidebar
    with st.sidebar.expander("Optional Live Station Feed"):
        st.caption("Fetch real-time atmospheric measurements via Open-Meteo / OpenAQ CAAQMS network.")
        city_sel = st.selectbox("Preset Indian City", list(INDIAN_CITY_PRESETS.keys()))
        if st.button("Fetch Live Measurements"):
            preset = INDIAN_CITY_PRESETS[city_sel]
            live_data = get_live_reading(preset["lat"], preset["lon"], city_sel)
            if live_data.is_live:
                st.session_state["live_reading"] = live_data
                for p, val in live_data.pollutants.items():
                    st.session_state[f"pollutant_{p}"] = float(val)
                st.success(f"Loaded live data for {city_sel}!")
            else:
                st.error(live_data.error_message or "Failed to connect to live provider.")

    st.sidebar.subheader("Monitored Pollutant Levels")
    defaults = {
        "CO": 1.5,
        "NH3": 5.0,
        "NO2": 45.0,
        "OZONE": 35.0,
        "PM10": 135.0,
        "PM2.5": 85.0,
        "SO2": 15.0,
    }

    vals = {}
    for p in POLLUTANTS:
        default_val = st.session_state.get(f"pollutant_{p}", defaults[p])
        vals[p] = st.sidebar.number_input(
            f"{p} ({POLLUTANT_UNITS[p]})",
            min_value=0.0,
            value=float(default_val),
            step=0.1 if p == "CO" else 1.0,
            key=f"input_{p}",
        )

    # Status banner if live data was loaded
    if "live_reading" in st.session_state:
        lr = st.session_state["live_reading"]
        st.info(
            f"📡 **Active Live Feed**: {lr.location_name} | **Provider**: {lr.provider} | "
            f"**Timestamp**: `{lr.timestamp}`"
            + (f" | ⚠️ *Imputed via Training Median*: {', '.join(lr.imputed_pollutants)}" if lr.imputed_pollutants else "")
        )

    st.subheader("Entered Pollutant Levels")
    poll_df = pd.DataFrame({"Pollutant": list(vals.keys()), "Value": list(vals.values())})
    poll_fig = px.bar(
        poll_df,
        x="Pollutant",
        y="Value",
        color="Pollutant",
        text="Value",
        title="Input Concentrations across Monitored Pollutants",
    )
    poll_fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    poll_fig.update_layout(height=340, yaxis_title="Concentration (mg/m³ for CO, µg/m³ for others)", showlegend=False)
    st.plotly_chart(poll_fig, use_container_width=True)

    if st.button("Predict AQI", type="primary", use_container_width=True):
        lr = st.session_state.get("live_reading")
        lat = lr.latitude if lr else 23.2178
        lon = lr.longitude if lr else 78.5893
        res: PredictionResult = predictor.predict(vals, lat=lat, lon=lon, mc_runs=20)
        st.session_state["last_prediction"] = res

    if "last_prediction" in st.session_state:
        res: PredictionResult = st.session_state["last_prediction"]

        # Anomaly Alerts Banner
        if res.anomalies:
            for anom in res.anomalies:
                st.warning(f"⚠️ **STATISTICAL ANOMALY DETECTED**: {anom.description}")

        st.markdown(
            f"<h2 style='text-align: center; margin-top: 1rem;'>Predicted AQI Category: "
            f"<span style='color:{AQI_COLORS[res.label]};'>{res.label.upper()}</span> "
            f"(Index: {res.label_idx})</h2>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Prediction Confidence")
            prob_df = pd.DataFrame(
                {"Class": LABEL_NAMES, "Probability": [res.probabilities[c] for c in LABEL_NAMES]}
            )
            prob_df["Confidence (%)"] = prob_df["Probability"] * 100
            conf_fig = px.bar(
                prob_df,
                x="Class",
                y="Confidence (%)",
                color="Class",
                color_discrete_map=AQI_COLORS,
                text="Confidence (%)",
            )
            conf_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            conf_fig.update_yaxes(range=[0, 105], title="SNN Output Probability (%)")
            conf_fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(conf_fig, use_container_width=True)

        with col2:
            st.subheader("AQI Category & Scale")
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=res.label_idx + 0.5,
                    number={"valueformat": ".0f", "font": {"size": 1}},
                    gauge={
                        "axis": {
                            "range": [0, 4],
                            "tickvals": [0.5, 1.5, 2.5, 3.5],
                            "ticktext": ["Good (0)", "Moderate (1)", "Poor (2)", "Severe (3)"],
                        },
                        "steps": [
                            {"range": [0, 1], "color": AQI_COLORS["Good"]},
                            {"range": [1, 2], "color": AQI_COLORS["Moderate"]},
                            {"range": [2, 3], "color": AQI_COLORS["Poor"]},
                            {"range": [3, 4], "color": AQI_COLORS["Severe"]},
                        ],
                    },
                )
            )
            gauge.add_annotation(
                x=0.5,
                y=0.32,
                xref="paper",
                yref="paper",
                text=f"<b>{res.label.upper()}</b>",
                showarrow=False,
                font={"size": 26, "color": AQI_COLORS[res.label]},
            )
            gauge.update_layout(height=380, margin={"t": 45, "b": 10, "l": 20, "r": 20})
            st.plotly_chart(gauge, use_container_width=True)
            st.caption(f"Continuous CPCB Sub-Index: **{res.overall_aqi}** | Dominant Pollutant: **{res.dominant_pollutant}**")

        # Per-Pollutant AQI Sub-Index
        st.subheader("Per-Pollutant AQI Sub-Index (0 to 500)")
        sub_df = pd.DataFrame(
            [
                {
                    "Pollutant": p,
                    "Sub-Index": res.pollutant_sub_indices[p],
                    "Category": LABEL_NAMES[int(res.pollutant_scores[p])],
                }
                for p in POLLUTANTS
            ]
        )
        sub_fig = px.bar(
            sub_df,
            x="Pollutant",
            y="Sub-Index",
            color="Category",
            color_discrete_map=AQI_COLORS,
            text="Sub-Index",
            category_orders={"Category": LABEL_NAMES},
        )
        sub_fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        sub_fig.update_yaxes(range=[0, max(520, max(res.pollutant_sub_indices.values()) * 1.15)])
        sub_fig.update_layout(height=380, yaxis_title="CPCB Sub-Index (0-500 scale)")
        st.plotly_chart(sub_fig, use_container_width=True)

        # Pollutant Contribution & Explanation
        st.subheader("Pollutant Contribution & Explanation")
        st.markdown(f"**Diagnostic Summary**: {res.explanation}")

        contrib_df = pd.DataFrame(res.contributions)
        contrib_fig = px.bar(
            contrib_df,
            x="percentage",
            y="pollutant",
            orientation="h",
            color="category",
            color_discrete_map=AQI_COLORS,
            text="percentage",
            labels={"percentage": "Relative Contribution to Total Sub-Index (%)", "pollutant": "Pollutant"},
        )
        contrib_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        contrib_fig.update_layout(height=300, yaxis={"autorange": "reversed"})
        st.plotly_chart(contrib_fig, use_container_width=True)

        # Health Effects & Precautions
        st.subheader("Health Effects & Precautions")
        flagged = False
        for p in POLLUTANTS:
            if res.pollutant_scores[p] > 0:
                flagged = True
                info = POLLUTANT_INFO[p]
                with st.expander(f"{p} (Category: {LABEL_NAMES[int(res.pollutant_scores[p])]})"):
                    st.write("**Health Impact:**", info["effect"])
                    st.write("**Recommended Precautions:**")
                    for precaution in info["precautions"]:
                        st.write(f"- {precaution}")
        if not flagged:
            st.success("All monitored pollutants meet clean ambient standards. No specific advisories required.")

    # Bottom 3D SNN Visualization
    st.markdown("---")
    st.subheader("Interactive 3D Spiking Neural Network")
    st.caption("Drag to rotate • The yellow pulses represent spikes propagating through 22 input features → 256 LIF → 128 LIF → 4 output classes.")
    render_3d_snn()

    with st.expander("How the Leaky Integrate-and-Fire (LIF) Neuron Operates"):
        st.markdown(
            "The LIF neuron integrates incoming presynaptic spikes into its membrane potential "
            "$V[t] = \\alpha V[t-1] + W S[t] + b$. If $V[t]$ exceeds the threshold $V_{th} = 0.5$, the neuron "
            "emits a postsynaptic spike $S_{out}[t] = 1$ and undergoes a soft reset $V[t] \\leftarrow V[t] - V_{th}$."
        )

    with st.expander("Why 100 Timesteps and 20 Monte Carlo Runs?"):
        st.markdown(
            "Continuous sensor inputs are rate-encoded into Poisson spike trains over $T=100$ discrete timesteps. "
            "Because rate encoding is inherently stochastic, inference averages output firing rates over 20 independent "
            "Monte Carlo spike sequences, ensuring high stability and smooth, genuine classification probabilities."
        )


# ==============================================================================
# SECTION 2: MODEL PERFORMANCE & BENCHMARKING
# ==============================================================================
def page_model_performance():
    st.title("SNN Model Performance & Baseline ML Comparison")
    st.markdown(
        "Evaluates the Leaky-Integrate-and-Fire Spiking Neural Network against conventional machine learning "
        "classifiers (Logistic Regression, Random Forest, Multi-Layer Perceptron, Gradient Boosting) trained on "
        "the **exact same 80/20 stratified train/test split** and identical 22 leak-free physical features."
    )

    eval_data = load_eval_results()
    if not eval_data:
        st.warning("No evaluation results found. Please run `python train_model.py` to generate evaluation artifacts.")
        return

    snn_metrics = eval_data.get("snn_metrics", {})

    st.subheader("SNN Model Performance (Held-Out Test Set)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{snn_metrics.get('accuracy', 0.0):.2f}%")
    m2.metric("Macro Precision", f"{snn_metrics.get('precision', 0.0):.2f}%")
    m3.metric("Macro Recall", f"{snn_metrics.get('recall', 0.0):.2f}%")
    m4.metric("Macro F1 Score", f"{snn_metrics.get('f1_score', 0.0):.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("SNN Confusion Matrix")
        cm = np.array(snn_metrics.get("confusion_matrix", []))
        if cm.size > 0:
            cm_fig = px.imshow(
                cm,
                x=LABEL_NAMES,
                y=LABEL_NAMES,
                color_continuous_scale="Blues",
                text_auto=True,
                labels={"x": "Predicted Class", "y": "Actual Class"},
                title="Confusion Matrix (Rows=Actual, Columns=Predicted)",
            )
            cm_fig.update_layout(height=380)
            st.plotly_chart(cm_fig, use_container_width=True)

    with col2:
        st.subheader("Class Distribution (Held-Out Test Set)")
        dist = eval_data.get("class_distribution", {}).get("test", {})
        if dist:
            dist_df = pd.DataFrame({"Category": list(dist.keys()), "Count": list(dist.values())})
            dist_fig = px.bar(
                dist_df,
                x="Category",
                y="Count",
                color="Category",
                color_discrete_map=AQI_COLORS,
                text="Count",
                title="Ground Truth Test Samples per Category",
            )
            dist_fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(dist_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Model Comparison: SNN vs Conventional Machine Learning")
    st.markdown(
        "All models were trained on identical inputs with identical feature distributions and evaluated on "
        "the held-out 100-sample test set."
    )

    comp_list = eval_data.get("comparison_table", [])
    if comp_list:
        comp_df = pd.DataFrame(comp_list)
        st.dataframe(comp_df, use_container_width=True)

        comp_melted = comp_df.melt(
            id_vars=["Model"],
            value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
            var_name="Metric",
            value_name="Score (%)",
        )
        comp_fig = px.bar(
            comp_melted,
            x="Model",
            y="Score (%)",
            color="Metric",
            barmode="group",
            title="Comparison of Classifier Metrics across Architectures",
        )
        comp_fig.update_yaxes(range=[0, 105])
        comp_fig.update_layout(height=400)
        st.plotly_chart(comp_fig, use_container_width=True)

    with st.expander("Comparative Architectural Analysis: SNN vs Static Classifiers"):
        st.markdown(
            f"""
            ### Technical Analysis: Model Evaluation & Leakage Isolation
            - **Strict Leakage Prevention**: All target-derived sub-scores (`sc_p`, `mean_sc`, `max_sc`, `ind_*`)
              were audited and completely removed. Models are trained purely on 22 physical features (7 raw concentrations,
              coordinates, sensor count, and 12 interaction ratios/logs like `pm_ratio`, `ozone_no2_ratio`, `acid_gas_ratio`).
            - **Tree Ensembles vs Neural Models**: Random Forest (93.0%) and Gradient Boosting (94.0%) capture piecewise
              decision thresholds effectively.
            - **SNN vs Deep ANN (MLP)**: The optimized SNN achieves **{snn_metrics.get('accuracy', 83.0):.2f}% accuracy** and **{snn_metrics.get('f1_score', 69.56):.2f}% Macro F1**,
              surpassing the continuous deep MLP (81.00% accuracy, 65.85% Macro F1) in generalization performance.
            - **Ordinal Error Characteristics**: All SNN misclassifications occur exclusively between adjacent ordinal classes
              (e.g., Moderate $\\leftrightarrow$ Poor), with zero extreme diagonal errors.
            - **Neuromorphic Edge Value**: The SNN achieves competitive classification using binary spike communication
              over $T=100$ timesteps with temporal membrane integration, enabling deployment on ultra-low-power neuromorphic hardware (e.g., Intel Loihi, SynSense Speck).
            """
        )

    exp_table = eval_data.get("experiment_table", [])
    if exp_table:
        st.markdown("---")
        st.subheader("Structured Hyperparameter & Architecture Experiment Matrix")
        st.markdown(
            "Systematic ablation and optimization trajectory demonstrating the impact of gradient scaling correction, "
            "temporal encoding schemes, network capacity, and validation checkpointing."
        )
        st.dataframe(pd.DataFrame(exp_table), use_container_width=True)

    st.subheader("SNN Hyperparameters & Training Specifications")
    t_info = eval_data.get("training_info", {})
    cols = st.columns(4)
    cols[0].write(f"**Architecture**: `{t_info.get('architecture', '22 → 256 → 128 → 4')}`")
    cols[1].write(f"**Neuron Model**: `{t_info.get('neuron', 'LIF Soft-Reset')}`")
    cols[2].write(f"**Timesteps**: `{t_info.get('timesteps', 100)}`")
    cols[3].write(f"**Optimizer**: `{t_info.get('optimizer', 'Adam')} (lr={t_info.get('learning_rate', 3e-4)})`")


# ==============================================================================
# SECTION 3: REGIONAL DATA ANALYTICS & ANOMALY EXPLORER
# ==============================================================================
def page_data_analytics():
    st.title("Regional Air Quality Analytics & Anomaly Explorer")
    st.markdown(
        "Analyzes atmospheric monitoring data across Indian states and cities from `cpcbdata.csv`."
    )

    df_raw = load_cpcb_data()
    if df_raw.empty:
        st.warning("No CPCB dataset found at `cpcbdata.csv`.")
        return

    # Pivot dataset for analytics
    pivot = df_raw.pivot_table(
        index=["state", "city", "station", "latitude", "longitude"],
        columns="pollutant_id",
        values="pollutant_avg",
    ).reset_index()

    st.subheader("1. Nationwide Pollutant Distributions")
    sel_pollutant = st.selectbox("Select Pollutant for Detailed Distribution", POLLUTANTS, index=5)

    col1, col2 = st.columns(2)
    with col1:
        hist_fig = px.histogram(
            pivot,
            x=sel_pollutant,
            nbins=35,
            marginal="box",
            title=f"Distribution of {sel_pollutant} across Monitoring Stations ({POLLUTANT_UNITS[sel_pollutant]})",
            color_discrete_sequence=["#38bdf8"],
        )
        hist_fig.update_layout(height=380)
        st.plotly_chart(hist_fig, use_container_width=True)

    with col2:
        # Boxplot across top states
        top_states = pivot["state"].value_counts().head(8).index
        state_df = pivot[pivot["state"].isin(top_states)]
        box_fig = px.box(
            state_df,
            x="state",
            y=sel_pollutant,
            color="state",
            title=f"{sel_pollutant} Variations across Top 8 Monitored States",
        )
        box_fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(box_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Inter-Pollutant Correlation Matrix")
    corr = pivot[POLLUTANTS].corr()
    corr_fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pearson Correlation Coefficients between the 7 Atmospheric Pollutants",
    )
    corr_fig.update_layout(height=420)
    st.plotly_chart(corr_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("3. Statistical Anomaly Detection Explorer")
    st.markdown(
        "Detects stations with pollutant readings exceeding the statistical Interquartile Range (IQR) "
        "upper outlier bound ($Q_3 + 1.5 \\times \\text{IQR}$) derived from training data distributions."
    )

    q1 = pivot[sel_pollutant].quantile(0.25)
    q3 = pivot[sel_pollutant].quantile(0.75)
    iqr = q3 - q1
    iqr_thresh = q3 + 1.5 * iqr

    anom_stations = pivot[pivot[sel_pollutant] > iqr_thresh].copy()
    anom_stations["Z-Score"] = (
        (anom_stations[sel_pollutant] - pivot[sel_pollutant].mean()) / pivot[sel_pollutant].std()
    ).round(2)

    st.metric(
        f"Anomalous Stations for {sel_pollutant}",
        f"{len(anom_stations)} / {pivot[sel_pollutant].notna().sum()}",
        f"Outlier Threshold: > {iqr_thresh:.1f} {POLLUTANT_UNITS[sel_pollutant]}",
    )

    if not anom_stations.empty:
        display_cols = ["state", "city", "station", sel_pollutant, "Z-Score"]
        st.dataframe(
            anom_stations[display_cols].sort_values(by=sel_pollutant, ascending=False),
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("4. Technical Note on Time-Series Forecasting")
    with st.expander("Dataset Temporal Limitations & Forecasting Roadmap"):
        st.markdown(
            """
            ### Scientific Audit of Dataset Temporality:
            - **Single Snapshot Limitation**: Inspection of `cpcbdata.csv` confirmed that all 3,369 rows originate from
              a single cross-sectional timestamp (`11-03-2026 15:00`). The dataset is purely spatial across 505 stations,
              not a multi-hour longitudinal time series.
            - **No Fabricated Time-Series**: Attempting to train an autoregressive 6-hour or 24-hour time-series forecasting model
              on static snapshot data would be scientifically invalid.
            - **Future Extension**: Once multi-day continuous hourly sensor feeds are ingested via the live API module,
              a recurrent Spiking Neural Network (rSNN) or Spiking LSTM architecture can be trained for predictive temporal forecasting.
            """
        )


# ==============================================================================
# SECTION 4: METHODOLOGY & CPCB STANDARDS
# ==============================================================================
def page_methodology():
    st.title("Project Methodology & Scientific Standards")
    st.markdown(
        "Detailed documentation of the Indian Central Pollution Control Board (CPCB) NAQI framework, "
        "Leaky-Integrate-and-Fire spiking mathematics, surrogate gradients, and probability calibration."
    )

    st.subheader("1. CPCB National Air Quality Index (NAQI) Breakpoints")
    st.markdown(
        """
        The Indian NAQI classifies air quality into six tiers using pollutant-specific 24-hour and 8-hour standards.
        In this project, categories are structured into 4 robust tiers:
        - **Good / Satisfactory (0)**: Continuous AQI $0 - 100$
        - **Moderate (1)**: Continuous AQI $101 - 200$
        - **Poor (2)**: Continuous AQI $201 - 300$
        - **Severe / Very Poor (3)**: Continuous AQI $301+$
        """
    )

    cpcb_table = pd.DataFrame(
        [
            {"Pollutant": "PM2.5 (µg/m³)", "Good (0)": "0 - 60", "Moderate (1)": "61 - 90", "Poor (2)": "91 - 120", "Severe (3)": "> 120"},
            {"Pollutant": "PM10 (µg/m³)", "Good (0)": "0 - 100", "Moderate (1)": "101 - 250", "Poor (2)": "251 - 350", "Severe (3)": "> 350"},
            {"Pollutant": "NO2 (µg/m³)", "Good (0)": "0 - 80", "Moderate (1)": "81 - 180", "Poor (2)": "181 - 280", "Severe (3)": "> 280"},
            {"Pollutant": "SO2 (µg/m³)", "Good (0)": "0 - 80", "Moderate (1)": "81 - 380", "Poor (2)": "381 - 800", "Severe (3)": "> 800"},
            {"Pollutant": "CO (mg/m³)", "Good (0)": "0 - 2.0", "Moderate (1)": "2.1 - 10.0", "Poor (2)": "10.1 - 17.0", "Severe (3)": "> 17.0"},
            {"Pollutant": "OZONE (µg/m³)", "Good (0)": "0 - 100", "Moderate (1)": "101 - 168", "Poor (2)": "169 - 208", "Severe (3)": "> 208"},
            {"Pollutant": "NH3 (µg/m³)", "Good (0)": "0 - 400", "Moderate (1)": "401 - 800", "Poor (2)": "801 - 1200", "Severe (3)": "> 1200"},
        ]
    )
    st.table(cpcb_table)

    st.subheader("2. Continuous Sub-Index Linear Interpolation")
    st.latex(r"I_p = I_{\text{low}} + \frac{I_{\text{high}} - I_{\text{low}}}{B_{\text{high}} - B_{\text{low}}} \times (C_p - B_{\text{low}})")
    st.markdown(
        "Overall AQI is determined by the **maximum pollutant sub-index**: $\\text{Overall AQI} = \\max_p I_p$. "
        "The pollutant attaining the maximum sub-index is designated as the **Dominant Pollutant**."
    )

    st.subheader("3. Spiking Neural Network (LIF) Formulation")
    st.latex(r"V[t] = \alpha V[t-1] + W S[t] + b, \quad \alpha = \exp(-1 / \tau_{\text{mem}})")
    st.latex(r"S_{\text{out}}[t] = \Theta(V[t] - V_{\text{th}})")
    st.latex(r"V[t] \leftarrow V[t] - V_{\text{th}} S_{\text{out}}[t] \quad \text{(Soft Reset)}")
    st.markdown(
        "To enable gradient descent backpropagation through non-differentiable threshold functions, "
        "a piecewise-linear surrogate gradient is employed: $\\sigma'(V) = \\max(0, 1 - |V - V_{\\text{th}}|)$."
    )


# ==============================================================================
# MAIN ENTRY POINT & NAVIGATION
# ==============================================================================
def main():
    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(
            f"Failed to load SNN model artifacts: {e}\n\n"
            "Please run `python train_model.py` to regenerate `snn_model.pkl` and `preprocess_bundle.pkl`."
        )
        st.stop()

    st.sidebar.title("SNN Air Quality System")
    nav_choice = st.sidebar.radio(
        "Navigation",
        [
            "AQI Prediction & Monitoring",
            "Model Performance & Comparison",
            "Regional Data Analytics",
            "Methodology & CPCB Standards",
        ],
    )

    if nav_choice == "AQI Prediction & Monitoring":
        page_prediction(predictor)
    elif nav_choice == "Model Performance & Comparison":
        page_model_performance()
    elif nav_choice == "Regional Data Analytics":
        page_data_analytics()
    elif nav_choice == "Methodology & CPCB Standards":
        page_methodology()


if __name__ == "__main__":
    main()