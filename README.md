# Spiking Neural Network (SNN) Air Quality Assessment & Analysis System

An end-to-end neuromorphic machine learning system for classifying, evaluating, and explaining Indian ambient air quality using a 3-layer Leaky-Integrate-and-Fire (LIF) Spiking Neural Network trained on Central Pollution Control Board (CPCB) continuous monitoring data.

---

## 1. Project Objective

The primary objective of this project is to develop a biologically plausible, event-driven Spiking Neural Network (SNN) capable of accurately assessing ambient air quality from multi-pollutant sensor telemetry. While conventional deep learning architectures (e.g. standard ANNs, Random Forests) process static floating-point vectors instantaneously, biological nervous systems process information through sparse temporal action potentials (spikes). 

This system bridges neuromorphic computing and environmental monitoring by:
- Converting multi-sensor atmospheric readings into temporal spike trains via rate coding.
- Processing spatio-temporal dynamics using Leaky Integrate-and-Fire (LIF) neurons.
- Providing genuine confidence probabilities via calibrated temperature scaling.
- Comparing neuromorphic performance against traditional machine learning baselines (Logistic Regression, Random Forest, MLP, Gradient Boosting).
- Offering transparent pollutant contribution explainability and statistical anomaly detection.
- Enabling real-time live air-quality ingestion across Indian cities via Open-Meteo and OpenAQ APIs.

---

## 2. Dataset

The model is trained on `cpcbdata.csv`, representing nationwide ambient monitoring data recorded across Central Pollution Control Board (CPCB) Continuous Ambient Air Quality Monitoring Stations (CAAQMS) throughout India:
- **Spatial Coverage**: 505 stations across 260 cities and 30 Indian states/UTs.
- **Station Snapshot**: 499 complete multi-pollutant station profiles after deduplication and pivoting.
- **Measured Parameters**: Minimum, maximum, and average concentrations for 7 criteria pollutants, accompanied by geographic coordinates (latitude and longitude).
- **Temporal Nature**: A spatial cross-sectional nationwide snapshot. (See *Project Limitations* regarding time-series forecasting).

---

## 3. Seven Monitored Criteria Pollutants

The system ingests and processes all seven primary criteria air pollutants monitored under the Indian National Ambient Air Quality Standards (NAAQS):

| Pollutant | Name / Form | Reporting Unit | Health Impact |
|---|---|---|---|
| **PM2.5** | Fine Particulate Matter ($<2.5\,\mu\text{m}$) | $\mu\text{g/m}^3$ | Deep alveolar and cardiovascular penetration, systemic inflammation |
| **PM10** | Inhalable Coarse Particles ($<10\,\mu\text{m}$) | $\mu\text{g/m}^3$ | Upper respiratory tract irritation, exacerbates asthma and bronchitis |
| **NO2** | Nitrogen Dioxide | $\mu\text{g/m}^3$ | Bronchoconstriction, airway inflammation, secondary aerosol precursor |
| **SO2** | Sulfur Dioxide | $\mu\text{g/m}^3$ | Bronchospasms, mucosal irritation, acid rain precursor |
| **CO** | Carbon Monoxide | $\text{mg/m}^3$ | Carboxyhemoglobin formation, hypoxia, neurological dizziness |
| **OZONE** | Ground-Level Tropospheric Ozone | $\mu\text{g/m}^3$ | Pulmonary tissue oxidation, reduced vital lung capacity, chest pain |
| **NH3** | Ammonia | $\mu\text{g/m}^3$ | Eye, nasal, and pharyngeal irritation from agricultural/industrial sources |

*Note on Units*: In raw CAAQMS telemetry, CO is commonly transmitted in decigrams ($0.1\,\text{mg/m}^3$). The preprocessing pipeline normalizes CO to standard $\text{mg/m}^3$ ($\div 10$), aligning training distributions with user inputs and international standards.

---

## 4. 22 Leak-Free Engineered Features

To eliminate target leakage while providing rich physical representations, the model utilizes 22 clean features:

```
Index 00-06: Raw Pollutant Concentrations:
             [CO, NH3, NO2, OZONE, PM10, PM2.5, SO2]
Index 07-08: Geographic Coordinates:
             [latitude, longitude]
Index 09:    Sensor Quality & Data Availability:
             [n_avail] (Count of active non-missing reporting sensors)
Index 10-21: Physical Aerosol & Atmospheric Interaction Features:
             [pm_ratio]            = PM2.5 / (PM10 + ε) (Fine-to-coarse ratio)
             [pm_diff]             = max(0, PM10 - PM2.5) (Coarse aerosol fraction)
             [pm_total]            = PM2.5 + PM10 (Total particulate mass)
             [no2_so2_ratio]       = NO2 / (SO2 + ε) (Mobile vs stationary emission tracer)
             [co_no2_ratio]        = CO / (NO2 + ε) (Incomplete combustion ratio)
             [ozone_no2_ratio]     = OZONE / (NO2 + ε) (Photochemical oxidation indicator)
             [total_oxidants]      = OZONE + NO2 (Total atmospheric oxidant load Ox)
             [sulfur_nitrogen_sum] = SO2 + NO2 (Acid gas precursor sum)
             [acid_gas_ratio]      = (SO2 + NO2) / (NH3 + ε) (Neutralization potential)
             [log_pm25]            = ln(1 + PM2.5) (Log-scale particulate intensity)
             [log_pm10]            = ln(1 + PM10) (Log-scale coarse particulate intensity)
             [log_no2]             = ln(1 + NO2) (Log-scale nitrogen dioxide intensity)
```

---

## 5. Data Leakage Check & Preprocessing Isolation

A comprehensive audit was performed to guarantee strict scientific validity:

1. **Target Derivation**: The target category `aqi_label` is derived from CPCB's piecewise sub-index breakpoints via the worst-case pollutant rule: $y = \max_p S_p \in \{0, 1, 2, 3\}$.
2. **Leakage Elimination**:
   - `max_sc`, which was previously identical to $y$, was **completely removed** from classifier inputs.
   - `ind_good`, `ind_poor`, `ind_severe` (one-hot target indicators) were **completely removed**.
   - `mean_sc` and individual discrete sub-scores (`sc_CO` .. `sc_SO2`) were **completely removed**.
   - The classifier receives **zero** target-derived features and operates purely on raw sensor concentrations and domain interaction ratios.
3. **Train/Test Isolation**:
   - The 80/20 stratified train/test split is executed **before** any preprocessing.
   - `SimpleImputer` (median) and `StandardScaler` are fitted **strictly on `X_train`**; the test set is only transformed.
   - Normalization bounds ($mn, mx$) are computed **strictly from `X_train_sc`**.
   - Class-balanced oversampling is applied **strictly to the training set** (`X_train_snn`), leaving the held-out test set untouched until final inference evaluation.

---

## 6. SNN Architecture

```
Input Features (22 Clean Physical Features) 
       ↓ (Poisson Rate Encoding over T=100 timesteps)
Spike Trains (100, B, 22)
       ↓
LIF Layer 1 (256 Neurons, α=0.9512, V_th=0.5)
       ↓ (Hidden Spikes)
LIF Layer 2 (128 Neurons, α=0.9512, V_th=0.5)
       ↓ (Hidden Spikes)
LIF Layer 3 (4 Output Neurons: Good, Moderate, Poor, Severe)
       ↓ (Spike Rate Averaging + Temperature Scaling β=10.0)
Softmax Firing Probability Distribution
```

- **Hidden Layer 1**: 256 LIF neurons ($W_1 \in \mathbb{R}^{22 \times 256}$, He normal initialization).
- **Hidden Layer 2**: 128 LIF neurons ($W_2 \in \mathbb{R}^{256 \times 128}$).
- **Output Layer**: 4 LIF neurons ($W_3 \in \mathbb{R}^{128 \times 4}$).
- **Simulation Window**: $T = 100$ discrete temporal timesteps.

---

## 7. Leaky Integrate-and-Fire (LIF) Neuron Model

Each neuron follows discrete-time Leaky Integrate-and-Fire dynamics with soft reset:

$$V[t] = \alpha V[t-1] + \sum_{j} W_{ij} S_j[t] + b_i$$

Where:
- $\alpha = \exp(-1 / \tau_{\text{mem}}) \approx 0.9512$ with membrane time constant $\tau_{\text{mem}} = 20.0\,\text{ms}$.
- Spike emission occurs when membrane potential exceeds threshold $V_{\text{th}} = 0.5$:
  $$S[t] = \Theta(V[t] - V_{\text{th}})$$
- **Soft Reset**: Rather than hard-resetting to 0, excess potential is retained to prevent information loss:
  $$V_{\text{after}}[t] = V[t] - V_{\text{th}} S[t]$$

### Surrogate Gradient Backpropagation
Because the step function $\Theta(\cdot)$ has zero derivative almost everywhere, a piecewise-linear surrogate gradient is used during the backward pass:
$$\sigma'(V) = \max(0, 1 - |V - V_{\text{th}}|)$$

Weight and bias updates are executed using the Adam optimizer ($b_1=0.9, b_2=0.999, \epsilon=10^{-8}$) over temporal BPTT.

---

## 8. Rate Encoding & Probability Calibration

- **Rate Encoding**: Each continuous normalized feature $x_k \in [0, 1]$ is translated into a Bernoulli spike train over $T=100$ timesteps:
  $$S_k[t] \sim \text{Bernoulli}(x_k), \quad t \in \{1, \dots, T\}$$
- **Probability Calibration**: Output firing rates $r_k = \frac{1}{T}\sum_{t=1}^T S_k^{\text{out}}[t] \in [0, 1]$ are scaled by calibration parameter $\beta = 10.0$:
  $$P(y = k) = \frac{\exp(\beta r_k)}{\sum_j \exp(\beta r_j)}$$
  This resolves probability saturation where unscaled softmax with rates $\le 1.0$ artificially bounded maximum confidence to $47.54\%$, ensuring realistic probabilities spanning $0\%$ to $100\%$.

---

## 9. CPCB AQI Methodology

The system adheres strictly to the official Indian Central Pollution Control Board (CPCB) National Air Quality Index (NAQI) protocol:

### Breakpoint Structure
Continuous sub-indices ($I_p \in [0, 500]$) are calculated via linear interpolation across CPCB breakpoints:
$$I_p = I_{\text{low}} + \frac{I_{\text{high}} - I_{\text{low}}}{B_{\text{high}} - B_{\text{low}}} \times (C_p - B_{\text{low}})$$

| Category | Continuous AQI Range | Project Class Index | PM2.5 ($\mu\text{g/m}^3$) | PM10 ($\mu\text{g/m}^3$) | NO2 ($\mu\text{g/m}^3$) | CO ($\text{mg/m}^3$) |
|---|---|---|---|---|---|---|
| **Good / Satisfactory** | 0 – 100 | **0** | 0 – 60 | 0 – 100 | 0 – 80 | 0 – 2.0 |
| **Moderate** | 101 – 200 | **1** | 61 – 90 | 101 – 250 | 81 – 180 | 2.1 – 10.0 |
| **Poor** | 201 – 300 | **2** | 91 – 120 | 251 – 350 | 181 – 280 | 10.1 – 17.0 |
| **Severe / Very Poor** | 301+ | **3** | > 120 | > 350 | > 280 | > 17.0 |

### Worst-Case Pollutant Rule
In accordance with CPCB NAQI guidelines, the overall category is governed by the **maximum sub-index** among monitored pollutants:
$$\text{Overall AQI} = \max_{p} I_p, \quad \text{Category} = \max_{p} S_p$$

---

## 10. Model Evaluation & Benchmark Comparison (Leak-Free)

All models were evaluated on the **exact same held-out 100-sample test split (20%)** stratified across all 4 classes:

| Model Architecture | Model Paradigm | Test Accuracy | Macro Precision | Macro Recall | Macro F1 Score |
|---|---|---|---|---|---|
| **SNN (3-Layer LIF)** | Neuromorphic Spiking | **77.00%** | **69.15%** | **66.78%** | **67.29%** |
| **Logistic Regression** | Linear / Generalized Linear | 82.00% | 73.14% | 68.89% | 70.48% |
| **Random Forest (100 Trees)** | Ensemble Bagging | 93.00% | 96.43% | 78.57% | 82.26% |
| **MLP (256-128 ANN)** | Deep Continuous Artificial | 81.00% | 69.42% | 64.21% | 65.85% |
| **Gradient Boosting** | Ensemble Boosting | 94.00% | 89.81% | 84.52% | 86.03% |

### SNN Confusion Matrix (Held-Out Test Set)
```
                  Predicted
               Good  Mod  Poor  Sev
Actual  Good [   3    4     0    0 ]
        Mod  [   2   35     5    0 ]
        Poor [   0    3     8    3 ]
        Sev  [   0    0     6   31 ]
```

### Technical Commentary on Model Comparison
- **Genuine Benchmarks**: Eliminating target leakage confirms that air quality classification from raw atmospheric concentrations is a non-trivial classification problem.
- **Tree Ensembles**: Random Forest ($93.00\%$) and Gradient Boosting ($94.00\%$) perform well because orthogonal decision trees naturally replicate piecewise rectangular CPCB concentration threshold boundaries.
- **Neuromorphic Spiking Value**: The SNN achieves competitive performance ($77.00\%$ Accuracy, $67.29\%$ Macro F1), surpassing the continuous deep MLP on Macro F1 ($67.29\%$ vs $65.85\%$). Crucially, error distribution shows misclassifications occur exclusively between adjacent ordinal air quality classes (Good $\leftrightarrow$ Moderate, Moderate $\leftrightarrow$ Poor, Poor $\leftrightarrow$ Severe). This validates event-driven neuromorphic processing for real-time edge micro-controllers.

---

## 11. Pollutant Contribution & Explainability

Rather than fabricating black-box saliency maps, the system provides transparent, scientifically defensible explainability derived directly from CPCB sub-index calculations:
- **Dominant Pollutant**: Designated as the pollutant achieving the highest continuous sub-index ($\arg\max_p I_p$).
- **Contribution Breakdown**: Ranks each pollutant by relative percentage of total sub-index load.
- **Narrative Diagnostic**:
  > *"The predicted air quality category is 'Moderate'. The dominant contributor is PM2.5 with a sub-index of 150.0. Primary drivers of elevated pollution: PM2.5 (sub-index 150.0), PM10 (sub-index 113.3)."*

---

## 12. Statistical Anomaly Detection

To detect sensor malfunctions, industrial flaring, or extreme pollution episodes, the system implements distribution-based anomaly detection using training data bounds:
- **Method 1: Interquartile Range (IQR)**:
  Flags readings exceeding the upper outlier fence $Q_3 + 1.5 \times \text{IQR}$. Extreme outliers are flagged at $Q_3 + 3.0 \times \text{IQR}$.
- **Method 2: Z-Score**:
  Computes standardized deviation from the training mean: $Z = (x - \mu) / \sigma$.
- **UI Alerts**: If a user enters or live-fetches an abnormal concentration (e.g., $\text{PM2.5} = 310\,\mu\text{g/m}^3$), an alert banner highlights:
  > *⚠️ STATISTICAL ANOMALY DETECTED: PM2.5 (310.0 µg/m³) exceeds the training IQR outlier bound (296.8 µg/m³, Z-score: +2.6σ).*

---

## 13. Optional Live AQI Monitoring Integration

The application includes optional real-time atmospheric data ingestion:
1. **Primary Keyless Provider (Open-Meteo)**: Zero-configuration live feeds for any coordinates in India, ingesting real-time PM2.5, PM10, CO, NO2, SO2, and Ozone.
2. **Authenticated Provider (OpenAQ v3)**: Supports official CAAQMS station telemetry when an API key is provided via `.env` (`OPENAQ_API_KEY`) or Streamlit secrets.
3. **Honest Missing-Value Handling**: Open-Meteo does not measure ammonia (NH3). NH3 is imputed using the training set median ($5.0\,\mu\text{g/m}^3$) and clearly badged in the UI as *Imputed via Training Median*.

---

## 14. Project Limitations

1. **Dataset Temporality**: `cpcbdata.csv` represents a single spatial snapshot taken at `11-03-2026 15:00`. Because longitudinal multi-hour or multi-day time-series readings are not present, time-series forecasting cannot be trained legitimately and is documented as a future enhancement once multi-day sensor feeds are accumulated.

---

## 15. Installation & Execution

### Prerequisites
- Python 3.10 or 3.11

### Setup
```bash
# 1. Clone or navigate to the project directory
cd inference

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Provide OpenAQ API key in .env
# OPENAQ_API_KEY=your_key_here

# 5. Run the application
streamlit run app.py
```

Open your browser at `http://localhost:8501`.
