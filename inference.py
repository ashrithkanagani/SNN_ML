"""
Inference utilities for the SNN Air Quality classifier.

Single source of truth for turning raw pollutant readings into the exact
22 leak-free features the model was trained on:
Raw concentrations + Coordinates + Data availability + Aerosol/Atmospheric ratios ->
Median-Imputation -> Standard-Scaling -> Min-Max Normalization to [0, 1].

Includes:
- SNN inference with Monte Carlo spike averaging
- Continuous CPCB AQI sub-index calculations (0-500) for post-prediction explainability
- Dominant pollutant and relative impact breakdown
- Statistical anomaly detection (IQR & Z-score)
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from train_model import (  # noqa: F401
    CLEAN_FEATURE_ORDER,
    CPCB_4TIER,
    CPCB_CONTINUOUS_BREAKS,
    LABEL_NAMES,
    POLLUTANTS,
    LIFLayer,
    SNN,
    calculate_cpcb_sub_index,
    pollutant_score,
)

# snn_model.pkl was created by running train_model.py directly, so pickle
# recorded its classes under the "__main__" module. Alias them here so
# `pickle.load` can resolve SNN/LIFLayer no matter which module is the
# real __main__ (Streamlit, a script, a notebook, etc.).
sys.modules["__main__"].SNN = SNN
sys.modules["__main__"].LIFLayer = LIFLayer

MODEL_PATH = Path(__file__).parent / "snn_model.pkl"
BUNDLE_PATH = Path(__file__).parent / "preprocess_bundle.pkl"

POLLUTANT_UNITS = {
    "CO": "mg/m³",
    "NH3": "µg/m³",
    "NO2": "µg/m³",
    "OZONE": "µg/m³",
    "PM10": "µg/m³",
    "PM2.5": "µg/m³",
    "SO2": "µg/m³",
}


@dataclass
class AnomalyReport:
    pollutant: str
    current_value: float
    unit: str
    iqr_threshold: float
    z_score: float
    severity: str  # "High" or "Extreme"
    description: str


@dataclass
class PredictionResult:
    label: str
    label_idx: int
    probabilities: dict[str, float]
    pollutant_scores: dict[str, float]  # discrete category 0..3
    pollutant_sub_indices: dict[str, float]  # continuous CPCB AQI 0..500
    overall_aqi: float  # continuous max sub-index
    dominant_pollutant: str
    contributions: list[dict[str, any]]
    explanation: str
    anomalies: list[AnomalyReport]


class AQIPredictor:
    """Loads the trained SNN plus its fitted preprocessing pipeline once,
    and exposes a single `predict` entry point for raw sensor inputs."""

    def __init__(self, model_path: Path = MODEL_PATH, bundle_path: Path = BUNDLE_PATH):
        with open(model_path, "rb") as f:
            self.model: SNN = pickle.load(f)

        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)
        self.imputer = bundle["imputer"]
        self.scaler = bundle["scaler"]
        self.mn = bundle["mn"]
        self.mx = bundle["mx"]
        self.rng = bundle["rng"]
        self.feat_cols = bundle["feat_cols"]
        self.anomaly_stats = bundle.get("anomaly_stats", {})

        if list(self.feat_cols) != list(CLEAN_FEATURE_ORDER):
            raise ValueError(
                "Preprocessing feature order does not match the clean feature order: "
                f"expected {CLEAN_FEATURE_ORDER}, got {self.feat_cols}"
            )

    def _engineer_features(self, pollutant_values: dict[str, float], lat: float, lon: float) -> np.ndarray:
        co = float(pollutant_values["CO"])
        nh3 = float(pollutant_values["NH3"])
        no2 = float(pollutant_values["NO2"])
        ozone = float(pollutant_values["OZONE"])
        pm10 = float(pollutant_values["PM10"])
        pm25 = float(pollutant_values["PM2.5"])
        so2 = float(pollutant_values["SO2"])

        row = [
            co,
            nh3,
            no2,
            ozone,
            pm10,
            pm25,
            so2,
            float(lat),
            float(lon),
            7.0,  # n_avail
            pm25 / (pm10 + 1e-4),  # pm_ratio
            max(0.0, pm10 - pm25),  # pm_diff
            pm25 + pm10,  # pm_total
            no2 / (so2 + 1e-4),  # no2_so2_ratio
            co / (no2 + 1e-4),  # co_no2_ratio
            ozone / (no2 + 1e-4),  # ozone_no2_ratio
            ozone + no2,  # total_oxidants
            so2 + no2,  # sulfur_nitrogen_sum
            (so2 + no2) / (nh3 + 1e-4),  # acid_gas_ratio
            float(np.log1p(max(0.0, pm25))),  # log_pm25
            float(np.log1p(max(0.0, pm10))),  # log_pm10
            float(np.log1p(max(0.0, no2))),  # log_no2
        ]
        assert len(row) == len(self.feat_cols), (
            f"feature count mismatch: built {len(row)}, model expects {len(self.feat_cols)}"
        )
        return np.array(row, dtype=float).reshape(1, -1)

    def _preprocess(self, X_raw: np.ndarray) -> np.ndarray:
        """Apply the exact fitted imputer -> scaler -> min-max clip used at train time."""
        X = self.scaler.transform(self.imputer.transform(X_raw))
        X = np.clip((X - self.mn) / self.rng, 0, 1).astype(np.float32)
        return X

    def detect_anomalies(self, pollutant_values: dict[str, float]) -> list[AnomalyReport]:
        """Detects pollutant readings that are unusually high compared with the training data distribution."""
        anomalies = []
        for p in POLLUTANTS:
            if p not in pollutant_values or p not in self.anomaly_stats:
                continue
            val = float(pollutant_values[p])
            stat = self.anomaly_stats[p]
            iqr_upper = stat["iqr_upper"]
            iqr_extreme = stat["iqr_extreme"]
            mean = stat["mean"]
            std = stat["std"] if stat["std"] > 1e-6 else 1.0
            z_score = (val - mean) / std

            if val > iqr_upper:
                severity = "Extreme" if val > iqr_extreme else "High"
                desc = (
                    f"{p} ({val:.1f} {POLLUTANT_UNITS[p]}) exceeds the training IQR outlier bound "
                    f"({iqr_upper:.1f} {POLLUTANT_UNITS[p]}, Z-score: +{z_score:.1f}σ)."
                )
                anomalies.append(
                    AnomalyReport(
                        pollutant=p,
                        current_value=val,
                        unit=POLLUTANT_UNITS[p],
                        iqr_threshold=iqr_upper,
                        z_score=round(z_score, 2),
                        severity=severity,
                        description=desc,
                    )
                )
        return anomalies

    def predict(
        self,
        pollutant_values: dict[str, float],
        lat: float = 23.2178,
        lon: float = 78.5893,
        mc_runs: int = 20,
    ) -> PredictionResult:
        """Predict the AQI category for one reading.

        `pollutant_values` must have a float entry for every name in POLLUTANTS.
        `mc_runs` controls how many stochastic spike-train samples are averaged.
        """
        missing = [p for p in POLLUTANTS if p not in pollutant_values]
        if missing:
            raise ValueError(f"Missing pollutant readings: {missing}")
        negative = {p: pollutant_values[p] for p in POLLUTANTS if pollutant_values[p] < 0}
        if negative:
            raise ValueError(f"Pollutant readings cannot be negative: {negative}")

        X_raw = self._engineer_features(pollutant_values, lat, lon)
        X = self._preprocess(X_raw)

        probs = self.model.predict_proba(X, mc_runs=mc_runs)
        pred_idx = int(probs.argmax(1)[0])

        # Compute continuous and discrete sub-indices according to official CPCB methodology
        discrete_scores = {p: pollutant_score(pollutant_values[p], p) for p in POLLUTANTS}
        continuous_sub_indices = {p: calculate_cpcb_sub_index(pollutant_values[p], p) for p in POLLUTANTS}
        overall_aqi = max(continuous_sub_indices.values())

        # Dominant pollutant: pollutant with highest continuous sub-index
        sorted_pollutants = sorted(continuous_sub_indices.items(), key=lambda item: item[1], reverse=True)
        dominant_pollutant, dominant_sub_idx = sorted_pollutants[0]

        total_sub = sum(continuous_sub_indices.values()) or 1.0
        contributions = [
            {
                "pollutant": p,
                "sub_index": round(sub_val, 1),
                "concentration": pollutant_values[p],
                "unit": POLLUTANT_UNITS[p],
                "category": LABEL_NAMES[int(discrete_scores[p])],
                "percentage": round((sub_val / total_sub) * 100, 1),
            }
            for p, sub_val in sorted_pollutants
        ]

        # Natural language explanation of contribution
        top_two = [f"{item['pollutant']} (sub-index {item['sub_index']})" for item in contributions[:2] if item['sub_index'] > 50]
        if top_two:
            explanation = (
                f"The predicted air quality category is '{LABEL_NAMES[pred_idx]}'. "
                f"The dominant contributor is {dominant_pollutant} with a sub-index of {dominant_sub_idx:.1f}. "
                f"Primary drivers of elevated pollution: {', '.join(top_two)}."
            )
        else:
            explanation = (
                f"The predicted air quality category is '{LABEL_NAMES[pred_idx]}'. "
                "All monitored pollutant concentrations remain within acceptable ambient standards."
            )

        anomalies = self.detect_anomalies(pollutant_values)

        return PredictionResult(
            label=LABEL_NAMES[pred_idx],
            label_idx=pred_idx,
            probabilities={LABEL_NAMES[i]: float(probs[0, i]) for i in range(len(LABEL_NAMES))},
            pollutant_scores=discrete_scores,
            pollutant_sub_indices=continuous_sub_indices,
            overall_aqi=round(overall_aqi, 1),
            dominant_pollutant=dominant_pollutant,
            contributions=contributions,
            explanation=explanation,
            anomalies=anomalies,
        )
