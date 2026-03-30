"""
Model Drift Monitoring Component
---------------------------------
Detects two types of model degradation:

1. Data drift   — PSI (Population Stability Index) across numerical features.
                  A per-feature PSI > 0.2 signals significant drift.
                  Aggregate mean PSI > 0.5 triggers the Prometheus alert.

2. Accuracy drop — rolling R² computed from ground-truth feedback rows.
                   A drop of >0.05 below the baseline R² signals degradation.

Baseline stats are saved to `monitoring/baseline_stats.yaml` the first time
the training pipeline runs (call `save_baseline_stats()` after training).
Subsequently, `ModelMonitor` loads them on construction.
"""

import os
import yaml
import json
import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import r2_score, mean_absolute_error

from flightdelay.logging.logger import logger
from flightdelay.utils.prometheus_utils import (
    model_drift_score,
    rolling_mae,
    rolling_r2,
    data_quality_score,
)

BASELINE_STATS_PATH = os.path.join("monitoring", "baseline_stats.yaml")
FEEDBACK_CSV_PATH = os.path.join("prediction_output", "feedback.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between two continuous distributions.

    PSI < 0.1  → no significant change
    PSI 0.1–0.2 → moderate change
    PSI > 0.2  → significant drift
    """
    # Build common bin edges from the expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # remove duplicates (constant features)
    if len(breakpoints) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    # Convert to proportions, clipping zeros to avoid log(0)
    eps = 1e-8
    expected_pct = np.clip(expected_counts / len(expected), eps, None)
    actual_pct = np.clip(actual_counts / len(actual), eps, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def save_baseline_stats(
    train_df: pd.DataFrame,
    numerical_features: list,
    baseline_r2: float,
    baseline_mae: float,
) -> None:
    """
    Persist baseline distribution statistics after training.
    Called once at the end of a successful training run.
    """
    os.makedirs("monitoring", exist_ok=True)

    stats: Dict = {
        "r2":    baseline_r2,
        "mae":   baseline_mae,
        "features": {},
    }
    for col in numerical_features:
        if col not in train_df.columns:
            continue
        values = train_df[col].dropna().values.tolist()
        stats["features"][col] = {
            "mean":   float(np.mean(values)),
            "std":    float(np.std(values)),
            "min":    float(np.min(values)),
            "max":    float(np.max(values)),
            # Store raw sample (max 5 000 points) for PSI later
            "sample": values[:5000],
        }

    with open(BASELINE_STATS_PATH, "w") as f:
        yaml.dump(stats, f, allow_unicode=True)

    logger.info(f"Baseline stats saved to {BASELINE_STATS_PATH}")


def _load_baseline_stats() -> Optional[Dict]:
    """Load baseline stats from disk; return None if not yet available."""
    if not os.path.exists(BASELINE_STATS_PATH):
        logger.warning(
            f"Baseline stats not found at {BASELINE_STATS_PATH}. "
            "Run the training pipeline first to generate them."
        )
        return None
    with open(BASELINE_STATS_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main monitor class
# ---------------------------------------------------------------------------

class ModelMonitor:
    """
    Drift monitor that exposes two public methods:

    * `calculate_drift_score(production_data)`  → float (mean PSI)
    * `check_accuracy_drop(feedback_data)`      → bool  (True = retrain needed)
    * `assess_data_quality(df)`                 → float (quality score 0–1)
    """

    def __init__(self):
        self.baseline_stats = _load_baseline_stats()
        if self.baseline_stats:
            self.numerical_features = list(self.baseline_stats.get("features", {}).keys())
        else:
            self.numerical_features = []

    # ------------------------------------------------------------------ #
    # Data drift                                                           #
    # ------------------------------------------------------------------ #

    def calculate_drift_score(self, production_data: pd.DataFrame) -> float:
        """
        Compute PSI for each numerical feature and set the Prometheus gauge.

        Returns:
            Mean PSI across all available features (0.0 if no baseline).
        """
        if not self.baseline_stats:
            logger.warning("No baseline stats — skipping drift calculation.")
            return 0.0

        psi_scores: Dict[str, float] = {}

        for feature in self.numerical_features:
            if feature not in production_data.columns:
                logger.debug(f"Feature '{feature}' missing from production data, skipping.")
                continue

            baseline_sample = np.array(
                self.baseline_stats["features"][feature]["sample"], dtype=float
            )
            production_sample = production_data[feature].dropna().values.astype(float)

            if len(production_sample) < 10:
                logger.debug(f"Too few production samples for '{feature}', skipping.")
                continue

            psi = _calculate_psi(baseline_sample, production_sample)
            psi_scores[feature] = psi

            if psi > 0.2:
                logger.warning(f"[DRIFT] Feature '{feature}' PSI={psi:.3f} > 0.2 — significant drift.")
            elif psi > 0.1:
                logger.info(f"[DRIFT] Feature '{feature}' PSI={psi:.3f} — moderate drift.")

        if not psi_scores:
            return 0.0

        aggregate_drift = float(np.mean(list(psi_scores.values())))
        model_drift_score.set(aggregate_drift)

        logger.info(
            f"Drift assessment: mean_PSI={aggregate_drift:.4f} "
            f"| features_checked={len(psi_scores)}"
        )

        # Persist per-feature breakdown for debugging
        report_path = os.path.join("monitoring", "drift_report.json")
        os.makedirs("monitoring", exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(
                {
                    "mean_psi": aggregate_drift,
                    "threshold_alert": 0.5,
                    "features": psi_scores,
                },
                f,
                indent=2,
            )

        return aggregate_drift

    # ------------------------------------------------------------------ #
    # Accuracy drop                                                        #
    # ------------------------------------------------------------------ #

    def check_accuracy_drop(self, feedback_data: pd.DataFrame) -> bool:
        """
        Compute rolling R² and MAE from labeled feedback rows.

        Args:
            feedback_data: DataFrame with columns 'prediction' and 'actual_delay'.

        Returns:
            True if R² has dropped below (baseline_r2 - 0.05), False otherwise.
        """
        if len(feedback_data) < 100:
            logger.info(
                f"Only {len(feedback_data)} feedback rows — need ≥100 to assess accuracy."
            )
            return False

        if not self.baseline_stats:
            logger.warning("No baseline stats — cannot assess accuracy drop.")
            return False

        predictions = feedback_data["prediction"].values
        actuals = feedback_data["actual_delay"].values

        mae = float(mean_absolute_error(actuals, predictions))
        r2 = float(r2_score(actuals, predictions))

        rolling_mae.set(mae)
        rolling_r2.set(r2)

        baseline_r2: float = self.baseline_stats.get("r2", 0.0)
        drop_threshold = baseline_r2 - 0.05

        logger.info(
            f"Accuracy check: rolling_r2={r2:.4f}, baseline_r2={baseline_r2:.4f}, "
            f"threshold={drop_threshold:.4f}, rolling_mae={mae:.2f}"
        )

        if r2 < drop_threshold:
            logger.warning(
                f"[ACCURACY DROP] R²={r2:.4f} < threshold={drop_threshold:.4f} — retraining recommended."
            )
            return True

        return False

    # ------------------------------------------------------------------ #
    # Data quality                                                         #
    # ------------------------------------------------------------------ #

    def assess_data_quality(self, df: pd.DataFrame) -> float:
        """
        Quick data quality score: fraction of non-null values across all columns.

        Also validates numerical features are within [min - 3σ, max + 3σ] baseline range.

        Returns:
            float in [0, 1] — 1.0 means perfect quality.
        """
        if df.empty:
            data_quality_score.set(0.0)
            return 0.0

        # Completeness (non-null fraction)
        completeness = float(df.notna().values.mean())

        # Range validity (only for features we have baseline stats for)
        range_scores = []
        for feature in self.numerical_features:
            if feature not in df.columns or not self.baseline_stats:
                continue
            stats = self.baseline_stats["features"].get(feature)
            if not stats:
                continue
            low = stats["mean"] - 3 * stats["std"]
            high = stats["mean"] + 3 * stats["std"]
            col = df[feature].dropna()
            if len(col) == 0:
                continue
            in_range = float(((col >= low) & (col <= high)).mean())
            range_scores.append(in_range)

        range_validity = float(np.mean(range_scores)) if range_scores else 1.0
        score = 0.5 * completeness + 0.5 * range_validity

        data_quality_score.set(score)
        logger.debug(
            f"Data quality: completeness={completeness:.3f}, "
            f"range_validity={range_validity:.3f}, score={score:.3f}"
        )
        return score

    # ------------------------------------------------------------------ #
    # Convenience: run full check from feedback CSV                        #
    # ------------------------------------------------------------------ #

    def run_full_check(self, production_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Load feedback CSV + run both drift and accuracy checks.

        Args:
            production_data: Optional DataFrame of recent prediction inputs for drift.

        Returns:
            dict with keys: drift_score, accuracy_drop, should_retrain
        """
        result = {"drift_score": 0.0, "accuracy_drop": False, "should_retrain": False}

        # Drift check
        if production_data is not None and not production_data.empty:
            result["drift_score"] = self.calculate_drift_score(production_data)

        # Accuracy check from feedback file
        if os.path.exists(FEEDBACK_CSV_PATH):
            try:
                feedback_df = pd.read_csv(FEEDBACK_CSV_PATH)
                # Feedback CSV columns: request_id, actual_delay, user_feedback, timestamp
                # Predictions CSV may be joined by request_id in future;
                # for now we rely on 'prediction' column if present
                if "prediction" in feedback_df.columns and "actual_delay" in feedback_df.columns:
                    result["accuracy_drop"] = self.check_accuracy_drop(feedback_df)
            except Exception as exc:
                logger.warning(f"Could not read feedback CSV: {exc}")

        result["should_retrain"] = (
            result["drift_score"] > 0.5 or result["accuracy_drop"]
        )

        return result
