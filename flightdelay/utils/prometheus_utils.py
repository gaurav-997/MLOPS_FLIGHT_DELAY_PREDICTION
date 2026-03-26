"""
Centralised Prometheus metrics for the Flight Delay Prediction service.
path - flightdelay/utils/prometheus_utils.py

Import this module wherever metrics need to be updated; each metric is a
module-level singleton so it is registered only once in the default registry.

Usage:
    from flightdelay.utils.prometheus_utils import (
        predictions_total, prediction_latency, delay_class_predictions,
        model_drift_score, rolling_mae, rolling_r2, model_errors, data_quality_score
    )
"""

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Prediction metrics
# ---------------------------------------------------------------------------

predictions_total = Counter(
    "flight_predictions_total",
    "Total number of predictions served",
    ["model_version", "endpoint"],
)

prediction_latency = Histogram(
    "flight_prediction_latency_seconds",
    "End-to-end prediction latency in seconds",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

delay_class_predictions = Counter(
    "flight_delay_predictions",
    "Prediction counts split by delay class",
    ["delay_class"],
)

# ---------------------------------------------------------------------------
# Drift metrics
# ---------------------------------------------------------------------------

model_drift_score = Gauge(
    "flight_model_drift_score",
    "Aggregated data drift score (mean PSI across numerical features)",
)

rolling_mae = Gauge(
    "flight_rolling_mae",
    "Rolling Mean Absolute Error computed from ground-truth feedback",
)

rolling_r2 = Gauge(
    "flight_rolling_r2",
    "Rolling R² score computed from ground-truth feedback",
)

# ---------------------------------------------------------------------------
# Error tracking
# ---------------------------------------------------------------------------

model_errors = Counter(
    "flight_model_errors_total",
    "Total model / inference errors",
    ["error_type"],
)

# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

data_quality_score = Gauge(
    "flight_data_quality_score",
    "Input data quality score (fraction of non-null, in-range values)",
)
