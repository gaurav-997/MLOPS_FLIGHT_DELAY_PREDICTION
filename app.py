"""
FastAPI Prediction Service — Flight Delay Prediction API

Endpoints:
  GET  /           → redirect to /docs
  GET  /health     → liveness check
  GET  /train      → trigger training pipeline
  POST /predict    → CSV upload → predictions HTML table
  GET  /metrics    → Prometheus metrics scrape
  POST /feedback   → submit ground-truth for drift monitoring
  POST /webhook/retrain → Grafana alert → background retrain

Run:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import time
import io

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from flightdelay.logging.logger import logger
from flightdelay.utils.main_utils import load_object
from flightdelay.utils.ml_utils.model.estimator import FlightDelayModel
from flightdelay.utils.prometheus_utils import (
    predictions_total,
    prediction_latency,
    delay_class_predictions,
    model_drift_score,
    rolling_mae as prom_rolling_mae,
    rolling_r2 as prom_rolling_r2,
    model_errors,
    data_quality_score,
)
from flightdelay.components.modelmonitoring import ModelMonitor
from flightdelay.pipeline.retraining_manager import RetrainingManager
from flightdelay.components.feedback_collector import FeedbackCollector

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Flight Delay Prediction API",
    description="MLOps Flight Delay Prediction — upload a CSV to get predictions",
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------------------------
# Metrics imported from prometheus_utils (see flightdelay/utils/main_utils/)
# ---------------------------------------------------------------------------
# Model — loaded once at startup
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join("final_model", "model.pkl")
PREPROCESSOR_PATH = os.path.join("final_model", "preprocessor.pkl")

network_model: FlightDelayModel | None = None


def _load_model() -> FlightDelayModel | None:
    """Attempt to load production model; return None if not yet available."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        logger.warning(
            "Production model not found at final_model/. "
            "Run the training pipeline first (GET /train)."
        )
        return None
    try:
        preprocessor = load_object(PREPROCESSOR_PATH)
        model = load_object(MODEL_PATH)
        logger.info("Production model loaded successfully.")
        return FlightDelayModel(preprocessor, model)
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        return None


monitor: ModelMonitor | None = None
retrain_manager: RetrainingManager | None = None
feedback_collector: FeedbackCollector | None = None


@app.on_event("startup")
def startup_event():
    global network_model, monitor, retrain_manager, feedback_collector
    network_model = _load_model()
    monitor = ModelMonitor()
    retrain_manager = RetrainingManager()
    feedback_collector = FeedbackCollector()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    request_id: str
    actual_delay: float
    user_feedback: str = ""


class AlertWebhook(BaseModel):
    alertname: str = "unknown"
    reason: str = "drift_detected"
    severity: str = "warning"





# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["ops"])
async def health():
    return {
        "status": "healthy",
        "model_loaded": network_model is not None,
        "model_path": MODEL_PATH,
    }


@app.get("/train", tags=["ops"])
async def train_model():
    """Synchronously run the full training pipeline and reload the model."""
    try:
        logger.info("Training triggered via /train endpoint.")
        from training_pipeline import TrainingPipeline
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()

        global network_model
        network_model = _load_model()

        return {"message": "Training completed successfully", "model_loaded": network_model is not None}
    except Exception as exc:
        logger.error(f"/train failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict", tags=["prediction"])
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Upload a CSV file with flight features.
    Returns an HTML table of predictions.
    """
    if network_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run GET /train to train a model first.",
        )

    start_time = time.time()

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        predictions = network_model.predict(df)
        df["PREDICTED_DELAY"] = predictions
        df["DELAY_CLASS"] = (predictions > 15).astype(int)

        elapsed = time.time() - start_time
        predictions_total.labels(model_version="v1.0", endpoint="/predict").inc(len(df))
        prediction_latency.observe(elapsed)
        delay_class_predictions.labels(delay_class="delayed").inc(int((df["DELAY_CLASS"] == 1).sum()))
        delay_class_predictions.labels(delay_class="ontime").inc(int((df["DELAY_CLASS"] == 0).sum()))

        # Assess input data quality and update gauge
        if monitor:
            monitor.assess_data_quality(df)

        # Persist predictions
        os.makedirs("prediction_output", exist_ok=True)
        output_path = os.path.join("prediction_output", "predictions.csv")
        df.to_csv(output_path, index=False)

        return templates.TemplateResponse(
            "table.html",
            {
                "request": request,
                "data": df.head(100).to_dict("records"),
                "columns": df.columns.tolist(),
                "total_flights": len(df),
                "delayed_count": int((df["DELAY_CLASS"] == 1).sum()),
                "ontime_count": int((df["DELAY_CLASS"] == 0).sum()),
                "latency_ms": round(elapsed * 1000, 1),
            },
        )

    except Exception as exc:
        model_errors.labels(error_type="prediction_error").inc()
        logger.error(f"/predict failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics", tags=["ops"])
async def metrics():
    """Prometheus metrics scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/drift", tags=["monitoring"])
async def check_drift(file: UploadFile = File(...)):
    """
    Upload a CSV of recent production inputs.
    Runs PSI drift detection and returns per-feature scores + aggregate.
    """
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not initialised.")
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        drift_score = monitor.calculate_drift_score(df)
        return {
            "drift_score": drift_score,
            "threshold": 0.5,
            "alert": drift_score > 0.5,
        }
    except Exception as exc:
        logger.error(f"/drift failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/feedback", tags=["monitoring"])
async def submit_feedback(feedback: FeedbackRequest):
    """
    Accept ground-truth feedback for a previous prediction.
    Persisted to SQLite via FeedbackCollector; also checks accuracy-drop trigger.
    """
    try:
        # Store in SQLite via FeedbackCollector
        if feedback_collector:
            feedback_collector.update_ground_truth(
                request_id=feedback.request_id,
                actual_delay=feedback.actual_delay,
                user_feedback=feedback.user_feedback,
            )
            coverage = feedback_collector.label_coverage()
            should_retrain = feedback_collector.should_trigger_retraining()
            feedback_count = coverage["labeled"]
        else:
            # Fallback: simple CSV append
            row = {
                "request_id": feedback.request_id,
                "actual_delay": feedback.actual_delay,
                "user_feedback": feedback.user_feedback,
                "timestamp": time.time(),
            }
            feedback_path = os.path.join("prediction_output", "feedback.csv")
            os.makedirs("prediction_output", exist_ok=True)
            pd.DataFrame([row]).to_csv(
                feedback_path, mode="a",
                header=not os.path.exists(feedback_path),
                index=False,
            )
            should_retrain = False
            feedback_count = 0

        # Auto-trigger retraining in background if threshold exceeded
        if should_retrain and retrain_manager:
            retrain_manager.trigger_retraining_async(reason="accuracy_drop")

        return {
            "status": "success",
            "should_retrain": should_retrain,
            "feedback_count": feedback_count,
        }

    except Exception as exc:
        logger.error(f"/feedback failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/webhook/retrain", tags=["monitoring"])
async def retrain_webhook(alert: AlertWebhook):
    """
    Webhook endpoint — triggered by Grafana / Prometheus alertmanager.
    Spawns a background retraining job via RetrainingManager.
    """
    logger.info(f"Retrain webhook received: alertname={alert.alertname}, reason={alert.reason}")
    if retrain_manager:
        retrain_manager.trigger_retraining_async(reason=alert.reason)
    return {"status": "retraining_initiated", "reason": alert.reason, "alertname": alert.alertname}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
