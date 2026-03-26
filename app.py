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
import sys
import time
import threading
import io

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from flightdelay.logging.logger import logger
from flightdelay.utils.main_utils import load_object
from flightdelay.utils.ml_utils.model.estimator import FlightDelayModel

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
# Prometheus metrics
# ---------------------------------------------------------------------------

predictions_total = Counter(
    "flight_predictions_total",
    "Total number of predictions served",
    ["model_version"],
)
prediction_latency = Histogram(
    "flight_prediction_latency_seconds",
    "End-to-end prediction latency in seconds",
)
delay_predictions = Counter(
    "flight_delay_predictions",
    "Prediction counts split by delay class",
    ["delay_class"],
)
model_errors = Counter(
    "flight_model_errors_total",
    "Errors during prediction",
    ["error_type"],
)

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


@app.on_event("startup")
def startup_event():
    global network_model
    network_model = _load_model()


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
# Background retraining helper
# ---------------------------------------------------------------------------

_retrain_lock = threading.Lock()


def _trigger_retraining(reason: str):
    """Run training pipeline in a background thread (non-blocking)."""
    if not _retrain_lock.acquire(blocking=False):
        logger.info("Retraining already in progress — skipping duplicate trigger.")
        return
    try:
        logger.info(f"Background retraining started. Reason: {reason}")
        from training_pipeline import TrainingPipeline
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()

        # Reload production model after successful retrain
        global network_model
        network_model = _load_model()
        logger.info("Background retraining completed; model reloaded.")
    except Exception as exc:
        logger.error(f"Background retraining failed: {exc}")
    finally:
        _retrain_lock.release()


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
        predictions_total.labels(model_version="v1.0").inc(len(df))
        prediction_latency.observe(elapsed)
        delay_predictions.labels(delay_class="delayed").inc(int((df["DELAY_CLASS"] == 1).sum()))
        delay_predictions.labels(delay_class="ontime").inc(int((df["DELAY_CLASS"] == 0).sum()))

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


@app.post("/feedback", tags=["monitoring"])
async def submit_feedback(feedback: FeedbackRequest):
    """
    Accept ground-truth feedback for a previous prediction.
    Stores to prediction_output/feedback.csv for future drift analysis.
    """
    try:
        row = {
            "request_id": feedback.request_id,
            "actual_delay": feedback.actual_delay,
            "user_feedback": feedback.user_feedback,
            "timestamp": time.time(),
        }
        feedback_path = os.path.join("prediction_output", "feedback.csv")
        os.makedirs("prediction_output", exist_ok=True)

        row_df = pd.DataFrame([row])
        if os.path.exists(feedback_path):
            row_df.to_csv(feedback_path, mode="a", header=False, index=False)
        else:
            row_df.to_csv(feedback_path, index=False)

        # Simple drift heuristic: if >100 feedback rows accumulated, flag for retraining
        feedback_df = pd.read_csv(feedback_path)
        should_retrain = len(feedback_df) >= 100

        return {"status": "success", "should_retrain": should_retrain, "feedback_count": len(feedback_df)}

    except Exception as exc:
        logger.error(f"/feedback failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/webhook/retrain", tags=["monitoring"])
async def retrain_webhook(alert: AlertWebhook):
    """
    Webhook endpoint — triggered by Grafana / Prometheus alertmanager.
    Spawns a background retraining job.
    """
    logger.info(f"Retrain webhook received: alertname={alert.alertname}, reason={alert.reason}")
    thread = threading.Thread(
        target=_trigger_retraining,
        args=(alert.reason,),
        daemon=True,
    )
    thread.start()
    return {"status": "retraining_initiated", "reason": alert.reason, "alertname": alert.alertname}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
