"""
Feedback Collector — stores predictions and ground-truth labels in SQLite.

Workflow:
  1. At prediction time, call `store_prediction(request_id, features, prediction, model_version)`
  2. When the actual delay is known, call `update_ground_truth(request_id, actual_delay)`
  3. Periodically call `should_trigger_retraining()` to check if MAE has degraded.
  4. Call `get_labeled_data()` to retrieve all rows with ground truth for retraining.

DB path defaults to `feedback_data/feedback.db` — created automatically.
"""

import os
import json
import sqlite3
import threading

import pandas as pd
from sklearn.metrics import mean_absolute_error

from flightdelay.logging.logger import logger

DEFAULT_DB_PATH = os.path.join("feedback_data", "feedback.db")
BASELINE_MAE = 10.0  # minutes — update after first training run
RETRAIN_MAE_MULTIPLIER = 1.20  # trigger if rolling MAE > baseline * 1.20
MIN_LABELED_ROWS = 100  # minimum labeled rows before assessing


class FeedbackCollector:
    """
    Thread-safe SQLite-backed store for prediction feedback.
    """

    _lock = threading.Lock()

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._create_table()
        logger.info(f"FeedbackCollector initialised. DB: {db_path}")

    # ------------------------------------------------------------------ #
    # Schema                                                               #
    # ------------------------------------------------------------------ #

    def _create_table(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    request_id      TEXT PRIMARY KEY,
                    features        TEXT,
                    prediction      REAL,
                    actual_delay    REAL,
                    model_version   TEXT,
                    timestamp       REAL,
                    user_feedback   TEXT
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    # ------------------------------------------------------------------ #
    # Write operations                                                     #
    # ------------------------------------------------------------------ #

    def store_prediction(
        self,
        request_id: str,
        features: dict,
        prediction: float,
        model_version: str = "v1.0",
    ) -> None:
        """
        Persist a single prediction.

        Args:
            request_id:     Unique ID for this prediction request.
            features:       Dict of input feature values.
            prediction:     Predicted arrival delay in minutes.
            model_version:  String tag for the model that produced the prediction.
        """
        import time
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO feedback
                  (request_id, features, prediction, model_version, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    json.dumps(features),
                    float(prediction),
                    model_version,
                    time.time(),
                ),
            )
        logger.debug(f"Stored prediction for request_id={request_id}")

    def update_ground_truth(
        self,
        request_id: str,
        actual_delay: float,
        user_feedback: str = "",
    ) -> None:
        """
        Record the actual arrival delay for a previously stored prediction.

        Args:
            request_id:    Must match a previously stored request_id.
            actual_delay:  Observed arrival delay in minutes.
            user_feedback: Optional free-text from a user (e.g. "correct").
        """
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE feedback
                   SET actual_delay = ?,
                       user_feedback = ?
                 WHERE request_id = ?
                """,
                (float(actual_delay), user_feedback, request_id),
            )
        logger.debug(f"Updated ground truth for request_id={request_id}")

    # ------------------------------------------------------------------ #
    # Read operations                                                      #
    # ------------------------------------------------------------------ #

    def get_labeled_data(self) -> pd.DataFrame:
        """Return all feedback rows that have an actual_delay recorded."""
        with self._connect() as conn:
            df = pd.read_sql(
                "SELECT * FROM feedback WHERE actual_delay IS NOT NULL ORDER BY timestamp",
                conn,
            )
        df["features"] = df["features"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else {}
        )
        return df

    def get_all_predictions(self) -> pd.DataFrame:
        """Return every stored row (labeled and unlabeled)."""
        with self._connect() as conn:
            return pd.read_sql("SELECT * FROM feedback ORDER BY timestamp", conn)

    def label_coverage(self) -> dict:
        """Return counts of total vs labeled predictions."""
        with self._connect() as conn:
            total   = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            labeled = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE actual_delay IS NOT NULL"
            ).fetchone()[0]
        return {"total": total, "labeled": labeled, "coverage_pct": round(100 * labeled / max(total, 1), 1)}

    # ------------------------------------------------------------------ #
    # Retraining heuristic                                                 #
    # ------------------------------------------------------------------ #

    def should_trigger_retraining(self, baseline_mae: float = BASELINE_MAE) -> bool:
        """
        Returns True if rolling MAE on labeled data exceeds baseline * 1.20.

        Only evaluated when ≥ MIN_LABELED_ROWS labeled rows are available.
        """
        labeled = self.get_labeled_data()
        n = len(labeled)

        if n < MIN_LABELED_ROWS:
            logger.info(
                f"FeedbackCollector: only {n} labeled rows (need {MIN_LABELED_ROWS}) — skipping retraining check."
            )
            return False

        mae = float(mean_absolute_error(labeled["actual_delay"], labeled["prediction"]))
        threshold = baseline_mae * RETRAIN_MAE_MULTIPLIER

        logger.info(
            f"FeedbackCollector: rolling_MAE={mae:.2f}, baseline={baseline_mae:.2f}, "
            f"threshold={threshold:.2f}, labeled_rows={n}"
        )

        if mae > threshold:
            logger.warning(
                f"FeedbackCollector: MAE {mae:.2f} > threshold {threshold:.2f} — retraining recommended."
            )
            return True

        return False

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

    def export_to_csv(self, path: str = "feedback_data/feedback_export.csv") -> str:
        """Dump the full feedback table to CSV and return the path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = self.get_all_predictions()
        df.to_csv(path, index=False)
        logger.info(f"Feedback exported to {path} ({len(df)} rows)")
        return path
