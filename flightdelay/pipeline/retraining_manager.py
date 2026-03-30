"""
Retraining Manager — orchestrates the full retrain → evaluate → promote cycle.

Triggers:
  1. Scheduled (CronJob / GitHub Actions) → reason="scheduled"
  2. Drift alert (Grafana webhook via /webhook/retrain) → reason="drift_detected"
  3. Feedback-based (FeedbackCollector.should_trigger_retraining()) → reason="accuracy_drop"
  4. Manual (workflow_dispatch / POST /webhook/retrain) → reason="manual"

Usage:
    from flightdelay.pipeline.retraining_manager import RetrainingManager
    manager = RetrainingManager()
    manager.trigger_retraining(reason="drift_detected")
"""

import os
import subprocess
import threading

from flightdelay.logging.logger import logger

_retrain_lock = threading.Lock()


class RetrainingManager:
    """
    Coordinates data versioning (DVC), pipeline execution, and post-train
    artefact promotion.
    """

    def __init__(self):
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def trigger_retraining(self, reason: str = "manual") -> bool:
        """
        Run the full retraining cycle synchronously.

        Steps:
          1. Pull latest data from DVC remote
          2. Run training pipeline (data ingestion → model evaluation)
          3. Version new model artefacts with DVC
          4. Push artefacts to DVC remote
          5. Log completion

        Args:
            reason: Human-readable trigger reason (logged + committed).

        Returns:
            True if retraining succeeded, False otherwise.
        """
        if not _retrain_lock.acquire(blocking=False):
            logger.warning("RetrainingManager: retraining already in progress — skipping.")
            return False

        try:
            logger.info(f"{'='*60}")
            logger.info(f" RETRAINING STARTED  |  reason: {reason}")
            logger.info(f"{'='*60}")

            # 1. Pull latest data
            self._dvc_pull()

            # 2. Merge any new labeled feedback into the training set (optional)
            self._merge_feedback_data()

            # 3. Run the full training pipeline
            self._run_training_pipeline()

            # 4. Version updated model with DVC
            self._dvc_add_model()

            # 5. Push updated artefacts
            self._dvc_push()

            logger.info(f"{'='*60}")
            logger.info(f" RETRAINING COMPLETED  |  reason: {reason}")
            logger.info(f"{'='*60}")

            return True

        except Exception as exc:
            logger.error(f"RetrainingManager: retraining failed — {exc}")
            return False

        finally:
            _retrain_lock.release()

    # ------------------------------------------------------------------ #
    # Async variant (fire-and-forget from FastAPI webhook)                 #
    # ------------------------------------------------------------------ #

    def trigger_retraining_async(self, reason: str = "drift_detected") -> threading.Thread:
        """Launch retraining in a daemon thread and return the thread object."""
        thread = threading.Thread(
            target=self.trigger_retraining,
            args=(reason,),
            daemon=True,
            name=f"retrain-{reason}",
        )
        thread.start()
        logger.info(f"RetrainingManager: async retraining started (thread={thread.name})")
        return thread

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _run_cmd(self, cmd: list, description: str = "") -> None:
        """Run a shell command; raise on non-zero exit code."""
        logger.info(f"  $ {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        if result.returncode != 0:
            logger.error(f"Command failed ({description}):\n{result.stderr}")
            raise RuntimeError(f"{description} failed: {result.stderr[:500]}")
        if result.stdout:
            logger.debug(result.stdout[:2000])

    def _dvc_pull(self) -> None:
        logger.info("[1/4] Pulling data from DVC remote ...")
        try:
            self._run_cmd(["dvc", "pull"], "dvc pull")
        except RuntimeError as exc:
            logger.warning(f"dvc pull failed (non-fatal, continuing): {exc}")

    def _merge_feedback_data(self) -> None:
        """
        Optionally extend the training set with high-confidence feedback rows.
        Skipped silently if the feedback DB doesn't exist yet.
        """
        feedback_db = os.path.join(self.project_root, "feedback_data", "feedback.db")
        if not os.path.exists(feedback_db):
            logger.info("[2/4] No feedback DB found — skipping feedback merge.")
            return

        try:
            from flightdelay.components.feedback_collector import FeedbackCollector
            collector = FeedbackCollector(feedback_db)
            labeled = collector.get_labeled_data()
            logger.info(f"[2/4] {len(labeled)} labeled feedback rows available.")
            # Future: merge labeled rows into delay_data/ and re-run dvc add
        except Exception as exc:
            logger.warning(f"[2/4] Feedback merge skipped: {exc}")

    def _run_training_pipeline(self) -> None:
        logger.info("[3/4] Running training pipeline ...")
        # Import here to avoid circular imports at module load time
        from training_pipeline import TrainingPipeline
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()

    def _dvc_add_model(self) -> None:
        logger.info("[4/4] Versioning updated model with DVC ...")
        final_model_path = os.path.join(self.project_root, "final_model")
        if not os.path.exists(final_model_path):
            logger.warning("final_model/ not found — skipping dvc add.")
            return
        try:
            self._run_cmd(["dvc", "add", "final_model/"], "dvc add final_model")
        except RuntimeError as exc:
            logger.warning(f"dvc add skipped: {exc}")

    def _dvc_push(self) -> None:
        try:
            self._run_cmd(["dvc", "push"], "dvc push")
            logger.info("Model artefacts pushed to DVC remote.")
        except RuntimeError as exc:
            logger.warning(f"dvc push failed (artefacts not uploaded): {exc}")
