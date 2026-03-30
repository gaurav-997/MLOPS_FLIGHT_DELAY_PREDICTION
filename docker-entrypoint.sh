#!/bin/bash
# docker-entrypoint.sh — selects the runtime mode via CMD
set -e

case "$1" in
  api)
    echo ">>> Starting Flight Delay Prediction API ..."
    exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
    ;;
  train)
    echo ">>> Running training pipeline ..."
    exec python training_pipeline.py
    ;;
  retrain)
    echo ">>> Running scheduled retraining ..."
    exec python -c "
from flightdelay.pipeline.retraining_manager import RetrainingManager
RetrainingManager().trigger_retraining(reason='scheduled')
"
    ;;
  test)
    echo ">>> Running tests ..."
    exec pytest tests/ -v --tb=short
    ;;
  *)
    exec "$@"
    ;;
esac
