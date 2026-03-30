"""Tests for flightdelay.utils.ml_utils.model.estimator"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from flightdelay.utils.ml_utils.model.estimator import FlightDelayModel


class TestFlightDelayModel:
    def _make_model(self):
        """Return a FlightDelayModel backed by mock preprocessor + regressor."""
        preprocessor = MagicMock()
        preprocessor.transform.side_effect = lambda df: df.values

        regressor = MagicMock()
        regressor.predict.return_value = np.array([10.0, 20.0, 30.0])

        return FlightDelayModel(preprocessor, regressor)

    def test_predict_returns_ndarray(self):
        model = self._make_model()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = model.predict(df)
        assert isinstance(result, np.ndarray)

    def test_predict_length_matches_input(self):
        model = self._make_model()
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = model.predict(df)
        assert len(result) == 3

    def test_predict_calls_preprocessor(self):
        preprocessor = MagicMock()
        preprocessor.transform.return_value = np.zeros((2, 3))
        regressor = MagicMock()
        regressor.predict.return_value = np.array([0.0, 1.0])

        model = FlightDelayModel(preprocessor, regressor)
        df = pd.DataFrame({"x": [1, 2]})
        model.predict(df)

        preprocessor.transform.assert_called_once()

    def test_predict_calls_regressor(self):
        preprocessor = MagicMock()
        preprocessor.transform.return_value = np.zeros((2, 3))
        regressor = MagicMock()
        regressor.predict.return_value = np.array([5.0, 10.0])

        model = FlightDelayModel(preprocessor, regressor)
        df = pd.DataFrame({"x": [1, 2]})
        model.predict(df)

        regressor.predict.assert_called_once()

    def test_stores_preprocessor_and_model(self):
        preprocessor = MagicMock()
        regressor = MagicMock()
        model = FlightDelayModel(preprocessor, regressor)
        assert model.preprocessor is preprocessor
        assert model.model is regressor
