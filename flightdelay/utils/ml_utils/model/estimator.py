"""
FlightDelayModel — wraps preprocessor + trained model for consistent inference.
Used by both the FastAPI service and any batch prediction scripts.
"""

import pandas as pd
import numpy as np


class FlightDelayModel:
    """
    Combines a fitted preprocessor and a trained regressor into a single
    predict-able object that can be serialised and loaded at serving time.
    """

    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        """
        Transform raw feature dataframe and return predicted arrival delays (minutes).

        Args:
            dataframe: Raw input DataFrame (same schema used during training).

        Returns:
            1-D numpy array of float predictions.
        """
        transformed = self.preprocessor.transform(dataframe)
        predictions = self.model.predict(transformed)
        return predictions
