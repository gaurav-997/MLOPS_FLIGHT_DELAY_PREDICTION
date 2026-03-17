import pandas as pd
from dotenv import load_dotenv
import os
import sys
from flightdelay.logging import logger
from flightdelay.exceptions import CustomException

class DataIngestion:

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise CustomException(e, sys)
