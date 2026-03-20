"""
Utility functions for saving/loading objects and numpy arrays
"""

import os
import sys
import dill
import numpy as np
from flightdelay.exception.exception import CustomException
from flightdelay.logging.logger import logger


def save_object(file_path: str, obj: object) -> None:
    """
    Save Python object using dill
    
    Args:
        file_path: Path to save the object
        obj: Object to save (model, preprocessor, etc.)
    """
    try:
        logger.info(f"Saving object to {file_path}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
        logger.info(f"Object saved successfully: {file_path}")
    
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> object:
    """
    Load Python object using dill
    
    Args:
        file_path: Path to load the object from
    
    Returns:
        Loaded object
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")
        
        logger.info(f"Loading object from {file_path}")
        
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        
        logger.info(f"Object loaded successfully: {file_path}")
        return obj
    
    except Exception as e:
        raise CustomException(e, sys)


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Save numpy array to file
    
    Args:
        file_path: Path to save the array
        array: Numpy array to save
    """
    try:
        logger.info(f"Saving numpy array to {file_path}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
        
        logger.info(f"Numpy array saved successfully: {file_path}, shape: {array.shape}")
    
    except Exception as e:
        raise CustomException(e, sys)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load numpy array from file
    
    Args:
        file_path: Path to load the array from
    
    Returns:
        Loaded numpy array
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")
        
        logger.info(f"Loading numpy array from {file_path}")
        
        with open(file_path, "rb") as file_obj:
            array = np.load(file_obj)
        
        logger.info(f"Numpy array loaded successfully: {file_path}, shape: {array.shape}")
        return array
    
    except Exception as e:
        raise CustomException(e, sys)
