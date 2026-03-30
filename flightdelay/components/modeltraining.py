"""
Model Training Component
Trains multiple regression models and selects the best one
Evaluates: Random Forest, Gradient Boosting, XGBoost, LightGBM, Linear Regression, Ridge
"""

import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from flightdelay.logging.logger import logger
from flightdelay.exception.exception import CustomException
from flightdelay.entity.config_entity import ModelTrainerConfig
from flightdelay.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, RegressionMetricArtifact
from flightdelay.utils.main_utils import save_object, load_numpy_array_data
from flightdelay.utils.ml_utils.mlflow_utils import log_model_training


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def get_model_dict(self) -> dict:
        """
        Define all models to evaluate
        Optimized for high-dimensional data (4780 features)
        """
        try:
            models = {
                "Random Forest": RandomForestRegressor(
                    n_estimators=50,  # Reduced from 100
                    max_depth=10,  # Reduced from 15
                    max_features='sqrt',  # Limit features per split
                    min_samples_split=10,  # Faster splits
                    min_samples_leaf=4,  # Faster leafs
                    random_state=42,
                    n_jobs=-1
                ),
                "Gradient Boosting": GradientBoostingRegressor(
                    n_estimators=50,  # Reduced from 100
                    max_depth=4,  # Reduced from 5
                    learning_rate=0.1,
                    subsample=0.8,  # Sample 80% of data
                    max_features='sqrt',  # Limit features
                    random_state=42
                ),
                "XGBoost": XGBRegressor(
                    n_estimators=50,  # Reduced from 100
                    max_depth=5,  # Reduced from 7
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,  # Limit features per tree
                    random_state=42,
                    n_jobs=-1
                ),
                "LightGBM": LGBMRegressor(
                    n_estimators=50,  # Reduced from 100
                    max_depth=5,  # Reduced from 7
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ),
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(alpha=1.0)
            }

            return models

        except Exception as e:
            raise CustomException(e, sys)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetricArtifact:
        """
        Calculate regression metrics
        """
        try:
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)

            return RegressionMetricArtifact(
                r2_score=r2,
                mae=mae,
                rmse=rmse,
                mse=mse
            )

        except Exception as e:
            raise CustomException(e, sys)

    def train_and_evaluate_models(
            self, X_train: np.ndarray, y_train: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray) -> tuple:
        """
        Train all models and return the best one
        """
        try:
            logger.info("Starting model training and evaluation...")

            models = self.get_model_dict()

            best_model = None
            best_model_name = None
            best_score = -float('inf')
            best_train_metric = None
            best_test_metric = None

            model_report = {}

            print(f"\nTraining {len(models)} models...")
            print("="*70)

            for model_name, model in models.items():
                logger.info(f"Training {model_name}...")
                print(f"\n[{list(models.keys()).index(model_name) + 1}/{len(models)}] Training {model_name}...")

                # Train model
                model.fit(X_train, y_train)

                # Predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Calculate metrics
                train_metrics = self.calculate_metrics(y_train, y_train_pred)
                test_metrics = self.calculate_metrics(y_test, y_test_pred)

                # Log results
                logger.info(f"{model_name} Results:")
                logger.info(f"  Train R²: {train_metrics.r2_score:.4f} | MAE: {train_metrics.mae:.2f} | RMSE: {train_metrics.rmse:.2f}")
                logger.info(f"  Test  R²: {test_metrics.r2_score:.4f} | MAE: {test_metrics.mae:.2f} | RMSE: {test_metrics.rmse:.2f}")

                print(f"  Train R²: {train_metrics.r2_score:.4f} | MAE: {train_metrics.mae:.2f} | RMSE: {train_metrics.rmse:.2f}")
                print(f"  Test  R²: {test_metrics.r2_score:.4f} | MAE: {test_metrics.mae:.2f} | RMSE: {test_metrics.rmse:.2f}")

                # Store report
                model_report[model_name] = {
                    "train_r2": train_metrics.r2_score,
                    "test_r2": test_metrics.r2_score,
                    "train_mae": train_metrics.mae,
                    "test_mae": test_metrics.mae,
                    "train_rmse": train_metrics.rmse,
                    "test_rmse": test_metrics.rmse
                }

                # Track best model based on test R²
                if test_metrics.r2_score > best_score:
                    best_score = test_metrics.r2_score
                    best_model = model
                    best_model_name = model_name
                    best_train_metric = train_metrics
                    best_test_metric = test_metrics

            logger.info("="*70)
            logger.info(f"BEST MODEL: {best_model_name}")
            logger.info(f"  Test R²: {best_score:.4f}")
            logger.info("="*70)

            print("\n" + "="*70)
            print(f"BEST MODEL: {best_model_name}")
            print(f"  Test R²: {best_score:.4f} | MAE: {best_test_metric.mae:.2f} | RMSE: {best_test_metric.rmse:.2f}")
            print("="*70)

            return best_model, best_model_name, best_train_metric, best_test_metric, model_report

        except Exception as e:
            raise CustomException(e, sys)

    def check_model_acceptance(
            self, train_metric: RegressionMetricArtifact,
            test_metric: RegressionMetricArtifact,
            model_name: str) -> bool:
        """
        Check if model meets acceptance criteria:
        1. Test R² > expected score
        2. Overfitting check (train-test gap < threshold)
        """
        try:
            logger.info("Checking model acceptance criteria...")

            # Check 1: Minimum performance threshold
            if test_metric.r2_score < self.model_trainer_config.expected_score:
                logger.error(f"Model R² {test_metric.r2_score:.4f} < threshold {self.model_trainer_config.expected_score:.4f}")
                print("\n[FAIL] Model performance below threshold:")
                print(f"  Test R²: {test_metric.r2_score:.4f} < Required: {self.model_trainer_config.expected_score:.4f}")
                return False

            # Check 2: Overfitting threshold
            overfitting_gap = train_metric.r2_score - test_metric.r2_score
            if overfitting_gap > self.model_trainer_config.overfitting_threshold:
                logger.error(f"Overfitting detected: train-test gap {overfitting_gap:.4f} > threshold {self.model_trainer_config.overfitting_threshold:.4f}")
                print("\n[FAIL] Overfitting detected:")
                print(f"  Train R²: {train_metric.r2_score:.4f}")
                print(f"  Test R²: {test_metric.r2_score:.4f}")
                print(f"  Gap: {overfitting_gap:.4f} > Threshold: {self.model_trainer_config.overfitting_threshold:.4f}")
                return False

            logger.info(f"[PASS] Model {model_name} accepted")
            logger.info(f"  Test R²: {test_metric.r2_score:.4f} >= {self.model_trainer_config.expected_score:.4f}")
            logger.info(f"  Overfitting gap: {overfitting_gap:.4f} <= {self.model_trainer_config.overfitting_threshold:.4f}")

            print(f"\n[PASS] Model acceptance criteria met:")
            print(f"  ✓ Test R²: {test_metric.r2_score:.4f} >= {self.model_trainer_config.expected_score:.4f}")
            print(f"  ✓ Overfitting gap: {overfitting_gap:.4f} <= {self.model_trainer_config.overfitting_threshold:.4f}")

            return True

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_training(self) -> ModelTrainerArtifact:
        """
        Main method to train and evaluate models
        """
        try:
            logger.info("="*70)
            logger.info("STARTING MODEL TRAINING PROCESS")
            logger.info("="*70)

            # Create directories
            os.makedirs(self.model_trainer_config.trained_model_dir, exist_ok=True)

            # Step 1: Load transformed data
            print("\n[Step 1/4] Loading transformed data...")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            # Separate features and target
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logger.info(f"Train shape: X={X_train.shape}, y={y_train.shape}")
            logger.info(f"Test shape: X={X_test.shape}, y={y_test.shape}")
            print(f"  Train: X={X_train.shape}, y={y_train.shape}")
            print(f"  Test: X={X_test.shape}, y={y_test.shape}")
            print("  [PASS] Data loaded successfully")

            # Step 2: Train and evaluate models
            print("\n[Step 2/4] Training and evaluating models...")
            best_model, best_model_name, train_metrics, test_metrics, model_report = self.train_and_evaluate_models(
                X_train, y_train, X_test, y_test
            )
            print("  [PASS] Model training completed")

            # Step 3: Check acceptance criteria
            print("\n[Step 3/4] Checking model acceptance criteria...")
            is_accepted = self.check_model_acceptance(train_metrics, test_metrics, best_model_name)

            if not is_accepted:
                raise Exception("Model did not meet acceptance criteria")

            # Step 4: Save best model
            print("\n[Step 4/4] Saving best model...")
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logger.info(f"Best model saved: {self.model_trainer_config.trained_model_file_path}")
            print(f"  Model saved: {self.model_trainer_config.trained_model_file_path}")
            print("  [PASS] Model saved successfully")

            # Step 5: Log training run to MLflow / DagsHub
            print("\n[Step 5/5] Logging run to MLflow...")
            try:
                mlflow_run_id = log_model_training(
                    best_model=best_model,
                    best_model_name=best_model_name,
                    train_r2=train_metrics.r2_score,
                    test_r2=test_metrics.r2_score,
                    test_mae=test_metrics.mae,
                    test_rmse=test_metrics.rmse,
                    test_mse=test_metrics.mse,
                    expected_score=self.model_trainer_config.expected_score,
                    overfitting_threshold=self.model_trainer_config.overfitting_threshold,
                    features_count=X_train.shape[1],
                )
                print(f"  MLflow run id: {mlflow_run_id}")
                print("  [PASS] MLflow logging completed")
            except Exception as mlflow_err:
                # MLflow logging is non-critical – log warning but do not abort
                logger.warning(f"MLflow logging failed (non-fatal): {mlflow_err}")
                mlflow_run_id = None

            # Create artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics,
                best_model_name=best_model_name,
                best_model_score=test_metrics.r2_score,
                mlflow_run_id=mlflow_run_id,
            )

            logger.info("="*70)
            logger.info("[PASS] MODEL TRAINING COMPLETED")
            logger.info(f"Best Model: {best_model_name}")
            logger.info(f"Test R²: {test_metrics.r2_score:.4f} | MAE: {test_metrics.mae:.2f} | RMSE: {test_metrics.rmse:.2f}")
            logger.info("="*70)

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
