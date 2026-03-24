import os
import sys
import yaml
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from flightdelay.logging.logger import logger
from flightdelay.exception.exception import CustomException
from flightdelay.entity.config_entity import ModelEvaluationConfig
from flightdelay.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    RegressionMetricArtifact
)
from flightdelay.utils.main_utils import load_object, load_numpy_array_data, save_object
from flightdelay.utils.ml_utils.mlflow_utils import log_model_evaluation, promote_model_to_production, register_model


class ModelEvaluation:
    """
    Model Evaluation Component
    
    Compares newly trained model with production model (if exists).
    Accepts new model only if it improves R² by at least CHANGED_THRESHOLD.
    Saves best model to final_model/ directory for production use.
    """
    
    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact,
        model_evaluation_config: ModelEvaluationConfig
    ):
        try:
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_evaluation_config = model_evaluation_config
        except Exception as e:
            raise CustomException(e, sys)
    
    def calculate_metrics(self, y_true, y_pred) -> RegressionMetricArtifact:
        """
        Calculate regression metrics
        
        Args:
            y_true: True target values
            y_pred: Predicted values
        
        Returns:
            RegressionMetricArtifact with r2, mae, rmse, mse
        """
        try:
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mse = mean_squared_error(y_true, y_pred)
            
            return RegressionMetricArtifact(
                r2_score=r2,
                mae=mae,
                rmse=rmse,
                mse=mse
            )
        except Exception as e:
            raise CustomException(e, sys)
    
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> RegressionMetricArtifact:
        """
        Evaluate a model on test data
        
        Args:
            model: Trained model object
            X_test: Test features
            y_test: Test target
        
        Returns:
            RegressionMetricArtifact
        """
        try:
            logger.info("Evaluating model on test data...")
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred)
            
            logger.info(f"Model Metrics - R²: {metrics.r2_score:.4f}, MAE: {metrics.mae:.2f}, RMSE: {metrics.rmse:.2f}")
            
            return metrics
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_evaluation_report(
        self,
        is_model_accepted: bool,
        improvement: float,
        trained_metrics: RegressionMetricArtifact,
        best_model_metrics: RegressionMetricArtifact = None
    ):
        """
        Save evaluation report as YAML
        
        Args:
            is_model_accepted: Whether new model was accepted
            improvement: R² improvement score
            trained_metrics: Metrics for trained model
            best_model_metrics: Metrics for best/production model (if exists)
        """
        try:
            # Create report directory
            os.makedirs(self.model_evaluation_config.model_evaluation_dir, exist_ok=True)
            
            # Prepare report data
            report = {
                'model_evaluation': {
                    'is_model_accepted': bool(is_model_accepted),
                    'improvement': float(improvement),
                    'change_threshold': float(self.model_evaluation_config.change_threshold),
                    'trained_model': {
                        'r2_score': float(trained_metrics.r2_score),
                        'mae': float(trained_metrics.mae),
                        'rmse': float(trained_metrics.rmse),
                        'mse': float(trained_metrics.mse)
                    }
                }
            }
            
            # Add production model metrics if available
            if best_model_metrics:
                report['model_evaluation']['production_model'] = {
                    'r2_score': float(best_model_metrics.r2_score),
                    'mae': float(best_model_metrics.mae),
                    'rmse': float(best_model_metrics.rmse),
                    'mse': float(best_model_metrics.mse)
                }
            
            # Save report
            with open(self.model_evaluation_config.report_file_path, 'w') as f:
                yaml.dump(report, f, default_flow_style=False)
            
            logger.info(f"Evaluation report saved to: {self.model_evaluation_config.report_file_path}")
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Main method to evaluate and compare models
        
        Returns:
            ModelEvaluationArtifact with evaluation results
        """
        try:
            logger.info("="*70)
            logger.info("Starting Model Evaluation...")
            logger.info("="*70)
            
            # Step 1: Load test data
            logger.info("Step 1/5: Loading test data...")
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]
            logger.info(f"Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
            
            # Step 2: Load trained model
            logger.info("Step 2/5: Loading trained model...")
            trained_model = load_object(self.model_trainer_artifact.trained_model_file_path)
            logger.info(f"Trained model loaded: {self.model_trainer_artifact.best_model_name}")
            
            # Step 3: Evaluate trained model
            logger.info("Step 3/5: Evaluating trained model...")
            trained_metrics = self.evaluate_model(trained_model, X_test, y_test)
            trained_r2 = trained_metrics.r2_score
            
            # Step 4: Check if production model exists
            logger.info("Step 4/5: Checking for production model...")
            best_model_file_path = self.model_evaluation_config.best_model_file_path
            
            if os.path.exists(best_model_file_path):
                logger.info(f"Production model found at: {best_model_file_path}")
                
                # Load and evaluate production model
                production_model = load_object(best_model_file_path)
                production_metrics = self.evaluate_model(production_model, X_test, y_test)
                production_r2 = production_metrics.r2_score
                
                # Compare models
                improvement = trained_r2 - production_r2
                is_model_accepted = improvement >= self.model_evaluation_config.change_threshold
                
                logger.info(f"\n{'='*70}")
                logger.info(f"MODEL COMPARISON:")
                logger.info(f"  Production R²: {production_r2:.4f}")
                logger.info(f"  New Model R²:  {trained_r2:.4f}")
                logger.info(f"  Improvement:   {improvement:.4f} (Threshold: {self.model_evaluation_config.change_threshold})")
                logger.info(f"  Decision:      {'ACCEPTED [YES]' if is_model_accepted else 'REJECTED [NO]'}")
                logger.info(f"{'='*70}\n")
                
                best_model_metric = production_metrics
                
            else:
                # No production model exists - accept first model
                logger.info("No production model found. This is the first model.")
                is_model_accepted = True
                improvement = trained_r2
                best_model_metric = trained_metrics
                
                logger.info(f"\n{'='*70}")
                logger.info(f"FIRST MODEL EVALUATION:")
                logger.info(f"  New Model R²: {trained_r2:.4f}")
                logger.info(f"  Decision:     ACCEPTED [YES] (First model)")
                logger.info(f"{'='*70}\n")
            
            # Step 5: Save best model if accepted
            logger.info("Step 5/5: Saving best model...")
            if is_model_accepted:
                # Create best model directory
                os.makedirs(self.model_evaluation_config.best_model_dir, exist_ok=True)
                
                # Save trained model as best model
                save_object(best_model_file_path, trained_model)
                logger.info(f"Best model saved to: {best_model_file_path}")
                
                # Also save preprocessor
                preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
                save_object(self.model_evaluation_config.best_preprocessor_file_path, preprocessor)
                logger.info(f"Preprocessor saved to: {self.model_evaluation_config.best_preprocessor_file_path}")
                
                best_model_path = best_model_file_path
            else:
                logger.info("Model not accepted. Production model remains unchanged.")
                best_model_path = best_model_file_path
            
            # Save evaluation report
            self.save_evaluation_report(
                is_model_accepted=is_model_accepted,
                improvement=improvement,
                trained_metrics=trained_metrics,
                best_model_metrics=best_model_metric
            )

            # Log evaluation results to MLflow and (if accepted) promote to Production
            run_id = getattr(self.model_trainer_artifact, "mlflow_run_id", None)
            try:
                prod_metrics = best_model_metric if os.path.exists(
                    self.model_evaluation_config.best_model_file_path
                ) and best_model_metric is not trained_metrics else None

                log_model_evaluation(
                    run_id=run_id,
                    is_model_accepted=is_model_accepted,
                    improvement=improvement,
                    trained_r2=trained_metrics.r2_score,
                    trained_mae=trained_metrics.mae,
                    trained_rmse=trained_metrics.rmse,
                    production_r2=prod_metrics.r2_score if prod_metrics else None,
                    production_mae=prod_metrics.mae if prod_metrics else None,
                    production_rmse=prod_metrics.rmse if prod_metrics else None,
                )

                if is_model_accepted and run_id:
                    version = register_model(run_id)
                    promote_model_to_production(version)

            except Exception as mlflow_err:
                logger.warning(f"MLflow evaluation logging failed (non-fatal): {mlflow_err}")

            # Create artifact
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_score=improvement,
                best_model_path=best_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                train_metric=self.model_trainer_artifact.train_metric_artifact,
                best_model_metric=best_model_metric,
                report_file_path=self.model_evaluation_config.report_file_path
            )
            
            logger.info("="*70)
            logger.info("Model Evaluation Completed Successfully!")
            logger.info("="*70)
            
            return model_evaluation_artifact
            
        except Exception as e:
            raise CustomException(e, sys)
