"""
Training Pipeline Orchestrator

This runs all components in sequence:
1. Data Ingestion
2. Data Validation  
3. Data Transformation
4. Model Training
5. Model Evaluation

Each component receives artifacts from previous step.
Run once to complete full pipeline.
"""

import sys
from flightdelay.logging.logger import logger
from flightdelay.exception.exception import CustomException
from flightdelay.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)
from flightdelay.components.dataingestion import DataIngestion
from flightdelay.components.datavalidation import DataValidation
from flightdelay.components.datatransformation import DataTransformation
from flightdelay.components.modeltraining import ModelTrainer
from flightdelay.components.modelevaluation import ModelEvaluation


class TrainingPipeline:
    """
    Complete ML Training Pipeline
    Orchestrates all components from data ingestion to model evaluation
    """
    
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
    
    def start_data_ingestion(self):
        """Step 1: Data Ingestion"""
        try:
            logger.info("\n" + "="*80)
            logger.info("[STEP 1/5] DATA INGESTION")
            logger.info("="*80)
            
            data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            
            logger.info(f"Data Ingestion completed. Artifacts saved to: {data_ingestion_config.data_ingestion_dir}")
            return data_ingestion_artifact
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_data_validation(self, data_ingestion_artifact):
        """Step 2: Data Validation"""
        try:
            logger.info("\n" + "="*80)
            logger.info("[STEP 2/5] DATA VALIDATION")
            logger.info("="*80)
            
            data_validation_config = DataValidationConfig(self.training_pipeline_config)
            data_validation = DataValidation(data_validation_config, data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()
            
            logger.info(f"Data Validation completed. Status: {data_validation_artifact.data_validation_status}")
            return data_validation_artifact
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_data_transformation(self, data_validation_artifact):
        """Step 3: Data Transformation"""
        try:
            logger.info("\n" + "="*80)
            logger.info("[STEP 3/5] DATA TRANSFORMATION + FEATURE ENGINEERING")
            logger.info("="*80)
            
            data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
            data_transformation = DataTransformation(
                data_transformation_config,
                data_validation_artifact
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            
            logger.info(f"Data Transformation completed. Preprocessor saved.")
            return data_transformation_artifact
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_model_training(self, data_transformation_artifact):
        """Step 4: Model Training"""
        try:
            logger.info("\n" + "="*80)
            logger.info("[STEP 4/5] MODEL TRAINING")
            logger.info("="*80)
            
            model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_training()
            
            logger.info(f"Model Training completed. Best model: {model_trainer_artifact.best_model_name}")
            return model_trainer_artifact
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_model_evaluation(self, model_trainer_artifact, data_transformation_artifact):
        """Step 5: Model Evaluation"""
        try:
            logger.info("\n" + "="*80)
            logger.info("[STEP 5/5] MODEL EVALUATION")
            logger.info("="*80)
            
            model_evaluation_config = ModelEvaluationConfig(self.training_pipeline_config)
            model_evaluation = ModelEvaluation(
                model_trainer_artifact,
                data_transformation_artifact,
                model_evaluation_config
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            
            logger.info(f"Model Evaluation completed. Accepted: {model_evaluation_artifact.is_model_accepted}")
            return model_evaluation_artifact
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def run_pipeline(self):
        """
        Run complete training pipeline
        All steps executed in sequence
        """
        try:
            logger.info("\n" + "="*80)
            logger.info(" STARTING COMPLETE TRAINING PIPELINE")
            logger.info(f" Artifacts Directory: {self.training_pipeline_config.artifact_dir}")
            logger.info("="*80 + "\n")
            
            # Step 1: Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            
            # Step 2: Data Validation
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            
            # Step 3: Data Transformation
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact
            )
            
            # Step 4: Model Training
            model_trainer_artifact = self.start_model_training(data_transformation_artifact)
            
            # Step 5: Model Evaluation
            model_evaluation_artifact = self.start_model_evaluation(
                model_trainer_artifact,
                data_transformation_artifact
            )
            
            # Summary
            logger.info("\n" + "="*80)
            logger.info(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info(f"\nFINAL RESULTS:")
            logger.info(f"  - Data Validation: {'PASSED' if data_validation_artifact.data_validation_status else 'FAILED'}")
            logger.info(f"  - Best Model: {model_trainer_artifact.best_model_name}")
            logger.info(f"  - Test R²: {model_trainer_artifact.test_metric_artifact.r2_score:.4f}")
            logger.info(f"  - Test MAE: {model_trainer_artifact.test_metric_artifact.mae:.2f} minutes")
            logger.info(f"  - Model Accepted: {'YES' if model_evaluation_artifact.is_model_accepted else 'NO'}")
            logger.info(f"  - R² Improvement: {model_evaluation_artifact.improved_score:.4f}")
            
            if model_evaluation_artifact.is_model_accepted:
                logger.info(f"\n  Production Model: {model_evaluation_artifact.best_model_path}")
            
            logger.info(f"\n  All artifacts saved to: {self.training_pipeline_config.artifact_dir}")
            logger.info("="*80 + "\n")
            
            return model_evaluation_artifact
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        logger.info("="*80)
        logger.info(" MLOPS FLIGHT DELAY PREDICTION - TRAINING PIPELINE")
        logger.info("="*80 + "\n")
        
        # Create and run pipeline
        pipeline = TrainingPipeline()
        model_evaluation_artifact = pipeline.run_pipeline()
        
        # Final message
        print("\n" + "="*80)
        print(" >>> TRAINING PIPELINE COMPLETED SUCCESSFULLY! <<<")
        print("="*80)
        print(f"\nProduction model ready at: {model_evaluation_artifact.best_model_path}")
        print(f"Evaluation report: {model_evaluation_artifact.report_file_path}")
        print("\nNext steps:")
        print("  1. Check final_model/ directory for production model")
        print("  2. Run pipeline again to test model comparison logic")
        print("  3. Implement Model Pusher (Step 10) to deploy model")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print("\n>>> PIPELINE FAILED <<<")
        print(f"Error: {str(e)}")
        sys.exit(1)
