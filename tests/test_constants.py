"""Tests for flightdelay.constant.common_constants"""
from flightdelay.constant import common_constants


class TestCommonConstants:
    def test_pipeline_name_defined(self):
        assert common_constants.PIPELINE_NAME == "flight_delay_prediction_pipeline"

    def test_artifact_dir_name(self):
        assert common_constants.ARTIFACT_DIR_NAME == "Artifacts"

    def test_target_column(self):
        assert common_constants.TARGET_COLUMN == "ARRIVAL_DELAY"

    def test_data_ingestion_dir(self):
        assert common_constants.DATA_INGESTION_DIR_NAME == "data_ingestion"

    def test_data_validation_dir(self):
        assert common_constants.DATA_VALIDATION_DIR_NAME == "data_validation"

    def test_data_transformation_dir(self):
        assert common_constants.DATA_TRANSFORMATION_DIR_NAME == "data_transformation"

    def test_model_trainer_dir(self):
        assert common_constants.MODEL_TRAINER_DIR_NAME == "model_trainer"

    def test_model_evaluation_dir(self):
        assert common_constants.MODEL_EVALUATION_DIR_NAME == "model_evaluation"

    def test_best_model_dir(self):
        assert common_constants.BEST_MODEL_DIR == "final_model"

    def test_expected_score_is_float(self):
        assert isinstance(common_constants.MODEL_TRAINER_EXPECTED_SCORE, float)
        assert 0.0 < common_constants.MODEL_TRAINER_EXPECTED_SCORE < 1.0

    def test_overfitting_threshold_is_float(self):
        assert isinstance(common_constants.MODEL_TRAINER_OVERFITTING_THRESHOLD, float)
        assert common_constants.MODEL_TRAINER_OVERFITTING_THRESHOLD > 0.0

    def test_change_threshold_is_non_negative(self):
        assert common_constants.MODEL_EVALUATION_CHANGED_THRESHOLD >= 0.0

    def test_file_names_are_strings(self):
        assert common_constants.BEST_MODEL_FILE_NAME.endswith(".pkl")
        assert common_constants.BEST_PREPROCESSOR_FILE_NAME.endswith(".pkl")
        assert common_constants.PREPROCESSING_OBJECT_FILE_NAME.endswith(".pkl")
        assert common_constants.TRANSFORMED_TRAIN_FILE_NAME.endswith(".npy")
        assert common_constants.TRANSFORMED_TEST_FILE_NAME.endswith(".npy")
