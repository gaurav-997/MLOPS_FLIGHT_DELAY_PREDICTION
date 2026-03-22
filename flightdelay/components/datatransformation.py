"""
Data Transformation Component
Joins flights + airlines + airports + holidays + weather
Creates preprocessing pipeline for ML training
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from flightdelay.logging.logger import logger
from flightdelay.exception.exception import CustomException
from flightdelay.entity.config_entity import DataTransformationConfig
from flightdelay.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from flightdelay.constant import common_constants
from flightdelay.utils.main_utils import save_object, save_numpy_array_data


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, 
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)
    
    def load_validated_data(self) -> tuple:
        """Load all validated data sources"""
        try:
            logger.info("Loading validated data from all sources...")
            
            flights_df = pd.read_csv(self.data_validation_artifact.validated_flights_dir)
            airports_df = pd.read_csv(self.data_validation_artifact.validated_airports_dir)
            airlines_df = pd.read_csv(self.data_validation_artifact.validated_airlines_dir)
            holidays_df = pd.read_csv(self.data_validation_artifact.validated_holidays_dir)
            weather_df = pd.read_csv(self.data_validation_artifact.validated_weather_dir)
            
            logger.info(f"Loaded flights: {len(flights_df)} rows")
            logger.info(f"Loaded airports: {len(airports_df)} rows")
            logger.info(f"Loaded airlines: {len(airlines_df)} rows")
            logger.info(f"Loaded holidays: {len(holidays_df)} rows")
            logger.info(f"Loaded weather: {len(weather_df)} rows")
            
            return flights_df, airports_df, airlines_df, holidays_df, weather_df
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def join_simple_data(self, flights_df: pd.DataFrame, airlines_df: pd.DataFrame, 
                         airports_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 6.1: Join flights with airlines and airports (origin + destination)
        """
        try:
            logger.info("Starting simple joins (flights + airlines + airports)...")
            
            # Join 1: Airlines data
            logger.info("  [1/3] Joining with airlines data...")
            flights_df = flights_df.merge(
                airlines_df,
                left_on="AIRLINE",
                right_on="IATA_CODE",
                how="left",
                suffixes=("", "_airline")
            )
            flights_df.rename(columns={"AIRLINE_x": "AIRLINE_CODE", "AIRLINE_y": "AIRLINE_NAME"}, inplace=True)
            logger.info(f"    After airline join: {len(flights_df)} rows")
            
            # Join 2: Origin airports
            logger.info("  [2/3] Joining with origin airport data...")
            flights_df = flights_df.merge(
                airports_df,
                left_on="ORIGIN_AIRPORT",
                right_on="IATA_CODE",
                how="left",
                suffixes=("", "_origin")
            )
            # Rename origin airport columns
            flights_df.rename(columns={
                "AIRPORT": "ORIGIN_AIRPORT_NAME",
                "CITY": "ORIGIN_CITY",
                "STATE": "ORIGIN_STATE",
                "COUNTRY": "ORIGIN_COUNTRY",
                "LATITUDE": "ORIGIN_LAT",
                "LONGITUDE": "ORIGIN_LON"
            }, inplace=True)
            logger.info(f"    After origin airport join: {len(flights_df)} rows")
            
            # Join 3: Destination airports
            logger.info("  [3/3] Joining with destination airport data...")
            flights_df = flights_df.merge(
                airports_df,
                left_on="DESTINATION_AIRPORT",
                right_on="IATA_CODE",
                how="left",
                suffixes=("", "_dest")
            )
            # Rename destination airport columns
            flights_df.rename(columns={
                "AIRPORT": "DEST_AIRPORT_NAME",
                "CITY": "DEST_CITY",
                "STATE": "DEST_STATE",
                "COUNTRY": "DEST_COUNTRY",
                "LATITUDE": "DEST_LAT",
                "LONGITUDE": "DEST_LON"
            }, inplace=True)
            logger.info(f"    After destination airport join: {len(flights_df)} rows")
            
            logger.info("[PASS] Simple joins completed")
            return flights_df
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def join_temporal_data(self, flights_df: pd.DataFrame, holidays_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 6.2: Temporal join with holidays data
        """
        try:
            logger.info("Starting temporal join (flights + holidays)...")
            
            # Create date column in flights
            flights_df['date'] = pd.to_datetime(flights_df[['YEAR', 'MONTH', 'DAY']])
            logger.info(f"Created date column: {flights_df['date'].min()} to {flights_df['date'].max()}")
            
            # Ensure holidays date is datetime
            holidays_df['date'] = pd.to_datetime(holidays_df['date'])
            
            # Join with holidays
            flights_df = flights_df.merge(
                holidays_df,
                on="date",
                how="left"
            )
            
            # Fill missing holiday values (non-holidays)
            flights_df['is_holiday'] = flights_df['is_holiday'].fillna(0).astype(int)
            flights_df['holiday_name'] = flights_df['holiday_name'].fillna('None')
            
            holiday_count = (flights_df['is_holiday'] == 1).sum()
            logger.info(f"  Holiday flights: {holiday_count} ({holiday_count/len(flights_df)*100:.2f}%)")
            logger.info("[PASS] Temporal join completed")
            
            return flights_df
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def join_weather_data(self, flights_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 6.3: Weather join (simplified - by date only)
        """
        try:
            logger.info("Starting weather join (flights + weather)...")
            
            # Ensure weather date is datetime
            weather_df['date'] = pd.to_datetime(weather_df['date'])
            
            # Join with weather
            initial_rows = len(flights_df)
            flights_df = flights_df.merge(
                weather_df,
                on="date",
                how="left"
            )
            logger.info(f"  After weather join: {len(flights_df)} rows (delta: {len(flights_df) - initial_rows})")
            
            # Fill missing weather values with median
            weather_cols = ['TMAX', 'TMIN', 'PRCP', 'AWND']
            for col in weather_cols:
                if col in flights_df.columns:
                    missing_before = flights_df[col].isnull().sum()
                    if missing_before > 0:
                        median_val = flights_df[col].median()
                        flights_df[col].fillna(median_val, inplace=True)
                        logger.info(f"  Filled {missing_before} missing {col} with median: {median_val:.2f}")
            
            logger.info("[PASS] Weather join completed")
            return flights_df
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def clean_joined_data(self, flights_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 6.4: Data cleaning after joins
        """
        try:
            logger.info("Starting data cleaning...")
            
            initial_rows = len(flights_df)
            
            # Drop rows with missing target variable ( arrival delay)
            flights_df = flights_df.dropna(subset=[common_constants.TARGET_COLUMN])
            after_target_drop = len(flights_df)
            logger.info(f"  Dropped {initial_rows - after_target_drop} rows with missing {common_constants.TARGET_COLUMN}")
            
            # Drop duplicate IATA_CODE columns from joins
            cols_to_drop = []
            for col in flights_df.columns:
                if 'IATA_CODE' in col and col != 'IATA_CODE':
                    cols_to_drop.append(col)
            
            if cols_to_drop:
                flights_df = flights_df.drop(columns=cols_to_drop, errors='ignore')
                logger.info(f"  Dropped duplicate IATA_CODE columns: {cols_to_drop}")
            
            # Handle categorical nulls
            if 'CANCELLATION_REASON' in flights_df.columns:
                flights_df['CANCELLATION_REASON'] = flights_df['CANCELLATION_REASON'].fillna('None')
            
            # Drop duplicate rows
            before_dedup = len(flights_df)
            flights_df = flights_df.drop_duplicates()
            after_dedup = len(flights_df)
            logger.info(f"  Dropped {before_dedup - after_dedup} duplicate rows")
            
            logger.info(f"[PASS] Data cleaning completed. Final rows: {len(flights_df)}")
            return flights_df
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Step 6.5: Create preprocessing pipeline with KNN imputer + scaler + encoder
        """
        try:
            logger.info("Creating preprocessing pipeline...")
            
            # Define numerical columns
            numerical_cols = [
                'DISTANCE', 'SCHEDULED_TIME', 'ELAPSED_TIME', 
                'ORIGIN_LAT', 'ORIGIN_LON', 'DEST_LAT', 'DEST_LON',
                'TMAX', 'TMIN', 'PRCP', 'AWND',
                'DEPARTURE_DELAY'  # Feature for prediction
            ]
            
            # Define categorical columns
            categorical_cols = [
                'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
                'DAY_OF_WEEK', 'MONTH'
            ]
            
            logger.info(f"  Numerical features: {len(numerical_cols)} columns")
            logger.info(f"  Categorical features: {len(categorical_cols)} columns")
            
            # Numerical pipeline: KNN imputer + Standard scaler
            num_pipeline = Pipeline([
                ('imputer', KNNImputer(n_neighbors=3)),
                ('scaler', StandardScaler())
            ])
            
            # Categorical pipeline: Most frequent imputer + OneHot encoder
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            # Combined preprocessor
            preprocessor = ColumnTransformer([
                ('num', num_pipeline, numerical_cols),
                ('cat', cat_pipeline, categorical_cols)
            ])
            
            logger.info("[PASS] Preprocessing pipeline created")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def prepare_features_target(self, df: pd.DataFrame) -> tuple:
        """Separate features and target variable"""
        try:
            logger.info("Preparing features and target...")
            
            # Define feature columns to keep
            feature_cols = [
                # Time features
                'YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK',
                # Flight identifiers
                'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
                # Flight metrics
                'DISTANCE', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'DEPARTURE_DELAY',
                # Geographic features
                'ORIGIN_LAT', 'ORIGIN_LON', 'DEST_LAT', 'DEST_LON',
                # Weather features
                'TMAX', 'TMIN', 'PRCP', 'AWND',
                # Holiday feature
                'is_holiday'
            ]
            
            # Filter to available columns
            available_features = [col for col in feature_cols if col in df.columns]
            missing_features = set(feature_cols) - set(available_features)
            
            if missing_features:
                logger.warning(f"  Missing features: {missing_features}")
            
            # Prepare X (features) and y (target)
            X = df[available_features].copy()
            y = df[common_constants.TARGET_COLUMN].copy()
            
            logger.info(f"  Features shape: {X.shape}")
            logger.info(f"  Target shape: {y.shape}")
            logger.info(f"  Target stats - Mean: {y.mean():.2f}, Median: {y.median():.2f}, Std: {y.std():.2f}")
            
            return X, y
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Main orchestration method for data transformation
        """
        try:
            logger.info("="*70)
            logger.info("STARTING DATA TRANSFORMATION PROCESS")
            logger.info("="*70)
            
            # Create directories
            os.makedirs(self.data_transformation_config.transformed_data_dir, exist_ok=True)
            os.makedirs(self.data_transformation_config.transformed_object_dir, exist_ok=True)
            
            # Step 1: Load validated data
            print("\n[Step 1/7] Loading validated data...")
            flights_df, airports_df, airlines_df, holidays_df, weather_df = self.load_validated_data()
            print("  [PASS] Validated data loaded")
            
            # Step 2: Simple joins (airlines + airports)
            print("\n[Step 2/7] Performing simple joins...")
            flights_df = self.join_simple_data(flights_df, airlines_df, airports_df)
            print(f"  [PASS] Simple joins completed: {len(flights_df)} rows")
            
            # Step 3: Temporal join (holidays)
            print("\n[Step 3/7] Performing temporal join...")
            flights_df = self.join_temporal_data(flights_df, holidays_df)
            print(f"  [PASS] Temporal join completed: {len(flights_df)} rows")
            
            # Step 4: Weather join
            print("\n[Step 4/7] Performing weather join...")
            flights_df = self.join_weather_data(flights_df, weather_df)
            print(f"  [PASS] Weather join completed: {len(flights_df)} rows")
            
            # Step 5: Data cleaning
            print("\n[Step 5/7] Cleaning joined data...")
            flights_df = self.clean_joined_data(flights_df)
            print(f"  [PASS] Data cleaning completed: {len(flights_df)} rows")
            
            # Save joined data
            flights_df.to_csv(self.data_transformation_config.final_joined_data_path, index=False)
            logger.info(f"Joined data saved: {self.data_transformation_config.final_joined_data_path}")
            print(f"  [PASS] Joined data saved: {self.data_transformation_config.final_joined_data_path}")
            
            # Step 6: Prepare features and target
            print("\n[Step 6/7] Preparing features and target...")
            X, y = self.prepare_features_target(flights_df)
            print(f"  [PASS] Features: {X.shape}, Target: {y.shape}")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            logger.info(f"Train-test split: {len(X_train)} train, {len(X_test)} test")
            print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
            
            # Step 7: Preprocessing pipeline
            print("\n[Step 7/7] Creating and applying preprocessing pipeline...")
            preprocessor = self.get_preprocessing_pipeline()
            
            # Fit and transform
            logger.info("Fitting preprocessor on training data...")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            
            logger.info(f"Transformed train shape: {X_train_transformed.shape}")
            logger.info(f"Transformed test shape: {X_test_transformed.shape}")
            print(f"  Transformed train: {X_train_transformed.shape}")
            print(f"  Transformed test: {X_test_transformed.shape}")
            
            # Save preprocessor
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            logger.info(f"Preprocessor saved: {self.data_transformation_config.transformed_object_file_path}")
            print(f"  [PASS] Preprocessor saved")
            
            # Combine X and y for saving
            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]
            
            # Save transformed arrays
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            
            logger.info(f"Transformed train saved: {self.data_transformation_config.transformed_train_file_path}")
            logger.info(f"Transformed test saved: {self.data_transformation_config.transformed_test_file_path}")
            print(f"  [PASS] Transformed data saved")
            
            # Create artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                joined_data_file_path=self.data_transformation_config.final_joined_data_path
            )
            
            logger.info("="*70)
            logger.info("[PASS] DATA TRANSFORMATION COMPLETED")
            logger.info("="*70)
            
            return data_transformation_artifact
        
        except Exception as e:
            raise CustomException(e, sys)
