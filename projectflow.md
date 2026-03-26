Batch data  → flights.csv
Static data → airports.csv, airlines.csv
API data    → weather.py
Generated   → holidays_data.py

JOINING ALL DATA SOURCES

This is:

💥 Hardest part
💥 Most important part
💥 What makes your project “10x MLOps”

You are about to learn:

Temporal joins (flight time ↔ weather time)
Multi-source joins (airport ↔ metadata)
Feature engineering


Excellent project scope! This is a production-grade MLOps system with real complexity. Let me summarize what you're building:

🎯 Your Project Architecture:
Data Sources (4 heterogeneous types):
Technical Challenges You'll Solve:
Multi-schema validation

4 different data formats
Schema drift detection
Complex joins:

Spatial: Flight → Airport (by IATA code)
Temporal: Flight date → Holiday date
Spatio-temporal: Flight + Airport location → Weather station + time
API constraints:

Rate limiting (NOAA: 1000/day)
Incremental updates vs full reload
Feature engineering:

Time-series: rolling averages, lag features
Derived: hour_of_day, airport_congestion, historical_delay_rate
Production MLOps:

Data drift detection
Auto-retraining triggers
Model monitoring (target: 85%+ accuracy)
Expected Resume Impact:
✅ Multi-source data ingestion
✅ Schema validation & versioning
✅ Temporal/spatial joins
✅ Feature engineering pipeline
✅ Drift detection
✅ Automated retraining
✅ Production deployment (K8s target)

***************************************************************

. Add ONE thing to make it elite

👉 Add Feature Store concept (even basic)

Even simple:

store features in parquet / redis

reuse during inference

👉 This will push you into top-tier MLOps


1. Environment Setup
   └─ Fix API key → .env file
   └─ Create requirements.txt
   └─ Virtual environment

# Read → Validate → Clean → Join → Feature → Train
# flights
   ↓
+ airlines
   ↓
+ airports
   ↓
+ holidays
   ↓
+ weather
   ↓
final_dataset.csv

# 2. Data Exploration & Profiling
    flights.csv        ✅ MAIN dataset
    airports.csv       ✅ metadata
    airlines.csv       ✅ metadata
    holidays_data.py   ✅ generates holiday data
    weather.py         ✅ API-based weather

   └─ Understand flight.csv(main) dataset structure
   └─ Check data quality (nulls, duplicates, outliers)
   └─ Profile: pandas-profiling or ydata-profiling
   └─ Document findings
    Output of Step 2
    ✔ which columns are useful
    ✔ which columns are noisy
    ✔ where nulls exist

# 3. Schema Definition ✅ DONE ( why - If data changes → pipeline fails early )
   └─ Define expected schemas/structure for each source
   e.g {
 "YEAR": int,
 "MONTH": int,
 "DAY": int,
 "AIRLINE": str,
 "ORIGIN_AIRPORT": str,
 "DESTINATION_AIRPORT": str,
 "DEPARTURE_DELAY": float,
 "ARRIVAL_DELAY": float
}

   └─ Create schema validation logic BEFORE ingestion ✅ 
   └─ Implementation: Simple validation (flightdelay/components/schema_validation.py)
   └─ Test: python test_schema.py
   
   Files created:
   ✔ flightdelay/components/schema_validation.py - Core validation functions
   ✔ test_schema.py - Test validation
   ✔ example_pipeline_with_validation.py - Integration example

# 4. Data Ingestion Component ✅ DONE (flightdelay/components/dataingestion.py)
   └─ Read raw data from multiple sources 
   ( ✔ flights.csv - delay_data/flights_sample.csv
     ✔ airports.csv - delay_data/airports.csv
     ✔ airlines.csv - delay_data/airlines.csv
     ✔ holidays.csv - delay_data/holidays.csv
     ✔ weather.csv - delay_data/weather_data.csv )

   └─ Apply schema validation ✅ (using validate_all_schemas)
   └─ Log validation errors ✅
   └─ Save validated data to artifacts ✅ (Artifacts/timestamp/data_ingestion/ingested/)
   └─ Test: python test_data_ingestion.py
   
   Files:
   ✔ flightdelay/components/dataingestion.py - Main ingestion with validation
   ✔ test_data_ingestion.py - Test ingestion flow
   ✔ DataIngestionArtifact - Returns paths to all ingested files

# 5. Data Validation Component ( check on required columns that we choosen in schmea validation)
   └─ Detect data drift (compare ingested data vs baseline)
   └─ Validate data quality (missing values, outliers, invalid ranges)
   └─ Generate drift report (JSON/HTML)
   
   Constants (flightdelay/constant/training_pipeline/__init__.py):
   - DATA_VALIDATION_DIR_NAME = 'data_validation'
   - DATA_DRIFT_REPORT_FILE_NAME = 'drift_report.yaml'
   - DATA_VALIDATION_VALID_DIR = 'validated'
   - DATA_VALIDATION_INVALID_DIR = 'invalidated'
   
   Config (flightdelay/entity/config_entity.py):
   - DataValidationConfig: validation_dir, report_file_path, valid_dir, invalid_dir
   
   Artifact (flightdelay/entity/artifact_entity.py):
   - DataValidationArtifact: validation_status, drift_report_path, valid_data_dir
   
   Implementation (flightdelay/components/datavalidation.py):
   1. Load ingested data from DataIngestionArtifact
   2. Check data quality:
      - Missing value analysis (ARRIVAL_DELAY critical)
      - Invalid airport codes (not in airports.csv)
      - Date range validation (valid flight dates)
      - Coordinate validation (lat/long in valid ranges)
   3. Detect data drift (using scipy.stats.ks_2samp):
      - Compare numerical columns distribution
      - Compare categorical value frequencies
      - Generate drift score per column
   4. Generate drift report (YAML format)

# 6. Data Joining & Transformation Component
   
   Constants:
   - DATA_TRANSFORMATION_DIR_NAME = 'data_transformation'
   - DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR = 'transformed'
   - DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR = 'transformed_object'
   - PREPROCESSING_OBJECT_FILE_NAME = 'preprocessor.pkl'
   - FINAL_JOINED_DATA_FILE_NAME = 'flights_joined.csv'
   
   Config:
   - DataTransformationConfig: transformation_dir, transformed_data_dir, 
     transformed_object_file_path, final_joined_data_path
   
   Artifact:
   - DataTransformationArtifact: transformed_object_file_path, 
     transformed_train_file_path, transformed_test_file_path,
     joined_data_file_path
   
   Implementation (flightdelay/components/datatransformation.py):
   
   ✅ Step 6.1: SIMPLE JOINS (flights + airlines + airports)
   ```python
   # Join 1: Airlines data
   flights = flights.merge(
       airlines,
       left_on="AIRLINE",
       right_on="IATA_CODE",
       how="left",
       suffixes=("", "_airline")
   )
   # Add: AIRLINE_NAME column
   
   # Join 2: Origin airports
   flights = flights.merge(
       airports,
       left_on="ORIGIN_AIRPORT",
       right_on="IATA_CODE",
       how="left",
       suffixes=("", "_origin")
   )
   # Rename: AIRPORT → ORIGIN_AIRPORT_NAME
   # Rename: CITY → ORIGIN_CITY, STATE → ORIGIN_STATE
   # Rename: LATITUDE → ORIGIN_LAT, LONGITUDE → ORIGIN_LON
   
   # Join 3: Destination airports  
   flights = flights.merge(
       airports,
       left_on="DESTINATION_AIRPORT",
       right_on="IATA_CODE",
       how="left",
       suffixes=("", "_dest")
   )
   # Rename: AIRPORT → DEST_AIRPORT_NAME
   # Rename: CITY → DEST_CITY, STATE → DEST_STATE
   # Rename: LATITUDE → DEST_LAT, LONGITUDE → DEST_LON
   ```
   
   ✅ Step 6.2: TEMPORAL JOIN (+ holidays)
   ```python
   # Create date column
   flights['date'] = pd.to_datetime(
       flights[['YEAR', 'MONTH', 'DAY']]
   )
   
   # Join holidays
   flights = flights.merge(
       holidays_df,
       on="date",
       how="left"
   )
   
   # Fill missing (non-holidays)
   flights['is_holiday'] = flights['is_holiday'].fillna(0).astype(int)
   flights['holiday_name'] = flights['holiday_name'].fillna('None')
   ```
   
   ✅ Step 6.3: WEATHER JOIN (Simplified - by date only)
   ```python
   # For now: simplified join by date
   # Weather data already has 'date' column
   flights = flights.merge(
       weather_df,
       on="date",
       how="left"
   )
   
   # Fill missing weather values with median
   weather_cols = ['TMAX', 'TMIN', 'PRCP', 'AWND']
   for col in weather_cols:
       flights[col].fillna(flights[col].median(), inplace=True)
   ```
   
   Note: Advanced spatio-temporal join (airport + nearest weather station)
   can be implemented later with airport-to-station mapping
   
   ✅ Step 6.4: Data Cleaning
   ```python
   # Drop rows with missing target
   flights = flights.dropna(subset=['ARRIVAL_DELAY'])
   
   # Drop duplicate IATA_CODE columns from joins
   flights = flights.drop(columns=['IATA_CODE_origin', 'IATA_CODE_dest', 
                                     'IATA_CODE_airline'], errors='ignore')
   
   # Handle categorical nulls
   flights['CANCELLATION_REASON'].fillna('None', inplace=True)
   
   # Save joined data
   flights.to_csv('flights_joined.csv', index=False)
   ```
   
   ✅ Step 6.5: Preprocessing Pipeline (sklearn)
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.impute import SimpleImputer, KNNImputer
   from sklearn.preprocessing import StandardScaler, OneHotEncoder
   from sklearn.compose import ColumnTransformer
   
   # Define column types
   numerical_cols = ['DISTANCE', 'SCHEDULED_TIME', 'ORIGIN_LAT', 
                     'ORIGIN_LON', 'DEST_LAT', 'DEST_LON',
                     'TMAX', 'TMIN', 'PRCP', 'AWND']
   
   categorical_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
                       'DAY_OF_WEEK', 'MONTH']
   
   # Numerical pipeline
   num_pipeline = Pipeline([
       ('imputer', KNNImputer(n_neighbors=3)),
       ('scaler', StandardScaler())
   ])
   
   # Categorical pipeline
   cat_pipeline = Pipeline([
       ('imputer', SimpleImputer(strategy='most_frequent')),
       ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
   ])
   
   # Combined preprocessor
   preprocessor = ColumnTransformer([
       ('num', num_pipeline, numerical_cols),
       ('cat', cat_pipeline, categorical_cols)
   ])
   
   # Fit and transform
   X_train_transformed = preprocessor.fit_transform(X_train)
   X_test_transformed = preprocessor.transform(X_test)
   
   # Save preprocessor
   save_object(preprocessor, 'preprocessor.pkl')
   ```

# 7. Feature Engineering Component
   
   Implementation separate feature_engineering.py:
   
   ✅ Temporal Features:
   ```python
   # Extract from timestamps
   flights['HOUR'] = flights['SCHEDULED_DEPARTURE'] // 100
   flights['DAY_OF_WEEK'] = flights['DAY_OF_WEEK']  # Already present
   flights['MONTH'] = flights['MONTH']  # Already present
   flights['IS_WEEKEND'] = flights['DAY_OF_WEEK'].isin([6, 7]).astype(int)
   flights['QUARTER'] = (flights['MONTH'] - 1) // 3 + 1
   ```
   
   ✅ Derived Features:
   ```python
   # Holiday flag (from join)
   flights['IS_HOLIDAY'] = flights['is_holiday']
   
   # Create delay target (binary classification)
   flights['DELAYED'] = (flights['ARRIVAL_DELAY'] > 15).astype(int)
   
   # Flight duration vs scheduled
   flights['DURATION_RATIO'] = flights['ELAPSED_TIME'] / flights['SCHEDULED_TIME']
   
   # Distance bins
   flights['DISTANCE_BIN'] = pd.cut(flights['DISTANCE'], 
                                      bins=[0, 500, 1000, 2000, 5000], 
                                      labels=['short', 'medium', 'long', 'ultra'])
   ```
   
   ✅ Advanced Aggregation Features (Historical):
   ```python
   # Airport historical delay rates
   airport_delay_stats = flights.groupby('ORIGIN_AIRPORT')['ARRIVAL_DELAY'].agg([
       ('ORIGIN_AVG_DELAY', 'mean'),
       ('ORIGIN_DELAY_STD', 'std'),
       ('ORIGIN_DELAY_RATE', lambda x: (x > 15).mean())
   ]).reset_index()
   
   flights = flights.merge(airport_delay_stats, on='ORIGIN_AIRPORT', how='left')
   
   # Airline historical delay rates  
   airline_delay_stats = flights.groupby('AIRLINE')['ARRIVAL_DELAY'].agg([
       ('AIRLINE_AVG_DELAY', 'mean'),
       ('AIRLINE_DELAY_RATE', lambda x: (x > 15).mean())
   ]).reset_index()
   
   flights = flights.merge(airline_delay_stats, on='AIRLINE', how='left')
   
   # Route historical delay rates
   flights['ROUTE'] = flights['ORIGIN_AIRPORT'] + '_' + flights['DESTINATION_AIRPORT']
   route_delay_stats = flights.groupby('ROUTE')['ARRIVAL_DELAY'].agg([
       ('ROUTE_AVG_DELAY', 'mean')
   ]).reset_index()
   
   flights = flights.merge(route_delay_stats, on='ROUTE', how='left')
   ```
   
   ✅ Weather Features:
   ```python
   # Temperature range
   flights['TEMP_RANGE'] = flights['TMAX'] - flights['TMIN']
   
   # Extreme weather flags
   flights['EXTREME_COLD'] = (flights['TMIN'] < -10).astype(int)
   flights['EXTREME_HOT'] = (flights['TMAX'] > 35).astype(int)
   flights['HEAVY_RAIN'] = (flights['PRCP'] > 50).astype(int)
   flights['HIGH_WIND'] = (flights['AWND'] > 15).astype(int)
   ```
   
   Final Feature List (50+ features):
   - Original: YEAR, MONTH, DAY, DAY_OF_WEEK, AIRLINE, FLIGHT_NUMBER, etc.
   - Temporal: HOUR, IS_WEEKEND, QUARTER
   - Derived: DELAYED, DURATION_RATIO, DISTANCE_BIN
   - Historical: ORIGIN_AVG_DELAY, AIRLINE_DELAY_RATE, ROUTE_AVG_DELAY
   - Weather: TMAX, TMIN, PRCP, AWND, TEMP_RANGE, EXTREME_* flags
   - Holiday: IS_HOLIDAY
   - Geographic: ORIGIN_LAT, ORIGIN_LON, DEST_LAT, DEST_LON
   
   Target: ARRIVAL_DELAY (regression) or DELAYED (classification)

# 8. Model Training Component

   Constants:
   - MODEL_TRAINER_DIR_NAME = 'model_trainer'
   - MODEL_TRAINER_TRAINED_MODEL_DIR = 'trained_model'
   - MODEL_TRAINER_TRAINED_MODEL_NAME = 'model.pkl'
   - MODEL_TRAINER_EXPECTED_SCORE = 0.75  # Min R² for regression or F1 for classification
   - MODEL_TRAINER_OVERFITTING_THRESHOLD = 0.05
   
   Config:
   - ModelTrainerConfig: model_trainer_dir, trained_model_file_path, 
     expected_score, overfitting_threshold
   
   Artifact:
   - RegressionMetricArtifact: r2_score, mae, rmse, mape
   - ClassificationMetricArtifact: f1_score, precision, recall, accuracy
   - ModelTrainerArtifact: trained_model_file_path, train_metric_artifact, test_metric_artifact
   
   Implementation (flightdelay/components/modeltraining.py):
   
   ```python
   # Load transformed data
   train_arr = load_numpy_array(transformation_artifact.transformed_train_file_path)
   test_arr = load_numpy_array(transformation_artifact.transformed_test_file_path)
   
   X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
   X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
   
   # Define models to evaluate
   models = {
       "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
       "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
       "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
       "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
       "Linear Regression": LinearRegression(),
       "Ridge": Ridge(alpha=1.0)
   }
   
   # Train and evaluate each model
   best_model = None
   best_score = -float('inf')
   
   for model_name, model in models.items():
       model.fit(X_train, y_train)
       
       # Evaluate
       train_pred = model.predict(X_train)
       test_pred = model.predict(X_test)
       
       train_r2 = r2_score(y_train, train_pred)
       test_r2 = r2_score(y_test, test_pred)
       test_mae = mean_absolute_error(y_test, test_pred)
       test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
       
       logger.info(f"{model_name}: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}, MAE={test_mae:.2f}")
       
       if test_r2 > best_score:
           best_score = test_r2
           best_model = model
           best_model_name = model_name
   
   # Check acceptance criteria
   if best_score < expected_score:
       raise Exception(f"Best model R² {best_score:.3f} < threshold {expected_score}")
   
   # Check overfitting
   train_score = r2_score(y_train, best_model.predict(X_train))
   if (train_score - best_score) > overfitting_threshold:
       raise Exception(f"Overfitting detected: train-test gap {train_score - best_score:.3f}")
   
   # Save best model
   save_object(best_model, trained_model_file_path)
   
   return ModelTrainerArtifact(...)
   ```

# 9. Model Evaluation Component
## Overview
Successfully implemented complete Model Evaluation Component that compares newly trained models with production models and promotes better models.

   Constants:
   - MODEL_EVALUATION_DIR_NAME = 'model_evaluation'
   - MODEL_EVALUATION_REPORT_NAME = 'report.yaml'
   - MODEL_EVALUATION_CHANGED_THRESHOLD = 0.02  # Min R² improvement
   - BEST_MODEL_DIR = 'final_model'
   - BEST_MODEL_FILE_NAME = 'model.pkl'
   
   Config:
   - ModelEvaluationConfig: evaluation_dir, report_file_path, change_threshold,
     best_model_dir, best_model_file_path
   
   Artifact:
   - ModelEvaluationArtifact: is_model_accepted, improved_score, 
     best_model_path, trained_model_path, train_metric, best_model_metric
   
   Implementation (flightdelay/components/modelevaluation.py):
   ```python
   # Load test data
   test_arr = load_numpy_array(transformation_artifact.transformed_test_file_path)
   X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
   
   # Load trained model
   trained_model = load_object(model_trainer_artifact.trained_model_file_path)
   trained_pred = trained_model.predict(X_test)
   trained_r2 = r2_score(y_test, trained_pred)

   **Key Methods:**
- `calculate_metrics(y_true, y_pred)` - Calculate R², MAE, RMSE, MSE
- `evaluate_model(model, X_test, y_test)` - Evaluate a model on test data
- `save_evaluation_report(...)` - Save evaluation results as YAML
- `initiate_model_evaluation()` - Main orchestration method
   
  **Workflow:**
1. Load test data from transformation artifact
2. Load newly trained model
3. Evaluate trained model on test data
4. Check if production model exists:
   - **If exists**: Compare models, accept only if R² improvement >= threshold
   - **If not exists**: Accept first model automatically
5. If accepted:
   - Save model to `final_model/model.pkl`
   - Save preprocessor to `final_model/preprocessor.pkl`
6. Generate evaluation report (YAML)
   ```

# 10. Model Pusher Component

   Constants:
   - MODEL_PUSHER_DIR_NAME = 'model_pusher'
   - SAVED_MODEL_DIR = 'saved_models'
   - TRAINING_BUCKET_NAME = 'flightdelay-mlops'
   
   Config:
   - ModelPusherConfig: pusher_dir, saved_model_dir, model_file_path
   
   Artifact:
   - ModelPusherArtifact: is_pushed, model_dir, saved_model_path, s3_model_path
   
   Implementation (flightdelay/components/modelpusher.py):
   ```python
   # Only push if model was accepted
   if not model_evaluation_artifact.is_model_accepted:
       logger.info("Model not accepted, skipping push")
       return ModelPusherArtifact(is_pushed=False, ...)
   
   # Copy to saved_models/ with version
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   saved_model_path = os.path.join(SAVED_MODEL_DIR, timestamp)
   os.makedirs(saved_model_path, exist_ok=True)
   
   shutil.copy(best_model_file_path, os.path.join(saved_model_path, 'model.pkl'))
   shutil.copy(preprocessor_path, os.path.join(saved_model_path, 'preprocessor.pkl'))
   
   # Push to S3/cloud (optional)
   if boto3_available:
       s3_sync.sync_folder_to_s3(
           folder_path=saved_model_path,
           bucket_name=TRAINING_BUCKET_NAME,
           s3_folder_name=f"models/FlightDelay/{timestamp}/"
       )
       logger.info(f"Model pushed to S3: s3://{TRAINING_BUCKET_NAME}/models/FlightDelay/{timestamp}/")
   
   return ModelPusherArtifact(is_pushed=True, ...)
   ```

# 11. MLflow + DVC + DagsHub Tracking

   ## MLflow - Experiment Tracking & Model Registry
   path - flightdelay/utils/ml_utils/mlflow_utils.py
   
   Setup:
   ```python
   import mlflow
   import mlflow.sklearn
   
   # Set tracking URI
   mlflow.set_tracking_uri("https://dagshub.com/<user>/mlops_flight_delay_prediction.mlflow")
   mlflow.set_experiment("FlightDelay_Prediction")
   ```
   
   Integration in ModelTraining:
   ```python
   with mlflow.start_run():
       # Log parameters
       mlflow.log_param("model_name", best_model_name)
       mlflow.log_param("expected_r2", expected_score)
       mlflow.log_param("overfitting_threshold", overfitting_threshold)
       mlflow.log_param("features_count", X_train.shape[1])
       
       # Log metrics
       mlflow.log_metric("train_r2", train_r2)
       mlflow.log_metric("test_r2", test_r2)
       mlflow.log_metric("test_mae", test_mae)
       mlflow.log_metric("test_rmse", test_rmse)
       mlflow.log_metric("test_mape", test_mape)
       
       # Log model
       mlflow.sklearn.log_model(best_model, "model")
       
       # Log artifacts
       mlflow.log_artifact(confusion_matrix_plot)
       mlflow.log_artifact(feature_importance_plot)
   ```
   
   Model Registry:
   ```python
   # Register trained model
   model_uri = f"runs:/{run.info.run_id}/model"
   mlflow.register_model(model_uri, "FlightDelayModel")
   
   # Promote to production if accepted
   if model_evaluation_artifact.is_model_accepted:
       client = mlflow.tracking.MlflowClient()
       client.transition_model_version_stage(
           name="FlightDelayModel",
           version=latest_version,
           stage="Production"
       )
   ```
   
   ## DagsHub - Unified MLOps Hub ✅ DONE
   Repo: https://dagshub.com/chauhan7gaurav/MLOPS_FLIGHT_DELAY_PREDICTION

   ### STEP A — Create DagsHub repo (manual, one-time)
   1. Go to https://dagshub.com → Sign in → New Repository
   2. Name: MLOPS_FLIGHT_DELAY_PREDICTION
   3. Click "Connect a repo" → Choose GitHub → Select mlops_flight_delay_prediction
   4. DagsHub auto-enables:
      - MLflow tracking server  → <repo>.mlflow
      - DVC remote storage      → <repo>.dvc

   ### STEP B — Generate access token (manual, one-time)
   1. DagsHub → top-right avatar → Settings → Access Tokens
   2. Click "Generate new token" → Name it (e.g. FlightDelay_Prediction)
   3. Copy the token immediately (shown only once)
   Token used: 065997e88f5fdedb8f25c0050c2668f6e4a97596  (name: FlightDelay_Prediction)

   ### STEP C — Set environment variables (run every new terminal session)
   ```powershell
   $env:DAGSHUB_USER             = "chauhan7gaurav"
   $env:MLFLOW_TRACKING_URI      = "https://dagshub.com/chauhan7gaurav/MLOPS_FLIGHT_DELAY_PREDICTION.mlflow"
   $env:MLFLOW_TRACKING_USERNAME = "chauhan7gaurav"
   $env:MLFLOW_TRACKING_PASSWORD = "065997e88f5fdedb8f25c0050c2668f6e4a97596"
   ```
   Tip: Add these to a `.env` file and load with python-dotenv so they persist.

   ### STEP D — Init DVC & configure remote (one-time, already done ✅)
   ```powershell
   dvc init -f                          # force if .dvc/ already existed
   dvc remote add origin https://dagshub.com/chauhan7gaurav/MLOPS_FLIGHT_DELAY_PREDICTION.dvc
   dvc remote default origin
   dvc remote modify origin --local auth basic
   dvc remote modify origin --local user chauhan7gaurav
   dvc remote modify origin --local password 065997e88f5fdedb8f25c0050c2668f6e4a97596
   ```
   Note: --local flags write to .dvc/config.local (gitignored) — credentials never committed.

   ### STEP E — Track raw data with DVC (one-time, already done ✅)
   ```powershell
   # Remove from git first (can't be tracked by both)
   git rm -r --cached delay_data/flights_sample.csv delay_data/weather_data.csv
   git commit -m "untrack raw data from git (moving to DVC)"

   # Hand off to DVC
   dvc add delay_data/flights_sample.csv delay_data/weather_data.csv

   # Commit DVC pointer files
   git add "delay_data\flights_sample.csv.dvc" "delay_data\.gitignore" "delay_data\weather_data.csv.dvc" dvc.yaml params.yaml .dvc/config
   git commit -m "track raw data with DVC, add dvc.yaml pipeline"

   # Push data to DagsHub storage
   dvc push
   # → 2 files pushed ✅
   ```

   ### STEP F — Verify MLflow → DagsHub connection (already done ✅)
   ```powershell
   python -c "
   import mlflow, os
   mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
   mlflow.set_experiment('FlightDelay_Prediction')
   with mlflow.start_run(run_name='connectivity_test') as run:
       mlflow.log_param('test', 'dagshub_connection')
       mlflow.log_metric('status', 1.0)
       print('Run ID:', run.info.run_id)
       print('MLflow → DagsHub: SUCCESS')
   "
   # Experiment 'FlightDelay_Prediction' created on DagsHub ✅
   # View at: https://dagshub.com/chauhan7gaurav/MLOPS_FLIGHT_DELAY_PREDICTION/experiments
   ```

   ### STEP G — Run training pipeline (generates real MLflow runs)
   ```powershell
   # Env vars must be set (STEP C) before running
   python training_pipeline.py
   # Each run auto-logs: params, metrics (R², MAE, RMSE), model artifact
   # Model registered as 'FlightDelayModel' in MLflow Registry
   # If accepted → promoted to Production stage automatically
   ```

   ## DVC - Data & Artifact Versioning ✅ DONE
   path - dvc.yaml, .dvc/config

   Pipeline defined in dvc.yaml (single stage wrapping full pipeline):
   ```yaml
   stages:
     training_pipeline:
       cmd: python training_pipeline.py
       deps:
         - delay_data/flights_sample.csv
         - delay_data/airports.csv
         - delay_data/airlines.csv
         - delay_data/holidays.csv
         - delay_data/weather_data.csv
         - flightdelay/components/modeltraining.py
         - flightdelay/components/modelevaluation.py
       params:
         - params.yaml
       outs:
         - final_model:
             persist: true
   ```
   Run pipeline via DVC (tracks deps/outs, skips unchanged stages):
   ```powershell
   dvc repro        # re-runs only changed stages
   dvc push         # push new artifacts to DagsHub
   ```

# 12. Training Pipeline Orchestrator


   path - flightdelay/pipeline/training_pipeline.py
   
   ```python
   class TrainingPipeline:
       def __init__(self):
           self.training_pipeline_config = TrainingPipelineConfig()
       
       def run_pipeline(self):
           try:
               logger.info("="*70)
               logger.info("STARTING TRAINING PIPELINE")
               logger.info("="*70)
               
               # 1. Data Ingestion
               logger.info("\n[Stage 1/8] Data Ingestion")
               data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
               data_ingestion = DataIngestion(data_ingestion_config)
               data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
               
               # 2. Data Validation
               logger.info("\n[Stage 2/8] Data Validation")
               data_validation_config = DataValidationConfig(self.training_pipeline_config)
               data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
               data_validation_artifact = data_validation.initiate_data_validation()
               
               # 3. Data Transformation
               logger.info("\n[Stage 3/8] Data Transformation")
               data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
               data_transformation = DataTransformation(
                   data_ingestion_artifact, 
                   data_validation_artifact,
                   data_transformation_config
               )
               data_transformation_artifact = data_transformation.initiate_data_transformation()
               
               # 4. Model Training
               logger.info("\n[Stage 4/8] Model Training")
               model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
               model_trainer = ModelTrainer(data_transformation_artifact, model_trainer_config)
               model_trainer_artifact = model_trainer.initiate_model_training()
               
               # 5. Model Evaluation
               logger.info("\n[Stage 5/8] Model Evaluation")
               model_evaluation_config = ModelEvaluationConfig(self.training_pipeline_config)
               model_evaluation = ModelEvaluation(
                   model_trainer_artifact,
                   data_transformation_artifact,
                   model_evaluation_config
               )
               model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
               
               # 6. Model Pusher (if accepted)
               logger.info("\n[Stage 6/8] Model Pusher")
               if model_evaluation_artifact.is_model_accepted:
                   model_pusher_config = ModelPusherConfig(self.training_pipeline_config)
                   model_pusher = ModelPusher(model_evaluation_artifact, model_pusher_config)
                   model_pusher_artifact = model_pusher.initiate_model_pushing()
                   
                   # 7. Sync to S3
                   logger.info("\n[Stage 7/8] Sync Artifacts to S3")
                   self.sync_artifact_dir_to_s3()
                   self.sync_saved_model_dir_to_s3()
               else:
                   logger.info("Model not accepted, skipping push and S3 sync")
               
               logger.info("\n" + "="*70)
               logger.info("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
               logger.info("="*70)
               
           except Exception as e:
               logger.error(f"Training pipeline failed: {str(e)}")
               raise CustomException(e, sys)
   ```
   
   Main entry point (main.py):
   ```python
   if __name__ == "__main__":
       try:
           pipeline = TrainingPipeline()
           pipeline.run_pipeline()
       except Exception as e:
           logger.error(f"Pipeline execution failed: {str(e)}")
           raise
   ```

# 13. FastAPI Prediction Service

   path - app.py
   
   ```python
   from fastapi import FastAPI, File, UploadFile, HTTPException
   from fastapi.responses import HTMLResponse
   from fastapi.templating import Jinja2Templates
   from prometheus_client import Counter, Histogram, generate_latest
   import uvicorn
   
   app = FastAPI(title="Flight Delay Prediction API")
   templates = Jinja2Templates(directory="templates")
   
   # Prometheus metrics
   predictions_total = Counter('predictions_total', 'Total predictions', ['model_version'])
   prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
   delay_predictions = Counter('delay_predictions', 'Delay predictions by class', ['delay_class'])
   
   # Load model and preprocessor
   model = load_object("final_model/model.pkl")
   preprocessor = load_object("final_model/preprocessor.pkl")
   network_model = FlightDelayModel(preprocessor, model)
   
   @app.get("/")
   async def root():
       return RedirectResponse(url="/docs")
   
   @app.get("/health")
   async def health():
       return {"status": "healthy", "model_loaded": model is not None}
   
   @app.get("/train")
   async def train_model():
       try:
           pipeline = TrainingPipeline()
           pipeline.run_pipeline()
           return {"message": "Training completed successfully"}
       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))
   
   @app.post("/predict")
   async def predict(file: UploadFile = File(...)):
       start_time = time.time()
       
       try:
           # Read CSV
           df = pd.read_csv(file.file)
           
           # Predict
           predictions = network_model.predict(df)
           df['PREDICTED_DELAY'] = predictions
           df['DELAY_CLASS'] = (predictions > 15).astype(int)
           
           # Track metrics
           predictions_total.labels(model_version='v1.0').inc(len(df))
           prediction_latency.observe(time.time() - start_time)
           delay_predictions.labels(delay_class='delayed').inc((df['DELAY_CLASS'] == 1).sum())
           delay_predictions.labels(delay_class='ontime').inc((df['DELAY_CLASS'] == 0).sum())
           
           # Save predictions
           output_path = "prediction_output/predictions.csv"
           df.to_csv(output_path, index=False)
           
           # Render HTML table
           return templates.TemplateResponse("table.html", {
               "request": {},
               "data": df.head(100).to_dict('records'),
               "total_flights": len(df),
               "delayed_count": (df['DELAY_CLASS'] == 1).sum(),
               "ontime_count": (df['DELAY_CLASS'] == 0).sum()
           })
           
       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))
   
   @app.get("/metrics")
   async def metrics():
       return Response(content=generate_latest(), media_type="text/plain")
   
   @app.post("/feedback")
   async def submit_feedback(request: FeedbackRequest):
       # Store ground truth for retraining
       feedback_collector.update_ground_truth(
           request_id=request.request_id,
           actual_delay=request.actual_delay,
           user_feedback=request.user_feedback
       )
       
       should_retrain = feedback_collector.should_trigger_retraining()
       return {"status": "success", "should_retrain": should_retrain}
   
   @app.post("/webhook/retrain")
   async def retrain_webhook(alert: AlertWebhook):
       # Triggered by Grafana drift alert
       logger.info(f"Retraining triggered by alert: {alert.alertname}")
       
       # Spawn background retraining
       threading.Thread(target=trigger_retraining, args=(alert.reason,)).start()
       
       return {"status": "retraining_initiated", "reason": alert.reason}
   
   if __name__ == "__main__":
       uvicorn.run(app, host="0.0.0.0", port=8000)
   ```
   
   FlightDelayModel wrapper (flightdelay/utils/ml_utils/model/estimator.py):
   ```python
   class FlightDelayModel:
       def __init__(self, preprocessor, model):
           self.preprocessor = preprocessor
           self.model = model
       
       def predict(self, dataframe):
           # Apply preprocessing
           transformed = self.preprocessor.transform(dataframe)
           
           # Predict
           predictions = self.model.predict(transformed)
           
           return predictions
   ```

# 14. Model Drift Monitoring (Prometheus + Grafana)

   ## Prometheus Metrics (flightdelay/utils/main_utils/prometheus_utils.py)
   
   Custom metrics:
   ```python
   from prometheus_client import Counter, Gauge, Histogram
   
   # Prediction metrics
   predictions_total = Counter('flight_predictions_total', 'Total predictions', ['model_version', '=endpoint'])
   prediction_latency = Histogram('flight_prediction_latency_seconds', 'Prediction latency')
   delay_class_predictions = Counter('flight_delay_predictions', 'Predictions by delay class', ['delay_class'])
   
   # Drift metrics
   model_drift_score = Gauge('flight_model_drift_score', 'Model drift score (PSI/KS)')
   rolling_mae = Gauge('flight_rolling_mae', 'Rolling MAE when ground truth available')
   rolling_r2 = Gauge('flight_rolling_r2', 'Rolling R² score')
   
   # Error tracking
   model_errors = Counter('flight_model_errors', 'Model errors', ['error_type'])
   
   # Data quality
   data_quality_score = Gauge('flight_data_quality_score', 'Data quality score')
   ```
   
   ## Drift Detection (flightdelay/components/modelmonitoring.py)
   ```python
   class ModelMonitor:
       def __init__(self):
           self.baseline_stats = load_baseline_stats()
       
       def calculate_drift_score(self, production_data):
           # PSI (Population Stability Index)
           psi_scores = {}
           for feature in numerical_features:
               expected = self.baseline_stats[feature]['distribution']
               actual = production_data[feature].value_counts(normalize=True)
               psi = calculate_psi(expected, actual)
               psi_scores[feature] = psi
           
           # Aggregate drift score
           drift_score = np.mean(list(psi_scores.values()))
           model_drift_score.set(drift_score)
           
           return drift_score
       
       def check_accuracy_drop(self, feedback_data):
           if len(feedback_data) < 100:
               return False
           
           predictions = feedback_data['prediction']
           actuals = feedback_data['actual_delay']
           
           mae = mean_absolute_error(actuals, predictions)
           r2 = r2_score(actuals, predictions)
           
           rolling_mae.set(mae)
           rolling_r2.set(r2)
           
           baseline_r2 = self.baseline_stats['r2']
           return r2 < (baseline_r2 - 0.05)  # 5% drop threshold
   ```
   
   ## Kubernetes ServiceMonitor (k8s/servicemonitor.yaml)
   ```yaml
   apiVersion: monitoring.coreos.com/v1
   kind: ServiceMonitor
   metadata:
     name: flight-delay-api-monitor
     namespace: flight-delay
   spec:
     selector:
       matchLabels:
         app: flight-delay-api
     endpoints:
       - port: http
         path: /metrics
         interval: 15s
   ```
   
   ## PrometheusRule - Alerts (k8s/prometheusrule.yaml)
   ```yaml
   apiVersion: monitoring.coreos.com/v1
   kind: PrometheusRule
   metadata:
     name: flight-delay-alerts
     namespace: flight-delay
   spec:
     groups:
       - name: model_alerts
         rules:
           - alert: DataDriftDetected
             expr: flight_model_drift_score > 0.5
             for: 10m
             severity: critical
             annotations:
               summary: "Data drift detected"
               description: "Drift score {{ $value }} exceeds threshold"
             labels:
               action: trigger_retraining
           
           - alert: ModelAccuracyDrop
             expr: flight_rolling_r2 < 0.70
             for: 15m
             severity: critical
             annotations:
               summary: "Model accuracy dropped"
             labels:
               action: trigger_retraining
           
           - alert: HighPredictionLatency
             expr: histogram_quantile(0.99, rate(flight_prediction_latency_seconds_bucket[5m])) > 1.0
             for: 5m
             severity: warning
   ```
   
   ## Grafana Dashboards
   Access: `kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80`
   
   Panels:
   - Prediction volume over time
   - Latency distribution (p50, p95, p99)
   - Delay predictions ratio (delayed vs on-time)
   - Model drift score timeline
   - Rolling MAE/R² (when feedback available)
   - Error rates by type

# 15. Docker & Kubernetes Deployment

   ## Dockerfile (Multi-stage)
   ```dockerfile
   FROM python:3.11-slim AS builder
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   FROM python:3.11-slim
   WORKDIR /app
   COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
   COPY . .
   
   RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
   USER appuser
   
   COPY docker-entrypoint.sh /usr/local/bin/
   ENTRYPOINT ["docker-entrypoint.sh"]
   CMD ["api"]
   ```
   
   ## docker-entrypoint.sh
   ```bash
   #!/bin/bash
   case "$1" in
     api)    uvicorn app:app --host 0.0.0.0 --port 8000 ;;
     train)  python main.py ;;
     test)   pytest tests/ ;;
     retrain) python scheduled_retrain.py ;;
     *)      exec "$@" ;;
   esac
   ```
   
   ## Kubernetes Deployment (k8s/deployment.yaml)
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: flight-delay-api
     namespace: flight-delay
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: flight-delay-api
     template:
       metadata:
         labels:
           app: flight-delay-api
       spec:
         containers:
           - name: api
             image: <registry>/flight-delay-api:latest
             ports:
               - containerPort: 8000
             livenessProbe:
               httpGet:
                 path: /health
                 port: 8000
               initialDelaySeconds: 30
               periodSeconds: 10
             readinessProbe:
               httpGet:
                 path: /health
                 port: 8000
               initialDelaySeconds: 10
               periodSeconds: 5
             resources:
               requests:
                 cpu: "500m"
                 memory: "1Gi"
               limits:
                 cpu: "2"
                 memory: "4Gi"
             volumeMounts:
               - name: model-data
                 mountPath: /app/final_model
         volumes:
           - name: model-data
             persistentVolumeClaim:
               claimName: model-data-pvc
   ```
   
   ## HPA (Horizontal Pod Autoscaler)
   ```yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: flight-delay-hpa
     namespace: flight-delay
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: flight-delay-api
     minReplicas: 3
     maxReplicas: 20
     metrics:
       - type: Resource
         resource:
           name: cpu
           target:
             type: Utilization
             averageUtilization: 70
       - type: Resource
         resource:
           name: memory
           target:
             type: Utilization
             averageUtilization: 80
   ```
   
   ## CronJob for Scheduled Training
   ```yaml
   apiVersion: batch/v1
   kind: CronJob
   metadata:
     name: flight-delay-training
     namespace: flight-delay
   spec:
     schedule: "0 2 * * 0"  # Sunday 2 AM
     jobTemplate:
       spec:
         template:
           spec:
             containers:
               - name: trainer
                 image: <registry>/flight-delay-api:latest
                 command: ["docker-entrypoint.sh", "train"]
                 resources:
                   requests:
                     cpu: "4"
                     memory: "16Gi"
             restartPolicy: OnFailure
   ```

# 16. CI/CD Pipeline (GitHub Actions)

   ## CI Workflow (.github/workflows/ci.yaml)
   ```yaml
   name: CI
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - uses: actions/setup-python@v4
           with:
             python-version: '3.11'
         - run: pip install -r requirements.txt
         - run: pytest tests/ --cov=flightdelay
         - run: flake8 flightdelay/
     
     security:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - run: pip install safety bandit
         - run: safety check
         - run: bandit -r flightdelay/
   ```
   
   ## CD Workflow (.github/workflows/cd.yaml)
   ```yaml
   name: CD
   on:
     push:
       branches: [main]
       tags: ['v*.*.*']
   jobs:
     build-and-push:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - uses: docker/login-action@v2
           with:
             username: ${{ secrets.DOCKER_USERNAME }}
             password: ${{ secrets.DOCKER_PASSWORD }}
         - uses: docker/build-push-action@v4
           with:
             push: true
             tags: <registry>/flight-delay-api:latest, <registry>/flight-delay-api:${{ github.sha }}
     
     deploy-staging:
       needs: build-and-push
       runs-on: ubuntu-latest
       steps:
         - uses: aws-actions/configure-aws-credentials@v2
         - run: aws eks update-kubeconfig --name flight-delay-staging
         - run: |
             helm upgrade --install flight-delay-api ./helm/flight-delay \
               --namespace flight-delay \
               --values ./helm/flight-delay/values-staging.yaml \
               --set image.tag=${{ github.sha }}
     
     deploy-production:
       needs: deploy-staging
       if: startsWith(github.ref, 'refs/tags/')
       runs-on: ubuntu-latest
       steps:
         - run: |
             helm upgrade --install flight-delay-api ./helm/flight-delay \
               --namespace flight-delay \
               --values ./helm/flight-delay/values-production.yaml \
               --set image.tag=${{ github.sha }}
   ```
   
   ## Training Pipeline Workflow (.github/workflows/training_pipeline.yaml)
   ```yaml
   name: Training Pipeline
   on:
     schedule:
       - cron: '0 2 * * 0'  # Weekly Sunday 2 AM
     workflow_dispatch:  # Manual trigger
     repository_dispatch:
       types: [drift-alert]  # Webhook from Grafana
   jobs:
     train:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
        - uses: iterative/setup-dvc@v1
         - run: dvc pull
         - run: python main.py
         - run: dvc add final_model/
         - run: |
             git add final_model.dvc
             git commit -m "chore: retrained model"
             git push
         - run: dvc push
         - uses: act10ns/slack@v1
           with:
             message: 'Model retraining completed'
   ```

# 17. Model Retraining System

   ## Retraining Triggers (4 types)
   
   1. **Scheduled**: CronJob (weekly) or GitHub Actions schedule
   2. **Drift-based**: Grafana alert → Webhook → /webhook/retrain → Spawn training
   3. **Feedback-based**: When labeled samples > threshold
   4. **Manual**: POST /manual-retrain or workflow_dispatch
   
   ## Feedback Collector (flightdelay/components/feedback_collector.py)
   ```python
   class FeedbackCollector:
       def __init__(self, db_path="feedback_data/feedback.db"):
           self.conn = sqlite3.connect(db_path)
           self.create_table()
       
       def store_prediction(self, request_id, features, prediction, model_version):
           # Store prediction with features
           pass
       
       def update_ground_truth(self, request_id, actual_delay):
           # Update with actual delay when available
           pass
       
       def get_labeled_data(self):
           # Return all predictions with ground truth
           return pd.read_sql("SELECT * FROM feedback WHERE actual_delay IS NOT NULL", self.conn)
       
       def should_trigger_retraining(self):
           labeled = self.get_labeled_data()
           
           # Check thresholds
           if len(labeled) < 100:
               return False
           
           # Check accuracy drop
           mae = mean_absolute_error(labeled['actual_delay'], labeled['prediction'])
           baseline_mae = 10.0  # baseline
           
           return mae > (baseline_mae * 1.2)  # 20% worse
   ```
   
   ## Retraining Manager (flightdelay/pipeline/retraining_manager.py)
   ```python
   class RetrainingManager:
       def trigger_retraining(self, reason="manual"):
           logger.info(f"Retraining triggered: {reason}")
           
           # Prepare data
           feedback_data = feedback_collector.get_labeled_data()
           training_data = self.merge_with_historical(feedback_data)
           
           # Version with DVC
           save_data(training_data, "training_data.csv")
           os.system("dvc add training_data.csv")
           os.system("git add training_data.csv.dvc")
           os.system(f"git commit -m 'retrain: {reason}'")
           
           # Run training pipeline
           pipeline = TrainingPipeline()
           pipeline.run_pipeline()
           
           # Notify
           send_slack_notification(f"Retraining completed. Reason: {reason}")
   ```

# 18. Documentation & Deployment Guides

   Create these files:
   - README.md - Project overview and quickstart
   - K8S_DEPLOYMENT.md - Kubernetes deployment guide
   - MONITORING_SETUP.md - Prometheus/Grafana setup
   - RETRAINING_GUIDE.md - How retraining works
   - API_DOCUMENTATION.md - FastAPI endpoints

# 19. Final Checklist

   ✅ Setup & Infrastructure:
   - [x] Logger and exception handling
   - [x] Constants and config entities
   - [x] Virtual environment and requirements.txt
   - [ ] .env file for secrets
   
   ✅ Data Pipeline:
   - [x] Schema validation
   - [x] Data ingestion
   - [ ] Data validation and drift detection
   - [ ] Data transformation and joins
   - [ ] Feature engineering
   
   ✅ Model Pipeline:
   - [ ] Model training
   - [ ] Model evaluation
   - [ ] Model pusher
   - [ ] Training pipeline orchestrator
   
   ✅ Tracking & Versioning:
   - [ ] MLflow experiment tracking
   - [ ] DVC data versioning
   - [ ] DagsHub integration
   - [ ] Model registry
   
   ✅ Deployment & Serving:
   - [ ] FastAPI prediction service
   - [ ] Docker containerization
   - [ ] Kubernetes deployment
   - [ ] Helm charts
   
   ✅ Monitoring & Retraining:
   - [ ] Prometheus metrics
   - [ ] Grafana dashboards
   - [ ] Drift detection
   - [ ] Feedback collection
   - [ ] Automated retraining
   
   ✅ CI/CD:
   - [ ] GitHub Actions CI workflow
   - [ ] GitHub Actions CD workflow
   - [ ] Training pipeline automation
   
   ✅ Documentation:
   - [ ] README and setup guides
   - [ ] API documentation
   - [ ] Deployment guides

**Next Steps:**
1. Test data ingestion: ✅ DONE
2. Implement data validation component
3. Implement data transformation and joins
4. Implement feature engineering
5. Implement model training
6. Set up MLflow tracking
7. Create FastAPI prediction service
8. Deploy to Kubernetes
9. Set up monitoring
10. Implement retraining system
   ```