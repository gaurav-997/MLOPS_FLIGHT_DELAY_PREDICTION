import os
import sys
import pandas as pd
import numpy as np
import yaml
from scipy.stats import ks_2samp
from typing import Dict, List
from datetime import datetime
from flightdelay.logging.logger import logger
from flightdelay.exception.exception import CustomException
from flightdelay.entity.config_entity import DataValidationConfig
from flightdelay.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_errors = []
            self.drift_report = {}
        except Exception as e:
            raise CustomException(e, sys)

    def load_data(self, filepath: str) -> pd.DataFrame:
        try:
            logger.info(f"Loading data from: {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def check_required_columns(self, df: pd.DataFrame, df_name: str, required_cols: List[str]) -> Dict:
        """Check if all required columns from schema validation are present"""
        try:
            logger.info(f"Checking required columns in {df_name}")

            existing_columns = set(df.columns)
            required_columns = set(required_cols)
            missing_columns = required_columns - existing_columns
            extra_columns = existing_columns - required_columns

            column_check = {
                "dataset": df_name,
                "required_columns_count": len(required_columns),
                "existing_columns_count": len(existing_columns),
                "missing_columns": list(missing_columns),
                "missing_columns_count": len(missing_columns),
                "extra_columns": list(extra_columns),
                "all_required_present": len(missing_columns) == 0
            }

            if missing_columns:
                error_msg = f"{df_name} is missing required columns: {list(missing_columns)}"
                self.validation_errors.append(error_msg)
                logger.error(error_msg)
            else:
                logger.info(f"  [PASS] All {len(required_columns)} required columns present in {df_name}")

            if extra_columns:
                logger.info(f"  Note: {df_name} has {len(extra_columns)} additional columns beyond schema")

            return column_check

        except Exception as e:
            raise CustomException(e, sys)

    def check_missing_values(self, df: pd.DataFrame, df_name: str, critical_cols: List[str] = None) -> Dict:
        """Check for missing values with special attention to critical columns"""
        try:
            logger.info(f"Checking missing values in {df_name}")

            missing_summary = {
                "total_rows": len(df),
                "columns_with_missing": {},
                "critical_missing": {}
            }

            # Check all columns
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    missing_pct = (missing_count / len(df)) * 100
                    missing_summary["columns_with_missing"][col] = {
                        "count": int(missing_count),
                        "percentage": round(missing_pct, 2)
                    }
                    logger.warning(f"  {df_name}.{col}: {missing_count} missing ({missing_pct:.2f}%)")

            # Check critical columns
            if critical_cols:
                for col in critical_cols:
                    if col in df.columns:
                        missing_count = df[col].isnull().sum()
                        if missing_count > 0:
                            missing_summary["critical_missing"][col] = int(missing_count)
                            self.validation_errors.append(
                                f"CRITICAL: {df_name}.{col} has {missing_count} missing values"
                            )

            return missing_summary

        except Exception as e:
            raise CustomException(e, sys)

    def check_duplicates(self, df: pd.DataFrame, df_name: str) -> Dict:
        """Check for duplicate rows"""
        try:
            duplicate_count = df.duplicated().sum()
            duplicate_summary = {
                "total_rows": len(df),
                "duplicate_count": int(duplicate_count),
                "duplicate_percentage": round((duplicate_count / len(df)) * 100, 2) if len(df) > 0 else 0
            }

            if duplicate_count > 0:
                logger.warning(f"{df_name} has {duplicate_count} duplicate rows ({duplicate_summary['duplicate_percentage']}%)")
            else:
                logger.info(f"{df_name} has no duplicate rows")

            return duplicate_summary

        except Exception as e:
            raise CustomException(e, sys)

    def validate_airport_codes(self, flights_df: pd.DataFrame, airports_df: pd.DataFrame) -> Dict:
        """Validate that all airport codes in flights exist in airports reference"""
        try:
            logger.info("Validating airport codes...")

            valid_airports = set(airports_df['IATA_CODE'].unique())
            origin_airports = set(flights_df['ORIGIN_AIRPORT'].unique())
            dest_airports = set(flights_df['DESTINATION_AIRPORT'].unique())

            invalid_origins = origin_airports - valid_airports
            invalid_dests = dest_airports - valid_airports

            validation_summary = {
                "valid_airports_count": len(valid_airports),
                "unique_origin_airports": len(origin_airports),
                "unique_dest_airports": len(dest_airports),
                "invalid_origins": list(invalid_origins),
                "invalid_destinations": list(invalid_dests),
                "invalid_origin_count": len(invalid_origins),
                "invalid_dest_count": len(invalid_dests)
            }

            if invalid_origins:
                logger.warning(f"Found {len(invalid_origins)} invalid origin airports: {list(invalid_origins)[:10]}")
                self.validation_errors.append(f"Invalid origin airports: {len(invalid_origins)} codes not in reference")

            if invalid_dests:
                logger.warning(f"Found {len(invalid_dests)} invalid destination airports: {list(invalid_dests)[:10]}")
                self.validation_errors.append(f"Invalid destination airports: {len(invalid_dests)} codes not in reference")

            return validation_summary

        except Exception as e:
            raise CustomException(e, sys)

    def validate_date_ranges(self, flights_df: pd.DataFrame) -> Dict:
        """Validate flight dates are in reasonable ranges"""
        try:
            logger.info("Validating date ranges...")

            # Check year range
            year_min = flights_df['YEAR'].min()
            year_max = flights_df['YEAR'].max()

            # Check month range
            month_min = flights_df['MONTH'].min()
            month_max = flights_df['MONTH'].max()

            # Check day range
            day_min = flights_df['DAY'].min()
            day_max = flights_df['DAY'].max()

            validation_summary = {
                "year_range": f"{year_min}-{year_max}",
                "month_range": f"{month_min}-{month_max}",
                "day_range": f"{day_min}-{day_max}",
                "valid_year_range": bool(2000 <= year_min and year_max <= 2030),
                "valid_month_range": bool(1 <= month_min and month_max <= 12),
                "valid_day_range": bool(1 <= day_min and day_max <= 31)
            }

            # Validate ranges
            if not (2000 <= year_min and year_max <= 2030):
                self.validation_errors.append(f"Invalid year range: {year_min}-{year_max} (expected 2000-2030)")

            if not (1 <= month_min and month_max <= 12):
                self.validation_errors.append(f"Invalid month range: {month_min}-{month_max} (expected 1-12)")

            if not (1 <= day_min and day_max <= 31):
                self.validation_errors.append(f"Invalid day range: {day_min}-{day_max} (expected 1-31)")

            logger.info(f"Date range: {year_min}-{year_max}, Months: {month_min}-{month_max}, Days: {day_min}-{day_max}")

            return validation_summary

        except Exception as e:
            raise CustomException(e, sys)

    def validate_coordinates(self, airports_df: pd.DataFrame) -> Dict:
        """Validate latitude and longitude are in valid ranges"""
        try:
            logger.info("Validating geographic coordinates...")

            invalid_lat = airports_df[(airports_df['LATITUDE'] < -90) | (airports_df['LATITUDE'] > 90)]
            invalid_lon = airports_df[(airports_df['LONGITUDE'] < -180) | (airports_df['LONGITUDE'] > 180)]

            validation_summary = {
                "total_airports": len(airports_df),
                "invalid_latitude_count": len(invalid_lat),
                "invalid_longitude_count": len(invalid_lon),
                "latitude_range": f"{airports_df['LATITUDE'].min():.2f} to {airports_df['LATITUDE'].max():.2f}",
                "longitude_range": f"{airports_df['LONGITUDE'].min():.2f} to {airports_df['LONGITUDE'].max():.2f}"
            }

            if len(invalid_lat) > 0:
                self.validation_errors.append(f"Invalid latitude values: {len(invalid_lat)} airports")
                logger.warning(f"Found {len(invalid_lat)} airports with invalid latitude")

            if len(invalid_lon) > 0:
                self.validation_errors.append(f"Invalid longitude values: {len(invalid_lon)} airports")
                logger.warning(f"Found {len(invalid_lon)} airports with invalid longitude")

            logger.info(f"Coordinate ranges - Lat: {validation_summary['latitude_range']}, Lon: {validation_summary['longitude_range']}")

            return validation_summary

        except Exception as e:
            raise CustomException(e, sys)

    def detect_data_drift_numerical(
            self, reference_df: pd.DataFrame, current_df: pd.DataFrame,
            column: str, threshold: float = 0.05) -> Dict:
        """Detect drift in numerical column using Kolmogorov-Smirnov test"""
        try:
            # Remove NaN values
            reference_data = reference_df[column].dropna()
            current_data = current_df[column].dropna()

            # Perform KS test
            ks_statistic, p_value = ks_2samp(reference_data, current_data)

            # Drift detected if p-value < threshold
            drift_detected = p_value < threshold

            drift_info = {
                "column": column,
                "type": "numerical",
                "ks_statistic": round(float(ks_statistic), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": bool(drift_detected),
                "threshold": threshold,
                "reference_mean": round(float(reference_data.mean()), 2),
                "current_mean": round(float(current_data.mean()), 2),
                "reference_std": round(float(reference_data.std()), 2),
                "current_std": round(float(current_data.std()), 2)
            }

            if drift_detected:
                logger.warning(f"Drift detected in {column}: KS={ks_statistic:.4f}, p={p_value:.4f}")

            return drift_info

        except Exception as e:
            logger.error(f"Error detecting drift in {column}: {str(e)}")
            return {"column": column, "error": str(e)}

    def detect_data_drift_categorical(
            self, reference_df: pd.DataFrame, current_df: pd.DataFrame,
            column: str, threshold: float = 0.1) -> Dict:
        """Detect drift in categorical column by comparing value distributions"""
        try:
            # Get value distributions
            reference_dist = reference_df[column].value_counts(normalize=True)
            current_dist = current_df[column].value_counts(normalize=True)

            # Calculate Chi-square-like distance
            all_categories = set(reference_dist.index) | set(current_dist.index)

            distance = 0
            for category in all_categories:
                ref_freq = reference_dist.get(category, 0)
                curr_freq = current_dist.get(category, 0)
                distance += abs(ref_freq - curr_freq)

            # Normalize distance
            drift_score = distance / 2  # Divide by 2 to get range [0, 1]
            drift_detected = drift_score > threshold

            drift_info = {
                "column": column,
                "type": "categorical",
                "drift_score": round(float(drift_score), 4),
                "drift_detected": bool(drift_detected),
                "threshold": threshold,
                "reference_unique_values": len(reference_dist),
                "current_unique_values": len(current_dist),
                "new_categories": list(set(current_dist.index) - set(reference_dist.index)),
                "missing_categories": list(set(reference_dist.index) - set(current_dist.index))
            }

            if drift_detected:
                logger.warning(f"Drift detected in categorical {column}: score={drift_score:.4f}")

            return drift_info

        except Exception as e:
            logger.error(f"Error detecting categorical drift in {column}: {str(e)}")
            return {"column": column, "error": str(e)}

    def perform_drift_detection(
            self, reference_df: pd.DataFrame, current_df: pd.DataFrame,
            df_name: str) -> Dict:
        """Perform comprehensive drift detection on all columns"""
        try:
            logger.info(f"Performing drift detection on {df_name}...")

            drift_results = {
                "dataset": df_name,
                "timestamp": datetime.now().isoformat(),
                "numerical_columns": {},
                "categorical_columns": {},
                "overall_drift_score": 0.0,
                "columns_with_drift": []
            }

            # Identify column types
            numerical_cols = reference_df.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = reference_df.select_dtypes(include=['object']).columns

            drift_scores = []

            # Check numerical columns
            for col in numerical_cols:
                if col in current_df.columns:
                    drift_info = self.detect_data_drift_numerical(reference_df, current_df, col)
                    drift_results["numerical_columns"][col] = drift_info

                    if drift_info.get("drift_detected"):
                        drift_results["columns_with_drift"].append(col)
                        drift_scores.append(drift_info.get("ks_statistic", 0))

            # Check categorical columns (limit to important ones to avoid overhead)
            important_categorical = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
                                     'DAY_OF_WEEK', 'IATA_CODE']

            for col in categorical_cols:
                if col in important_categorical and col in current_df.columns:
                    drift_info = self.detect_data_drift_categorical(reference_df, current_df, col)
                    drift_results["categorical_columns"][col] = drift_info

                    if drift_info.get("drift_detected"):
                        drift_results["columns_with_drift"].append(col)
                        drift_scores.append(drift_info.get("drift_score", 0))

            # Calculate overall drift score
            if drift_scores:
                drift_results["overall_drift_score"] = round(float(np.mean(drift_scores)), 4)

            logger.info(f"Drift detection completed for {df_name}. Overall score: {drift_results['overall_drift_score']:.4f}")
            logger.info(f"Columns with drift: {len(drift_results['columns_with_drift'])}")

            return drift_results

        except Exception as e:
            raise CustomException(e, sys)

    def generate_drift_report(self, drift_data: Dict):
        """Generate YAML drift report"""
        try:
            logger.info("Generating drift report...")

            # Create directory if not exists
            os.makedirs(os.path.dirname(self.data_validation_config.data_drift_report_file_path), exist_ok=True)

            # Write YAML report
            with open(self.data_validation_config.data_drift_report_file_path, 'w') as file:
                yaml.dump(drift_data, file, default_flow_style=False, sort_keys=False)

            logger.info(f"Drift report saved to: {self.data_validation_config.data_drift_report_file_path}")

        except Exception as e:
            raise CustomException(e, sys)

    def save_validated_data(self, df: pd.DataFrame, filename: str, is_valid: bool):
        """Save data to validated or invalidated directory"""
        try:
            if is_valid:
                save_dir = self.data_validation_config.data_validation_valid_dir
            else:
                save_dir = self.data_validation_config.data_validation_invalid_dir

            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)

            df.to_csv(save_path, index=False)
            logger.info(f"Saved {'validated' if is_valid else 'invalidated'} data to: {save_path}")

            return save_path

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info("="*70)
            logger.info("STARTING DATA VALIDATION PROCESS")
            logger.info("="*70)

            # Step 1: Load ingested data
            print("\n[Step 1/4] Loading ingested data for validation...")
            flights_df = self.load_data(self.data_ingestion_artifact.ingested_flights_path)
            airports_df = self.load_data(self.data_ingestion_artifact.ingested_airports_path)
            airlines_df = self.load_data(self.data_ingestion_artifact.ingested_airlines_path)
            holidays_df = self.load_data(self.data_ingestion_artifact.ingested_holidays_path)
            weather_df = self.load_data(self.data_ingestion_artifact.ingested_weather_path)
            print("  [PASS] All ingested data loaded successfully")

            # Step 2: Data Quality Checks
            print("\n[Step 2/4] Performing data quality checks...")
            quality_report = {
                "validation_timestamp": datetime.now().isoformat(),
                "data_quality_checks": {}
            }

            # Required columns check (from schema_validation.py)
            print("  [2.0] Checking required columns presence...")

            required_flights_cols = [
                "YEAR", "MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER",
                "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE",
                "DEPARTURE_TIME", "DEPARTURE_DELAY", "SCHEDULED_ARRIVAL",
                "ARRIVAL_TIME", "ARRIVAL_DELAY", "DISTANCE"
            ]
            required_airports_cols = [
                "IATA_CODE", "AIRPORT", "CITY", "STATE", "COUNTRY",
                "LATITUDE", "LONGITUDE"
            ]
            required_airlines_cols = ["IATA_CODE", "AIRLINE"]
            required_holidays_cols = ["date", "holiday_name", "is_holiday"]
            required_weather_cols = ["date", "AWND", "TMAX", "TMIN", "PRCP"]

            quality_report["data_quality_checks"]["flights_columns"] = self.check_required_columns(
                flights_df, "flights", required_flights_cols
            )
            quality_report["data_quality_checks"]["airports_columns"] = self.check_required_columns(
                airports_df, "airports", required_airports_cols
            )
            quality_report["data_quality_checks"]["airlines_columns"] = self.check_required_columns(
                airlines_df, "airlines", required_airlines_cols
            )
            quality_report["data_quality_checks"]["holidays_columns"] = self.check_required_columns(
                holidays_df, "holidays", required_holidays_cols
            )
            quality_report["data_quality_checks"]["weather_columns"] = self.check_required_columns(
                weather_df, "weather", required_weather_cols
            )

            # Missing values check
            print("  [2.1] Checking missing values...")
            quality_report["data_quality_checks"]["flights_missing"] = self.check_missing_values(
                flights_df, "flights", critical_cols=['ARRIVAL_DELAY', 'DEPARTURE_DELAY']
            )
            quality_report["data_quality_checks"]["airports_missing"] = self.check_missing_values(
                airports_df, "airports", critical_cols=['LATITUDE', 'LONGITUDE']
            )
            quality_report["data_quality_checks"]["weather_missing"] = self.check_missing_values(
                weather_df, "weather", critical_cols=['TMAX', 'TMIN']
            )

            # Duplicates check
            print("  [2.2] Checking duplicate rows...")
            quality_report["data_quality_checks"]["flights_duplicates"] = self.check_duplicates(flights_df, "flights")
            quality_report["data_quality_checks"]["airports_duplicates"] = self.check_duplicates(airports_df, "airports")

            # Airport codes validation
            print("  [2.3] Validating airport codes...")
            quality_report["data_quality_checks"]["airport_codes"] = self.validate_airport_codes(flights_df, airports_df)

            # Date range validation
            print("  [2.4] Validating date ranges...")
            quality_report["data_quality_checks"]["date_ranges"] = self.validate_date_ranges(flights_df)

            # Coordinate validation
            print("  [2.5] Validating coordinates...")
            quality_report["data_quality_checks"]["coordinates"] = self.validate_coordinates(airports_df)

            print("  [PASS] Data quality checks completed")

            # Step 3: Data Drift Detection
            print("\n[Step 3/4] Performing data drift detection...")

            # For drift detection, compare first half vs second half of flights data
            # In production, you'd compare against baseline/reference data
            mid_point = len(flights_df) // 2
            reference_flights = flights_df.iloc[:mid_point]
            current_flights = flights_df.iloc[mid_point:]

            print("  [3.1] Detecting drift in flights data...")
            flights_drift = self.perform_drift_detection(reference_flights, current_flights, "flights")

            print("  [3.2] Detecting drift in weather data...")
            if len(weather_df) > 10:
                mid_weather = len(weather_df) // 2
                reference_weather = weather_df.iloc[:mid_weather]
                current_weather = weather_df.iloc[mid_weather:]
                weather_drift = self.perform_drift_detection(reference_weather, current_weather, "weather")
            else:
                weather_drift = {"dataset": "weather", "message": "Insufficient data for drift detection"}

            # Combine drift report
            drift_report = {
                "validation_timestamp": datetime.now().isoformat(),
                "drift_detection": {
                    "flights": flights_drift,
                    "weather": weather_drift
                },
                "summary": {
                    "total_columns_checked": len(flights_drift.get("numerical_columns", {})) + len(flights_drift.get("categorical_columns", {})),
                    "columns_with_drift": flights_drift.get("columns_with_drift", []),
                    "overall_drift_score": flights_drift.get("overall_drift_score", 0.0),
                    "drift_threshold_exceeded": flights_drift.get("overall_drift_score", 0.0) > 0.3
                }
            }

            print(f"  [PASS] Drift detection completed. Overall score: {drift_report['summary']['overall_drift_score']:.4f}")

            # Step 4: Generate Reports and Save Validated Data
            print("\n[Step 4/4] Generating reports and saving validated data...")

            # Generate drift report
            self.generate_drift_report(drift_report)
            print(f"  [PASS] Drift report saved: {self.data_validation_config.data_drift_report_file_path}")

            # Determine validation status
            validation_status = len(self.validation_errors) == 0

            if validation_status:
                print("\n  [PASS] Data validation PASSED")
                # Save to validated directory
                validated_flights_path = self.save_validated_data(flights_df, "flights.csv", True)
                validated_airports_path = self.save_validated_data(airports_df, "airports.csv", True)
                validated_airlines_path = self.save_validated_data(airlines_df, "airlines.csv", True)
                validated_holidays_path = self.save_validated_data(holidays_df, "holidays.csv", True)
                validated_weather_path = self.save_validated_data(weather_df, "weather.csv", True)
            else:
                print(f"\n  [FAIL] Data validation FAILED with {len(self.validation_errors)} errors:")
                for error in self.validation_errors:
                    print(f"    - {error}")
                # Save to invalidated directory
                validated_flights_path = self.save_validated_data(flights_df, "flights.csv", False)
                validated_airports_path = self.save_validated_data(airports_df, "airports.csv", False)
                validated_airlines_path = self.save_validated_data(airlines_df, "airlines.csv", False)
                validated_holidays_path = self.save_validated_data(holidays_df, "holidays.csv", False)
                validated_weather_path = self.save_validated_data(weather_df, "weather.csv", False)

            # Create artifact
            data_validation_artifact = DataValidationArtifact(
                validated_flights_dir=validated_flights_path,
                data_validation_status=validation_status,
                validated_airports_dir=validated_airports_path,
                validated_airlines_dir=validated_airlines_path,
                validated_holidays_dir=validated_holidays_path,
                validated_weather_dir=validated_weather_path,
                data_drift_report_path=self.data_validation_config.data_drift_report_file_path
            )

            logger.info("="*70)
            logger.info("[PASS] DATA VALIDATION COMPLETED")
            logger.info("="*70)

            return data_validation_artifact

        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise CustomException(e, sys)
