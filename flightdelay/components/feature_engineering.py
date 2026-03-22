"""
Feature Engineering Component
Creates advanced features from joined flight data
"""

import sys
import pandas as pd
import numpy as np
from flightdelay.logging.logger import logger
from flightdelay.exception.exception import CustomException


class FeatureEngineering:
    """
    Creates engineered features for flight delay prediction
    - Temporal features (hour, weekend, quarter)
    - Derived features (delay flags, duration ratios, distance bins)
    - Historical aggregations (airport, airline, route delay stats)
    - Weather features (temperature range, extreme weather flags)
    """
    
    def __init__(self):
        pass
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from timestamps
        """
        try:
            logger.info("Creating temporal features...")
            
            # Extract hour from scheduled departure (format: HHMM)
            if 'SCHEDULED_DEPARTURE' in df.columns:
                df['HOUR'] = df['SCHEDULED_DEPARTURE'] // 100
                logger.info(f"  Created HOUR feature (range: {df['HOUR'].min()}-{df['HOUR'].max()})")
            
            # Weekend flag (Saturday=6, Sunday=7)
            if 'DAY_OF_WEEK' in df.columns:
                df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([6, 7]).astype(int)
                weekend_pct = (df['IS_WEEKEND'].sum() / len(df)) * 100
                logger.info(f"  Created IS_WEEKEND feature ({weekend_pct:.1f}% weekend flights)")
            
            # Quarter (Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec)
            if 'MONTH' in df.columns:
                df['QUARTER'] = ((df['MONTH'] - 1) // 3 + 1).astype(int)
                logger.info(f"  Created QUARTER feature (1-4)")
            
            logger.info("[PASS] Temporal features created")
            return df
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from existing columns
        """
        try:
            logger.info("Creating derived features...")
            
            # Holiday flag (rename from is_holiday for consistency)
            if 'is_holiday' in df.columns:
                df['IS_HOLIDAY'] = df['is_holiday']
                holiday_pct = (df['IS_HOLIDAY'].sum() / len(df)) * 100
                logger.info(f"  Created IS_HOLIDAY feature ({holiday_pct:.1f}% holiday flights)")
            
            # Binary delay classification (>15 minutes = delayed)
            if 'ARRIVAL_DELAY' in df.columns:
                df['DELAYED'] = (df['ARRIVAL_DELAY'] > 15).astype(int)
                delayed_pct = (df['DELAYED'].sum() / len(df)) * 100
                logger.info(f"  Created DELAYED target ({delayed_pct:.1f}% delayed flights)")
            
            # Duration ratio (actual vs scheduled)
            if 'ELAPSED_TIME' in df.columns and 'SCHEDULED_TIME' in df.columns:
                # Avoid division by zero
                df['DURATION_RATIO'] = np.where(
                    df['SCHEDULED_TIME'] > 0,
                    df['ELAPSED_TIME'] / df['SCHEDULED_TIME'],
                    1.0
                )
                logger.info(f"  Created DURATION_RATIO feature (mean: {df['DURATION_RATIO'].mean():.2f})")
            
            # Distance bins (short, medium, long, ultra)
            if 'DISTANCE' in df.columns:
                df['DISTANCE_BIN'] = pd.cut(
                    df['DISTANCE'],
                    bins=[0, 500, 1000, 2000, 5000],
                    labels=['short', 'medium', 'long', 'ultra']
                )
                logger.info(f"  Created DISTANCE_BIN feature")
                logger.info(f"    Distribution: {df['DISTANCE_BIN'].value_counts().to_dict()}")
            
            logger.info("[PASS] Derived features created")
            return df
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create historical aggregation features
        WARNING: This creates target leakage for training, should use proper time-based splits
        For production: Use historical data from previous time periods only
        """
        try:
            logger.info("Creating aggregation features (historical delay stats)...")
            
            if 'ARRIVAL_DELAY' not in df.columns:
                logger.warning("ARRIVAL_DELAY not found, skipping aggregation features")
                return df
            
            # Airport historical delay rates (origin)
            if 'ORIGIN_AIRPORT' in df.columns:
                logger.info("  Computing origin airport delay statistics...")
                airport_delay_stats = df.groupby('ORIGIN_AIRPORT')['ARRIVAL_DELAY'].agg([
                    ('ORIGIN_AVG_DELAY', 'mean'),
                    ('ORIGIN_DELAY_STD', 'std'),
                    ('ORIGIN_DELAY_RATE', lambda x: (x > 15).mean())
                ]).reset_index()
                
                # Merge back to main dataframe
                df = df.merge(airport_delay_stats, on='ORIGIN_AIRPORT', how='left')
                logger.info(f"    ORIGIN_AVG_DELAY: mean={df['ORIGIN_AVG_DELAY'].mean():.2f}")
                logger.info(f"    ORIGIN_DELAY_RATE: mean={df['ORIGIN_DELAY_RATE'].mean():.2%}")
            
            # Airline historical delay rates
            if 'AIRLINE' in df.columns:
                logger.info("  Computing airline delay statistics...")
                airline_delay_stats = df.groupby('AIRLINE')['ARRIVAL_DELAY'].agg([
                    ('AIRLINE_AVG_DELAY', 'mean'),
                    ('AIRLINE_DELAY_RATE', lambda x: (x > 15).mean())
                ]).reset_index()
                
                df = df.merge(airline_delay_stats, on='AIRLINE', how='left')
                logger.info(f"    AIRLINE_AVG_DELAY: mean={df['AIRLINE_AVG_DELAY'].mean():.2f}")
                logger.info(f"    AIRLINE_DELAY_RATE: mean={df['AIRLINE_DELAY_RATE'].mean():.2%}")
            
            # Route historical delay rates
            if 'ORIGIN_AIRPORT' in df.columns and 'DESTINATION_AIRPORT' in df.columns:
                logger.info("  Computing route delay statistics...")
                df['ROUTE'] = df['ORIGIN_AIRPORT'] + '_' + df['DESTINATION_AIRPORT']
                
                route_delay_stats = df.groupby('ROUTE')['ARRIVAL_DELAY'].agg([
                    ('ROUTE_AVG_DELAY', 'mean')
                ]).reset_index()
                
                df = df.merge(route_delay_stats, on='ROUTE', how='left')
                logger.info(f"    ROUTE_AVG_DELAY: mean={df['ROUTE_AVG_DELAY'].mean():.2f}")
                logger.info(f"    Unique routes: {df['ROUTE'].nunique()}")
            
            logger.info("[PASS] Aggregation features created")
            logger.warning("NOTE: Aggregation features may cause target leakage in training!")
            logger.warning("      For production, use time-based splits or historical-only data")
            
            return df
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-related features
        """
        try:
            logger.info("Creating weather features...")
            
            # Temperature range
            if 'TMAX' in df.columns and 'TMIN' in df.columns:
                df['TEMP_RANGE'] = df['TMAX'] - df['TMIN']
                logger.info(f"  Created TEMP_RANGE feature (mean: {df['TEMP_RANGE'].mean():.2f}°F)")
            
            # Extreme cold (below -10°F)
            if 'TMIN' in df.columns:
                df['EXTREME_COLD'] = (df['TMIN'] < -10).astype(int)
                cold_pct = (df['EXTREME_COLD'].sum() / len(df)) * 100
                logger.info(f"  Created EXTREME_COLD feature ({cold_pct:.1f}% of flights)")
            
            # Extreme hot (above 95°F / 35°C equivalent ~95°F)
            if 'TMAX' in df.columns:
                df['EXTREME_HOT'] = (df['TMAX'] > 95).astype(int)
                hot_pct = (df['EXTREME_HOT'].sum() / len(df)) * 100
                logger.info(f"  Created EXTREME_HOT feature ({hot_pct:.1f}% of flights)")
            
            # Heavy rain (>50mm precipitation)
            if 'PRCP' in df.columns:
                df['HEAVY_RAIN'] = (df['PRCP'] > 50).astype(int)
                rain_pct = (df['HEAVY_RAIN'].sum() / len(df)) * 100
                logger.info(f"  Created HEAVY_RAIN feature ({rain_pct:.1f}% of flights)")
            
            # High wind (>15 m/s or mph depending on data)
            if 'AWND' in df.columns:
                df['HIGH_WIND'] = (df['AWND'] > 15).astype(int)
                wind_pct = (df['HIGH_WIND'].sum() / len(df)) * 100
                logger.info(f"  Created HIGH_WIND feature ({wind_pct:.1f}% of flights)")
            
            logger.info("[PASS] Weather features created")
            return df
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to create all engineered features
        
        Args:
            df: Joined dataframe with flights + airlines + airports + holidays + weather
        
        Returns:
            DataFrame with engineered features
        """
        try:
            logger.info("="*70)
            logger.info("STARTING FEATURE ENGINEERING")
            logger.info("="*70)
            
            initial_cols = len(df.columns)
            logger.info(f"Initial columns: {initial_cols}")
            
            # Step 1: Temporal features
            print("\n[Feature Engineering 1/4] Creating temporal features...")
            df = self.create_temporal_features(df)
            
            # Step 2: Derived features
            print("[Feature Engineering 2/4] Creating derived features...")
            df = self.create_derived_features(df)
            
            # Step 3: Historical aggregation features
            print("[Feature Engineering 3/4] Creating aggregation features...")
            df = self.create_aggregation_features(df)
            
            # Step 4: Weather features
            print("[Feature Engineering 4/4] Creating weather features...")
            df = self.create_weather_features(df)
            
            final_cols = len(df.columns)
            new_features = final_cols - initial_cols
            
            logger.info("="*70)
            logger.info(f"[PASS] FEATURE ENGINEERING COMPLETED")
            logger.info(f"Added {new_features} new features ({initial_cols} -> {final_cols})")
            logger.info("="*70)
            
            print(f"\n[PASS] Feature engineering completed: {new_features} new features added")
            
            return df
        
        except Exception as e:
            raise CustomException(e, sys)


def get_feature_list() -> dict:
    """
    Return categorized list of all engineered features
    """
    return {
        "temporal_features": [
            "HOUR",
            "IS_WEEKEND",
            "QUARTER"
        ],
        "derived_features": [
            "IS_HOLIDAY",
            "DELAYED",
            "DURATION_RATIO",
            "DISTANCE_BIN"
        ],
        "aggregation_features": [
            "ORIGIN_AVG_DELAY",
            "ORIGIN_DELAY_STD",
            "ORIGIN_DELAY_RATE",
            "AIRLINE_AVG_DELAY",
            "AIRLINE_DELAY_RATE",
            "ROUTE_AVG_DELAY",
            "ROUTE"
        ],
        "weather_features": [
            "TEMP_RANGE",
            "EXTREME_COLD",
            "EXTREME_HOT",
            "HEAVY_RAIN",
            "HIGH_WIND"
        ],
        "original_features": [
            "YEAR", "MONTH", "DAY", "DAY_OF_WEEK",
            "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
            "DISTANCE", "SCHEDULED_TIME", "ELAPSED_TIME",
            "DEPARTURE_DELAY",
            "ORIGIN_LAT", "ORIGIN_LON", "DEST_LAT", "DEST_LON",
            "TMAX", "TMIN", "PRCP", "AWND"
        ],
        "target": "ARRIVAL_DELAY",
        "target_binary": "DELAYED"
    }
