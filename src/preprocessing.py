"""
AaroSense - Data Validation, Preprocessing, and Seasonal Splitting Pipeline.
Designed for Delhi PM2.5 MLOps Pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AaroSense.Preprocessing")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration dataclass for preprocessing and feature engineering."""
    
    # Metadata columns to drop from feature sets
    metadata_cols_to_drop: List[str] = field(
        default_factory=lambda: ["latitude", "longitude"]
    )
    
    # Primary identifiers & target
    timestamp_col: str = "timestamp"
    target_col: str = "target_pm25"
    season_col: str = "season"
    
    # Periodic features and their theoretical periods for cyclical encoding
    cyclical_features: Dict[str, float] = field(
        default_factory=lambda: {
            "hour": 24.0,
            "day_of_week": 7.0,
            "month": 12.0,
            "day_of_year": 365.25,
        }
    )
    
    # Wind vector configuration
    wind_speed_col: str = "wind_speed"
    wind_direction_col: str = "wind_direction"
    
    # Missing value handling strategy for lag/rolling columns
    drop_na_rows: bool = True
    
    # Train / Test split configuration
    train_ratio: float = 0.80
    
    # Baseline directory for drift detection reference sets
    baseline_dir: Path = Path("baselines")


@dataclass
class SeasonalSplits:
    """Container for seasonal training and testing sets."""
    train: pd.DataFrame
    test: pd.DataFrame
    
    @property
    def X_train(self) -> pd.DataFrame:
        """Features for training (excluding target and timestamp)."""
        exclude_cols = ["target_pm25", "timestamp", "season"]
        return self.train.drop(columns=[c for c in exclude_cols if c in self.train.columns])

    @property
    def y_train(self) -> pd.Series:
        """Target series for training."""
        return self.train["target_pm25"]

    @property
    def X_test(self) -> pd.DataFrame:
        """Features for testing (excluding target and timestamp)."""
        exclude_cols = ["target_pm25", "timestamp", "season"]
        return self.test.drop(columns=[c for c in exclude_cols if c in self.test.columns])

    @property
    def y_test(self) -> pd.Series:
        """Target series for testing."""
        return self.test["target_pm25"]


class DataValidator:
    """Validates schema integrity and data sanity for Delhi air quality data."""

    REQUIRED_COLUMNS = [
        "timestamp", "latitude", "longitude",
        "temperature", "humidity", "wind_speed", "wind_direction", "pressure", "precipitation",
        "pm25", "pm25_lag_1", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
        "pm25_mean_3h", "pm25_mean_6h", "pm25_mean_12h", "pm25_mean_24h",
        "hour", "day_of_week", "month", "day_of_year", "season",
        "target_pm25",
    ]

    @classmethod
    def validate_schema(cls, df: pd.DataFrame) -> None:
        """
        Verify that all expected columns exist in the DataFrame.

        Args:
            df: Input raw DataFrame.

        Raises:
            ValueError: If required columns are missing.
        """
        missing = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Schema validation failed. Missing required columns: {missing}")
        logger.info("Schema validation passed successfully.")

    @staticmethod
    def validate_ranges(df: pd.DataFrame) -> None:
        """
        Sanity check on physical feature bounds.
        
        Args:
            df: Cleaned or raw DataFrame.
        """
        if (df["humidity"] < 0).any() or (df["humidity"] > 100).any():
            logger.warning("Relative humidity values found outside [0, 100]% range.")
        if (df["wind_direction"] < 0).any() or (df["wind_direction"] > 360).any():
            logger.warning("Wind direction values found outside [0, 360] degrees range.")
        if (df["pm25"] < 0).any():
            logger.warning("Negative PM2.5 observations detected.")


class FeatureEngineer:
    """Applies domain-specific transformations (cyclical encoding, wind vectors)."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config

    def encode_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms periodic integer/float columns into sine and cosine components.
        
        Formula:
            sin_feat = sin(2 * pi * x / period)
            cos_feat = cos(2 * pi * x / period)
        
        Raw periodic columns are dropped to eliminate non-linear boundary discontinuities.
        """
        df_encoded = df.copy()
        
        for col, period in self.config.cyclical_features.items():
            if col in df_encoded.columns:
                radians = 2.0 * np.pi * df_encoded[col] / period
                df_encoded[f"{col}_sin"] = np.sin(radians)
                df_encoded[f"{col}_cos"] = np.cos(radians)
                df_encoded.drop(columns=[col], inplace=True)
                logger.debug("Applied cyclical sine/cosine transformation to '%s' (period=%.1f).", col, period)
            else:
                logger.warning("Cyclical feature '%s' not present in DataFrame.", col)
                
        return df_encoded

    def compute_wind_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decomposes scalar wind speed and direction into orthogonal vector components.
        
        Wind direction is meteorologically defined as the direction FROM which wind blows.
        Vector components:
            wind_x = wind_speed * cos(rad(wind_direction))  (East-West component)
            wind_y = wind_speed * sin(rad(wind_direction))  (North-South component)
        """
        df_trans = df.copy()
        ws_col = self.config.wind_speed_col
        wd_col = self.config.wind_direction_col

        if ws_col in df_trans.columns and wd_col in df_trans.columns:
            # Convert degrees to radians
            rad = np.deg2rad(df_trans[wd_col])
            df_trans["wind_x"] = df_trans[ws_col] * np.cos(rad)
            df_trans["wind_y"] = df_trans[ws_col] * np.sin(rad)
            logger.debug("Computed wind_x and wind_y vector components.")
        else:
            logger.warning("Wind speed or direction column missing; skipping vector decomposition.")
            
        return df_trans


class PM25DataPreprocessor:
    """
    End-to-end preprocessing pipeline for Delhi PM2.5 tabular dataset.
    Handles cleaning, transformations, seasonal slicing, and MLOps baseline export.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()
        self.feature_engineer = FeatureEngineer(self.config)

    def clean_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs data cleaning:
        1. Ensures datetime timestamp and chronological sorting.
        2. Drops static coordinates (latitude, longitude).
        3. Removes initial NaN warmup rows created by rolling/lag windowing without leakage.
        4. Applies cyclical feature encoding and wind vector decomposition.

        Args:
            df: Raw input DataFrame.

        Returns:
            Preprocessed and chronologically ordered DataFrame.
        """
        logger.info("Starting data cleaning and feature engineering...")
        DataValidator.validate_schema(df)
        
        cleaned_df = df.copy()
        
        # 1. Ensure timestamp is datetime and sort chronologically
        if not pd.api.types.is_datetime64_any_dtype(cleaned_df[self.config.timestamp_col]):
            cleaned_df[self.config.timestamp_col] = pd.to_datetime(cleaned_df[self.config.timestamp_col])
        
        cleaned_df.sort_values(by=self.config.timestamp_col, ascending=True, inplace=True)
        cleaned_df.reset_index(drop=True, inplace=True)

        # 2. Drop constant metadata coordinates
        cols_to_drop = [c for c in self.config.metadata_cols_to_drop if c in cleaned_df.columns]
        if cols_to_drop:
            cleaned_df.drop(columns=cols_to_drop, inplace=True)
            logger.info("Dropped metadata columns: %s", cols_to_drop)

        # 3. Handle missing values
        # In time-series lag/rolling features, NaN values occur at the beginning of the series.
        # Dropping rows with NaNs avoids synthetic leakage or imputation distortions on lag windows.
        initial_len = len(cleaned_df)
        if self.config.drop_na_rows:
            cleaned_df.dropna(inplace=True)
            cleaned_df.reset_index(drop=True, inplace=True)
            dropped_rows = initial_len - len(cleaned_df)
            logger.info("Dropped %d rows with NaNs (e.g., warm-up lag/rolling periods). Remaining rows: %d", dropped_rows, len(cleaned_df))

        # Sanity validation
        DataValidator.validate_ranges(cleaned_df)

        # 4. Feature Transformations
        cleaned_df = self.feature_engineer.compute_wind_components(cleaned_df)
        cleaned_df = self.feature_engineer.encode_cyclical_features(cleaned_df)

        logger.info("Data cleaning and feature transformations complete. Final feature count: %d", cleaned_df.shape[1])
        return cleaned_df

    def split_by_season(
        self, df: pd.DataFrame
    ) -> Dict[str, SeasonalSplits]:
        """
        Splits cleaned DataFrame by season and performs chronological train/test splits.

        For each season:
          - Slices the data corresponding to that season.
          - Sorts chronologically by timestamp.
          - Applies an 80/20 train/test split preserving temporal order (no future leakage).

        Args:
            df: Cleaned and transformed DataFrame.

        Returns:
            Dictionary mapping season names (e.g., 'Winter', 'Summer') to SeasonalSplits objects.
        """
        logger.info("Splitting dataset into seasonal subsets and applying chronological train/test splits...")
        
        if self.config.season_col not in df.columns:
            raise KeyError(f"Season column '{self.config.season_col}' not found in DataFrame.")

        unique_seasons = df[self.config.season_col].dropna().unique()
        seasonal_datasets: Dict[str, SeasonalSplits] = {}

        for season in unique_seasons:
            season_df = df[df[self.config.season_col] == season].sort_values(by=self.config.timestamp_col).copy()
            season_df.reset_index(drop=True, inplace=True)
            
            n_total = len(season_df)
            if n_total == 0:
                logger.warning("Season '%s' has 0 rows, skipping.", season)
                continue
                
            split_idx = int(n_total * self.config.train_ratio)
            
            train_df = season_df.iloc[:split_idx].copy().reset_index(drop=True)
            test_df = season_df.iloc[split_idx:].copy().reset_index(drop=True)
            
            seasonal_datasets[str(season)] = SeasonalSplits(train=train_df, test=test_df)
            
            logger.info(
                "Season '%s': Total=%d | Train=%d (from %s to %s) | Test=%d (from %s to %s)",
                season,
                n_total,
                len(train_df),
                train_df[self.config.timestamp_col].min(),
                train_df[self.config.timestamp_col].max(),
                len(test_df),
                test_df[self.config.timestamp_col].min(),
                test_df[self.config.timestamp_col].max(),
            )

        return seasonal_datasets

    def export_drift_baselines(
        self,
        seasonal_splits: Dict[str, SeasonalSplits],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """
        Serializes training feature distributions for each season as Parquet files.
        These baseline distributions serve as reference data for Kolmogorov-Smirnov (KS)
        drift detection and data drift monitoring pipelines.

        Args:
            seasonal_splits: Dictionary of seasonal splits.
            output_dir: Directory to save Parquet baselines. Defaults to config.baseline_dir.

        Returns:
            Dictionary mapping season names to the saved Parquet file paths.
        """
        out_path = Path(output_dir) if output_dir else self.config.baseline_dir
        out_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths: Dict[str, Path] = {}
        
        for season_name, splits in seasonal_splits.items():
            baseline_features = splits.X_train.copy()
            
            # Format filename safely (e.g. baselines/winter_baseline.parquet)
            sanitized_name = season_name.strip().lower().replace(" ", "_")
            file_path = out_path / f"{sanitized_name}_baseline.parquet"
            
            # Export to Parquet
            baseline_features.to_parquet(file_path, index=False, engine="pyarrow")
            saved_paths[season_name] = file_path
            
            logger.info(
                "Exported baseline for '%s' (Shape: %s) to '%s'",
                season_name,
                baseline_features.shape,
                file_path.as_posix(),
            )
            
        return saved_paths


# Example usage demonstration
if __name__ == "__main__":
    # 1. Synthesize sample DataFrame matching user's exact schema
    np.random.seed(42)
    n_samples = 1500
    timestamps = pd.date_range(start="2023-01-01", periods=n_samples, freq="h")
    
    # Generate dummy seasons based on month
    def get_season(dt: pd.Timestamp) -> str:
        if dt.month in [12, 1, 2]:
            return "Winter"
        elif dt.month in [3, 4, 5]:
            return "Summer"
        elif dt.month in [6, 7, 8, 9]:
            return "Monsoon"
        else:
            return "Autumn"

    seasons = [get_season(ts) for ts in timestamps]

    sample_df = pd.DataFrame({
        "timestamp": timestamps,
        "latitude": 28.6139,
        "longitude": 77.2090,
        "temperature": np.random.uniform(10, 45, size=n_samples),
        "humidity": np.random.uniform(20, 95, size=n_samples),
        "wind_speed": np.random.uniform(0.5, 15.0, size=n_samples),
        "wind_direction": np.random.uniform(0, 360, size=n_samples),
        "pressure": np.random.uniform(995, 1020, size=n_samples),
        "precipitation": np.random.choice([0.0, 0.5, 2.0], size=n_samples, p=[0.85, 0.10, 0.05]),
        "pm25": np.random.uniform(30, 400, size=n_samples),
        "pm25_lag_1": np.random.uniform(30, 400, size=n_samples),
        "pm25_lag_3": np.random.uniform(30, 400, size=n_samples),
        "pm25_lag_6": np.random.uniform(30, 400, size=n_samples),
        "pm25_lag_12": np.random.uniform(30, 400, size=n_samples),
        "pm25_lag_24": np.random.uniform(30, 400, size=n_samples),
        "pm25_mean_3h": np.random.uniform(30, 400, size=n_samples),
        "pm25_mean_6h": np.random.uniform(30, 400, size=n_samples),
        "pm25_mean_12h": np.random.uniform(30, 400, size=n_samples),
        "pm25_mean_24h": np.random.uniform(30, 400, size=n_samples),
        "hour": timestamps.hour,
        "day_of_week": timestamps.dayofweek,
        "month": timestamps.month,
        "day_of_year": timestamps.dayofyear,
        "season": seasons,
        "target_pm25": np.random.uniform(30, 450, size=n_samples),
    })

    # Simulate warm-up NaNs in lag columns
    sample_df.loc[:24, ["pm25_lag_24", "pm25_mean_24h"]] = np.nan

    # 2. Execute Preprocessing Pipeline
    config = PreprocessingConfig(train_ratio=0.80, baseline_dir=Path("baselines"))
    preprocessor = PM25DataPreprocessor(config=config)
    
    # Clean & Transform
    cleaned_data = preprocessor.clean_and_prepare(sample_df)
    
    # Split by Season with chronological splits
    seasonal_data = preprocessor.split_by_season(cleaned_data)
    
    # Export MLOps drift baselines
    baseline_files = preprocessor.export_drift_baselines(seasonal_data)
    
    print("\n--- Pipeline Execution Summary ---")
    print(f"Cleaned columns: {list(cleaned_data.columns)}")
    for season, splits in seasonal_data.items():
        print(f"Season: {season} | Train shape: {splits.X_train.shape} | Test shape: {splits.X_test.shape}")
    print(f"Baseline files generated: {baseline_files}")
