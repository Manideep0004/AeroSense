"""
AaroSense - FastAPI Schemas (Milestone 4).

Pydantic v2 models for the prediction API, enforcing strict data validation
and providing OpenAPI documentation.
"""


from pydantic import BaseModel, ConfigDict, Field


class PM25InferenceFeatures(BaseModel):
    """
    Input schema for a single PM2.5 prediction request.
    Includes current meteorology, historical lags, and cyclical encodings.
    """

    # Exclude metadata like date/season from model features, though they might be in the request for routing
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Relative humidity %")
    wind_speed: float = Field(..., description="Wind speed in m/s")
    wind_direction: float = Field(..., description="Wind direction in degrees")
    pressure: float = Field(..., description="Atmospheric pressure in hPa")
    precipitation: float = Field(..., description="Precipitation in mm")

    # Current and historical PM2.5
    pm25: float = Field(..., description="Current PM2.5 concentration (µg/m³)")
    pm25_lag_1: float = Field(..., description="PM2.5 lag (T-1)")
    pm25_lag_3: float = Field(..., description="PM2.5 lag (T-3)")
    pm25_lag_6: float = Field(..., description="PM2.5 lag (T-6)")
    pm25_lag_12: float = Field(..., description="PM2.5 lag (T-12)")
    pm25_lag_24: float = Field(..., description="PM2.5 lag (T-24)")

    pm25_mean_3h: float = Field(..., description="3-hour rolling mean PM2.5")
    pm25_mean_6h: float = Field(..., description="6-hour rolling mean PM2.5")
    pm25_mean_12h: float = Field(..., description="12-hour rolling mean PM2.5")
    pm25_mean_24h: float = Field(..., description="24-hour rolling mean PM2.5")

    # Wind vectors
    wind_x: float = Field(..., description="Wind vector X component")
    wind_y: float = Field(..., description="Wind vector Y component")

    # Cyclical temporal features
    hour_sin: float = Field(..., description="Hour of day (sine component)")
    hour_cos: float = Field(..., description="Hour of day (cosine component)")
    day_of_week_sin: float = Field(..., description="Day of week (sine component)")
    day_of_week_cos: float = Field(..., description="Day of week (cosine component)")
    month_sin: float = Field(..., description="Month of year (sine component)")
    month_cos: float = Field(..., description="Month of year (cosine component)")
    day_of_year_sin: float = Field(..., description="Day of year (sine component)")
    day_of_year_cos: float = Field(..., description="Day of year (cosine component)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "temperature": 15.2,
                "humidity": 75.0,
                "wind_speed": 2.1,
                "wind_direction": 270.0,
                "pressure": 1012.5,
                "precipitation": 0.0,
                "pm25": 150.5,
                "pm25_lag_1": 145.2,
                "pm25_lag_3": 140.1,
                "pm25_lag_6": 125.0,
                "pm25_lag_12": 110.5,
                "pm25_lag_24": 160.2,
                "pm25_mean_3h": 145.3,
                "pm25_mean_6h": 138.2,
                "pm25_mean_12h": 125.1,
                "pm25_mean_24h": 140.0,
                "wind_x": -2.1,
                "wind_y": 0.0,
                "hour_sin": 0.5,
                "hour_cos": -0.866,
                "day_of_week_sin": 0.0,
                "day_of_week_cos": 1.0,
                "month_sin": 0.5,
                "month_cos": 0.866,
                "day_of_year_sin": 0.1,
                "day_of_year_cos": 0.99,
            }
        }
    )


class PredictRequest(BaseModel):
    season: str = Field(
        ..., description="Season for routing to the correct model (e.g., 'Winter')"
    )
    features: PM25InferenceFeatures
    batch_id: str | None = Field(
        None, description="Optional ID for tracking requests"
    )


class PredictBatchRequest(BaseModel):
    season: str = Field(..., description="Season for routing (e.g., 'Winter')")
    features_list: list[PM25InferenceFeatures] = Field(
        ..., min_length=1, max_length=1000
    )
    batch_id: str | None = Field(
        None, description="Optional ID for tracking requests"
    )


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted PM2.5 value (µg/m³)")
    model_used: str = Field(..., description="Algorithm used (e.g., 'lightgbm')")
    season: str = Field(..., description="Seasonal model used")
    batch_id: str | None = None


class PredictionBatchResponse(BaseModel):
    predictions: list[float] = Field(..., description="List of predicted PM2.5 values")
    model_used: str = Field(..., description="Algorithm used")
    season: str = Field(..., description="Seasonal model used")
    batch_id: str | None = None


class ModelInfo(BaseModel):
    season: str
    model_name: str
    status: str
    loaded: bool


class HealthResponse(BaseModel):
    status: str = "ok"
    active_models: list[ModelInfo]
    drift_status: dict
