"""Weather service interfacing with Open-Meteo API.
Provides live real-time conditions and accurate 7-day hourly/daily forecasts
without requiring proprietary API keys or paid tiers.
"""

from typing import Any, Dict, Optional
import httpx


def c_to_f(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return round((celsius * 9 / 5) + 32, 1)


def kmh_to_mph(kmh: float) -> float:
    """Convert km/h to mph."""
    return round(kmh * 0.621371, 1)


def fetch_weather_forecast(
    latitude: float,
    longitude: float,
    timezone: str = "auto",
    timeout: float = 8.0,
) -> Dict[str, Any]:
    """Fetch current weather and 7-day detailed forecast from Open-Meteo API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "temperature_2m",
            "apparent_temperature",
            "is_day",
            "precipitation",
            "rain",
            "showers",
            "snowfall",
            "weather_code",
            "cloud_cover",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
        ],
        "hourly": [
            "temperature_2m",
            "apparent_temperature",
            "precipitation_probability",
            "precipitation",
            "weather_code",
            "wind_speed_10m",
        ],
        "daily": [
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "uv_index_max",
        ],
        "forecast_days": 8,  # Today + 7 future days
        "timezone": timezone or "auto",
    }

    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
