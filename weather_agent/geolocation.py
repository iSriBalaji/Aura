"""Geolocation and geocoding services for Weather Agent.
Provides device location auto-detection and Open-Meteo geocoding without external API keys.
"""

import re
from typing import Any, Dict, Optional, Tuple
import httpx


def detect_device_location(timeout: float = 5.0) -> Dict[str, Any]:
    """Detect current device location via IP-based geolocation services.
    Falls back gracefully across multiple reliable endpoints.
    """
    # Service 1: ipwho.is (fast, no key required, includes timezone)
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get("https://ipwho.is/")
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success", False):
                    city = data.get("city") or "Current Location"
                    region = data.get("region") or ""
                    country = data.get("country") or ""
                    loc_parts = [p for p in [city, region, country] if p]
                    full_name = ", ".join(loc_parts) if loc_parts else "Current Device Location"
                    return {
                        "name": full_name,
                        "city": city,
                        "latitude": float(data["latitude"]),
                        "longitude": float(data["longitude"]),
                        "timezone": data.get("timezone", {}).get("id", "auto"),
                        "source": "ipwho.is",
                    }
    except Exception:
        pass

    # Service 2: ip-api.com fallback
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get("http://ip-api.com/json")
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "success":
                    city = data.get("city") or "Current Location"
                    region = data.get("regionName") or ""
                    country = data.get("country") or ""
                    loc_parts = [p for p in [city, region, country] if p]
                    full_name = ", ".join(loc_parts) if loc_parts else "Current Device Location"
                    return {
                        "name": full_name,
                        "city": city,
                        "latitude": float(data["lat"]),
                        "longitude": float(data["lon"]),
                        "timezone": data.get("timezone", "auto"),
                        "source": "ip-api.com",
                    }
    except Exception:
        pass

    # Safe fallback if network lookup fails
    return {
        "name": "Default Location (New York, US)",
        "city": "New York",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "timezone": "America/New_York",
        "source": "fallback",
    }


def geocode_location(query: str, timeout: float = 6.0) -> Optional[Dict[str, Any]]:
    """Geocode a location query string (e.g. 'Austin, TX', 'London', 'Tokyo')
    using Open-Meteo's free geocoding API.
    """
    clean_query = query.strip()
    if not clean_query:
        return None

    # Check if query is raw coordinates like "30.2672, -97.7431"
    coord_match = re.match(
        r"^\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*$", clean_query
    )
    if coord_match:
        lat = float(coord_match.group(1))
        lon = float(coord_match.group(2))
        return {
            "name": f"Coordinates ({lat:.4f}, {lon:.4f})",
            "city": f"{lat:.2f}, {lon:.2f}",
            "latitude": lat,
            "longitude": lon,
            "timezone": "auto",
            "source": "coordinates",
        }

    # Query Open-Meteo Geocoding
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": clean_query, "count": 1, "language": "en", "format": "json"}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results")
                if results and len(results) > 0:
                    top = results[0]
                    name = top.get("name", clean_query)
                    admin1 = top.get("admin1")
                    country = top.get("country")
                    parts = [name]
                    if admin1 and admin1 != name:
                        parts.append(admin1)
                    if country and country != admin1:
                        parts.append(country)
                    full_name = ", ".join(parts)

                    return {
                        "name": full_name,
                        "city": name,
                        "latitude": float(top["latitude"]),
                        "longitude": float(top["longitude"]),
                        "timezone": top.get("timezone", "auto"),
                        "country": country,
                        "source": "open-meteo-geocoding",
                    }
    except Exception:
        pass

    return None


def extract_location_from_text(text: str) -> Optional[str]:
    """Extract a location target from a user's natural language question.
    Examples:
    - "I'm in London, should I bike today?" -> "London"
    - "What should I wear in San Francisco?" -> "San Francisco"
    - "Can I bike today in Austin, TX?" -> "Austin, TX"
    - "Weather for Paris for the next week" -> "Paris"
    """
    clean_text = text.strip()

    patterns = [
        # "I am in London" / "currently in Paris"
        r"(?:(?:i'm|im|i\s+am|currently|living|staying|located)\s+(?:in|at))\s+([A-Za-z\s,.-]+?)(?=[,;.!?]|\b(?:should|can|could|will|how|what|is|today|tomorrow|this|next|for|and)\b|$)",
        # "weather in Paris", "forecast for Seattle"
        r"(?:weather|forecast|clothes|clothing|wear|layers|cycling|biking|ride)\s+(?:in|for|at|around)\s+([A-Za-z\s,.-]+?)(?=[,;.!?]|\b(?:today|tomorrow|this|next|for|and|should|can)\b|$)",
        # General "in <City>"
        r"\b(?:in|at|around)\s+([A-Z][a-zA-Z\s,.-]+?)(?=[,;.!?]|\b(?:today|tomorrow|this|next|should|can|could|will|how|what|is|and)\b|$)",
    ]

    for pat in patterns:
        m = re.search(pat, clean_text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            cleaned = re.sub(
                r"\b(today|tomorrow|this week|the next \d+ days|the next 7 days|the week|cycling|biking|layers|clothes|now)\b",
                "",
                candidate,
                flags=re.IGNORECASE,
            ).strip(" ,.?;:")
            if cleaned and len(cleaned) > 1 and cleaned.lower() not in ("my area", "here", "current location", "device"):
                test_geo = geocode_location(cleaned)
                if test_geo:
                    return cleaned

    return None
