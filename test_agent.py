"""Automated test suite for the Weather, Clothing & Cycling LangGraph Agent."""

import sys
from weather_agent.decision_engine import evaluate_clothing, evaluate_cycling, analyze_weather
from weather_agent.geolocation import detect_device_location, geocode_location, extract_location_from_text
from weather_agent.weather_service import fetch_weather_forecast
from weather_agent.graph import graph
from langchain_core.messages import HumanMessage


def test_decision_logic():
    print("Testing Clothing & Cycling Decision Logic...")

    # Sub-freezing cold
    cold_cloth = evaluate_clothing(apparent_temp_c=-8.0, wind_speed_kmh=20.0, precip_prob=10.0)
    assert cold_cloth["layer_count"] == 4, f"Expected 4 layers, got {cold_cloth['layer_count']}"
    assert "thermal" in cold_cloth["base_layer"].lower()

    # Mild rain
    mild_rain_cloth = evaluate_clothing(apparent_temp_c=14.0, wind_speed_kmh=15.0, precip_prob=70.0, precip_mm=3.0)
    assert any("umbrella" in acc.lower() or "rain" in acc.lower() for acc in mild_rain_cloth["accessories"])

    # Cycling tests
    # Ideal conditions
    ideal_cyc = evaluate_cycling(
        temp_c=20.0,
        apparent_temp_c=20.0,
        wind_speed_kmh=8.0,
        wind_gusts_kmh=12.0,
        precip_prob=0.0,
        precip_mm=0.0,
        weather_code=0,
    )
    assert ideal_cyc["score"] >= 85, f"Expected >= 85 for ideal cycling, got {ideal_cyc['score']}"
    assert ideal_cyc["rating"] == "Excellent"

    # Hazardous storm
    storm_cyc = evaluate_cycling(
        temp_c=15.0,
        apparent_temp_c=15.0,
        wind_speed_kmh=45.0,
        wind_gusts_kmh=65.0,
        precip_prob=90.0,
        precip_mm=12.0,
        weather_code=95,
    )
    assert storm_cyc["score"] < 30, f"Expected hazardous score (<30), got {storm_cyc['score']}"
    assert storm_cyc["rating"] == "Hazardous"
    print("✅ Decision logic passed successfully!\n")


def test_geolocation_and_weather():
    print("Testing Geolocation & Weather Fetching...")
    # Geocoding
    geo = geocode_location("Austin, TX")
    assert geo is not None, "Failed to geocode Austin, TX"
    assert "Austin" in geo["name"]
    print(f"Geocoded: {geo['name']} -> ({geo['latitude']}, {geo['longitude']})")

    # Weather fetch
    weather = fetch_weather_forecast(geo["latitude"], geo["longitude"])
    assert "current" in weather, "Missing 'current' in weather response"
    assert "daily" in weather, "Missing 'daily' in weather response"
    assert len(weather["daily"]["time"]) >= 7, "Expected at least 7 days in daily forecast"
    print(f"Weather fetched: {len(weather['daily']['time'])} days, current temp: {weather['current']['temperature_2m']}°C")

    # Full analysis
    analysis = analyze_weather(weather)
    assert len(analysis["daily"]) >= 7
    assert len(analysis["best_cycling_rankings"]) >= 7
    print("✅ Geolocation and Weather fetch passed successfully!\n")


def test_graph_end_to_end():
    print("Testing LangGraph End-to-End Execution...")

    # Test 1: Explicit location query
    print("--- Test 1: Specified Location ('Tokyo') ---")
    res1 = graph.invoke({"location_query": "Tokyo"})
    assert res1.get("final_report"), "Graph did not generate a final report"
    assert len(res1["messages"]) > 0
    print(f"Location resolved: {res1['location_name']}")
    print("Sample Output snippet:")
    print("\n".join(res1["final_report"].split("\n")[:8]))

    # Test 2: Natural Language input with coordinates or device
    print("\n--- Test 2: Natural language input ('Should I ride my bike in Seattle today?') ---")
    res2 = graph.invoke({
        "messages": [HumanMessage(content="Should I ride my bike in Seattle today and what clothes do I need?")]
    })
    assert res2.get("final_report")
    assert "Seattle" in res2["location_name"]
    print(f"Location resolved from message: {res2['location_name']}")

    # Test 3: Device location fallback
    print("\n--- Test 3: Auto-detect Device Location ---")
    res3 = graph.invoke({})
    assert res3.get("final_report")
    print(f"Device Location resolved: {res3['location_name']}")

    print("\n🎉 All LangGraph End-to-End Tests Passed Successfully!")


if __name__ == "__main__":
    try:
        test_decision_logic()
        test_geolocation_and_weather()
        test_graph_end_to_end()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
