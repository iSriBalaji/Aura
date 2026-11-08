"""Automated test suite for the Weather, Activity & Trending News LangGraph Agent."""

import sys
from weather_agent.decision_engine import evaluate_clothing, evaluate_cycling, analyze_weather
from weather_agent.geolocation import detect_device_location, geocode_location, extract_location_from_text
from weather_agent.weather_service import fetch_weather_forecast
from weather_agent.news_service import fetch_trending_news, search_news_topic
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

    # Cycling tests: Ideal
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

    # Cycling tests: Storm
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


def test_news_service():
    print("Testing Trending News Service...")
    news = fetch_trending_news(categories=["technology", "science", "politics", "general"], limit_per_cat=2)
    assert "technology" in news
    assert "science" in news
    assert "politics" in news
    assert "general" in news
    assert len(news["technology"]) > 0, "No tech news retrieved"
    print(f"Tech Headline Sample: {news['technology'][0]['title']}")

    # Deep dive search
    deepdive = search_news_topic("NASA Artemis", limit=3)
    assert len(deepdive) > 0, "No deep dive results for NASA Artemis"
    print(f"Deep-dive Result Sample: {deepdive[0]['title']}")
    print("✅ News service passed successfully!\n")


def test_geolocation_and_weather():
    print("Testing Geolocation & Weather Fetching...")
    geo = geocode_location("Austin, TX")
    assert geo is not None, "Failed to geocode Austin, TX"
    assert "Austin" in geo["name"]

    weather = fetch_weather_forecast(geo["latitude"], geo["longitude"])
    assert "current" in weather
    assert "daily" in weather
    assert len(weather["daily"]["time"]) >= 7

    analysis = analyze_weather(weather)
    assert len(analysis["daily"]) >= 7
    print("✅ Geolocation and Weather fetch passed successfully!\n")


def test_graph_end_to_end():
    print("Testing LangGraph End-to-End Execution...")

    # Test 1: Weather query
    print("--- Test 1: Weather Intent ('Austin, TX') ---")
    res1 = graph.invoke({"location_query": "Austin, TX"})
    assert res1.get("final_report"), "Graph did not generate a final report"
    assert "Austin" in res1["location_name"]

    # Test 2: News Digest query
    print("\n--- Test 2: News Digest Intent ---")
    res2 = graph.invoke({
        "messages": [HumanMessage(content="What are the trending important news stories in technology and science?")]
    })
    assert res2.get("final_report")
    assert "Trending Important News" in res2["final_report"] or "Technology" in res2["final_report"]
    print("Sample News Output snippet:")
    print("\n".join(res2["final_report"].split("\n")[:7]))

    # Test 3: News Deep Dive query
    print("\n--- Test 3: News Deep Dive Intent ---")
    res3 = graph.invoke({
        "messages": [HumanMessage(content="Tell me more about NASA Artemis moon mission")]
    })
    assert res3.get("final_report")
    assert "In-Depth News Brief" in res3["final_report"] or "Artemis" in res3["final_report"]
    print("Sample Deep-Dive Output snippet:")
    print("\n".join(res3["final_report"].split("\n")[:7]))

    print("\n🎉 All LangGraph End-to-End Tests Passed Successfully!")


if __name__ == "__main__":
    try:
        test_decision_logic()
        test_news_service()
        test_geolocation_and_weather()
        test_graph_end_to_end()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
