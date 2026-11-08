"""Automated test suite for Weather, Wardrobe, Cycling, News, Events & Sports LangGraph Agent."""

import sys
from weather_agent.decision_engine import evaluate_clothing, evaluate_cycling, analyze_weather, find_optimal_outdoor_windows
from weather_agent.geolocation import detect_device_location, geocode_location
from weather_agent.weather_service import fetch_weather_forecast
from weather_agent.news_service import fetch_trending_news, search_news_topic
from weather_agent.events_service import (
    fetch_city_events,
    fetch_major_us_events,
    fetch_sports_briefing,
)
from weather_agent.memory_service import (
    add_custom_memory,
    format_profile_markdown,
    load_profile,
    remove_custom_memory,
)
from weather_agent.tts_service import clean_markdown_for_speech
from weather_agent.graph import graph
from langchain_core.messages import HumanMessage


def test_news_service():
    print("Testing News Service with 2-3 Line Summaries & 6 Categories...")
    news = fetch_trending_news(location_name="Prospect Heights, Illinois", limit_per_cat=2)
    expected_cats = ["general", "technology", "politics", "business", "science", "local"]
    for c in expected_cats:
        assert c in news, f"Missing category: {c}"
        assert len(news[c]) > 0, f"No stories found for category: {c}"
        for item in news[c]:
            assert "what_happened" in item, f"Missing 'what_happened' in {item}"
            assert "key_context" in item, f"Missing 'key_context' in {item}"
            assert "what_to_know" in item, f"Missing 'what_to_know' in {item}"

    print(f"Sample Tech Story: {news['technology'][0]['title']}")
    print(f"  • What happened: {news['technology'][0]['what_happened']}")
    print(f"  • Key context: {news['technology'][0]['key_context']}")
    print(f"  • What to know: {news['technology'][0]['what_to_know']}")
    print("✅ News service passed successfully!\n")


def test_events_and_sports_service():
    print("Testing Events & Sports Service...")
    # US Events
    us_events = fetch_major_us_events()
    assert "conferences" in us_events
    assert "music_festivals" in us_events
    print(f"Sample US Event: {us_events['conferences'][0]['title'] if us_events['conferences'] else 'None'}")

    # Sports
    sports = fetch_sports_briefing(["football", "f1", "cricket"])
    assert "football" in sports
    assert "f1" in sports
    assert "cricket" in sports
    print(f"Sample F1 Headline: {sports['f1'][0]['title'] if sports['f1'] else 'None'}")
    print(f"Sample Cricket Headline: {sports['cricket'][0]['title'] if sports['cricket'] else 'None'}")

    # City events
    city_ev = fetch_city_events("Chicago")
    assert len(city_ev) > 0, "No Chicago events found"
    print(f"Sample Chicago Event: {city_ev[0]['title']}")
    print("✅ Events & Sports service passed successfully!\n")


def test_graph_end_to_end():
    print("Testing LangGraph End-to-End Execution...")

    # Test 1: Weather Intent
    print("--- Test 1: Weather Intent ('Austin, TX') ---")
    res1 = graph.invoke({"location_query": "Austin, TX"})
    assert res1.get("final_report")
    assert "Austin" in res1["location_name"]

    # Test 2: Events & Sports Intent
    print("\n--- Test 2: Events & Sports Intent ('Major events in the US and F1 cricket updates') ---")
    res2 = graph.invoke({
        "messages": [HumanMessage(content="What are the major events happening in the US and latest sports updates on F1 and cricket?")]
    })
    assert res2.get("final_report")
    assert "Major Events" in res2["final_report"] or "Sports" in res2["final_report"]
    print("Sample Events Output snippet:")
    print("\n".join(res2["final_report"].split("\n")[:8]))

    # Test 3: City-Specific Events Intent ('Events in Chicago')
    print("\n--- Test 3: City-Specific Events Intent ('Events in Chicago this weekend') ---")
    res3 = graph.invoke({
        "messages": [HumanMessage(content="What events and festivals are happening in Chicago?")]
    })
    assert res3.get("final_report")
    assert "Chicago" in res3["final_report"]
    print("Sample City Events Output snippet:")
    print("\n".join(res3["final_report"].split("\n")[:8]))

    # Test 4: News Intent with 2-3 Line Summaries & 6 Categories
    print("\n--- Test 4: News Digest Intent (All 6 Categories & 2-3 Line Summaries) ---")
    res4 = graph.invoke({
        "messages": [HumanMessage(content="Give me the trending news digest today with 2-3 line summaries")]
    })
    assert res4.get("final_report")
    assert "Technology" in res4["final_report"]
    assert "Business" in res4["final_report"]
    assert "Politics" in res4["final_report"]
    assert "Science" in res4["final_report"]
    assert "Local" in res4["final_report"]
    assert "What happened:" in res4["final_report"]
    assert "What to know:" in res4["final_report"]
    print("Sample News Output snippet:")
    print("\n".join(res4["final_report"].split("\n")[:14]))

    print("\n🎉 All LangGraph End-to-End Tests Passed Successfully!")


def test_optimal_window_service():
    print("Testing Hourly Optimal Outdoor & Cycling Window Finder...")
    hourly_mock = {
        "time": [f"2026-11-08T{h:02d}:00" for h in range(6, 21)],
        "temperature_2m": [6.0, 7.0, 9.0, 11.0, 13.0, 15.0, 16.0, 16.5, 15.0, 13.0, 11.0, 10.0, 8.0, 7.0, 6.0],
        "apparent_temperature": [4.0, 5.0, 7.0, 9.0, 12.0, 14.0, 15.0, 15.5, 14.0, 12.0, 9.0, 8.0, 6.0, 5.0, 4.0],
        "precipitation_probability": [0] * 15,
        "precipitation": [0.0] * 15,
        "weather_code": [1] * 15,
        "wind_speed_10m": [8.0, 9.0, 10.0, 11.0, 12.0, 12.0, 11.0, 10.0, 9.0, 8.0, 8.0, 7.0, 6.0, 6.0, 5.0],
        "wind_gusts_10m": [12.0, 14.0, 15.0, 16.0, 17.0, 17.0, 16.0, 15.0, 14.0, 12.0, 12.0, 10.0, 9.0, 9.0, 8.0],
    }
    opt = find_optimal_outdoor_windows(hourly_mock)
    assert opt["has_data"] is True
    best = opt["best_window"]
    assert "label" in best
    assert best["avg_score"] > 60
    assert "temp_f" in best
    assert "wind_mph" in best
    print(f"Optimal Window: {best['label']} (Score: {best['avg_score']}/100, Temp: {best['temp_f']}°F)")
    print("✅ Optimal cycling window finder passed successfully!\n")


def test_memory_system():
    print("Testing Baju's Persistent Personal Memory Service...")
    profile = load_profile()
    assert profile["user"]["nickname"] == "Baju"
    assert "Ferrari" in profile["sports"]["f1_teams"] or "Scuderia Ferrari" in profile["sports"]["f1_teams"]
    assert "Chicago Bears" in profile["sports"]["nfl_teams"]

    # Test adding a custom memory
    success = add_custom_memory("Test Memory: Baju prefers riding with sunglasses.")
    assert success is True

    # Test memory profile markdown
    md = format_profile_markdown()
    assert "Aura Memory & Personal Profile for Baju" in md
    assert "Test Memory" in md

    # Clean up test note
    p2 = load_profile()
    notes = p2.get("custom_notes", [])
    for idx, n in enumerate(notes):
        if "Test Memory" in n.get("note", ""):
            remove_custom_memory(idx)
            break

    print("✅ Memory system passed successfully!\n")


def test_tts_service():
    print("Testing Zero-Cost TTS Markdown Cleaner...")
    sample_md = """# 🌤️ Aura Daily Briefing for Baju
> Hello Baju! Here is your weather.
- **Condition:** ☀️ Clear sky
- **Temperature:** 15°C (59°F)
| Metric | Value |
| --- | --- |
| Rain | 0% |
"""
    clean_speech = clean_markdown_for_speech(sample_md)
    assert "Hello Baju" in clean_speech
    assert "#" not in clean_speech
    assert "|" not in clean_speech
    print(f"Cleaned Speech: '{clean_speech[:80]}...'")
    print("✅ TTS markdown cleaner passed successfully!\n")


def test_graph_memory_intents():
    print("Testing LangGraph Memory Intents...")
    # View memory
    res_view = graph.invoke({
        "messages": [HumanMessage(content="What do you remember about me?")]
    })
    assert res_view.get("final_report")
    assert "Aura Memory & Personal Profile for Baju" in res_view["final_report"]

    # Store memory
    res_store = graph.invoke({
        "messages": [HumanMessage(content="Remember that I prefer cycling on paved bike paths")]
    })
    assert res_store.get("final_report")
    assert "Memory Updated for Baju" in res_store["final_report"]
    print("✅ Memory graph intents passed successfully!\n")


if __name__ == "__main__":
    try:
        test_optimal_window_service()
        test_memory_system()
        test_tts_service()
        test_news_service()
        test_events_and_sports_service()
        test_graph_memory_intents()
        test_graph_end_to_end()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
