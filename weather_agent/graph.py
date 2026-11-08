"""LangGraph Workflow definition for the Weather, Clothing & Cycling Agent.
Compiles a production-ready StateGraph ready for local execution and LangGraph Cloud.
"""

from typing import Any, Dict
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from weather_agent.decision_engine import analyze_weather
from weather_agent.geolocation import (
    detect_device_location,
    extract_location_from_text,
    geocode_location,
)
from weather_agent.llm_synthesizer import synthesize_report
from weather_agent.state import WeatherAgentState
from weather_agent.weather_service import fetch_weather_forecast


def resolve_location_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Resolve location from explicit coordinates, user query string,
    message history, or auto-detect from device IP.
    """
    lat = state.get("latitude")
    lon = state.get("longitude")
    loc_name = state.get("location_name")
    tz = state.get("timezone") or "auto"

    # 1. If explicit latitude and longitude already supplied
    if lat is not None and lon is not None:
        if not loc_name:
            loc_name = f"Location ({lat:.2f}, {lon:.2f})"
        return {
            "latitude": float(lat),
            "longitude": float(lon),
            "location_name": loc_name,
            "timezone": tz,
        }

    # 2. Check if explicit location_query is set
    query = state.get("location_query")

    # 3. If no explicit location_query, check user messages
    if not query and state.get("messages"):
        last_msg = state["messages"][-1]
        msg_text = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
        query = extract_location_from_text(msg_text)

    # 4. If query found, geocode it
    if query:
        geo = geocode_location(query)
        if geo:
            return {
                "latitude": geo["latitude"],
                "longitude": geo["longitude"],
                "location_name": geo["name"],
                "timezone": geo["timezone"],
                "location_query": query,
            }

    # 5. Fallback: Auto-detect device location
    device = detect_device_location()
    return {
        "latitude": device["latitude"],
        "longitude": device["longitude"],
        "location_name": device["name"],
        "timezone": device["timezone"],
        "location_query": None,
    }


def fetch_weather_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Fetch live and 7-day weather forecast from Open-Meteo."""
    lat = state["latitude"]
    lon = state["longitude"]
    tz = state.get("timezone", "auto")

    weather_data = fetch_weather_forecast(lat, lon, tz)
    return {"weather_data": weather_data}


def decision_engine_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Calculate clothing layer decisions and cycling suitability metrics."""
    weather_data = state["weather_data"]
    analysis = analyze_weather(weather_data)
    return {"decision_analysis": analysis}


def synthesize_report_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Generate final comprehensive report and attach to message thread."""
    loc_name = state.get("location_name", "Your Location")
    analysis = state["decision_analysis"]

    user_query = None
    if state.get("messages"):
        last_msg = state["messages"][-1]
        if isinstance(last_msg, HumanMessage) or getattr(last_msg, "type", None) == "human":
            user_query = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

    final_report = synthesize_report(
        location_name=loc_name,
        analysis=analysis,
        user_query=user_query,
    )

    ai_message = AIMessage(content=final_report)
    return {
        "final_report": final_report,
        "messages": [ai_message],
    }


# Define StateGraph
builder = StateGraph(WeatherAgentState)

# Add Nodes
builder.add_node("resolve_location", resolve_location_node)
builder.add_node("fetch_weather", fetch_weather_node)
builder.add_node("decision_engine", decision_engine_node)
builder.add_node("synthesize_report", synthesize_report_node)

# Add Edges
builder.add_edge(START, "resolve_location")
builder.add_edge("resolve_location", "fetch_weather")
builder.add_edge("fetch_weather", "decision_engine")
builder.add_edge("decision_engine", "synthesize_report")
builder.add_edge("synthesize_report", END)

# Compile Graph (exportable for LangGraph Cloud and local imports)
graph = builder.compile()
