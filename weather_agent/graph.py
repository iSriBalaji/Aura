"""LangGraph Workflow definition for Weather, Wardrobe, Cycling & Trending News.
Supports multi-intent routing: weather decisions, trending news digests,
and interactive in-depth news follow-ups.
"""

import re
from typing import Any, Dict, List, Literal
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from weather_agent.decision_engine import analyze_weather
from weather_agent.geolocation import (
    detect_device_location,
    extract_location_from_text,
    geocode_location,
)
from weather_agent.llm_synthesizer import (
    synthesize_news_deepdive,
    synthesize_news_digest,
    synthesize_weather_report,
)
from weather_agent.news_service import (
    fetch_trending_news,
    search_news_topic,
)
from weather_agent.state import WeatherAgentState
from weather_agent.weather_service import fetch_weather_forecast


def route_intent_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Determine the intent: weather, news_digest, news_deepdive, or combined."""
    # If state already has intent explicitly set
    if state.get("intent"):
        return {"intent": state["intent"]}

    last_text = ""
    if state.get("messages"):
        last_msg = state["messages"][-1]
        last_text = (
            last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
        )

    # 1. Check for News Deep Dive / Follow-up patterns
    deep_dive_patterns = [
        r"(?:tell me more about|more details (?:on|about)|more context (?:on|about)|elaborate on|deep dive (?:on|into)|what happened with|details on)\s+['\"]?([^'\"?]+)['\"]?",
        r"(?:tell me about|info on|explain)\s+(?:item\s+|#)?(\d+|[A-Za-z0-9\s-]+)",
    ]
    for pat in deep_dive_patterns:
        m = re.search(pat, last_text, re.IGNORECASE)
        if m:
            query_target = m.group(1).strip()
            # If the user asked for a number like "#2" and we have previous news items
            # or if they gave a name
            return {
                "intent": "news_deepdive",
                "news_query": query_target,
            }

    # 2. Check for News vs Weather keywords
    is_news = bool(
        re.search(
            r"\b(news|trending|headlines|happening|talking about|politics|tech news|science news|world events)\b",
            last_text,
            re.IGNORECASE,
        )
    )
    is_weather = bool(
        re.search(
            r"\b(weather|forecast|rain|temperature|wear|clothes|clothing|layers|cycling|bike|biking|ride)\b",
            last_text,
            re.IGNORECASE,
        )
    )

    if is_news and is_weather:
        return {"intent": "combined"}
    elif is_news:
        # Check if specific category requested
        cats = ["general", "technology", "politics", "science"]
        requested_cats = []
        for c in cats:
            if re.search(rf"\b{c}\b", last_text, re.IGNORECASE):
                requested_cats.append(c)
        return {
            "intent": "news_digest",
            "news_categories": requested_cats or cats,
        }
    else:
        return {"intent": "weather"}


def resolve_location_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Resolve location from explicit coordinates, user query string,
    message history, or auto-detect from device IP.
    """
    lat = state.get("latitude")
    lon = state.get("longitude")
    loc_name = state.get("location_name")
    tz = state.get("timezone") or "auto"

    if lat is not None and lon is not None:
        if not loc_name:
            loc_name = f"Location ({lat:.2f}, {lon:.2f})"
        return {
            "latitude": float(lat),
            "longitude": float(lon),
            "location_name": loc_name,
            "timezone": tz,
        }

    query = state.get("location_query")
    if not query and state.get("messages"):
        last_msg = state["messages"][-1]
        msg_text = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
        query = extract_location_from_text(msg_text)

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

    # Fallback to device location
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


def fetch_news_digest_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Fetch trending news across technology, politics, science, and general."""
    cats = state.get("news_categories") or ["general", "technology", "politics", "science"]
    news_data = fetch_trending_news(categories=cats, limit_per_cat=3)
    return {"news_data": news_data}


def fetch_news_deepdive_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Fetch targeted news search results for an in-depth follow-up on a specific item."""
    query = state.get("news_query") or "trending news"
    items = search_news_topic(query, limit=5)
    return {"news_deepdive_items": items}


def synthesize_weather_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Generate final weather, clothing & cycling report."""
    loc_name = state.get("location_name", "Your Location")
    analysis = state["decision_analysis"]

    user_query = None
    if state.get("messages"):
        last_msg = state["messages"][-1]
        if isinstance(last_msg, HumanMessage) or getattr(last_msg, "type", None) == "human":
            user_query = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

    final_report = synthesize_weather_report(
        location_name=loc_name,
        analysis=analysis,
        user_query=user_query,
    )
    return {
        "final_report": final_report,
        "messages": [AIMessage(content=final_report)],
    }


def synthesize_news_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Generate trending news digest report."""
    news_data = state["news_data"]

    user_query = None
    if state.get("messages"):
        last_msg = state["messages"][-1]
        if isinstance(last_msg, HumanMessage) or getattr(last_msg, "type", None) == "human":
            user_query = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

    final_report = synthesize_news_digest(news_data, user_query=user_query)
    return {
        "final_report": final_report,
        "messages": [AIMessage(content=final_report)],
    }


def synthesize_deepdive_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Generate in-depth news follow-up report."""
    query = state.get("news_query", "News Story")
    items = state.get("news_deepdive_items", [])

    user_query = None
    if state.get("messages"):
        last_msg = state["messages"][-1]
        if isinstance(last_msg, HumanMessage) or getattr(last_msg, "type", None) == "human":
            user_query = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

    final_report = synthesize_news_deepdive(query, items, user_query=user_query)
    return {
        "final_report": final_report,
        "messages": [AIMessage(content=final_report)],
    }


def synthesize_combined_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Generate combined weather + trending news report."""
    loc_name = state.get("location_name", "Your Location")
    analysis = state["decision_analysis"]
    news_data = state["news_data"]

    weather_rep = synthesize_weather_report(location_name=loc_name, analysis=analysis)
    news_rep = synthesize_news_digest(news_data)

    combined = f"{weather_rep}\n\n---\n\n{news_rep}"
    return {
        "final_report": combined,
        "messages": [AIMessage(content=combined)],
    }


# Routing Condition Function
def choose_intent_branch(state: WeatherAgentState) -> Literal["weather_branch", "news_branch", "deepdive_branch", "combined_branch"]:
    intent = state.get("intent", "weather")
    if intent == "news_deepdive":
        return "deepdive_branch"
    elif intent == "news_digest":
        return "news_branch"
    elif intent == "combined":
        return "combined_branch"
    return "weather_branch"


# Construct StateGraph
builder = StateGraph(WeatherAgentState)

# Add Nodes
builder.add_node("route_intent", route_intent_node)
builder.add_node("resolve_location", resolve_location_node)
builder.add_node("fetch_weather", fetch_weather_node)
builder.add_node("decision_engine", decision_engine_node)
builder.add_node("fetch_news_digest", fetch_news_digest_node)
builder.add_node("fetch_news_deepdive", fetch_news_deepdive_node)
builder.add_node("synthesize_weather", synthesize_weather_node)
builder.add_node("synthesize_news", synthesize_news_node)
builder.add_node("synthesize_deepdive", synthesize_deepdive_node)
builder.add_node("synthesize_combined", synthesize_combined_node)

# Add Conditional Edges from Router
builder.add_edge(START, "route_intent")
builder.add_conditional_edges(
    "route_intent",
    choose_intent_branch,
    {
        "weather_branch": "resolve_location",
        "news_branch": "fetch_news_digest",
        "deepdive_branch": "fetch_news_deepdive",
        "combined_branch": "resolve_location",
    },
)

# Weather Branch: resolve_location -> fetch_weather -> decision_engine -> synthesize_weather -> END
builder.add_edge("resolve_location", "fetch_weather")
builder.add_edge("fetch_weather", "decision_engine")

# Check if combined or pure weather
def check_weather_or_combined(state: WeatherAgentState) -> Literal["pure_weather", "go_to_news"]:
    if state.get("intent") == "combined":
        return "go_to_news"
    return "pure_weather"

builder.add_conditional_edges(
    "decision_engine",
    check_weather_or_combined,
    {
        "pure_weather": "synthesize_weather",
        "go_to_news": "fetch_news_digest",
    },
)

# Combined path continues to synthesize_combined
def check_after_news(state: WeatherAgentState) -> Literal["synthesize_news", "synthesize_combined"]:
    if state.get("intent") == "combined":
        return "synthesize_combined"
    return "synthesize_news"

builder.add_conditional_edges(
    "fetch_news_digest",
    check_after_news,
    {
        "synthesize_news": "synthesize_news",
        "synthesize_combined": "synthesize_combined",
    },
)

# News Deep Dive Branch: fetch_news_deepdive -> synthesize_deepdive -> END
builder.add_edge("fetch_news_deepdive", "synthesize_deepdive")

# Terminal Edges to END
builder.add_edge("synthesize_weather", END)
builder.add_edge("synthesize_news", END)
builder.add_edge("synthesize_deepdive", END)
builder.add_edge("synthesize_combined", END)

# Compile Graph
graph = builder.compile()
