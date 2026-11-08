"""LangGraph Workflow definition for Weather, Wardrobe, Cycling, Trending News, Events & Sports.
Supports multi-intent routing: weather decisions, trending news digests,
news follow-ups, and US/global events & sports briefings.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Literal
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from weather_agent.decision_engine import analyze_weather
from weather_agent.events_service import (
    fetch_city_events,
    fetch_major_us_events,
    fetch_sports_briefing,
)
from weather_agent.geolocation import (
    detect_device_location,
    extract_location_from_text,
    geocode_location,
)
from weather_agent.llm_synthesizer import (
    synthesize_events_and_sports,
    synthesize_news_deepdive,
    synthesize_news_digest,
    synthesize_weather_report,
)
from weather_agent.memory_service import (
    add_custom_memory,
    detect_memory_intent,
    format_profile_markdown,
)
from weather_agent.news_service import (
    fetch_trending_news,
    search_news_topic,
)
from weather_agent.state import WeatherAgentState
from weather_agent.weather_service import fetch_weather_forecast


def route_intent_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Determine the user intent:
    - events_sports: US conferences, music festivals, sports (NFL, F1, Cricket), city events
    - news_deepdive: detailed follow-up on a specific headline
    - news_digest: general/tech/politics/science trending news
    - combined: weather + news/events
    - weather: weather, wardrobe, and cycling decisions (default)
    """
    if state.get("intent"):
        return {"intent": state["intent"]}

    last_text = ""
    if state.get("messages"):
        last_msg = state["messages"][-1]
        last_text = (
            last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
        )

    # 0. Personal Memory cues (view profile or store instruction)
    mem_intent = detect_memory_intent(last_text)
    if mem_intent:
        if mem_intent["action"] == "view":
            return {"intent": "memory_view"}
        elif mem_intent["action"] == "store":
            return {
                "intent": "memory_store",
                "memory_note": mem_intent.get("note", ""),
            }

    # 1. News Deep Dive / Follow-up cues
    deep_dive_patterns = [
        r"(?:tell me more about|more details (?:on|about)|more context (?:on|about)|elaborate on|deep dive (?:on|into)|what happened with|details on)\s+['\"]?([^'\"?]+)['\"]?",
        r"(?:tell me about|info on|explain)\s+(?:item\s+|#)?(\d+|[A-Za-z0-9\s-]+)",
    ]
    for pat in deep_dive_patterns:
        m = re.search(pat, last_text, re.IGNORECASE)
        if m:
            query_target = m.group(1).strip()
            return {
                "intent": "news_deepdive",
                "news_query": query_target,
            }

    # 2. Events & Sports cues
    is_events_or_sports = bool(
        re.search(
            r"\b(event|events|conference|conferences|festival|festivals|concert|concerts|expo|sports|sport|football|nfl|formula 1|formula one|f1|cricket|nba|super bowl|grand prix)\b",
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

    is_news = bool(
        re.search(
            r"\b(news|trending|headlines|happening in the world|talking about|politics|tech news|technology news|science news|business news|business|market|markets|economy|local news)\b",
            last_text,
            re.IGNORECASE,
        )
    )

    if is_events_or_sports:
        # Check sports filters
        sports_filter = []
        if re.search(r"\b(football|nfl|cfb)\b", last_text, re.IGNORECASE):
            sports_filter.append("football")
        if re.search(r"\b(f1|formula 1|formula one|grand prix)\b", last_text, re.IGNORECASE):
            sports_filter.append("f1")
        if re.search(r"\b(cricket|ipl|icc)\b", last_text, re.IGNORECASE):
            sports_filter.append("cricket")

        # Check if city was mentioned for local events
        city = extract_location_from_text(last_text) or state.get("location_query")

        return {
            "intent": "events_sports",
            "sports_filter": sports_filter or ["football", "f1", "cricket", "general_sports"],
            "location_query": city,
        }

    if is_news and is_weather:
        return {"intent": "combined"}
    elif is_news:
        cats = ["general", "technology", "politics", "business", "science", "local"]
        requested_cats = [c for c in cats if re.search(rf"\b{c}\b", last_text, re.IGNORECASE)]
        if re.search(r"\b(world|international)\b", last_text, re.IGNORECASE) and "general" not in requested_cats:
            requested_cats.append("general")
        if re.search(r"\b(market|markets|economy)\b", last_text, re.IGNORECASE) and "business" not in requested_cats:
            requested_cats.append("business")
        if re.search(r"\b(local)\b", last_text, re.IGNORECASE) and "local" not in requested_cats:
            requested_cats.append("local")

        return {
            "intent": "news_digest",
            "news_categories": requested_cats or cats,
        }
    else:
        return {"intent": "weather"}


def resolve_location_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Resolve location for weather queries."""
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
    """Fetch trending news across general, technology, politics, business, science, and local."""
    cats = state.get("news_categories") or ["general", "technology", "politics", "business", "science", "local"]
    loc_name = state.get("location_name") or state.get("location_query")
    if not loc_name and state.get("messages"):
        last_msg = state["messages"][-1]
        msg_text = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
        loc_name = extract_location_from_text(msg_text)
    if not loc_name:
        loc_name = "Prospect Heights, Illinois"

    news_data = fetch_trending_news(categories=cats, location_name=loc_name, limit_per_cat=2)
    return {"news_data": news_data, "location_name": loc_name}


def fetch_news_deepdive_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Fetch targeted news search results for an in-depth follow-up on a specific item."""
    query = state.get("news_query") or "trending news"
    items = search_news_topic(query, limit=5)
    return {"news_deepdive_items": items}


def fetch_events_and_sports_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Fetch major US events, sports updates (football, F1, cricket), and city events strictly within ±17 days of today."""
    ref_date = datetime.now()
    sports_filter = state.get("sports_filter") or ["football", "f1", "cricket", "general_sports"]
    us_events = fetch_major_us_events(ref_date=ref_date, window_days=17)
    sports = fetch_sports_briefing(sports_filter, ref_date=ref_date, window_days=17)

    city_name = state.get("location_query")
    city_events = []
    if city_name:
        city_events = fetch_city_events(city_name, ref_date=ref_date, window_days=17)

    return {
        "events_data": us_events,
        "sports_data": sports,
        "city_events": city_events,
    }


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
    """Generate trending news digest report with structured 2-3 line summaries."""
    news_data = state["news_data"]
    loc_name = state.get("location_name") or "Prospect Heights, Illinois"

    user_query = None
    if state.get("messages"):
        last_msg = state["messages"][-1]
        if isinstance(last_msg, HumanMessage) or getattr(last_msg, "type", None) == "human":
            user_query = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

    final_report = synthesize_news_digest(news_data, location_name=loc_name, user_query=user_query)
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


def synthesize_events_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Generate events, festivals, and sports conversation report."""
    us_events = state.get("events_data")
    sports = state.get("sports_data")
    city_name = state.get("location_query")
    city_events = state.get("city_events")
    ref_date = datetime.now()

    user_query = None
    if state.get("messages"):
        last_msg = state["messages"][-1]
        if isinstance(last_msg, HumanMessage) or getattr(last_msg, "type", None) == "human":
            user_query = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

    final_report = synthesize_events_and_sports(
        us_events=us_events,
        sports=sports,
        city_name=city_name,
        city_events=city_events,
        user_query=user_query,
        ref_date=ref_date,
    )
    return {
        "final_report": final_report,
        "messages": [AIMessage(content=final_report)],
    }


def synthesize_combined_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Generate combined weather + trending news report."""
    loc_name = state.get("location_name", "Prospect Heights, Illinois")
    analysis = state["decision_analysis"]
    news_data = state["news_data"]

    weather_rep = synthesize_weather_report(location_name=loc_name, analysis=analysis)
    news_rep = synthesize_news_digest(news_data, location_name=loc_name)

    combined = f"{weather_rep}\n\n---\n\n{news_rep}"
    return {
        "final_report": combined,
        "messages": [AIMessage(content=combined)],
    }


def handle_memory_node(state: WeatherAgentState) -> Dict[str, Any]:
    """Handle memory view or store actions for Baju."""
    intent = state.get("intent")
    if intent == "memory_store":
        note = state.get("memory_note", "").strip()
        if note:
            add_custom_memory(note)
            resp = (
                f"# 🧠 Memory Updated for Baju\n\n"
                f"> **Got it, Baju!** I've locked this into my persistent memory:\n\n"
                f"📌 *\"{note}\"*\n\n"
                f"I will keep this in mind across all your future weather, cycling, and news briefings. "
                f"You can view everything I remember anytime by asking *\"Show my memory\"* or *\"What do you remember about me?\"*."
            )
        else:
            resp = "Hello Baju! Please tell me what you'd like me to remember (e.g. *\"Remember that I prefer morning rides\"*)."
    else:
        resp = format_profile_markdown()

    return {
        "final_report": resp,
        "messages": [AIMessage(content=resp)],
    }


# Routing Condition Function
def choose_intent_branch(
    state: WeatherAgentState,
) -> Literal["weather_branch", "news_branch", "deepdive_branch", "events_branch", "combined_branch", "memory_branch"]:
    intent = state.get("intent", "weather")
    if intent in ("memory_view", "memory_store"):
        return "memory_branch"
    elif intent == "events_sports":
        return "events_branch"
    elif intent == "news_deepdive":
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
builder.add_node("fetch_events_and_sports", fetch_events_and_sports_node)
builder.add_node("handle_memory", handle_memory_node)
builder.add_node("synthesize_weather", synthesize_weather_node)
builder.add_node("synthesize_news", synthesize_news_node)
builder.add_node("synthesize_deepdive", synthesize_deepdive_node)
builder.add_node("synthesize_events", synthesize_events_node)
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
        "events_branch": "fetch_events_and_sports",
        "combined_branch": "resolve_location",
        "memory_branch": "handle_memory",
    },
)

# Weather Branch: resolve_location -> fetch_weather -> decision_engine
builder.add_edge("resolve_location", "fetch_weather")
builder.add_edge("fetch_weather", "decision_engine")

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

# Deep Dive Branch
builder.add_edge("fetch_news_deepdive", "synthesize_deepdive")

# Events & Sports Branch
builder.add_edge("fetch_events_and_sports", "synthesize_events")

# Memory Branch
builder.add_edge("handle_memory", END)

# Terminal Edges to END
builder.add_edge("synthesize_weather", END)
builder.add_edge("synthesize_news", END)
builder.add_edge("synthesize_deepdive", END)
builder.add_edge("synthesize_events", END)
builder.add_edge("synthesize_combined", END)

# Compile Graph
graph = builder.compile()
