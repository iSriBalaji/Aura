"""State definition for Weather, Wardrobe, Cycling, Trending News, Events & Sports agent."""

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class WeatherAgentState(TypedDict, total=False):
    """LangGraph state representation for weather, wardrobe, cycling, trending news, events, and sports."""
    # Conversation messages (standard LangGraph message list)
    messages: Annotated[List[AnyMessage], add_messages]

    # Intent routing: 'weather', 'news_digest', 'news_deepdive', 'events_sports', 'combined', 'memory_view', 'memory_store'
    intent: Optional[str]
    memory_note: Optional[str]

    # Geolocation / target location
    location_query: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    location_name: Optional[str]
    timezone: Optional[str]

    # Weather data & decisions
    weather_data: Optional[Dict[str, Any]]
    decision_analysis: Optional[Dict[str, Any]]

    # Trending news data
    news_categories: Optional[List[str]]
    news_data: Optional[Dict[str, Any]]
    news_query: Optional[str]
    news_deepdive_items: Optional[List[Dict[str, Any]]]

    # Events & Sports data
    events_data: Optional[Dict[str, Any]]        # US conferences, music festivals, etc.
    sports_data: Optional[Dict[str, Any]]        # NFL, F1, Cricket, general sports
    city_events: Optional[List[Dict[str, Any]]]  # Location-specific events/festivals
    sports_filter: Optional[List[str]]          # Specific sports requested (e.g. ['f1', 'cricket'])

    # Model selection & optional overrides
    model_name: Optional[str]

    # Final synthesized user-facing markdown response
    final_report: Optional[str]
