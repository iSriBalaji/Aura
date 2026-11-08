"""State definition for the Weather, Activity, and Trending News LangGraph agent."""

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class WeatherAgentState(TypedDict, total=False):
    """LangGraph state representation for weather, wardrobe, cycling, and trending news."""
    # Conversation messages (standard LangGraph message list)
    messages: Annotated[List[AnyMessage], add_messages]

    # Intent routing: 'weather', 'news_digest', 'news_deepdive', 'combined'
    intent: Optional[str]

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
    news_categories: Optional[List[str]]  # e.g. ['technology', 'science', 'politics', 'general']
    news_data: Optional[Dict[str, Any]]   # structured news digest
    news_query: Optional[str]             # query for deep dive
    news_deepdive_items: Optional[List[Dict[str, Any]]]

    # Final synthesized user-facing markdown response
    final_report: Optional[str]
