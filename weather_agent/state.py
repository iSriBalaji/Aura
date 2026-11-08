"""State definition for the Weather, Clothing & Cycling LangGraph agent."""

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class WeatherAgentState(TypedDict, total=False):
    """LangGraph state representation for the weather advisor."""
    # Conversation messages (standard LangGraph message list)
    messages: Annotated[List[AnyMessage], add_messages]

    # Geolocation / target location
    location_query: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    location_name: Optional[str]
    timezone: Optional[str]

    # Raw / structured weather data from Open-Meteo
    weather_data: Optional[Dict[str, Any]]

    # Algorithmic decision assessment for clothing and cycling
    decision_analysis: Optional[Dict[str, Any]]

    # Final synthesized user-facing markdown response
    final_report: Optional[str]
