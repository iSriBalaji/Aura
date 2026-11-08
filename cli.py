#!/usr/bin/env python3
"""Interactive CLI for Weather, Clothing & Cycling LangGraph Agent."""

import argparse
import sys
from langchain_core.messages import HumanMessage
from weather_agent.graph import graph


def run_agent(location: str = None, lat: float = None, lon: float = None, message: str = None):
    """Invoke the LangGraph agent with given parameters."""
    print("⏳ Invoking LangGraph Weather & Activity Advisor...")
    initial_state = {}

    if lat is not None and lon is not None:
        initial_state["latitude"] = lat
        initial_state["longitude"] = lon
    elif location:
        initial_state["location_query"] = location

    if message:
        initial_state["messages"] = [HumanMessage(content=message)]
    else:
        prompt_text = "What is the weather today and this week, what should I wear, and is it good for cycling?"
        if location:
            prompt_text += f" for {location}"
        initial_state["messages"] = [HumanMessage(content=prompt_text)]

    result = graph.invoke(initial_state)
    report = result.get("final_report")
    print("\n" + "=" * 70)
    print(report)
    print("=" * 70 + "\n")


def interactive_mode():
    """Run an interactive prompt loop."""
    print("\n🚴 Welcome to the Weather, Clothing & Cycling Advisor (LangGraph)!")
    print("Type your question or location (e.g. 'Can I cycle in Seattle tomorrow?', 'Austin, TX', or 'today' for device location).")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("🌦️ Your query > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            run_agent(message=user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Stay safe out there!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Reliable & Cost-Effective LangGraph Weather, Clothing, and Cycling Advisor."
    )
    parser.add_argument(
        "-l", "--location", type=str, help="Specific location name or address (e.g. 'Austin, TX', 'London', 'Tokyo')"
    )
    parser.add_argument(
        "--lat", type=float, help="Explicit latitude (e.g. 30.2672)"
    )
    parser.add_argument(
        "--lon", type=float, help="Explicit longitude (e.g. -97.7431)"
    )
    parser.add_argument(
        "-m", "--message", type=str, help="Custom question / prompt"
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Launch interactive chat mode"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    else:
        run_agent(location=args.location, lat=args.lat, lon=args.lon, message=args.message)


if __name__ == "__main__":
    main()
