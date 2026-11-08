#!/usr/bin/env python3
"""Interactive CLI for Weather, Wardrobe, Cycling & Trending News LangGraph Agent."""

import argparse
import sys
from langchain_core.messages import HumanMessage
from weather_agent.graph import graph


def run_agent(
    location: str = None,
    lat: float = None,
    lon: float = None,
    message: str = None,
    news: bool = False,
    news_topic: str = None,
    deepdive: str = None,
):
    """Invoke the LangGraph agent with given parameters."""
    print("⏳ Invoking LangGraph Advisor...")
    initial_state = {}

    if deepdive:
        initial_state["intent"] = "news_deepdive"
        initial_state["news_query"] = deepdive
        initial_state["messages"] = [HumanMessage(content=f"Tell me more about {deepdive}")]
    elif news or news_topic:
        initial_state["intent"] = "news_digest"
        if news_topic:
            initial_state["news_categories"] = [news_topic]
            initial_state["messages"] = [HumanMessage(content=f"What is trending in {news_topic}?")]
        else:
            initial_state["messages"] = [HumanMessage(content="What are the trending important news stories?")]
    elif message:
        initial_state["messages"] = [HumanMessage(content=message)]
        if location:
            initial_state["location_query"] = location
    else:
        if lat is not None and lon is not None:
            initial_state["latitude"] = lat
            initial_state["longitude"] = lon
        elif location:
            initial_state["location_query"] = location

        prompt_text = "What is the weather today and this week, what should I wear, and is it good for cycling?"
        if location:
            prompt_text += f" for {location}"
        initial_state["messages"] = [HumanMessage(content=prompt_text)]

    result = graph.invoke(initial_state)
    report = result.get("final_report", "No report generated.")
    print("\n" + "=" * 70)
    print(report)
    print("=" * 70 + "\n")


def interactive_mode():
    """Run an interactive prompt loop supporting weather, news, and deep-dives."""
    print("\n🌤️ 🚴 📰 Welcome to the Weather, Activity & Trending News Advisor (LangGraph)!")
    print("You can ask about:")
    print("  • Weather & Cycling: 'Can I cycle in Seattle tomorrow?', 'Prospect Heights, IL', 'Today'")
    print("  • Trending News: 'What is trending in news today?', 'Tech headlines', 'Science news'")
    print("  • Follow-Up Deep-Dive: 'Tell me more about NASA Artemis', 'Details on [headline]'")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("🌦️ / 📰 Your query > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            run_agent(message=user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Stay informed and safe out there!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Reliable & Cost-Effective LangGraph Weather, Wardrobe, Cycling & Trending News Advisor."
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
        "-n", "--news", action="store_true", help="Fetch current trending important news digest"
    )
    parser.add_argument(
        "--news-topic", type=str, choices=["technology", "politics", "science", "general"],
        help="Filter news by topic"
    )
    parser.add_argument(
        "--deepdive", type=str, help="Get additional background context on a specific news topic"
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Launch interactive chat mode"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    else:
        run_agent(
            location=args.location,
            lat=args.lat,
            lon=args.lon,
            message=args.message,
            news=args.news,
            news_topic=args.news_topic,
            deepdive=args.deepdive,
        )


if __name__ == "__main__":
    main()
