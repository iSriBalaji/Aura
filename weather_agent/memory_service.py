"""Zero-cost persistent personal memory service for Aura.
Stores Baju's preferences, sports favorites, cycling comfort thresholds,
and custom learned facts in a lightweight local JSON profile.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "baju_profile.json"


def get_default_profile() -> Dict[str, Any]:
    """Default fallback profile for Baju."""
    return {
        "user": {
            "full_name": "Sri Balaji",
            "nickname": "Baju",
            "home_location": "Prospect Heights, Illinois",
            "timezone": "America/Chicago",
            "units": "Fahrenheit / mph",
        },
        "sports": {
            "f1_teams": ["Scuderia Ferrari", "McLaren"],
            "f1_drivers": ["Lando Norris", "Charles Leclerc"],
            "nfl_teams": ["Chicago Bears"],
            "cricket_teams": ["India National Cricket Team", "Chennai Super Kings (CSK)"],
        },
        "cycling": {
            "bike_type": "Road bike (aerodynamic drop bars)",
            "min_comfortable_temp_f": 45.0,
            "max_acceptable_wind_mph": 18.0,
            "max_acceptable_gusts_mph": 24.0,
            "preferred_ride_window_hours": 2.5,
            "ideal_ride_time_of_day": "Early to mid-afternoon (warmest pavement)",
        },
        "news_and_interests": {
            "topics": [
                "Artificial Intelligence & Technology",
                "Formula 1 & Motorsport",
                "Global Economy & Financial Markets",
                "Space & Frontier Science",
            ],
            "digest_style": "Concise 2-3 line bullet takeaways (What happened, Context, What to know)",
        },
        "custom_notes": [
            {
                "timestamp": "2026-11-08T08:00:00",
                "note": "Prefers cycling in the early afternoon when the road warms up.",
            },
            {
                "timestamp": "2026-11-08T08:00:00",
                "note": "Always alert on wind gusts above 20 mph before outdoor rides.",
            },
            {
                "timestamp": "2026-11-08T08:00:00",
                "note": "Aura should always address him warmly as Baju.",
            },
        ],
    }


def load_profile() -> Dict[str, Any]:
    """Load the persistent JSON profile or initialize default if not found."""
    if not PROFILE_PATH.exists():
        default_data = get_default_profile()
        save_profile(default_data)
        return default_data

    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception:
        return get_default_profile()


def save_profile(data: Dict[str, Any]) -> bool:
    """Save profile dictionary to local JSON file safely."""
    try:
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def add_custom_memory(note: str) -> bool:
    """Append a new personal note or preference to Baju's memory."""
    note = note.strip()
    if not note:
        return False

    profile = load_profile()
    notes = profile.setdefault("custom_notes", [])
    notes.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "note": note,
    })
    return save_profile(profile)


def remove_custom_memory(index: int) -> bool:
    """Remove a custom memory by index."""
    profile = load_profile()
    notes = profile.get("custom_notes", [])
    if 0 <= index < len(notes):
        notes.pop(index)
        return save_profile(profile)
    return False


def get_memory_summary_prompt() -> str:
    """Format Baju's profile for injection into LLM system prompts."""
    p = load_profile()
    usr = p.get("user", {})
    sports = p.get("sports", {})
    cyc = p.get("cycling", {})
    news = p.get("news_and_interests", {})
    notes = p.get("custom_notes", [])

    lines = [
        "=== BAJU'S PERSONAL PROFILE & MEMORY (ZERO COST LOCAL CACHE) ===",
        f"• User: {usr.get('full_name')} (Preferred Nickname: {usr.get('nickname', 'Baju')})",
        f"• Home Base: {usr.get('home_location')} ({usr.get('timezone')})",
        f"• Preferred Units: {usr.get('units')}",
        f"• Favorite Sports & Teams:",
        f"  - Formula 1: Teams: {', '.join(sports.get('f1_teams', []))} | Drivers: {', '.join(sports.get('f1_drivers', []))}",
        f"  - NFL: {', '.join(sports.get('nfl_teams', []))}",
        f"  - Cricket: {', '.join(sports.get('cricket_teams', []))}",
        f"• Cycling Comfort Thresholds:",
        f"  - Minimum comfortable temp: {cyc.get('min_comfortable_temp_f')}°F",
        f"  - Maximum wind tolerance: {cyc.get('max_acceptable_wind_mph')} mph (Gust limit: {cyc.get('max_acceptable_gusts_mph')} mph)",
        f"  - Typical ride duration: {cyc.get('preferred_ride_window_hours')} hours",
        f"• Key News Focus: {', '.join(news.get('topics', []))}",
    ]

    if notes:
        lines.append("• Personal Notes & Past Instructions:")
        for n in notes[-5:]:  # Most recent 5 notes
            lines.append(f"  - {n.get('note')}")

    lines.append("================================================================")
    return "\n".join(lines)


def format_profile_markdown() -> str:
    """Return a clean, minimal Markdown presentation of Baju's profile."""
    p = load_profile()
    usr = p.get("user", {})
    sports = p.get("sports", {})
    cyc = p.get("cycling", {})
    news = p.get("news_and_interests", {})
    notes = p.get("custom_notes", [])

    lines = [
        "### Aura Memory & Personal Profile for Baju",
        "> Preferences, cycling thresholds, and sports teams stored locally.\n",
        "#### Personal Identity",
        f"- **Name:** {usr.get('full_name')} (**{usr.get('nickname')}**)",
        f"- **Home Location:** {usr.get('home_location')} ({usr.get('timezone')})",
        f"- **Preferred Units:** {usr.get('units')}\n",
        "#### Favorite Sports & Teams",
        f"- **Formula 1 Teams:** {', '.join(sports.get('f1_teams', []))}",
        f"- **Favorite Drivers:** {', '.join(sports.get('f1_drivers', []))}",
        f"- **NFL:** {', '.join(sports.get('nfl_teams', []))}",
        f"- **Cricket:** {', '.join(sports.get('cricket_teams', []))}\n",
        "#### Cycling Profile & Thresholds",
        f"- **Bike Style:** {cyc.get('bike_type')}",
        f"- **Cold Tolerance:** Min {cyc.get('min_comfortable_temp_f')}°F (Apparent temp)",
        f"- **Wind Tolerance:** Up to {cyc.get('max_acceptable_wind_mph')} mph (Gusts < {cyc.get('max_acceptable_gusts_mph')} mph)",
        f"- **Ideal Ride Window:** {cyc.get('preferred_ride_window_hours')} hours ({cyc.get('ideal_ride_time_of_day')})\n",
        "#### News & Focus Areas",
        f"- **Topics:** {', '.join(news.get('topics', []))}",
        f"- **Format:** {news.get('digest_style')}\n",
        "#### Stored Custom Memories",
    ]

    if notes:
        for idx, n in enumerate(notes, 1):
            ts = n.get("timestamp", "")
            lines.append(f"{idx}. {n.get('note')} *(Saved {ts})*")
    else:
        lines.append("*No custom memories stored yet. Tell Aura 'Remember that...' anytime!*")

    return "\n".join(lines)


def detect_memory_intent(text: str) -> Optional[Dict[str, Any]]:
    """Detect whether user wants to store a new memory or view their profile."""
    text_clean = text.strip()

    # Query / view memory
    view_patterns = [
        r"\b(?:what do you remember|show my profile|view my profile|what is my memory|what are my preferences|my profile|show memory)\b",
        r"\b(?:who am i|tell me what you know about me)\b",
    ]
    for pat in view_patterns:
        if re.search(pat, text_clean, re.IGNORECASE):
            return {"action": "view"}

    # Store memory
    store_patterns = [
        r"^(?:aura[, ]+)?(?:can you\s+)?(?:please\s+)?remember\s+(?:that\s+)?(.+)$",
        r"^(?:please\s+)?save\s+to\s+memory[:\s]+(.+)$",
        r"^(?:please\s+)?keep\s+in\s+mind\s+(?:that\s+)?(.+)$",
        r"^(?:note\s+down|take\s+note)[:\s]+(?:that\s+)?(.+)$",
    ]
    for pat in store_patterns:
        m = re.search(pat, text_clean, re.IGNORECASE)
        if m:
            note_content = m.group(1).strip()
            # Clean up trailing punctuation or quotes
            note_content = re.sub(r"^['\"]|['\"]$", "", note_content).strip()
            return {"action": "store", "note": note_content}

    return None
