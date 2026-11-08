"""Events and Sports Intelligence Service for US & Global talking points.
Fetches up-to-the-minute updates on major conferences, music festivals,
sports (American Football, Formula 1, Cricket, NBA), and city-specific happenings.
Zero-cost, highly reliable using live public RSS and search aggregation.
"""

import html
import re
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET
import httpx

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}


def clean_html(raw_html: str) -> str:
    """Strip HTML tags and unescape entities."""
    if not raw_html:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    return " ".join(text.split())


def parse_title_and_source(raw_title: str) -> Dict[str, str]:
    """Parse Google News title format ('Headline - Publisher Name')."""
    raw_title = html.unescape(raw_title or "")
    if " - " in raw_title:
        parts = raw_title.rsplit(" - ", 1)
        return {"title": parts[0].strip(), "source": parts[1].strip()}
    return {"title": raw_title.strip(), "source": "News"}


def parse_pub_date(raw_date: str) -> str:
    """Format RFC-822 date to a clean relative or standard string."""
    if not raw_date:
        return ""
    try:
        dt = datetime.strptime(raw_date.replace("GMT", "+0000").strip(), "%a, %d %b %Y %H:%M:%S %z")
        return dt.strftime("%b %d")
    except Exception:
        return raw_date.replace(" GMT", "")


def fetch_rss_items(url: str, limit: int = 3, timeout: float = 7.0) -> List[Dict[str, Any]]:
    """Helper to fetch and parse an RSS feed."""
    items = []
    try:
        with httpx.Client(headers=HTTP_HEADERS, follow_redirects=True, timeout=timeout) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                raw_items = root.findall(".//item")
                for it in raw_items[:limit]:
                    raw_title = it.findtext("title", "")
                    parsed = parse_title_and_source(raw_title)
                    link = it.findtext("link", "")
                    pub_date = it.findtext("pubDate", "")
                    clean_desc = clean_html(it.findtext("description", ""))

                    items.append({
                        "title": parsed["title"],
                        "source": parsed["source"],
                        "link": link,
                        "pub_date": parse_pub_date(pub_date),
                        "summary": clean_desc if clean_desc and clean_desc != parsed["title"] else "",
                    })
    except Exception:
        pass
    return items


def search_events_or_sports(query: str, limit: int = 3, timeout: float = 7.0) -> List[Dict[str, Any]]:
    """Search Google News for targeted event or sports topics."""
    encoded = urllib.parse.quote(query.strip())
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    return fetch_rss_items(url, limit=limit, timeout=timeout)


def fetch_sports_briefing(sports_filter: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch updates on American Football (NFL/CFB), Formula 1, Cricket, and General Sports."""
    sports = {}
    selected = sports_filter or ["football", "f1", "cricket", "general_sports"]

    if "football" in selected:
        sports["football"] = search_events_or_sports("NFL football highlights scores news", limit=3)
    if "f1" in selected:
        sports["f1"] = search_events_or_sports("Formula 1 Grand Prix qualifying race", limit=3)
    if "cricket" in selected:
        sports["cricket"] = search_events_or_sports("Cricket match tournament scores highlights", limit=3)
    if "general_sports" in selected:
        sports["general_sports"] = fetch_rss_items(
            "https://news.google.com/rss/headlines/section/topic/SPORTS?hl=en-US&gl=US&ceid=US:en",
            limit=3,
        )

    return sports


def fetch_major_us_events() -> Dict[str, List[Dict[str, Any]]]:
    """Fetch major US events: Conferences, Music Festivals, and Big Entertainment Events."""
    return {
        "conferences": search_events_or_sports("major tech conference or summit USA this week", limit=3),
        "music_festivals": search_events_or_sports("major music festival concert tour USA announced or this weekend", limit=3),
        "culture_entertainment": fetch_rss_items(
            "https://news.google.com/rss/headlines/section/topic/ENTERTAINMENT?hl=en-US&gl=US&ceid=US:en",
            limit=3,
        ),
    }


def fetch_city_events(city_name: str) -> List[Dict[str, Any]]:
    """Fetch events, festivals, concerts, or major sports games for a specific city."""
    clean_city = city_name.strip()
    query = f"events festivals concerts {clean_city} this week or weekend"
    return search_events_or_sports(query, limit=5)


def fallback_format_events_and_sports(
    us_events: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    sports: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    city_name: Optional[str] = None,
    city_events: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Deterministic, structured markdown report for Events and Sports talking points."""
    lines = []
    lines.append("# 🎟️ Major Events & Sports Conversation Briefing")
    lines.append("> **Quick Overview:** Highlights of major events in the US, key sports updates, and conversation-starter talking points you can share with people right now.\n")

    # City-specific events if requested
    if city_name and city_events:
        lines.append(f"## 📍 Happenings & Events in {city_name}")
        lines.append(f"*Great local talking points and weekend activities around {city_name}:*")
        for it in city_events:
            src = f" — *{it['source']}*" if it.get("source") else ""
            lines.append(f"- **{it['title']}**{src}")
        lines.append("")

    # Major US Events: Conferences & Festivals
    if us_events:
        lines.append("## 🎪 Major US Events (Conferences, Festivals & Entertainment)")

        confs = us_events.get("conferences", [])
        if confs:
            lines.append("### 💼 Tech & Industry Conferences")
            for it in confs:
                lines.append(f"- **{it['title']}** (*{it['source']}*)")

        music = us_events.get("music_festivals", [])
        if music:
            lines.append("### 🎸 Music Festivals & Major Tours")
            for it in music:
                lines.append(f"- **{it['title']}** (*{it['source']}*)")

        ent = us_events.get("culture_entertainment", [])
        if ent:
            lines.append("### 🎭 Entertainment & Cultural Buzz")
            for it in ent:
                lines.append(f"- **{it['title']}** (*{it['source']}*)")
        lines.append("")

    # Sports Highlights
    if sports:
        lines.append("## 🏆 Major Sports Highlights & Conversation Starters")

        fb = sports.get("football", [])
        if fb:
            lines.append("### 🏈 American Football (NFL & College)")
            for it in fb:
                lines.append(f"- **{it['title']}** (*{it['source']}*)")

        f1 = sports.get("f1", [])
        if f1:
            lines.append("### 🏎️ Formula 1 (Grand Prix)")
            for it in f1:
                lines.append(f"- **{it['title']}** (*{it['source']}*)")

        crick = sports.get("cricket", [])
        if crick:
            lines.append("### 🏏 Cricket (International & Tournaments)")
            for it in crick:
                lines.append(f"- **{it['title']}** (*{it['source']}*)")

        gen_sp = sports.get("general_sports", [])
        if gen_sp:
            lines.append("### 🏀 Top US Sports Buzz")
            for it in gen_sp:
                lines.append(f"- **{it['title']}** (*{it['source']}*)")
        lines.append("")

    lines.append("## 🗣️ Best Conversation Starters to Use Today:")
    lines.append("- *\"Did you catch the latest NFL highlights this weekend? Some crazy game finishes!\"*")
    lines.append("- *\"Are you following this F1 Grand Prix weekend? The qualifying shakeups were wild.\"*")
    if city_name:
        lines.append(f"- *\"Did you hear about what's happening around {city_name} this week?\"*")
    lines.append("- *\"Are you planning to attend any of the big tech expos or music festivals coming up?\"*")

    return "\n".join(lines)
