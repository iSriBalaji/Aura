"""Trending and breaking news service for general, technology, politics, and science.
Fetches up-to-the-minute, reliable news via Google News RSS feeds with zero API cost.
Supports category digests and targeted deep-dive queries for follow-up context.
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

CATEGORY_FEEDS = {
    "general": "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
    "technology": "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en",
    "science": "https://news.google.com/rss/headlines/section/topic/SCIENCE?hl=en-US&gl=US&ceid=US:en",
    "politics": "https://news.google.com/rss/headlines/section/topic/POLITICS?hl=en-US&gl=US&ceid=US:en",
}

CATEGORY_EMOJIS = {
    "general": "🌍 General / World",
    "technology": "💻 Technology & AI",
    "science": "🔬 Science & Discovery",
    "politics": "🏛️ Politics & Policy",
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
        # e.g., 'Sun, 08 Nov 2026 03:08:00 GMT'
        dt = datetime.strptime(raw_date.replace("GMT", "+0000").strip(), "%a, %d %b %Y %H:%M:%S %z")
        return dt.strftime("%b %d, %H:%M UTC")
    except Exception:
        return raw_date.replace(" GMT", "")


def fetch_category_feed(category: str, limit: int = 3, timeout: float = 7.0) -> List[Dict[str, Any]]:
    """Fetch and parse news items for a specific category feed."""
    url = CATEGORY_FEEDS.get(category.lower())
    if not url:
        return []

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
                    desc_html = it.findtext("description", "")
                    clean_desc = clean_html(desc_html)

                    items.append({
                        "category": category,
                        "title": parsed["title"],
                        "source": parsed["source"],
                        "link": link,
                        "pub_date": parse_pub_date(pub_date),
                        "summary": clean_desc if clean_desc and clean_desc != parsed["title"] else "",
                    })
    except Exception:
        pass

    return items


def fetch_trending_news(categories: Optional[List[str]] = None, limit_per_cat: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch trending news across specified categories (default: general, tech, science, politics)."""
    selected = categories or ["general", "technology", "politics", "science"]
    results = {}
    for cat in selected:
        if cat in CATEGORY_FEEDS:
            results[cat] = fetch_category_feed(cat, limit=limit_per_cat)
    return results


def search_news_topic(query: str, limit: int = 5, timeout: float = 7.0) -> List[Dict[str, Any]]:
    """Search Google News for deep-dive context on a specific topic or event."""
    encoded = urllib.parse.quote(query.strip())
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"

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
                        "summary": clean_desc,
                    })
    except Exception:
        pass

    return items


def fallback_format_news_digest(news_dict: Dict[str, List[Dict[str, Any]]]) -> str:
    """Format trending news digest into short, punchy conversation starters."""
    lines = []
    lines.append("# 📰 Trending Important News Digest")
    lines.append("> **Quick Overview:** The most discussed topics in Technology, Politics, Science, and World events over today and this past week. Use these points as quick conversational starters.\n")

    item_idx = 1
    for cat, items in news_dict.items():
        header = CATEGORY_EMOJIS.get(cat, f"📌 {cat.capitalize()}")
        lines.append(f"### {header}")
        if not items:
            lines.append("- *No recent stories retrieved for this category.*")
        for it in items:
            source_tag = f" — *{it['source']}* ({it['pub_date']})" if it.get("source") else ""
            lines.append(f"- **[{item_idx}] {it['title']}**{source_tag}")
            item_idx += 1
        lines.append("")

    lines.append("💡 *Tip: Want to know more about any topic? Ask: `Tell me more about #[number]` or `Give me details on [topic]` for an in-depth breakdown!*")
    return "\n".join(lines)


def fallback_format_news_deepdive(query: str, items: List[Dict[str, Any]]) -> str:
    """Format deep-dive background context when user asks about a specific news item."""
    lines = []
    lines.append(f"# 🔍 In-Depth News Brief: {query}")
    lines.append("> **Background & Talking Points:** Key developments and context to know when discussing this topic.\n")

    if not items:
        lines.append("- *No in-depth articles found for this topic query.*")
        return "\n".join(lines)

    top_item = items[0]
    lines.append(f"## 📌 Key Headline: {top_item['title']}")
    lines.append(f"- **Primary Source:** {top_item['source']} | **Reported:** {top_item['pub_date']}\n")

    lines.append("## 💬 What is Happening & Why People Are Talking:")
    for i, it in enumerate(items[:4], 1):
        lines.append(f"### {i}. {it['title']}")
        lines.append(f"- **Reported by:** {it['source']} ({it['pub_date']})")
        if it.get("summary") and it["summary"] != it["title"]:
            lines.append(f"- **Context:** {it['summary']}")
        lines.append("")

    lines.append("## 🗣️ Conversational Takeaways (What to know):")
    lines.append(f"- **The Core Story:** {top_item['title']}.")
    lines.append("- **Relevance:** Major development with broad societal, technological, or policy implications.")
    lines.append("- **Talking Point:** Highlight the most recent development and how different parties or analysts are responding.")
    return "\n".join(lines)
