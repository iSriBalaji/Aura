"""LLM synthesizer for Weather, Wardrobe, Cycling, Trending News, Events & Sports recommendations.
Interfaces with Gemini (gemini-2.5-flash), OpenAI (gpt-4o-mini), or uses deterministic
structured markdown fallbacks if no LLM credentials are set.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from weather_agent.events_service import (
    fallback_format_events_and_sports,
    get_current_window,
)
from weather_agent.memory_service import get_memory_summary_prompt
from weather_agent.news_service import (
    fallback_format_news_deepdive,
    fallback_format_news_digest,
)

load_dotenv()


def _safe_get_secret(key: str) -> Optional[str]:
    """Safely fetch a secret from Streamlit Cloud secrets without crashing if unconfigured."""
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            val = st.secrets.get(key)
            if val:
                return str(val).strip()
    except Exception:
        pass
    return None


def get_llm(model_override: Optional[str] = None, api_key: Optional[str] = None):
    """Instantiate a cost-effective LLM based on user selection, session API keys, or environment variables.
    
    Supports:
    - Built-in Deterministic Engine ($0, no keys)
    - Google Gemini: gemini-2.5-flash, gemini-2.5-flash-lite, gemini-1.5-flash, gemini-1.5-pro
    - Groq Cloud: llama-3.3-70b-versatile, llama-3.1-8b-instant
    - OpenAI: gpt-4o-mini, gpt-4o
    """
    target_model = model_override or os.getenv("LLM_MODEL") or "gemini-2.5-flash"
    m_lower = target_model.strip().lower()

    if m_lower in ("builtin", "built-in", "built-in engine (offline)", "offline", "none"):
        return None

    # Resolve keys per provider safely
    google_key = (
        api_key if (api_key and "gemini" in m_lower) else None
    ) or _safe_get_secret("GEMINI_API_KEY") or _safe_get_secret("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    groq_key = (
        api_key if (api_key and "llama" in m_lower) else None
    ) or _safe_get_secret("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

    openai_key = (
        api_key if (api_key and ("gpt" in m_lower or "o1" in m_lower)) else None
    ) or _safe_get_secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

    # Google Gemini models
    if "gemini" in m_lower:
        if google_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=target_model,
                    temperature=0.3,
                    google_api_key=google_key,
                )
            except Exception:
                pass
        return None

    # Groq models
    if "llama" in m_lower or "groq" in m_lower or "mixtral" in m_lower:
        if groq_key:
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=target_model,
                    temperature=0.3,
                    api_key=groq_key,
                    base_url="https://api.groq.com/openai/v1",
                )
            except Exception:
                pass
        return None

    # OpenAI models
    if "gpt" in m_lower or "o1" in m_lower or "o3" in m_lower:
        if openai_key:
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=target_model,
                    temperature=0.3,
                    api_key=openai_key,
                )
            except Exception:
                pass
        return None

    # Automatic fallback if key exists
    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3,
                google_api_key=google_key,
            )
        except Exception:
            pass

    return None


def fallback_format_report(
    location_name: str,
    analysis: Dict[str, Any],
    user_query: Optional[str] = None,
) -> str:
    """Deterministic, clean markdown weather report when LLM is offline or no key is set."""
    curr = analysis["current"]
    curr_cloth = curr["clothing"]
    curr_cyc = curr["cycling"]
    daily_list = analysis["daily"]
    opt_win = analysis.get("optimal_window", {})

    q_lower = (user_query or "").lower().strip()

    # If Baju specifically asked about cycling
    is_cycling_q = any(w in q_lower for w in ["cycle", "cycling", "bike", "biking", "ride", "riding"])
    is_full_q = any(w in q_lower for w in ["full", "7-day", "weekly", "all", "complete", "planner", "briefing"])
    
    if is_cycling_q and not is_full_q:
        lines = []
        lines.append(f"### Cycling Assessment for Baju")
        lines.append(f"**Location:** {location_name} | **Score:** {curr_cyc['score']}/100 ({curr_cyc['badge']})\n")
        lines.append(f"> **Verdict:** {curr_cyc['verdict']}\n")
        lines.append(f"- **Current Temp:** {curr['temp_f']}°F (Feels like {curr['apparent_temp_f']}°F)")
        lines.append(f"- **Wind:** {curr['wind_speed_mph']} mph (Gusts {curr['wind_gusts_kmh']} km/h)")
        lines.append(f"- **Roads:** {curr_cyc['precip_note']}")
        if curr_cyc.get("cycling_kit"):
            lines.append(f"- **Recommended Kit:** {', '.join(curr_cyc['cycling_kit'])}")
        if opt_win and opt_win.get("best_window"):
            best = opt_win["best_window"]
            lines.append(f"\n**Optimal Outdoor Window Today:**")
            lines.append(f"- **Prime Window:** **{best['label']}** ({best.get('badge', 'Good')}, Score: {best['avg_score']}/100)")
            lines.append(f"- **Conditions:** {best.get('temp_f', 0):.0f}°F, Wind: {best.get('wind_mph', 0):.0f} mph, Rain chance: {best.get('precip_prob', 0):.0f}%")
        return "\n".join(lines)

    # If Baju specifically asked about clothing or wearing
    is_clothing_q = any(w in q_lower for w in ["wear", "clothes", "clothing", "layer", "layers", "jacket", "shorts", "coat", "umbrella", "hoodie"])
    if is_clothing_q and not is_full_q:
        lines = []
        lines.append(f"### Clothing & Layering Recommendation for Baju")
        lines.append(f"**Location:** {location_name} | **Current:** {curr['temp_f']}°F ({curr['weather_name']})\n")
        lines.append(f"> **Summary:** {curr_cloth['summary']}\n")
        lines.append(f"- **Layers Needed:** {curr_cloth['layer_count']}")
        lines.append(f"- **Base Layer:** {curr_cloth['base_layer']}")
        if curr_cloth['mid_layer'] != "None":
            lines.append(f"- **Mid Layer:** {curr_cloth['mid_layer']}")
        if curr_cloth['outer_layer'] != "None":
            lines.append(f"- **Outer Layer:** {curr_cloth['outer_layer']}")
        lines.append(f"- **Bottoms:** {curr_cloth['bottoms']}")
        lines.append(f"- **Footwear:** {curr_cloth['footwear']}")
        if curr_cloth['accessories']:
            lines.append(f"- **Accessories:** {', '.join(curr_cloth['accessories'])}")
        return "\n".join(lines)

    # If Baju specifically asked about optimal window
    if ("window" in q_lower or "optimal" in q_lower or "best time" in q_lower) and not is_full_q:
        if opt_win and opt_win.get("best_window"):
            best = opt_win["best_window"]
            lines = []
            lines.append(f"### Optimal Outdoor & Cycling Window for Baju")
            lines.append(f"**Location:** {location_name}\n")
            lines.append(f"- **Best Window:** **{best['label']}** ({best.get('badge', 'Good')}, Score: {best['avg_score']}/100)")
            lines.append(f"- **Avg Temp:** {best.get('temp_f', 0):.0f}°F | **Wind:** {best.get('wind_mph', 0):.0f} mph | **Rain Chance:** {best.get('precip_prob', 0):.0f}%")
            lines.append(f"- **Analysis:** {best.get('summary', '')}")
            avoid = opt_win.get("avoid_windows", [])
            if avoid:
                lines.append(f"- **Times to Avoid:**")
                for av in avoid:
                    lines.append(f"  - **{av['label']}**: {av['reason']}")
            return "\n".join(lines)

    # Full structured daily briefing
    lines = []
    lines.append(f"# Aura Daily Briefing for Baju")
    lines.append(f"> Hello Baju! Here is your daily weather, wardrobe layering, and cycling assessment.\n")
    lines.append(f"**Location:** {location_name}\n")

    # Current Live Status
    lines.append("## Live Conditions")
    lines.append(f"- **Condition:** {curr['weather_icon']} {curr['weather_name']}")
    lines.append(f"- **Temperature:** {curr['temp_c']}°C ({curr['temp_f']}°F) | **Feels Like:** {curr['apparent_temp_c']}°C ({curr['apparent_temp_f']}°F)")
    lines.append(f"- **Wind:** {curr['wind_speed_kmh']} km/h ({curr['wind_speed_mph']} mph), Gusts up to {curr['wind_gusts_kmh']} km/h")
    lines.append(f"- **Precipitation:** {curr['precip_mm']} mm\n")

    # Clothing Decision
    lines.append("## Clothing & Layering Recommendation")
    lines.append(f"> **Summary:** {curr_cloth['summary']}")
    lines.append(f"- **Recommended Layers:** **{curr_cloth['layer_count']} Layer(s)**")
    lines.append(f"- **Base Layer:** {curr_cloth['base_layer']}")
    if curr_cloth['mid_layer'] != "None":
        lines.append(f"- **Mid Layer:** {curr_cloth['mid_layer']}")
    if curr_cloth['outer_layer'] != "None":
        lines.append(f"- **Outer Layer:** {curr_cloth['outer_layer']}")
    lines.append(f"- **Bottoms:** {curr_cloth['bottoms']}")
    lines.append(f"- **Footwear:** {curr_cloth['footwear']}")
    if curr_cloth['accessories']:
        lines.append(f"- **Accessories:** {', '.join(curr_cloth['accessories'])}\n")

    # Cycling Decision
    lines.append("## Cycling Assessment")
    lines.append(f"- **Cycling Score:** **{curr_cyc['score']}/100** ({curr_cyc['badge']})")
    lines.append(f"- **Verdict:** {curr_cyc['verdict']}")
    lines.append(f"- **Comfort & Factors:**")
    lines.append(f"  - Temperature: {curr_cyc['temp_note']}")
    lines.append(f"  - Roads / Rain: {curr_cyc['precip_note']}")
    lines.append(f"  - Wind Resistance: {curr_cyc['wind_note']}")

    if curr_cyc['cycling_kit']:
        lines.append(f"- **Suggested Cycling Kit:** {', '.join(curr_cyc['cycling_kit'])}")

    if curr_cyc['safety_tips']:
        lines.append(f"- **Safety Tips:** {', '.join(curr_cyc['safety_tips'])}")
    lines.append("")

    # Hourly Optimal Window Finder
    if opt_win and opt_win.get("best_window"):
        best = opt_win["best_window"]
        lines.append("## Optimal Outdoor Window")
        lines.append(f"- **Prime Window:** **{best['label']}** ({best.get('badge', 'Good')}, Score: **{best['avg_score']}/100**)")
        lines.append(f"  - Avg Temp: **{best.get('temp_f', 0):.0f}°F** | Rain Chance: **{best.get('precip_prob', 0):.0f}%** | Wind: **{best.get('wind_mph', 0):.0f} mph**")
        lines.append(f"  - Analysis: {best.get('summary', '')}")

        avoid = opt_win.get("avoid_windows", [])
        if avoid:
            lines.append(f"- **Times to Avoid:**")
            for av in avoid:
                lines.append(f"  - **{av['label']}**: {av['reason']}")
        lines.append("")

    # 7-Day Forecast Overview
    lines.append("## 7-Day Forecast & Planner")
    lines.append("| Date | Weather | Temp Range (C / F) | Rain Prob | Layers Needed | Cycling Score |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for d in daily_list:
        tag = "**Today**" if d["is_today"] else d["day_name"]
        w_col = f"{d['weather_icon']} {d['weather_name']}"
        t_col = f"{d['temp_min_c']}° - {d['temp_max_c']}°C<br>({d['temp_min_f']}° - {d['temp_max_f']}°F)"
        r_col = f"{d['precip_prob_max']:.0f}% ({d['precip_sum_mm']} mm)"
        l_col = f"{d['clothing']['layer_count']} layers"
        c_col = f"**{d['cycling']['score']}/100** ({d['cycling']['rating']})"
        lines.append(f"| {tag} | {w_col} | {t_col} | {r_col} | {l_col} | {c_col} |")

    lines.append("")

    # Cycling Week Highlights
    rankings = analysis.get("best_cycling_rankings", [])
    if rankings:
        best_day = rankings[0]
        worst_day = rankings[-1]
        lines.append(f"- **Best Day to Ride:** **{best_day[0]}** (Score: {best_day[1]}/100)")
        lines.append(f"- **Most Challenging Day:** **{worst_day[0]}** (Score: {worst_day[1]}/100)")

    return "\n".join(lines)


def synthesize_chat(
    user_query: str,
    location_name: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Conversational chat synthesis for direct human-like interaction with Baju."""
    llm = get_llm(model_override=model_name, api_key=api_key)
    if llm is None:
        q_lower = user_query.strip().lower()
        if any(w in q_lower for w in ["hi", "hello", "hey", "good morning", "good evening", "how are you", "who are you"]):
            return (
                "Hello Baju! I am Aura, your personal lifestyle and daily briefing copilot. "
                "I am right here with you! You can ask me questions naturally about the weather, "
                "cycling conditions, curated news, sports updates, or share anything you'd like me to remember."
            )
        return (
            f"Hello Baju! I'm here. You asked: \"{user_query}\". "
            "To enable general AI conversation for open questions, you can select Gemini 2.5 Flash in the sidebar "
            "with your free Google AI Studio key. Otherwise, I can instantly answer your questions on weather, "
            "clothing layers, cycling suitability, trending news, and sports!"
        )

    prompt = f"""You are Aura, the dedicated personal AI copilot to Baju (Sri Balaji).
Always address him warmly and directly as Baju.

{get_memory_summary_prompt()}

Active Location: {location_name or 'Prospect Heights, Illinois'}
Baju's Message: "{user_query}"

Instructions:
1. Listen carefully and converse naturally, warmly, and attentively with Baju.
2. Directly answer his exact question, thoughts, or greeting.
3. Be helpful, concise, and articulate (1-3 paragraphs unless he asked for a detailed explanation).
4. Keep formatting clean, readable, and human-friendly with minimal, tasteful formatting (avoid excessive emojis).
"""
    try:
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, str) and content.strip():
            return content.strip()
    except Exception:
        pass

    return f"Hello Baju! I'm right here with you. How can I help you today?"


def synthesize_weather_report(
    location_name: str,
    analysis: Dict[str, Any],
    user_query: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Synthesize weather, wardrobe, and cycling analysis using LLM if available."""
    llm = get_llm(model_override=model_name, api_key=api_key)
    if llm is None:
        return fallback_format_report(location_name, analysis, user_query=user_query)

    prompt = f"""You are Aura, the dedicated personal AI advisor to Baju (whose full name is Sri Balaji).
Always address him warmly and directly as Baju.

{get_memory_summary_prompt()}

Target Location: {location_name}
Baju's Question/Request: "{user_query or 'General forecast and advice'}"

Structured Decision Data:
- Current Weather: {analysis['current']}
- Optimal 2-3h Outdoor/Cycling Window: {analysis.get('optimal_window')}
- Clothing Recommendation: {analysis['current']['clothing']}
- Cycling Decision: {analysis['current']['cycling']}
- 7-Day Forecast & Cycling Scores: {analysis['daily']}

CRITICAL CONVERSATIONAL INSTRUCTIONS:
1. LISTEN TO BAJU AND ANSWER DIRECTLY: Look closely at what Baju asked: "{user_query}".
2. IF BAJU ASKED A SPECIFIC QUESTION (e.g. asking about wearing shorts, jacket, rain, cycling suitability, best time to ride, or current temperature):
   - Answer his EXACT question directly in a natural, friendly, conversational tone (1-3 paragraphs).
   - Use the live data above to give him the clear verdict right away.
   - DO NOT dump an unwanted 7-day forecast table or an exhaustive 6-section template. Only answer what he asked!
3. ONLY IF BAJU EXPLICITLY ASKED FOR A FULL BRIEFING / 7-DAY PLANNER (or clicked the complete planner button):
   - Provide a clean, structured overview covering:
     a) Live conditions & summary
     b) Wardrobe & layering recommendation
     c) Cycling suitability & optimal window
     d) 7-day forecast table with cycling ratings
4. Use minimal, tasteful emojis. Keep the formatting clean, modern, and human-friendly.
"""
    try:
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, str) and len(content.strip()) > 20:
            return content.strip()
    except Exception:
        pass

    return fallback_format_report(location_name, analysis, user_query=user_query)


def synthesize_news_digest(
    news_data: Dict[str, List[Dict[str, Any]]],
    location_name: str = "Prospect Heights, Illinois",
    user_query: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Synthesize trending news into exact 2-3 line reliable summaries across all 6 essential categories."""
    llm = get_llm(model_override=model_name, api_key=api_key)
    if llm is None:
        return fallback_format_news_digest(news_data, location_name=location_name)

    prompt = f"""You are Aura, the dedicated personal AI advisor to Baju (whose full name is Sri Balaji).
Always address him warmly and directly as Baju.

{get_memory_summary_prompt()}

Review the live trending news stories below across all 6 essential categories:
1. 🌍 General & World News
2. 💻 Technology & AI
3. 🏛️ Politics & Policy
4. 💼 Business & Economy
5. 🔬 Science & Discovery
6. 📍 Local News ({location_name})

User Query: {user_query or 'Executive news digest with 2-3 line summaries'}

Raw News Feed Data:
{news_data}

CRITICAL INSTRUCTIONS FOR NEWS STORIES:
1. Greet Baju warmly.
2. Structure your briefing by each of the 6 sections above.
3. For EVERY story, you MUST provide:
   - A bold headline with publisher source and publication date tag (e.g. `**[1] Headline Title** — *Publisher* (Date)`).
   - EXACTLY a 2 to 3-line structured summary formatted as bullets:
     • **What happened:** (1 concise, factual sentence on the core development)
     • **Key context:** (1 concise sentence with background, data, or impact)
     • **What to know:** (1 concise sentence giving Baju the key conversational talking point)
   DO NOT just provide titles. Every single story MUST have these reliable 2-3 lines so Baju knows exactly what happened and what to discuss.
4. Keep the styling clean, readable, minimal, and human-friendly.
"""
    try:
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, str) and len(content.strip()) > 50:
            return content.strip()
    except Exception:
        pass

    return fallback_format_news_digest(news_data, location_name=location_name)


def synthesize_news_deepdive(
    query: str,
    news_items: List[Dict[str, Any]],
    user_query: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Synthesize in-depth background context for a specific news item when asked by the user."""
    llm = get_llm(model_override=model_name, api_key=api_key)
    if llm is None:
        return fallback_format_news_deepdive(query, news_items)

    prompt = f"""You are Aura, the dedicated personal AI advisor to Baju (whose full name is Sri Balaji).
Always address him warmly and directly as Baju.
Baju wants additional context and background on a specific news topic from the recent trending headlines.

Topic: {query}
User Request: {user_query}

Recent News Articles / Coverage:
{news_items}

Please synthesize a comprehensive, easy-to-digest deep-dive briefing structured with:
1. Greet Baju and give an **Executive Overview**: What is the core development right now?
2. **Key Background & Context**: What led to this event over recent days or weeks?
3. **Why It Matters & Key Perspectives**: What are the major impacts, controversies, or points of view?
4. **Conversational Talking Points for Baju**: 2-3 memorable bullet points he can mention when discussing this topic with others.
Keep formatting clean and minimal.
"""
    try:
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, str) and len(content.strip()) > 50:
            return content.strip()
    except Exception:
        pass

    return fallback_format_news_deepdive(query, news_items)


def synthesize_events_and_sports(
    us_events: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    sports: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    city_name: Optional[str] = None,
    city_events: Optional[List[Dict[str, Any]]] = None,
    user_query: Optional[str] = None,
    ref_date: Optional[datetime] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Synthesize US & global events, sports highlights, and conversation starters anchored strictly to today ±17 days."""
    ref = ref_date or datetime.now()
    win = get_current_window(ref_date=ref, window_days=17)

    llm = get_llm(model_override=model_name, api_key=api_key)
    if llm is None:
        return fallback_format_events_and_sports(
            us_events=us_events,
            sports=sports,
            city_name=city_name,
            city_events=city_events,
            ref_date=ref,
        )

    prompt = f"""You are Aura, the dedicated personal AI advisor to Baju (whose full name is Sri Balaji).
Always address him warmly and directly as Baju.

{get_memory_summary_prompt()}

CRITICAL DATE AND TEMPORAL CONSTRAINT:
Today's exact date is: {win['today_str']}
Active temporal window: Strictly past week, today, or next week (±17 days: {win['window_str']}).
Baju specifically requires that all events, conferences, sports fixtures, and festivals MUST take place or have taken place strictly within this active ±17-day window.
- Under NO circumstances include events from past months, past years, or distant future dates.
- For every event, conference, and sports matchup, explicitly specify its exact date (e.g. "{ref.strftime('%b %d')}" or upcoming "{win['window_str']}").
- Ensure all talking points are fresh, timely, and conversation-ready for today.

User Request: {user_query or 'Major US events, conferences, music festivals, and sports updates'}
City Focus: {city_name or 'General US / Popular Globally'}

Data Collected (Within ±17 Days):
- City Events: {city_events if city_name else 'None'}
- US Events (Conferences, Festivals, Entertainment): {us_events}
- Sports Updates (American Football, Formula 1, Cricket, General): {sports}

Instructions:
1. Greet Baju warmly and explicitly note today's date ({win['today_str']}) and the active ±17-day window ({win['window_str']}).
2. Structure into clean, scannable sections:
   - 📍 **City Happenings** (if a city was mentioned: top festivals, concerts, or local buzz with specific dates).
   - 🎪 **Major US Events** (Conferences/Expos, Music Festivals, Major Entertainment occurring in this ±17-day window with dates).
   - 🏈 **Sports Pulse** (American Football / NFL / College highlights, 🏎️ Formula 1 Grand Prix status, 🏏 Cricket international matches with dates).
3. For each key event or sport:
   - Provide a short, factual 1-2 sentence breakdown of what happened or what's coming up within this window.
   - Include a natural "💬 **Conversation Starter for Baju**" quote he can naturally say in conversation with friends, colleagues, or clients.
4. Keep the tone vibrant, practical, timely, and conversational.
"""
    try:
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, str) and len(content.strip()) > 50:
            return content.strip()
    except Exception:
        pass

    return fallback_format_events_and_sports(
        us_events=us_events,
        sports=sports,
        city_name=city_name,
        city_events=city_events,
        ref_date=ref,
    )
