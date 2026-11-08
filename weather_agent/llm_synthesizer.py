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


def get_llm():
    """Instantiate a cost-effective LLM based on environment variables."""
    google_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("LLM_MODEL")

    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            m = model_name or "gemini-2.5-flash"
            return ChatGoogleGenerativeAI(
                model=m,
                temperature=0.3,
                google_api_key=google_key,
            )
        except Exception:
            pass

    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            m = model_name or "gpt-4o-mini"
            return ChatOpenAI(
                model=m,
                temperature=0.3,
                api_key=openai_key,
            )
        except Exception:
            pass

    return None


def fallback_format_report(
    location_name: str,
    analysis: Dict[str, Any],
) -> str:
    """Deterministic, beautifully styled markdown weather report when LLM is offline or no key is set."""
    curr = analysis["current"]
    curr_cloth = curr["clothing"]
    curr_cyc = curr["cycling"]
    daily_list = analysis["daily"]

    lines = []
    lines.append(f"# 🌤️ Aura Daily Briefing for Baju")
    lines.append(f"> **Hello Baju!** Here is your personalized weather, wardrobe layering, and cycling assessment.\n")
    lines.append(f"**📍 Location:** {location_name}\n")

    # Current Live Status
    lines.append("## 🔴 Live Current Conditions")
    lines.append(f"- **Condition:** {curr['weather_icon']} {curr['weather_name']}")
    lines.append(f"- **Temperature:** {curr['temp_c']}°C ({curr['temp_f']}°F) | **Feels Like:** {curr['apparent_temp_c']}°C ({curr['apparent_temp_f']}°F)")
    lines.append(f"- **Wind:** {curr['wind_speed_kmh']} km/h ({curr['wind_speed_mph']} mph), Gusts up to {curr['wind_gusts_kmh']} km/h")
    lines.append(f"- **Precipitation:** {curr['precip_mm']} mm\n")

    # Clothing Decision
    lines.append("## 👕 Today's Clothing & Layering Recommendation")
    lines.append(f"> **Quick Summary:** {curr_cloth['summary']}")
    lines.append(f"- **Recommended Layers:** **{curr_cloth['layer_count']} Layer(s)**")
    lines.append(f"- **Base Layer:** {curr_cloth['base_layer']}")
    if curr_cloth['mid_layer'] != "None":
        lines.append(f"- **Mid Layer:** {curr_cloth['mid_layer']}")
    if curr_cloth['outer_layer'] != "None":
        lines.append(f"- **Outer Layer:** {curr_cloth['outer_layer']}")
    lines.append(f"- **Bottoms:** {curr_cloth['bottoms']}")
    lines.append(f"- **Footwear:** {curr_cloth['footwear']}")
    if curr_cloth['accessories']:
        lines.append(f"- **Accessories & Essentials:** {', '.join(curr_cloth['accessories'])}\n")

    # Cycling Decision
    lines.append("## 🚴‍♂️ Cycling Suitability Assessment (Today)")
    lines.append(f"- **Cycling Score:** **{curr_cyc['score']}/100** ({curr_cyc['badge']})")
    lines.append(f"- **Verdict:** {curr_cyc['verdict']}")
    lines.append(f"- **Comfort & Factors:**")
    lines.append(f"  - 🌡️ Temperature: {curr_cyc['temp_note']}")
    lines.append(f"  - 🌧️ Roads / Rain: {curr_cyc['precip_note']}")
    lines.append(f"  - 💨 Wind Resistance: {curr_cyc['wind_note']}")

    if curr_cyc['cycling_kit']:
        lines.append(f"- **Suggested Cycling Kit:**")
        for kit_item in curr_cyc['cycling_kit']:
            lines.append(f"  - {kit_item}")

    if curr_cyc['safety_tips']:
        lines.append(f"- **Safety & Route Advice:**")
        for tip in curr_cyc['safety_tips']:
            lines.append(f"  - ⚠️ {tip}")
    lines.append("")

    # Hourly Optimal Window Finder
    opt_win = analysis.get("optimal_window", {})
    if opt_win and opt_win.get("best_window"):
        best = opt_win["best_window"]
        lines.append("## ⏱️ Optimal Cycling & Outdoor Window (Hourly Optimizer)")
        lines.append(f"- **🌟 Prime Window:** **{best['label']}** ({best.get('badge', '🟢 Good')}, Score: **{best['avg_score']}/100**)")
        lines.append(f"  - 🌡️ Avg Temp: **{best.get('temp_f', 0):.0f}°F** | 🌧️ Rain Chance: **{best.get('precip_prob', 0):.0f}%** | 💨 Wind: **{best.get('wind_mph', 0):.0f} mph** (Gusts: **{best.get('gusts_mph', 0):.0f} mph**)")
        lines.append(f"  - 💡 **Analysis:** {best.get('summary', '')}")

        avoid = opt_win.get("avoid_windows", [])
        if avoid:
            lines.append(f"- **⚠️ Times / Windows to Avoid:**")
            for av in avoid:
                lines.append(f"  - 🛑 **{av['label']}**: {av['reason']}")
        else:
            lines.append(f"- **✅ Avoid Windows:** None — overall daylight hours remain safe and rideable.")
        lines.append("")

    # 7-Day Forecast Overview
    lines.append("## 📅 7-Day Outlook & Decision Table")
    lines.append("| Date | Weather | Temp Range (C / F) | Rain Prob | Layers Needed | Cycling Score & Status |")
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
    lines.append("## 🏆 Weekly Cycling Planner")
    rankings = analysis.get("best_cycling_rankings", [])
    if rankings:
        best_day = rankings[0]
        worst_day = rankings[-1]
        lines.append(f"- **🌟 Best Day to Ride:** **{best_day[0]}** (Score: {best_day[1]}/100 - {best_day[2]})")
        lines.append(f"- **🛑 Most Challenging Day:** **{worst_day[0]}** (Score: {worst_day[1]}/100 - {worst_day[2]})")

    return "\n".join(lines)


def synthesize_weather_report(
    location_name: str,
    analysis: Dict[str, Any],
    user_query: Optional[str] = None,
) -> str:
    """Synthesize weather, wardrobe, and cycling analysis using LLM if available."""
    llm = get_llm()
    if llm is None:
        return fallback_format_report(location_name, analysis)

    prompt = f"""You are Aura, the dedicated personal AI advisor to Baju (whose full name is Sri Balaji).
Always address him warmly and directly as Baju.

{get_memory_summary_prompt()}

Analyze the structured live weather, clothing calculations, and cycling suitability metrics below, then produce an engaging, clear, and actionable Markdown report tailored for Baju (taking into account his personal cycling thresholds and preferences above).

Target Location: {location_name}
User Query: {user_query or 'General daily and weekly forecast and advice'}

Structured Decision Data:
- Current Live Weather: {analysis['current']}
- Daily Breakdown (Today + 7 Days): {analysis['daily']}
- Cycling Rankings: {analysis['best_cycling_rankings']}
- Hourly Optimal Outdoor Window: {analysis.get('optimal_window')}

Structure your response with:
1. 🌤️ Warm personal greeting to Baju, followed by Header & Location Summary with live metrics (Temp, Feels Like, Wind, Conditions).
2. 👕 Today's Wardrobe & Layering Strategy (Exact layers: Base, Mid, Outer shell, Bottoms, Footwear, and Accessories).
3. 🚴‍♂️ Today's Cycling Decision (Score /100, Rating Badge, kit, road/wind tips).
4. ⏱️ Hourly Optimal Cycling & Outdoor Window (Best contiguous 2-3 hour window, exact time range, weather metrics, why it's optimal, and any avoid periods).
5. 📅 7-Day Planning Table (Date, Weather, Temp C/F, Rain chance, Layers needed, Cycling score & status).
6. 🏆 Weekly Decision Takeaways (Best cycling day vs day to avoid).

Be direct, highly organized, warm, and practical.
"""
    try:
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, str) and len(content.strip()) > 50:
            return content.strip()
    except Exception:
        pass

    return fallback_format_report(location_name, analysis)


def synthesize_news_digest(
    news_data: Dict[str, List[Dict[str, Any]]],
    location_name: str = "Prospect Heights, Illinois",
    user_query: Optional[str] = None,
) -> str:
    """Synthesize trending news into exact 2-3 line reliable summaries across all 6 essential categories."""
    llm = get_llm()
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
4. Conclude with a helpful tip reminding Baju he can ask for a deep dive on any specific topic or headline number.
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
) -> str:
    """Synthesize in-depth background context for a specific news item when asked by the user."""
    llm = get_llm()
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
1. Greet Baju and give a 📌 **Executive Overview**: What is the core development right now?
2. 🔍 **Key Background & Context**: What led to this event over recent days or weeks?
3. 💬 **Why It Matters & Key Perspectives**: What are the major impacts, controversies, or points of view?
4. 🗣️ **Conversational Talking Points for Baju**: 2-3 memorable bullet points he can mention when discussing this topic with others.
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
) -> str:
    """Synthesize US & global events, sports highlights, and conversation starters anchored strictly to today ±17 days."""
    ref = ref_date or datetime.now()
    win = get_current_window(ref_date=ref, window_days=17)

    llm = get_llm()
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
