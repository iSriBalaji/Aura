"""LLM synthesizer for Weather, Clothing & Cycling recommendations.
Reliably interfaces with Gemini, OpenAI, or falls back to structured markdown
if no LLM credentials are configured.
"""

import os
from typing import Any, Dict, Optional
from dotenv import load_dotenv

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
    """Deterministic, beautifully styled markdown report when LLM is offline or no API key is set."""
    curr = analysis["current"]
    curr_cloth = curr["clothing"]
    curr_cyc = curr["cycling"]
    daily_list = analysis["daily"]

    lines = []
    lines.append(f"# 🌤️ Weather, Clothing & Cycling Decision Report")
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


def synthesize_report(
    location_name: str,
    analysis: Dict[str, Any],
    user_query: Optional[str] = None,
) -> str:
    """Synthesize decision analysis using LLM if available, otherwise fallback cleanly."""
    llm = get_llm()
    if llm is None:
        return fallback_format_report(location_name, analysis)

    prompt = f"""You are an elite, practical Weather, Wardrobe, and Cycling Decision Advisor.
Analyze the structured live weather, clothing calculations, and cycling suitability metrics below, then produce an engaging, clear, and actionable Markdown report.

Target Location: {location_name}
User Query / Context: {user_query or 'General daily and weekly forecast and advice'}

Structured Decision Data:
- Current Live Weather: {analysis['current']}
- Daily Breakdown (Today + 7 Days): {analysis['daily']}
- Cycling Rankings: {analysis['best_cycling_rankings']}

Structure your response with:
1. 🌤️ Header & Location Summary with live metrics (Temp, Feels Like, Wind, Conditions).
2. 👕 Today's Wardrobe & Layering Strategy:
   - Layer count and specific garments (Base layer, Mid layer, Outer shell, Bottoms, Footwear, and Accessories).
   - Explain why (wind chill, rain chance, or heat).
3. 🚴‍♂️ Today's Cycling Decision:
   - Cycling Score (out of 100) and Rating Badge (Optimal / Good / Moderate / Poor / Hazardous).
   - Road conditions, wind resistance, and specific cycling kit (jersey, tights, gilet, gloves).
   - Essential bike handling & safety tips (tire pressure, braking, wind direction).
4. 📅 7-Day Planning Table:
   - Neatly formatted Markdown table covering Date, Weather, Temp (C & F), Rain Chance, Recommended Layers, and Cycling Rating.
5. 🏆 Weekly Decision Takeaways:
   - Best days for long cycling rides vs days to avoid / take a rest day or indoor trainer.
   - Any significant weather shift mid-week to prepare for.

Be direct, highly organized, and practical. Do not make up fake weather data—use the provided metrics strictly.
"""

    try:
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, str) and len(content.strip()) > 50:
            return content.strip()
    except Exception:
        pass

    return fallback_format_report(location_name, analysis)
