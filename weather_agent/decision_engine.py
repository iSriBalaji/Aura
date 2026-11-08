"""Deterministic decision engine for Clothing Layering and Cycling Suitability.
Provides reliable, consistent, and scientifically grounded calculations
independent of LLM hallucinations while giving rich data for LLM synthesis.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from weather_agent.weather_service import c_to_f, kmh_to_mph

WMO_WEATHER_CODES: Dict[int, Dict[str, str]] = {
    0: {"name": "Clear sky", "icon": "☀️", "type": "clear"},
    1: {"name": "Mainly clear", "icon": "🌤️", "type": "partly_cloudy"},
    2: {"name": "Partly cloudy", "icon": "⛅", "type": "partly_cloudy"},
    3: {"name": "Overcast", "icon": "☁️", "type": "cloudy"},
    45: {"name": "Fog", "icon": "🌫️", "type": "fog"},
    48: {"name": "Depositing rime fog", "icon": "🌫️", "type": "fog"},
    51: {"name": "Light drizzle", "icon": "🌦️", "type": "drizzle"},
    53: {"name": "Moderate drizzle", "icon": "🌦️", "type": "drizzle"},
    55: {"name": "Dense drizzle", "icon": "🌧️", "type": "drizzle"},
    56: {"name": "Light freezing drizzle", "icon": "🌨️", "type": "freezing_rain"},
    57: {"name": "Dense freezing drizzle", "icon": "🌨️", "type": "freezing_rain"},
    61: {"name": "Slight rain", "icon": "🌦️", "type": "rain"},
    63: {"name": "Moderate rain", "icon": "🌧️", "type": "rain"},
    65: {"name": "Heavy rain", "icon": "🌧️", "type": "heavy_rain"},
    66: {"name": "Light freezing rain", "icon": "🌨️", "type": "freezing_rain"},
    67: {"name": "Heavy freezing rain", "icon": "🌨️", "type": "freezing_rain"},
    71: {"name": "Slight snow fall", "icon": "🌨️", "type": "snow"},
    73: {"name": "Moderate snow fall", "icon": "❄️", "type": "snow"},
    75: {"name": "Heavy snow fall", "icon": "❄️", "type": "heavy_snow"},
    77: {"name": "Snow grains", "icon": "❄️", "type": "snow"},
    80: {"name": "Slight rain showers", "icon": "🌦️", "type": "rain"},
    81: {"name": "Moderate rain showers", "icon": "🌧️", "type": "rain"},
    82: {"name": "Violent rain showers", "icon": "⛈️", "type": "heavy_rain"},
    85: {"name": "Slight snow showers", "icon": "🌨️", "type": "snow"},
    86: {"name": "Heavy snow showers", "icon": "❄️", "type": "heavy_snow"},
    95: {"name": "Thunderstorm", "icon": "⚡", "type": "thunderstorm"},
    96: {"name": "Thunderstorm with slight hail", "icon": "⛈️", "type": "thunderstorm"},
    99: {"name": "Thunderstorm with heavy hail", "icon": "⛈️", "type": "thunderstorm"},
}


def get_wmo_info(code: int) -> Dict[str, str]:
    """Retrieve human description and emoji for WMO weather code."""
    return WMO_WEATHER_CODES.get(
        code, {"name": f"Weather Code {code}", "icon": "🌡️", "type": "unknown"}
    )


def evaluate_clothing(
    apparent_temp_c: float,
    wind_speed_kmh: float,
    precip_prob: float,
    precip_mm: float = 0.0,
    uv_index: float = 3.0,
) -> Dict[str, Any]:
    """Calculate clothing layers and specific outfit advice based on feels-like
    temperature, wind chill factor, precipitation, and sun exposure.
    """
    temp_f = c_to_f(apparent_temp_c)
    accessories: List[str] = []

    # Rain & Moisture protection
    rainy = precip_prob >= 35 or precip_mm > 0.5
    windy = wind_speed_kmh >= 24.0

    if rainy:
        accessories.append("Compact travel umbrella or hooded rain jacket")
        accessories.append("Water-resistant footwear")
    if uv_index >= 5.0 and apparent_temp_c > 12.0:
        accessories.append(f"UV Sunglasses & Sunscreen (UV Index: {uv_index:.1f})")

    # Layering taxonomy based on apparent temperature
    if apparent_temp_c < -5.0:
        # Extreme Cold (< 23°F)
        layer_count = 4
        base_layer = "Heavyweight thermal base layer (Merino wool or synthetic moisture-wicking)"
        mid_layer = "Thick fleece sweater, wool knit, or insulated puffer vest"
        outer_layer = "Heavy insulated down parka / windproof & waterproof winter coat"
        bottoms = "Thermal-lined trousers or insulated snow pants over thermal leggings"
        footwear = "Insulated waterproof winter boots with thick wool socks"
        accessories.extend(["Thermal beanie/toque", "Insulated winter gloves/mittens", "Neck gaiter or scarf"])
        summary = "Sub-freezing extreme cold: 4 full layers needed with heavy insulation and wind protection."

    elif -5.0 <= apparent_temp_c < 5.0:
        # Very Cold (23°F - 41°F)
        layer_count = 3
        base_layer = "Long-sleeve thermal base layer (light merino or synthetic)"
        mid_layer = "Fleece pullover, cardigan, or lightweight down jacket"
        outer_layer = "Insulated winter coat, trench coat, or heavy wind-resistant shell"
        bottoms = "Heavy jeans, corduroys, or trousers (optional base layer leggings if outdoors long)"
        footwear = "Warm closed shoes or leather boots with crew wool socks"
        accessories.extend(["Warm beanie", "Fleece/leather gloves"])
        summary = "Cold winter conditions: 3 warm layers with effective insulation and outer wind-blocking."

    elif 5.0 <= apparent_temp_c < 12.0:
        # Chilly / Cool (41°F - 54°F)
        layer_count = 2
        base_layer = "Comfortable cotton or merino long-sleeve shirt or t-shirt"
        mid_layer = "Sweater, knit cardigan, or hoodie"
        if rainy or windy:
            layer_count = 3
            outer_layer = "Windbreaker jacket or waterproof raincoat shell"
        else:
            outer_layer = "Light puffer jacket, denim jacket, or leather jacket"
        bottoms = "Full-length trousers, chinos, or jeans"
        footwear = "Standard everyday sneakers or ankle boots"
        accessories.append("Light scarf or thin beanie if sensitive to breeze")
        summary = f"Chilly conditions: {layer_count} layers. A cozy mid-layer under a jacket is ideal."

    elif 12.0 <= apparent_temp_c < 18.0:
        # Mild / Crisp (54°F - 64°F)
        layer_count = 2
        base_layer = "Short-sleeve or long-sleeve cotton t-shirt"
        mid_layer = "None needed (or optional light cardigan for evening chill)"
        if rainy:
            outer_layer = "Light waterproof raincoat or water-repellent trench"
        elif windy:
            outer_layer = "Light windbreaker jacket"
        else:
            outer_layer = "Light zip-up jacket, hoodie, or denim overshirt"
        bottoms = "Comfortable jeans, chinos, or casual pants"
        footwear = "Sneakers, loafers, or everyday walking shoes"
        summary = "Mild weather: 2 comfortable layers. A light jacket over a t-shirt handles morning/evening coolness."

    elif 18.0 <= apparent_temp_c < 24.0:
        # Warm / Comfortable (64°F - 75°F)
        layer_count = 1
        base_layer = "Breathable short-sleeve t-shirt, linen shirt, or polo"
        mid_layer = "None"
        outer_layer = "Optional ultra-light jacket or overshirt if out early morning or late night"
        bottoms = "Light trousers, chinos, or comfortable shorts"
        footwear = "Breathable sneakers or canvas slip-ons"
        summary = "Pleasantly warm: 1 single breathable layer. Ideal temperature for outdoor activities."

    elif 24.0 <= apparent_temp_c < 30.0:
        # Hot (75°F - 86°F)
        layer_count = 1
        base_layer = "Lightweight, loose-fitting cotton, linen, or moisture-wicking synthetic tee/tank"
        mid_layer = "None"
        outer_layer = "None"
        bottoms = "Shorts, linen pants, or lightweight skirt"
        footwear = "Ventilated running shoes, sandals, or breathable sneakers"
        accessories.extend(["Sun cap or wide-brim hat", "Sunglasses"])
        summary = "Hot conditions: 1 lightweight layer. Focus on breathable fabrics, sun protection, and hydration."

    else:
        # Very Hot / Heatwave (> 86°F / > 30°C)
        layer_count = 1
        base_layer = "Ultra-light, sweat-wicking synthetic or sheer linen shirt"
        mid_layer = "None"
        outer_layer = "None"
        bottoms = "Light athletic shorts or flowy breathable bottoms"
        footwear = "Cushioned sandals or well-ventilated mesh shoes"
        accessories.extend(["UV protection hat", "Polarized sunglasses", "Broad-spectrum SPF 50+ sunscreen", "Water bottle"])
        summary = "High heat: 1 minimal layer. Prioritize light colors, UV protection, and frequent hydration."

    return {
        "apparent_temp_c": apparent_temp_c,
        "apparent_temp_f": temp_f,
        "layer_count": layer_count,
        "base_layer": base_layer,
        "mid_layer": mid_layer,
        "outer_layer": outer_layer,
        "bottoms": bottoms,
        "footwear": footwear,
        "accessories": accessories,
        "summary": summary,
    }


def evaluate_cycling(
    temp_c: float,
    apparent_temp_c: float,
    wind_speed_kmh: float,
    wind_gusts_kmh: float,
    precip_prob: float,
    precip_mm: float,
    weather_code: int,
) -> Dict[str, Any]:
    """Calculate cycling suitability score (0-100) and actionable riding advice
    based on aerodynamic drag, temperature, road traction, and hazardous events.
    """
    wmo = get_wmo_info(weather_code)
    wmo_type = wmo["type"]

    # 1. Temperature Score (Max 30)
    # Optimal cycling range is 15°C to 23°C (59°F - 73°F)
    if 15.0 <= apparent_temp_c <= 23.0:
        temp_score = 30
        temp_note = "Optimal cycling temperature (comfortable aerobic effort)"
    elif 11.0 <= apparent_temp_c < 15.0 or 23.0 < apparent_temp_c <= 27.0:
        temp_score = 24
        temp_note = "Good riding temperature with minor gear adjustment"
    elif 6.0 <= apparent_temp_c < 11.0 or 27.0 < apparent_temp_c <= 32.0:
        temp_score = 16
        temp_note = "Cool or hot: requires extra layers or abundant water"
    elif 0.0 <= apparent_temp_c < 6.0 or 32.0 < apparent_temp_c <= 36.0:
        temp_score = 8
        temp_note = "Challenging conditions: thermal gear or heat management mandatory"
    else:
        # < 0°C (black ice hazard) or > 36°C (heatstroke risk)
        temp_score = 0
        temp_note = "Extreme temperature: risk of black ice or severe heat exhaustion"

    # 2. Precipitation & Road Traction Score (Max 35)
    if precip_mm == 0.0 and precip_prob < 15:
        precip_score = 35
        precip_note = "Dry roads and clear visibility for maximum tire grip"
    elif precip_mm < 0.5 and precip_prob < 35:
        precip_score = 25
        precip_note = "Mostly dry with slight damp spots; watch painted lines"
    elif precip_mm < 2.0 or precip_prob < 60:
        precip_score = 14
        precip_note = "Wet roads / spray: reduced braking efficiency, fenders recommended"
    elif precip_mm < 6.0 or precip_prob < 80:
        precip_score = 5
        precip_note = "Slick, wet roads and rain: slow cornering, high stopping distance"
    else:
        precip_score = 0
        precip_note = "Heavy downpour or snow: hazardous road traction"

    # 3. Wind & Gust Score (Max 25)
    # Wind creates exponential aerodynamic resistance and dangerous side-drafts
    if wind_speed_kmh < 12.0 and wind_gusts_kmh < 20.0:
        wind_score = 25
        wind_note = "Calm to light breeze: effortless riding"
    elif wind_speed_kmh < 20.0 and wind_gusts_kmh < 30.0:
        wind_score = 20
        wind_note = "Moderate breeze: pleasant, minor headwind resistance"
    elif wind_speed_kmh < 28.0 and wind_gusts_kmh < 42.0:
        wind_score = 12
        wind_note = "Noticeable wind: draft-fighting effort; avoid deep-section aero wheels"
    elif wind_speed_kmh < 38.0 and wind_gusts_kmh < 55.0:
        wind_score = 5
        wind_note = "Strong winds: tough headwinds and sharp crosswind buffeting"
    else:
        wind_score = 0
        wind_note = "Severe winds/gusts (>38 km/h): dangerous stability risk"

    # Severe Weather Penalty & Overrides
    hazard_score = 10
    hazard_note = "No severe weather alerts"
    is_severe = False
    if wmo_type in ("thunderstorm", "freezing_rain", "heavy_snow"):
        hazard_score = 0
        is_severe = True
        hazard_note = f"Dangerous event ({wmo['name']}): lightning, ice, or zero visibility"
    elif wmo_type in ("heavy_rain", "snow"):
        hazard_score = 2
        hazard_note = f"Severe precipitation ({wmo['name']})"
    elif wmo_type == "fog":
        hazard_score = 5
        hazard_note = "Dense fog: high-lumen front and rear flashing lights essential"

    # Total Score (0 - 100)
    raw_score = temp_score + precip_score + wind_score + hazard_score
    if is_severe:
        # Severe weather (thunderstorm, freezing rain, blizzard) makes cycling dangerous regardless of temperature
        total_score = min(raw_score, 15)
    elif precip_score == 0 and wind_score <= 5:
        # Severe combination of storm-level rain and violent wind
        total_score = min(raw_score, 25)
    else:
        total_score = max(0, min(100, raw_score))

    # Rating Classification
    if total_score >= 85:
        rating = "Excellent"
        badge = "🟢 Optimal"
        verdict = "Fantastic cycling weather! Grab your bike and enjoy the ride."
    elif total_score >= 70:
        rating = "Good"
        badge = "🟢 Good"
        verdict = "Good riding conditions with standard equipment."
    elif total_score >= 50:
        rating = "Moderate"
        badge = "🟡 Moderate (Caution)"
        verdict = "Rideable, but requires specific gear (wind vest, rain shell, or warm layers)."
    elif total_score >= 30:
        rating = "Poor"
        badge = "🟠 Poor"
        verdict = "Unfavorable conditions (wind, wet roads, or chill). Consider indoor trainer."
    else:
        rating = "Hazardous"
        badge = "🔴 Hazardous / Avoid"
        verdict = "Dangerous cycling conditions (ice, heavy rain, or violent gusts). Avoid outdoor riding."

    # Specific cycling gear suggestions
    cycling_kit: List[str] = []
    if apparent_temp_c < 5.0:
        cycling_kit.extend([
            "Thermal roubaix winter jacket + thermal long-sleeve base layer",
            "Thermal wind-block bib tights",
            "Windproof winter cycling gloves & thermal toe/shoe covers",
            "Cycling skull cap under helmet",
        ])
    elif apparent_temp_c < 12.0:
        cycling_kit.extend([
            "Long-sleeve jersey or thermal jacket",
            "Arm warmers + knee/leg warmers or full bib tights",
            "Full-finger lightweight cycling gloves",
            "Wind gilet / vest for descents",
        ])
    elif apparent_temp_c < 18.0:
        cycling_kit.extend([
            "Short-sleeve jersey with mesh base layer",
            "Removable arm warmers & windproof gilet (easy to pocket as you warm up)",
            "Padded bib shorts",
            "Fingerless cycling gloves",
        ])
    else:
        cycling_kit.extend([
            "Ultralight breathable summer cycling jersey",
            "Padded bib shorts",
            "Cycling sunglasses (UV tinted)",
            "Sunscreen on arms/legs & sweat-wicking cycling cap",
        ])

    safety_tips: List[str] = []
    if precip_score < 25:
        safety_tips.append("Lower tire pressure by 5-10 PSI to increase tire contact patch and grip on wet asphalt.")
        safety_tips.append("Avoid sudden braking on painted road markings, metal drain covers, or wet leaves.")
        safety_tips.append("Clip on rear mudguards/fenders to protect yourself and following riders from road spray.")
    if wind_score < 20:
        safety_tips.append(f"Plan your route into the wind on the way out ({wind_speed_kmh:.0f} km/h) so you enjoy a tailwind home.")
        safety_tips.append("Hold the handlebars firmly; watch for crosswind gaps between buildings or open fields.")
    if wmo_type == "fog" or apparent_temp_c < 8.0:
        safety_tips.append("Run daytime visible blinking headlights (300+ lumens) and taillights.")

    return {
        "score": total_score,
        "rating": rating,
        "badge": badge,
        "verdict": verdict,
        "temp_note": temp_note,
        "precip_note": precip_note,
        "wind_note": wind_note,
        "hazard_note": hazard_note,
        "cycling_kit": cycling_kit,
        "safety_tips": safety_tips,
    }


def find_optimal_outdoor_windows(
    hourly_data: Dict[str, Any],
    target_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze hourly metrics to pinpoint the optimal 2-3 hour window for cycling/outdoor activities,
    along with periods to avoid (e.g. sharp wind gusts, cold chill, or rain showers).
    """
    times = hourly_data.get("time", [])
    if not times:
        return {
            "has_data": False,
            "best_window": {
                "label": "1:00 PM – 4:00 PM",
                "avg_score": 75,
                "badge": "🟢 Good",
                "temp_f": 58.0,
                "wind_mph": 10.0,
                "precip_prob": 5,
                "summary": "Midday is typically the most comfortable riding window with warmer ambient temperatures.",
            },
            "avoid_windows": [],
            "timeline": [],
        }

    # Determine date to evaluate
    if not target_date:
        target_date = times[0][:10]

    temps = hourly_data.get("temperature_2m", [])
    apparents = hourly_data.get("apparent_temperature", [])
    precip_probs = hourly_data.get("precipitation_probability", [])
    precips = hourly_data.get("precipitation", [])
    codes = hourly_data.get("weather_code", [])
    winds = hourly_data.get("wind_speed_10m", [])
    gusts = hourly_data.get("wind_gusts_10m", [])

    hours_list: List[Dict[str, Any]] = []

    for idx, t_str in enumerate(times):
        if not t_str.startswith(target_date):
            continue

        try:
            dt = datetime.fromisoformat(t_str)
        except Exception:
            continue

        # Focus on daytime/active hours: 06:00 to 21:00
        if not (6 <= dt.hour <= 21):
            continue

        t_c = temps[idx] if idx < len(temps) else 15.0
        app_c = apparents[idx] if idx < len(apparents) else t_c
        p_prob = precip_probs[idx] if idx < len(precip_probs) else 0.0
        p_mm = precips[idx] if idx < len(precips) else 0.0
        w_code = codes[idx] if idx < len(codes) else 0
        w_kmh = winds[idx] if idx < len(winds) else 10.0
        g_kmh = gusts[idx] if idx < len(gusts) else w_kmh

        cyc_eval = evaluate_cycling(
            temp_c=t_c,
            apparent_temp_c=app_c,
            wind_speed_kmh=w_kmh,
            wind_gusts_kmh=g_kmh,
            precip_prob=p_prob,
            precip_mm=p_mm,
            weather_code=w_code,
        )

        time_label = dt.strftime("%I:%M %p").lstrip("0")
        wmo = get_wmo_info(w_code)

        hours_list.append({
            "datetime": dt,
            "hour": dt.hour,
            "time_label": time_label,
            "temp_c": t_c,
            "temp_f": c_to_f(t_c),
            "apparent_temp_f": c_to_f(app_c),
            "wind_mph": kmh_to_mph(w_kmh),
            "gusts_mph": kmh_to_mph(g_kmh),
            "precip_prob": p_prob,
            "precip_mm": p_mm,
            "weather_icon": wmo["icon"],
            "weather_name": wmo["name"],
            "score": cyc_eval["score"],
            "rating": cyc_eval["rating"],
            "badge": cyc_eval["badge"],
        })

    if not hours_list:
        return {
            "has_data": False,
            "best_window": {
                "label": "1:00 PM – 4:00 PM",
                "avg_score": 75,
                "badge": "🟢 Good",
                "temp_f": 58.0,
                "wind_mph": 10.0,
                "precip_prob": 5,
                "summary": "Midday is typically the most comfortable window.",
            },
            "avoid_windows": [],
            "timeline": [],
        }

    # Evaluate 2-hour and 3-hour contiguous windows
    candidates = []
    for win_len in (3, 2):
        if len(hours_list) < win_len:
            continue
        for start_idx in range(len(hours_list) - win_len + 1):
            window_slice = hours_list[start_idx : start_idx + win_len]
            avg_score = round(sum(h["score"] for h in window_slice) / win_len, 1)
            max_rain = max(h["precip_prob"] for h in window_slice)
            avg_temp = round(sum(h["temp_f"] for h in window_slice) / win_len, 1)
            avg_wind = round(sum(h["wind_mph"] for h in window_slice) / win_len, 1)
            max_gust = max(h["gusts_mph"] for h in window_slice)

            start_dt = window_slice[0]["datetime"]
            end_dt = window_slice[-1]["datetime"] + timedelta(hours=1)
            label = f"{start_dt.strftime('%I:%M %p').lstrip('0')} – {end_dt.strftime('%I:%M %p').lstrip('0')}"

            # Bonus for 3-hour over 2-hour if scores are very close
            sort_weight = avg_score + (1.5 if win_len == 3 else 0.0)

            candidates.append({
                "label": label,
                "win_len": win_len,
                "avg_score": avg_score,
                "sort_weight": sort_weight,
                "max_rain": max_rain,
                "avg_temp": avg_temp,
                "avg_wind": avg_wind,
                "max_gust": max_gust,
                "start_hour": window_slice[0]["hour"],
                "start_time": window_slice[0]["time_label"],
                "end_time": end_dt.strftime("%I:%M %p").lstrip("0"),
                "icons": [h["weather_icon"] for h in window_slice],
            })

    # Pick the best candidate window
    candidates.sort(key=lambda c: (c["sort_weight"], -c["max_rain"], -c["max_gust"]), reverse=True)
    best_candidate = candidates[0] if candidates else None

    if best_candidate:
        score_val = best_candidate["avg_score"]
        if score_val >= 85:
            badge = "🟢 Prime Ride Window"
        elif score_val >= 70:
            badge = "🟢 Good Ride Window"
        elif score_val >= 50:
            badge = "🟡 Moderate Window"
        else:
            badge = "🟠 Challenging Window"

        summary = (
            f"Expected {best_candidate['avg_temp']:.0f}°F, {best_candidate['max_rain']:.0f}% rain chance, "
            f"and gentle winds around {best_candidate['avg_wind']:.0f} mph (gusts {best_candidate['max_gust']:.0f} mph). "
            f"Optimal tarmac traction and minimal aerodynamic resistance."
        )
        best_window = {
            "label": best_candidate["label"],
            "avg_score": best_candidate["avg_score"],
            "badge": badge,
            "temp_f": best_candidate["avg_temp"],
            "wind_mph": best_candidate["avg_wind"],
            "gusts_mph": best_candidate["max_gust"],
            "precip_prob": best_candidate["max_rain"],
            "summary": summary,
        }
    else:
        best_window = {
            "label": "1:00 PM – 4:00 PM",
            "avg_score": 70,
            "badge": "🟢 Good",
            "temp_f": 58.0,
            "wind_mph": 10.0,
            "precip_prob": 10,
            "summary": "Pleasant midday conditions.",
        }

    # Find Avoid Windows
    avoid_windows: List[Dict[str, Any]] = []
    current_avoid_run: List[Dict[str, Any]] = []

    for h in hours_list:
        is_bad = False
        reasons = []
        if h["score"] < 45:
            is_bad = True
            reasons.append(f"Low suitability score ({h['score']}/100)")
        if h["precip_prob"] >= 40 or h["precip_mm"] >= 1.0:
            is_bad = True
            reasons.append(f"High rain chance ({h['precip_prob']:.0f}%)")
        if h["gusts_mph"] >= 24.0 or h["wind_mph"] >= 18.0:
            is_bad = True
            reasons.append(f"Turbulent winds/gusts ({h['gusts_mph']:.0f} mph)")
        if h["apparent_temp_f"] < 38.0:
            is_bad = True
            reasons.append(f"Freezing wind chill ({h['apparent_temp_f']:.0f}°F)")

        if is_bad:
            h_copy = dict(h)
            h_copy["reasons"] = reasons
            current_avoid_run.append(h_copy)
        else:
            if current_avoid_run:
                # Close run
                s_dt = current_avoid_run[0]["datetime"]
                e_dt = current_avoid_run[-1]["datetime"] + timedelta(hours=1)
                lbl = f"{s_dt.strftime('%I:%M %p').lstrip('0')} – {e_dt.strftime('%I:%M %p').lstrip('0')}"
                # Combine unique reasons
                all_reasons = []
                for bad_h in current_avoid_run:
                    for r in bad_h["reasons"]:
                        if r not in all_reasons:
                            all_reasons.append(r)
                avoid_windows.append({
                    "label": lbl,
                    "reason": "; ".join(all_reasons),
                    "hours_count": len(current_avoid_run),
                })
                current_avoid_run = []

    if current_avoid_run:
        s_dt = current_avoid_run[0]["datetime"]
        e_dt = current_avoid_run[-1]["datetime"] + timedelta(hours=1)
        lbl = f"{s_dt.strftime('%I:%M %p').lstrip('0')} – {e_dt.strftime('%I:%M %p').lstrip('0')}"
        all_reasons = []
        for bad_h in current_avoid_run:
            for r in bad_h["reasons"]:
                if r not in all_reasons:
                    all_reasons.append(r)
        avoid_windows.append({
            "label": lbl,
            "reason": "; ".join(all_reasons),
            "hours_count": len(current_avoid_run),
        })

    # Timeline of daytime hours
    timeline = [
        {
            "time": h["time_label"],
            "temp_f": h["temp_f"],
            "wind_mph": h["wind_mph"],
            "precip_prob": h["precip_prob"],
            "icon": h["weather_icon"],
            "score": h["score"],
            "rating": h["rating"],
        }
        for h in hours_list
    ]

    return {
        "has_data": True,
        "best_window": best_window,
        "avoid_windows": avoid_windows,
        "timeline": timeline,
    }


def analyze_weather(weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform complete multi-day analysis: Current, Today, and the next 7 days."""
    current = weather_data.get("current", {})
    daily = weather_data.get("daily", {})

    curr_temp_c = current.get("temperature_2m", 0.0)
    curr_apparent_c = current.get("apparent_temperature", curr_temp_c)
    curr_wind_kmh = current.get("wind_speed_10m", 0.0)
    curr_gusts_kmh = current.get("wind_gusts_10m", curr_wind_kmh)
    curr_precip = current.get("precipitation", 0.0)
    curr_code = current.get("weather_code", 0)

    # Current Evaluation
    curr_wmo = get_wmo_info(curr_code)
    current_clothing = evaluate_clothing(
        apparent_temp_c=curr_apparent_c,
        wind_speed_kmh=curr_wind_kmh,
        precip_prob=100.0 if curr_precip > 0 else 0.0,
        precip_mm=curr_precip,
        uv_index=3.0,
    )
    current_cycling = evaluate_cycling(
        temp_c=curr_temp_c,
        apparent_temp_c=curr_apparent_c,
        wind_speed_kmh=curr_wind_kmh,
        wind_gusts_kmh=curr_gusts_kmh,
        precip_prob=100.0 if curr_precip > 0 else 0.0,
        precip_mm=curr_precip,
        weather_code=curr_code,
    )

    # Daily Forecast Evaluation (Days 0 to 7)
    daily_times = daily.get("time", [])
    daily_codes = daily.get("weather_code", [])
    temp_max = daily.get("temperature_2m_max", [])
    temp_min = daily.get("temperature_2m_min", [])
    app_max = daily.get("apparent_temperature_max", [])
    app_min = daily.get("apparent_temperature_min", [])
    precip_sums = daily.get("precipitation_sum", [])
    precip_probs = daily.get("precipitation_probability_max", [])
    wind_max = daily.get("wind_speed_10m_max", [])
    gust_max = daily.get("wind_gusts_10m_max", [])
    uv_max = daily.get("uv_index_max", [])

    daily_reports: List[Dict[str, Any]] = []
    best_cycling_days: List[Tuple[str, int, str]] = []

    for i in range(len(daily_times)):
        d_date_str = daily_times[i]
        d_obj = datetime.strptime(d_date_str, "%Y-%m-%d")
        d_name = d_obj.strftime("%A, %b %d")
        is_today = (i == 0)

        d_code = daily_codes[i] if i < len(daily_codes) else 0
        d_wmo = get_wmo_info(d_code)
        t_max_c = temp_max[i] if i < len(temp_max) else 20.0
        t_min_c = temp_min[i] if i < len(temp_min) else 10.0
        a_max_c = app_max[i] if i < len(app_max) else t_max_c
        a_min_c = app_min[i] if i < len(app_min) else t_min_c
        p_sum = precip_sums[i] if i < len(precip_sums) else 0.0
        p_prob = precip_probs[i] if i < len(precip_probs) else 0.0
        w_spd = wind_max[i] if i < len(wind_max) else 10.0
        w_gst = gust_max[i] if i < len(gust_max) else w_spd
        uv = uv_max[i] if i < len(uv_max) else 3.0

        # Representative daytime feels-like temperature
        daytime_apparent = round((a_max_c * 0.7) + (a_min_c * 0.3), 1)

        d_clothing = evaluate_clothing(
            apparent_temp_c=daytime_apparent,
            wind_speed_kmh=w_spd,
            precip_prob=p_prob,
            precip_mm=p_sum,
            uv_index=uv,
        )

        d_cycling = evaluate_cycling(
            temp_c=t_max_c,
            apparent_temp_c=a_max_c,
            wind_speed_kmh=w_spd,
            wind_gusts_kmh=w_gst,
            precip_prob=p_prob,
            precip_mm=p_sum,
            weather_code=d_code,
        )

        daily_reports.append({
            "day_index": i,
            "is_today": is_today,
            "date": d_date_str,
            "day_name": d_name,
            "weather_name": d_wmo["name"],
            "weather_icon": d_wmo["icon"],
            "temp_min_c": t_min_c,
            "temp_max_c": t_max_c,
            "temp_min_f": c_to_f(t_min_c),
            "temp_max_f": c_to_f(t_max_c),
            "apparent_min_c": a_min_c,
            "apparent_max_c": a_max_c,
            "apparent_min_f": c_to_f(a_min_c),
            "apparent_max_f": c_to_f(a_max_c),
            "precip_sum_mm": p_sum,
            "precip_prob_max": p_prob,
            "wind_speed_max_kmh": w_spd,
            "wind_speed_max_mph": kmh_to_mph(w_spd),
            "wind_gusts_max_kmh": w_gst,
            "wind_gusts_max_mph": kmh_to_mph(w_gst),
            "uv_index_max": uv,
            "clothing": d_clothing,
            "cycling": d_cycling,
        })

        best_cycling_days.append((d_name, d_cycling["score"], d_cycling["rating"]))

    # Sort days by cycling suitability score
    sorted_cycling = sorted(best_cycling_days, key=lambda x: x[1], reverse=True)

    # Hourly optimal outdoor/cycling window analysis
    hourly_data = weather_data.get("hourly", {})
    optimal_window = find_optimal_outdoor_windows(hourly_data)

    return {
        "current": {
            "temp_c": curr_temp_c,
            "temp_f": c_to_f(curr_temp_c),
            "apparent_temp_c": curr_apparent_c,
            "apparent_temp_f": c_to_f(curr_apparent_c),
            "wind_speed_kmh": curr_wind_kmh,
            "wind_speed_mph": kmh_to_mph(curr_wind_kmh),
            "wind_gusts_kmh": curr_gusts_kmh,
            "wind_gusts_mph": kmh_to_mph(curr_gusts_kmh),
            "precip_mm": curr_precip,
            "weather_code": curr_code,
            "weather_name": curr_wmo["name"],
            "weather_icon": curr_wmo["icon"],
            "clothing": current_clothing,
            "cycling": current_cycling,
        },
        "daily": daily_reports,
        "best_cycling_rankings": sorted_cycling,
        "optimal_window": optimal_window,
    }
