# 🌤️ Weather, Clothing & Cycling Advisor Agent (LangGraph)

A reliable, production-ready, and cost-effective AI agent built with **LangGraph**. It fetches live, real-time weather and a 7-day forecast to make critical daily and weekly decisions:
1. **Wardrobe & Layering**: Computes exact layer counts (1-4) and garment breakdowns (base, mid, outer, bottoms, footwear, weather accessories).
2. **Cycling Suitability**: Evaluates road conditions, aerodynamic drag/wind, temperature comfort, and severe weather risks to calculate a 0-100 Cycling Score, required cycling kit, tire pressure tweaks, and safety cautions.
3. **Location Intelligence**: Automatically detects current device location via IP geolocation fallback or geocodes any user-specified city, address, or latitude/longitude coordinates.
4. **LangGraph Cloud Ready**: Packaged with standard `langgraph.json` configuration, ready for seamless deployment to LangGraph Cloud and LangGraph Studio.

---

## 🏗️ Architecture & Cost-Effectiveness

```
                    ┌────────────────────────────┐
                    │       User / Device        │
                    │ (Auto-detect IP or Query)  │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ 1. resolve_location_node   │
                    │  (Open-Meteo Geocoding /   │
                    │   Device Geolocation)      │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ 2. fetch_weather_node      │
                    │  (Open-Meteo API: Live +   │
                    │   7-Day Hourly/Daily)      │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ 3. decision_engine_node    │
                    │  (Clothing Layer Rules +   │
                    │   Cycling Score [0-100])   │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ 4. synthesize_report_node  │
                    │  (Gemini 2.5 Flash /       │
                    │   GPT-4o-mini / Fallback)  │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │     Actionable Report      │
                    │  (Today & 7-Day Decisions) │
                    └────────────────────────────┘
```

### Why It's Cost-Effective & Reliable:
- **Zero API Weather Cost**: Utilizes Open-Meteo's open meteorological data and geocoding API (10,000 free calls/day, no credit card, no proprietary API keys needed).
- **Zero Hallucination Physics**: Layering rules and cycling physics (apparent temp, wind chill, wet tire friction, drag) are calculated deterministically in Python before LLM synthesis.
- **Ultra-Low Cost LLM Synthesis**: Supports **Gemini 2.5 Flash** (sub-cent cost or free tier) or **GPT-4o-mini**. If no LLM API key is configured, it falls back to a clean deterministic Markdown report without failing.

---

## 🚀 Quickstart & Local Testing

### 1. Environment Setup
The project uses Python 3.10+:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your API key (if you want LLM-synthesized narrative reports; otherwise the agent runs using the built-in deterministic formatter):
```env
GEMINI_API_KEY=your_gemini_api_key_here
# or
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run Automated Tests
```bash
python3 test_agent.py
```
This tests:
- Algorithmic clothing layering and cycling suitability scoring.
- Geocoding and weather API calls.
- Full end-to-end LangGraph execution with device location, city queries, and coordinates.

### 4. Interactive CLI Testing
Run with device auto-location:
```bash
python3 cli.py
```

Run for a specific city:
```bash
python3 cli.py --location "Austin, TX"
python3 cli.py --location "Tokyo"
```

Run with natural language query:
```bash
python3 cli.py -m "I'm in Seattle, what should I wear and can I bike this week?"
```

Run interactive conversational mode:
```bash
python3 cli.py --interactive
```

---

## ☁️ Deploying to LangGraph Cloud

The project is structured according to official LangGraph Cloud specifications:

### Files Provided:
- `langgraph.json`: Cloud graph configuration mapping `weather_agent` to `./weather_agent/graph.py:graph`.
- `pyproject.toml` and `requirements.txt`: Dependency specifications.
- `weather_agent/graph.py`: Exported compiled `graph` instance.

### Deployment Steps:
1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for Weather & Activity Advisor"
   git remote add origin https://github.com/<your-user>/<repo-name>.git
   git push -u origin main
   ```
2. **Connect to LangGraph Cloud**:
   - Go to [smith.langchain.com](https://smith.langchain.com/) and navigate to the **Deployments** section.
   - Click **New Deployment** and select your GitHub repository.
   - Set the configuration file path to `langgraph.json`.
   - Add environment variables (`GEMINI_API_KEY` or `OPENAI_API_KEY`).
   - Click **Deploy**.
3. **LangGraph Studio**:
   - You can also test locally with LangGraph Studio CLI:
     ```bash
     pip install langgraph-cli
     langgraph dev
     ```
   - This opens the LangGraph Studio UI in your browser to inspect nodes, visualize state transitions, and test execution.

---

## 📊 Decision Criteria Reference

### Clothing Layering Engine
- **Extreme Cold (< -5°C / < 23°F)**: 4 Layers (Thermal base, fleece/wool mid, insulated down parka outer, thermal pants, winter boots, beanie, insulated gloves).
- **Very Cold (-5°C to 5°C / 23°F to 41°F)**: 3 Layers (Thermal long-sleeve, sweater/fleece, insulated winter coat).
- **Chilly / Cool (5°C to 12°C / 41°F to 54°F)**: 2-3 Layers (Long sleeve shirt, knit sweater/hoodie, light jacket/windbreaker).
- **Mild (12°C to 18°C / 54°F to 64°F)**: 2 Layers (T-shirt, light jacket/overshirt).
- **Warm (18°C to 24°C / 64°F to 75°F)**: 1 Layer (Breathable tee, shorts/chinos).
- **Hot (24°C to 30°C / 75°F to 86°F)**: 1 Layer (Loose cotton/linen, sun cap, sunglasses).
- **Heatwave (> 30°C / > 86°F)**: 1 Minimal Layer (UV-protective fabrics, SPF 50+, hydration pack).
- **Precipitation Modifiers**: Auto-adds waterproof raincoat/hardshell, umbrella, and water-resistant footwear.

### Cycling Suitability Score (0 - 100)
- **Temperature Comfort (0 - 30 pts)**: Optimal 15°C - 23°C (59°F - 73°F). Severe penalties below 0°C (black ice) and above 36°C (heatstroke).
- **Precipitation & Road Grip (0 - 35 pts)**: Dry pavement (35 pts), damp (25 pts), wet/spray (14 pts), downpour (0 pts).
- **Wind & Aerodynamic Drag (0 - 25 pts)**: Calm < 12 km/h (25 pts), moderate 12-20 km/h (20 pts), strong 20-28 km/h (12 pts), hazardous > 38 km/h (0 pts).
- **Weather Hazards (0 - 10 pts)**: Clear (10 pts), fog (5 pts), storm/ice (0 pts with hazard override).
