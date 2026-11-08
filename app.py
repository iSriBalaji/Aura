"""Aura — Personal AI Lifestyle & Intelligence Copilot for Baju (Sri Balaji).
Streamlit Web App optimized for desktop and mobile browsers.
Powered by LangGraph.
"""

import os
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Load local environment if present
load_dotenv()

# Import the LangGraph agent and services
from weather_agent.graph import graph
from weather_agent.geolocation import detect_device_location
from weather_agent.memory_service import (
    add_custom_memory,
    format_profile_markdown,
    load_profile,
    remove_custom_memory,
)
from weather_agent.tts_service import clean_markdown_for_speech, generate_audio_bytes

# Page Configuration
st.set_page_config(
    page_title="Aura — Copilot for Baju",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for a polished, modern look on mobile and desktop
st.markdown(
    """
    <style>
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 950px;
    }
    
    /* Header hero */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    
    /* Quick Action Pill Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
        border: 1px solid #CBD5E1;
    }
    .stButton>button:hover {
        border-color: #3B82F6;
        color: #3B82F6;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15);
    }
    
    /* Chat message bubble refinement */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def run_aura_query(
    query: str,
    location_override: str = None,
    intent_override: str = None,
    model_override: str = None,
    api_key: str = None,
):
    """Invoke the LangGraph Aura agent and return markdown response."""
    state_input = {}
    
    if location_override and location_override.strip():
        state_input["location_query"] = location_override.strip()

    if intent_override:
        state_input["intent"] = intent_override

    if model_override:
        state_input["model_name"] = model_override

    if api_key and api_key.strip():
        state_input["api_key"] = api_key.strip()

    state_input["messages"] = [HumanMessage(content=query)]

    try:
        result = graph.invoke(state_input)
        return result.get("final_report", "I processed your request, but could not produce a report.")
    except Exception as e:
        return f"**Error running Aura**: {str(e)}\n\nPlease try again or check your location input."


def render_audio_player(markdown_text: str, key_suffix: str = "latest"):
    """Render zero-cost voice playback controls: Browser Web Speech API + optional MP3 audio."""
    speech_text = clean_markdown_for_speech(markdown_text)
    safe_js_text = (
        speech_text.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("\r", "")
    )

    # Browser-native Web Speech API widget: 100% free, zero lag, works on iOS/Android/Mac/PC
    html_code = f"""
    <div style="display: flex; align-items: center; gap: 10px; margin-top: 8px; margin-bottom: 6px; padding: 7px 12px; background: #F8FAFC; border-radius: 8px; border: 1px solid #E2E8F0;">
        <span style="font-size: 13px; font-weight: 600; color: #334155;">Voice:</span>
        <button id="playBtn_{key_suffix}" onclick="speakAura_{key_suffix}()" style="padding: 5px 12px; font-size: 12px; font-weight: 600; background: #2563EB; color: white; border: none; border-radius: 6px; cursor: pointer;">
            Speak
        </button>
        <button onclick="window.speechSynthesis.cancel()" style="padding: 5px 12px; font-size: 12px; font-weight: 600; background: #E2E8F0; color: #475569; border: none; border-radius: 6px; cursor: pointer;">
            Stop
        </button>
    </div>
    <script>
    function speakAura_{key_suffix}() {{
        window.speechSynthesis.cancel();
        const text = "{safe_js_text}";
        const utter = new SpeechSynthesisUtterance(text);
        utter.rate = 1.0;
        utter.pitch = 1.0;
        window.speechSynthesis.speak(utter);
    }}
    </script>
    """
    components.html(html_code, height=50)

    # Optional MP3 generation expander
    with st.expander("Generate Audio File (MP3)", expanded=False):
        if st.button("Generate MP3 Audio", key=f"gen_mp3_{key_suffix}"):
            with st.spinner("Generating MP3..."):
                audio_bytes = generate_audio_bytes(markdown_text)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                else:
                    st.info("Browser Speech above is active. Server audio requires an active internet connection.")


# Initialize Chat Session State
if "messages" not in st.session_state:
    st.session_state.messages = []


# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("### Aura")
    st.caption("Personal Copilot for Baju")
    
    st.markdown("---")
    st.markdown("#### Location")
    default_loc = st.text_input("Active Location / City", value="Prospect Heights, Illinois")
    
    if st.button("Auto-Detect Device Location", use_container_width=True):
        with st.spinner("Detecting location..."):
            detected = detect_device_location()
            st.session_state.detected_location = detected.get("name", "Unknown")
            st.success(f"Detected: {st.session_state.detected_location}")
            default_loc = st.session_state.detected_location

    st.markdown("---")
    st.markdown("#### Quick Briefings")
    st.caption("Tap any button for an instant report:")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        btn_weather = st.button("Weather & Layers", use_container_width=True)
        btn_optimal_window = st.button("Optimal Ride Window", use_container_width=True)
        btn_news = st.button("Trending News", use_container_width=True)
    with col_btn2:
        btn_cycling = st.button("Cycling Score", use_container_width=True)
        btn_events = st.button("Events & Sports", use_container_width=True)
        btn_memory = st.button("Saved Memory", use_container_width=True)

    btn_full = st.button("7-Day Complete Planner", use_container_width=True)

    st.markdown("---")
    with st.expander("Memory & Preferences", expanded=False):
        profile = load_profile()
        sports = profile.get("sports", {})
        cyc = profile.get("cycling", {})
        st.markdown(f"**F1:** {', '.join(sports.get('f1_teams', []))} (*{', '.join(sports.get('f1_drivers', []))}*)")
        st.markdown(f"**NFL:** {', '.join(sports.get('nfl_teams', []))}")
        st.markdown(f"**Cricket:** {', '.join(sports.get('cricket_teams', []))}")
        st.markdown(f"**Ride Comfort:** Min {cyc.get('min_comfortable_temp_f')}°F | Max wind {cyc.get('max_acceptable_wind_mph')} mph")
        
        st.markdown("---")
        st.markdown("**Add a New Memory Note:**")
        new_memory_input = st.text_input("Teach Aura something new:", placeholder="e.g. Prefer riding between 1 PM and 4 PM", key="sidebar_new_mem")
        if st.button("Save Memory Note", use_container_width=True):
            if new_memory_input.strip():
                if add_custom_memory(new_memory_input.strip()):
                    st.success("Saved to persistent memory!")
                    st.rerun()

        notes = profile.get("custom_notes", [])
        if notes:
            st.markdown(f"**Saved Notes ({len(notes)}):**")
            for idx, n in enumerate(notes):
                c_txt, c_del = st.columns([5, 1])
                with c_txt:
                    st.caption(f"• {n.get('note')}")
                with c_del:
                    if st.button("✕", key=f"del_mem_{idx}"):
                        remove_custom_memory(idx)
                        st.rerun()

    st.markdown("---")
    st.markdown("#### AI Engine & Model")

    MODEL_OPTIONS = {
        "Google Gemini 2.5 Flash (Recommended)": "gemini-2.5-flash",
        "Google Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
        "Google Gemini 1.5 Flash": "gemini-1.5-flash",
        "Google Gemini 1.5 Pro": "gemini-1.5-pro",
        "Groq Llama 3.3 70B": "llama-3.3-70b-versatile",
        "Groq Llama 3.1 8B": "llama-3.1-8b-instant",
        "OpenAI GPT-4o-mini": "gpt-4o-mini",
        "Built-in Engine (Offline)": "builtin",
    }

    selected_label = st.selectbox(
        "Active Model",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
        help="Choose the model used for conversational synthesis."
    )
    selected_model_id = MODEL_OPTIONS[selected_label]
    st.session_state.active_model = selected_model_id

    # Secure API Key Handling
    # Priority: 1. Streamlit Cloud Secrets (most secure, never exposed to visitors)
    #           2. Session state input (password masked, kept only in user's browser session)
    #           3. Local .env / environment variable
    active_api_key = None

    if "gemini" in selected_model_id:
        cloud_secret_key = None
        try:
            if hasattr(st, "secrets"):
                cloud_secret_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            cloud_secret_key = None

        if cloud_secret_key:
            active_api_key = cloud_secret_key
            st.success("API Key secured via Streamlit Cloud Secrets (hidden and protected from public visitors).")
        else:
            session_key = st.text_input(
                "Gemini API Key",
                type="password",
                value=st.session_state.get("session_gemini_key", "") or os.getenv("GEMINI_API_KEY", ""),
                placeholder="AIzaSy...",
                help="Your key is password-masked, isolated strictly to your active session, and never exposed to other visitors.",
            )
            if session_key:
                st.session_state.session_gemini_key = session_key.strip()
                active_api_key = session_key.strip()

        st.caption(
            "[Get Free Gemini API Key (Google AI Studio)](https://aistudio.google.com/) — 15 req/min free."
        )

    elif "llama" in selected_model_id:
        cloud_secret_key = None
        try:
            if hasattr(st, "secrets"):
                cloud_secret_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            cloud_secret_key = None

        if cloud_secret_key:
            active_api_key = cloud_secret_key
            st.success("Groq API Key secured via Streamlit Cloud Secrets.")
        else:
            session_key = st.text_input(
                "Groq API Key",
                type="password",
                value=st.session_state.get("session_groq_key", "") or os.getenv("GROQ_API_KEY", ""),
                placeholder="gsk_...",
                help="Password-masked and isolated strictly to your session.",
            )
            if session_key:
                st.session_state.session_groq_key = session_key.strip()
                active_api_key = session_key.strip()

        st.caption("[Get Free Groq API Key](https://console.groq.com/keys)")

    elif "gpt" in selected_model_id:
        cloud_secret_key = None
        try:
            if hasattr(st, "secrets"):
                cloud_secret_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            cloud_secret_key = None

        if cloud_secret_key:
            active_api_key = cloud_secret_key
            st.success("OpenAI API Key secured via Streamlit Cloud Secrets.")
        else:
            session_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.get("session_openai_key", "") or os.getenv("OPENAI_API_KEY", ""),
                placeholder="sk-proj-...",
                help="Password-masked and isolated strictly to your session.",
            )
            if session_key:
                st.session_state.session_openai_key = session_key.strip()
                active_api_key = session_key.strip()

        st.caption("[OpenAI API Platform](https://platform.openai.com/api-keys)")
    else:
        st.info("Built-in Engine: Running offline rule algorithms and live feeds with zero keys.")

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ===================== MAIN CONTENT =====================

# Header
st.markdown("<div class='hero-title'>Aura</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='hero-subtitle'>Good day, <b>Baju</b>. Active location: <code>{default_loc}</code></div>",
    unsafe_allow_html=True,
)

# Handle Quick Action Buttons
triggered_query = None
triggered_intent = None

if btn_weather:
    triggered_query = f"What is the weather today and what clothing layers should I wear in {default_loc}?"
    triggered_intent = "weather"
elif btn_optimal_window:
    triggered_query = f"What is the optimal 2-3 hour window for cycling today in {default_loc}, and what are the avoid periods?"
    triggered_intent = "weather"
elif btn_cycling:
    triggered_query = f"Can I cycle today in {default_loc}? What is the cycling score and recommended kit?"
    triggered_intent = "weather"
elif btn_news:
    triggered_query = f"Give me today's executive news digest with 2-3 line summaries across General & World, Technology, Politics, Business, Science, and Local news for {default_loc}."
    triggered_intent = "news_digest"
elif btn_events:
    triggered_query = f"What major events, conferences, festivals, and sports updates are happening right now (past week, today, or next week / ±17 days) in the US or in {default_loc}?"
    triggered_intent = "events_sports"
elif btn_memory:
    triggered_query = "What do you remember about me? Show my profile and memories."
    triggered_intent = "memory_view"
elif btn_full:
    triggered_query = f"Give me the full 7-day weather outlook, layer decisions, and weekly cycling planner for {default_loc}."
    triggered_intent = "weather"

if triggered_query:
    st.session_state.messages.append({"role": "user", "content": triggered_query})
    with st.spinner("Analyzing live data..."):
        bot_response = run_aura_query(
            query=triggered_query,
            location_override=default_loc,
            intent_override=triggered_intent,
            model_override=st.session_state.get("active_model"),
            api_key=active_api_key,
        )
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

# Render Existing Conversation Messages
if not st.session_state.messages:
    # Friendly Minimal Welcome Banner for Baju
    st.markdown(
        f"""
        > **Hello Baju.** I'm Aura, your personal lifestyle and daily briefing copilot.
        > 
        > Tap any of the quick briefings in the sidebar or ask me directly below:
        > - *"What is the best 2-3 hour window for cycling today?"*
        > - *"What should I wear today in Prospect Heights?"*
        > - *"Give me today's 2-3 line news summaries across tech, business, politics, science, and local news."*
        > - *"What major US events or sports updates are happening?"*
        > - *"Can I wear shorts today?"*
        """
    )
else:
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_audio_player(msg["content"], key_suffix=f"msg_{idx}")

# Chat Input Box (Fixed at Bottom, Mobile-Friendly)
if prompt := st.chat_input("Ask Aura anything, Baju..."):
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke Aura
    with st.chat_message("assistant"):
        with st.spinner("Aura is thinking..."):
            response_text = run_aura_query(
                query=prompt,
                location_override=default_loc,
                model_override=st.session_state.get("active_model"),
                api_key=active_api_key,
            )
            st.markdown(response_text)
            render_audio_player(response_text, key_suffix=f"msg_{len(st.session_state.messages)}")

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response_text})
