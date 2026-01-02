"""Adaptive Learning System - Streamlit Demo.

A multi-page Streamlit application demonstrating the adaptive learning system.

Run with: uv run --extra streamlit streamlit run demo/streamlit_app/app.py
"""

import os
import streamlit as st

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Adaptive Learning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Get API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Auto-configure LLM from environment variables on first load
if "llm_configured" not in st.session_state:
    st.session_state.llm_configured = False
    st.session_state.llm = None
    st.session_state.api_provider = None

    # Try OpenAI first, then Anthropic
    if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
        try:
            from langchain_openai import ChatOpenAI
            st.session_state.llm = ChatOpenAI(
                api_key=OPENAI_API_KEY,
                model="gpt-4o-mini",
                temperature=0.7,
            )
            st.session_state.llm_configured = True
            st.session_state.api_provider = "openai"
        except Exception:
            pass

    elif ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "your_anthropic_api_key_here":
        try:
            from langchain_anthropic import ChatAnthropic
            st.session_state.llm = ChatAnthropic(
                api_key=ANTHROPIC_API_KEY,
                model="claude-3-haiku-20240307",
                temperature=0.7,
            )
            st.session_state.llm_configured = True
            st.session_state.api_provider = "anthropic"
        except Exception:
            pass

# Sidebar - API Status
st.sidebar.title("🔑 API Status")

if st.session_state.llm_configured:
    provider_name = "OpenAI" if st.session_state.api_provider == "openai" else "Anthropic"
    st.sidebar.success(f"✓ {provider_name} configured from .env")
else:
    st.sidebar.warning("No API key found in .env")
    st.sidebar.info("Add OPENAI_API_KEY or ANTHROPIC_API_KEY to your .env file")

# Sidebar - Session Info
st.sidebar.divider()
st.sidebar.subheader("📊 Session Info")

if "module" in st.session_state and st.session_state.module:
    st.sidebar.write(f"**Module:** {st.session_state.module.name}")
    st.sidebar.write(f"**Cards:** {len(st.session_state.get('cards', []))}")
    st.sidebar.write(f"**Questions:** {len(st.session_state.get('questions', []))}")
else:
    st.sidebar.write("No module loaded")

# Main page content
st.title("🎓 Adaptive Learning System")
st.markdown("""
Welcome to the **Adaptive Learning System** demo! This application demonstrates
personalized e-learning with:

- **Bloom's Taxonomy** for cognitive level classification
- **Mastery Learning** for progression tracking
- **Item Response Theory (IRT)** for question evaluation
- **LangGraph** for orchestration

### Getting Started

1. **Configure API Key** (sidebar) - Required for question generation
2. **Upload Document** - Upload a PDF or paste text to learn from
3. **Start Learning** - Begin an adaptive learning session
4. **Track Progress** - View your mastery scores and quiz history

### Pages

| Page | Description |
|------|-------------|
| 📄 Upload Document | Upload PDF or text, create learning cards |
| 🎓 Learning Session | Interactive quizzes with adaptive card selection |
| 📊 Progress | Visualize mastery scores and learning history |
| 🔬 Algorithms | Explore the underlying algorithms interactively |

---

*Built with [Adaptive Learning Core](https://github.com/emfrg/adaptive-learning-core)*
""")

# Show quick stats if data is loaded
if "card_states" in st.session_state and st.session_state.card_states:
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    card_states = st.session_state.card_states
    avg_mastery = sum(s.mastery_score for s in card_states.values()) / len(card_states)

    with col1:
        st.metric("Cards", len(card_states))
    with col2:
        st.metric("Avg. Mastery", f"{avg_mastery:.0%}")
    with col3:
        learned = sum(1 for s in card_states.values() if s.mastery_score >= 0.8)
        st.metric("Mastered", f"{learned}/{len(card_states)}")
    with col4:
        quizzes = len(st.session_state.get("quiz_history", []))
        st.metric("Quizzes Taken", quizzes)
