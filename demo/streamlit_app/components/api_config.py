"""API configuration component for Streamlit sidebar."""

import streamlit as st
from typing import Optional

from langchain_core.language_models import BaseChatModel


def render_api_sidebar() -> None:
    """Render API configuration in the sidebar.

    This component handles:
    - Provider selection (OpenAI/Anthropic)
    - API key input
    - LLM instance creation
    - Connection status display
    """
    st.sidebar.title("🔑 API Configuration")

    provider = st.sidebar.selectbox(
        "LLM Provider",
        ["OpenAI", "Anthropic"],
        index=0 if st.session_state.get("api_provider", "openai") == "openai" else 1,
    )

    api_key = st.sidebar.text_input(
        f"{provider} API Key",
        type="password",
        value=st.session_state.get("api_key", ""),
        help=f"Enter your {provider} API key for question generation",
    )

    if api_key:
        st.session_state.api_key = api_key
        st.session_state.api_provider = provider.lower()

        try:
            if provider == "OpenAI":
                from langchain_openai import ChatOpenAI
                st.session_state.llm = ChatOpenAI(
                    api_key=api_key,
                    model="gpt-4o-mini",
                    temperature=0.7,
                )
            else:
                from langchain_anthropic import ChatAnthropic
                st.session_state.llm = ChatAnthropic(
                    api_key=api_key,
                    model="claude-3-haiku-20240307",
                    temperature=0.7,
                )
            st.session_state.llm_configured = True
            st.sidebar.success(f"✓ {provider} configured")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            st.session_state.llm_configured = False
    else:
        st.sidebar.warning("API key required for question generation")


def get_llm() -> Optional[BaseChatModel]:
    """Get the configured LLM instance.

    Returns:
        The configured LangChain chat model, or None if not configured.
    """
    if st.session_state.get("llm_configured", False):
        return st.session_state.get("llm")
    return None


def is_llm_configured() -> bool:
    """Check if LLM is configured and ready.

    Returns:
        True if LLM is configured, False otherwise.
    """
    return st.session_state.get("llm_configured", False)
