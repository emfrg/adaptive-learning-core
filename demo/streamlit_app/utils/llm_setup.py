"""LLM setup utilities for Streamlit."""

import asyncio
from typing import Any, Callable, Coroutine, Optional, TypeVar

import streamlit as st
from langchain_core.language_models import BaseChatModel


T = TypeVar("T")


def get_configured_llm() -> Optional[BaseChatModel]:
    """Get the configured LLM from session state.

    Returns:
        The configured LangChain chat model, or None if not configured.
    """
    if st.session_state.get("llm_configured", False):
        return st.session_state.get("llm")
    return None


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine in Streamlit.

    Streamlit doesn't natively support async, so we need to
    run coroutines in a new event loop.

    Args:
        coro: The coroutine to run.

    Returns:
        The result of the coroutine.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def run_async_with_spinner(
    coro: Coroutine[Any, Any, T],
    message: str = "Processing...",
) -> T:
    """Run an async coroutine with a Streamlit spinner.

    Args:
        coro: The coroutine to run.
        message: The spinner message to display.

    Returns:
        The result of the coroutine.
    """
    with st.spinner(message):
        return run_async(coro)
