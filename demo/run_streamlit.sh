#!/bin/bash
# Run the Streamlit demo with proper PYTHONPATH

cd "$(dirname "$0")/.."
PYTHONPATH=src uv run --extra streamlit streamlit run demo/streamlit_app/app.py "$@"
