"""
Launcher for the Streamlit dashboard (streamlit_app.py).

    uv run python main.py

Equivalent to ``streamlit run streamlit_app.py``. The production app (FastAPI
+ worker + React) is started separately; see README.md.
"""

import sys
from pathlib import Path


def main():
    """Entry point for running the Deep Agents Streamlit application."""
    app_path = Path(__file__).parent / "streamlit_app.py"
    print(f"🚀 Launching Deep Agents Conversational Chatbot from {app_path.name}...")

    # Imported here so merely importing this module stays cheap.
    from streamlit.web import cli as stcli

    # Streamlit's CLI reads its arguments from sys.argv.
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
