"""Compatibility entry point for ``uvicorn app:app``."""

# Keep legacy commands on the one authoritative API application.
from main import app
