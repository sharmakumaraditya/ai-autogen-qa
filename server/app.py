"""
OpenEnv-compatible server entry point.
Re-exports the FastAPI app from the root server module.
"""
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server import app  # noqa: F401
