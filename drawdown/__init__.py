# drawdown/__init__.py
from pathlib import Path
import importlib
import os

# --- Project paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
