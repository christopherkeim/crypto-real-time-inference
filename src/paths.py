"""
These path definitions will execute every time you import a path into
a script.
"""

import os
from pathlib import Path

PARENT_DIR: Path = Path(__file__).parent.resolve().parent
DATA_DIR: Path = PARENT_DIR / "data"
MODELS_DIR: Path = PARENT_DIR / "models"
GRAPHS_DIR: Path = PARENT_DIR / "graphs"

if not DATA_DIR.exists():
    os.mkdir(DATA_DIR)

if not MODELS_DIR.exists():
    os.mkdir(MODELS_DIR)

if not GRAPHS_DIR.exists():
    os.mkdir(GRAPHS_DIR)
