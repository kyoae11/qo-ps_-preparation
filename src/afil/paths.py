"""Project-wide filesystem paths.

All paths are derived from this file, so scripts work on both Windows and
Linux without machine-specific absolute paths.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "ws_dream" / "dataset1"
SIMULATED_DATA_DIR = PROJECT_ROOT / "data" / "simulated"
PILOT_DATA_DIR = SIMULATED_DATA_DIR / "pilot"
FULL_SIMULATED_DATA_DIR = SIMULATED_DATA_DIR / "full"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
RESULT_DIR = OUTPUT_DIR / "results"
FIGURE_DIR = OUTPUT_DIR / "figures"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"


def ensure_output_dirs():
    """Create ignored runtime-output directories when an experiment starts."""
    for directory in (LOG_DIR, RESULT_DIR, FIGURE_DIR, CHECKPOINT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
