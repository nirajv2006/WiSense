"""Central configuration for the WiSense pipeline.

Everything that is a *design decision* (class definitions, window length,
filter settings, power policy) lives here so it can be documented and changed
in one place.
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # .../python
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                              # labelled recordings
FEATURES_CSV = DATA_DIR / "features.csv"                # windowed feature table
DB_PATH = DATA_DIR / "wisense.db"                       # live predictions (SQLite)
LIVE_SNAPSHOT = DATA_DIR / "live_window.npz"            # latest CSI window for plots
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "wisense_model.joblib"
REPORTS_DIR = ROOT / "reports"

# ── Occupancy classes ────────────────────────────────────────────────────────
# Three ordinal bands. We do NOT count people; the people count typed during
# data collection is only used to check that recordings match their band.
CLASSES = ["low", "medium", "high"]
CLASS_DEFINITIONS = {
    "low": "0-1 people in the sensing zone",
    "medium": "2-3 people",
    "high": "4 or more people",
}
# Used only to sanity-check labels typed during collection (max None = no upper limit)
CLASS_PEOPLE_RANGE = {"low": (0, 1), "medium": (2, 3), "high": (4, None)}

# ── Acquisition ──────────────────────────────────────────────────────────────
BAUD = 921600
EXPECTED_RATE_HZ = 50          # transmitter sends one UDP packet every 20 ms

# ESP32 LLTF CSI layout: 64 subcarriers stored as (imag, real) int8 pairs.
# Index order is 0..31 then -32..-1. Index 0 is DC; 27..37 are guard bands.
N_RAW_SUBCARRIERS = 64
VALID_SUBCARRIERS = [i for i in range(64) if i != 0 and not (27 <= i <= 37)]   # 52

# ── Windowing ────────────────────────────────────────────────────────────────
WINDOW_S = 4.0                 # one classification uses 4 s of CSI
STEP_S = 1.0                   # a new decision every 1 s (windows overlap)
MIN_FILL = 0.6                 # skip windows with < 60% of expected packets

# ── Signal processing ────────────────────────────────────────────────────────
HAMPEL_HALF_WINDOW = 5         # samples on each side (≈0.2 s at 50 Hz)
HAMPEL_N_SIGMA = 3.0
LOWPASS_CUTOFF_HZ = 10.0       # human motion is well below 10 Hz
LOWPASS_ORDER = 4

# Frequency bands for spectral features (Hz)
BANDS = {
    "breath": (0.1, 0.6),      # breathing / very slow posture changes
    "motion": (0.6, 3.0),      # walking, gestures
    "fast": (3.0, 10.0),       # fast movement (and residual noise)
}

# ── Live inference ───────────────────────────────────────────────────────────
PROBA_SMOOTHING = 3            # average class probabilities over last N windows

# ── Power recommendation policy ──────────────────────────────────────────────
# Recommendations only - WiSense never switches any appliance.
POWER_LEVELS = {
    "low": {
        "level": "ECO",
        "share": 0.3,
        "actions": "Minimum lighting in occupied zone, fans/AC on lowest or eco setpoint, "
                   "ventilation at minimum fresh-air rate.",
    },
    "medium": {
        "level": "STANDARD",
        "share": 0.6,
        "actions": "Normal lighting, fans/AC at standard setpoint, ventilation at normal rate.",
    },
    "high": {
        "level": "FULL",
        "share": 1.0,
        "actions": "Full lighting, fans/AC at comfort setpoint, ventilation boosted for crowd.",
    },
}
POWER_ORDER = ["ECO", "STANDARD", "FULL"]
STEP_UP_AFTER_S = 5            # react quickly when more people arrive (comfort first)
STEP_DOWN_AFTER_S = 60         # wait longer before reducing (avoid flicker)
MIN_CONFIDENCE = 0.55          # below this, hold the current level
