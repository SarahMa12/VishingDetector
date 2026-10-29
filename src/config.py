"""Shared settings for the project.

Every script and notebook imports paths and audio settings from here,
so a value only ever has to be changed in one place.
"""

from pathlib import Path

# Project root = the folder that contains `src/`.
ROOT = Path(__file__).resolve().parent.parent

# Downloaded ASVspoof audio and label files. Ignored by git (large).
DATA_DIR = ROOT / "data"

# Saved feature files, one per feature group. Ignored by git (large).
FEATURES_DIR = ROOT / "features"

# Where download_data.py records the folder each dataset was saved to.
# The exact folder layout inside `data/` is set by kagglehub, so we read
# the real paths from this file instead of guessing them.
DATA_PATHS_FILE = ROOT / "data_paths.json"

# Audio settings. ASVspoof audio is 16 kHz FLAC.
SAMPLE_RATE = 16000

# Every clip is repeated (if short) or trimmed (if long) to this length,
# so all clips give features of the same size.
CLIP_SECONDS = 4
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS  # 64,000

# Silence at the start and end of a clip is cut off. Anything more than this
# many dB quieter than the loudest part counts as silence. (The leading
# silence differs between real and fake clips, so it would give the label away.)
TRIM_TOP_DB = 30

# Feature settings
N_MFCC = 20
N_LFCC = 20
N_MELS = 80
HOP_LENGTH = 512  # samples between frames; every frame-based feature uses it
