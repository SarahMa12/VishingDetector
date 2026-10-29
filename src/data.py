"""Helpers for finding ASVspoof files, reading their labels, and loading clips.

Always load clips from the label files, not from a folder listing:
the 2019 dev and eval folders contain extra clips that have no label.
"""

import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from config import CLIP_SECONDS, DATA_PATHS_FILE, FEATURES_DIR, SAMPLE_RATE, TRIM_TOP_DB

# 2019 LA split -> (folder name, label file name)
_LA_2019 = {
    "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt"),
    "dev": ("ASVspoof2019_LA_dev", "ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval": ("ASVspoof2019_LA_eval", "ASVspoof2019.LA.cm.eval.trl.txt"),
}


def dataset_paths():
    """Folders recorded by download_data.py: {"asvspoof2019": ..., "asvspoof2021": ...}."""
    return json.loads(DATA_PATHS_FILE.read_text())


def load_la_2019(split):
    """Read one 2019 LA split into a table.

    split: "train" or "dev". ("eval" works too, but it is kept for the
    final test only.)

    Columns: speaker, file, system (fake-making method, "-" for real),
    label (0 = real, 1 = fake), path (full path to the .flac file).
    """
    root = Path(dataset_paths()["asvspoof2019"]) / "LA" / "LA"
    folder, label_file = _LA_2019[split]

    rows = []
    with open(root / "ASVspoof2019_LA_cm_protocols" / label_file) as f:
        for line in f:
            speaker, name, _, system, label = line.split()
            rows.append((speaker, name, system, 0 if label == "bonafide" else 1))

    df = pd.DataFrame(rows, columns=["speaker", "file", "system", "label"])
    df["path"] = [str(root / folder / "flac" / f"{n}.flac") for n in df["file"]]
    return df


def load_clip(path, seconds=CLIP_SECONDS):
    """Read one clip, trim the silence at both ends, then fit it to `seconds`.

    Clips shorter than `seconds` are repeated, longer ones are cut. Every
    feature, notebook and test set must load clips through this function,
    so the trimming rule is the same everywhere.

    Returns a float32 array of seconds * SAMPLE_RATE samples.
    """
    y, sr = sf.read(path, dtype="float32")
    assert sr == SAMPLE_RATE, f"{path} is {sr} Hz, expected {SAMPLE_RATE}"
    y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    return np.resize(y, int(seconds * SAMPLE_RATE))


def load_features(feature, split, seconds=CLIP_SECONDS):
    """Join the saved batches of one feature for one 2019 LA split.

    Returns (x, labels). `x` has one row per clip (float32), in the same
    order as `labels`, the table from load_la_2019. Stops if the saved
    clip names do not match the label file.
    """
    folder = FEATURES_DIR / f"{seconds:g}s" / split / feature
    batches = [np.load(f) for f in sorted(folder.glob("batch_?????.npz"))]
    x = np.concatenate([b["x"] for b in batches]).astype(np.float32)
    files = np.concatenate([b["files"] for b in batches])

    labels = load_la_2019(split)
    assert np.array_equal(files, labels["file"].to_numpy()), f"{folder}: clips do not match the label file"
    return x, labels
