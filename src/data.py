"""Helpers for finding ASVspoof files and reading their labels.

Always load clips from the label files, not from a folder listing:
the 2019 dev and eval folders contain extra clips that have no label.
"""

import json
from pathlib import Path

import pandas as pd

from config import DATA_PATHS_FILE

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
