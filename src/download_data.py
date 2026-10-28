"""Download the ASVspoof datasets from Kaggle into ./data.

Usage (from the project root):
    python src/download_data.py                  # both datasets
    python src/download_data.py asvspoof2019     # one dataset

asvspoof2019: ~24 GB. Train, dev and eval for LA and PA. We train on LA train.
asvspoof2021: ~62.5 GB. Evaluation only (LA, DF, PA) plus label keys.
              The PA folders are not needed and can be deleted afterward.

Downloads are cached, so re-running skips files that already exist.
The folder each dataset ends up in is printed and saved to data_paths.json.
"""

import os
import json
import sys

from config import DATA_DIR, DATA_PATHS_FILE

# The download location is a `./data` folder in your local project.
# kagglehub reads KAGGLEHUB_CACHE when it is imported, so this must be
# set before the `import kagglehub` line below.
DATA_DIR.mkdir(exist_ok=True)
os.environ["KAGGLEHUB_CACHE"] = str(DATA_DIR)

import kagglehub

# Short name -> Kaggle dataset handle.
DATASETS = {
    "asvspoof2019": "awsaf49/asvpoof-2019-dataset",
    "asvspoof2021": "mohammedabdeldayem/avsspoof-2021",
}


def main(names):
    """Download the named datasets and record where each one was saved.

    names: keys of DATASETS, e.g. ["asvspoof2019"].
    data_paths.json is rewritten after each download, so a finished
    dataset is still recorded if a later one fails. Existing entries
    for datasets not in `names` are kept.
    """
    paths = {}
    if DATA_PATHS_FILE.exists():
        paths = json.loads(DATA_PATHS_FILE.read_text())

    for name in names:
        paths[name] = kagglehub.dataset_download(DATASETS[name])
        print(name, "->", paths[name])
        DATA_PATHS_FILE.write_text(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or list(DATASETS))
