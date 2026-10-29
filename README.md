# VishingDetector

Detecting AI-generated voices and voice phishing (vishing) calls, using the ASVspoof 2019 and 2021 datasets.

Each clip is real ("bonafide") or fake ("spoof"). We train and tune on 2019 LA (Logical Access) and test on 2021 LA and DF (Deepfake). 2019 eval and all 2021 data stay untouched until the final test.

## Repo Structure

```
VishingDetector/
  data/                   downloaded audio and labels (ignored by git)
  features/               saved features (ignored by git)
  notebooks/
    01_eda.ipynb          exploring the 2019 LA train and dev data
    02_feature_checks.ipynb  comparing real and fake on the extracted features
  src/
    config.py             paths and audio settings
    data.py               reads label files, loads and trims clips
    download_data.py      downloads the datasets from Kaggle into data/
    features.py           feature functions (MFCC, LFCC, log-mel, CQT, pitch, ...)
    extract.py            batched, restartable feature extraction into features/
  data_paths.json         where each dataset was saved (ignored by git)
  requirements.txt        pinned Python dependencies
```

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
```

The download is about 87 GB. Keep at least 100 GB free.
