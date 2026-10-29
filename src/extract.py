"""Extract features from 2019 LA clips and save them to ./features.

Usage (from the project root):
    python src/extract.py --split train --feature all
    python src/extract.py --split dev --feature mfcc --seconds 3
    python src/extract.py --split train --limit 100      # quick test on 100 random clips

Output: features/<seconds>s/<split>/<feature>/batch_00000.npz, ...
Each file holds `x` (the features, one row per clip) and `files` (clip names).
Rows follow the order of the label file, BATCH_SIZE clips per file.

Safe to stop and re-run: batches that are already saved are skipped.
Only train and dev can be extracted here. 2019 eval and 2021 are kept for the final test.
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import torch
from tqdm import tqdm

from config import CLIP_SECONDS, FEATURES_DIR
from data import load_clip, load_la_2019
from features import FEATURES

BATCH_SIZE = 500


def batch_file(out_dir, feature, batch_idx):
    return out_dir / feature / f"batch_{batch_idx:05d}.npz"


def init_worker():
    # One thread per worker; the parallelism comes from the worker processes.
    torch.set_num_threads(1)


def process_batch(job):
    """Load each clip in the batch once and save every feature that is not saved yet."""
    out_dir, seconds, batch_idx, files, paths, names = job

    todo = [n for n in names if not batch_file(out_dir, n, batch_idx).exists()]
    if not todo:
        return

    results = {n: [] for n in todo}
    for path in paths:
        y = load_clip(path, seconds)
        for n in todo:
            results[n].append(FEATURES[n][0](y))

    for n in todo:
        final = batch_file(out_dir, n, batch_idx)
        final.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary name first, so an interrupted run never leaves a half-written batch.
        tmp = final.with_name(final.stem + ".tmp.npz")
        np.savez(tmp, x=np.stack(results[n]).astype(FEATURES[n][1]), files=np.array(files))
        tmp.replace(final)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=["train", "dev"], required=True)
    parser.add_argument("--feature", default="all", choices=["all", *FEATURES])
    parser.add_argument("--seconds", type=float, default=CLIP_SECONDS,
                        help="clip length after trimming (default: CLIP_SECONDS in config.py)")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument("--limit", type=int, help="only use this many random clips (for testing)")
    args = parser.parse_args()

    df = load_la_2019(args.split)
    split_name = args.split
    if args.limit:
        df = df.sample(args.limit, random_state=0).reset_index(drop=True)
        split_name += f"_n{args.limit}"

    out_dir = FEATURES_DIR / f"{args.seconds:g}s" / split_name
    names = list(FEATURES) if args.feature == "all" else [args.feature]

    jobs = []
    for i in range(0, len(df), BATCH_SIZE):
        chunk = df.iloc[i : i + BATCH_SIZE]
        jobs.append((out_dir, args.seconds, i // BATCH_SIZE,
                     chunk["file"].tolist(), chunk["path"].tolist(), names))

    print(f"{len(df):,} clips, {len(jobs)} batches, features: {', '.join(names)}")
    print(f"saving to {out_dir}")

    start = time.perf_counter()
    with Pool(args.workers, initializer=init_worker) as pool:
        for _ in tqdm(pool.imap_unordered(process_batch, jobs), total=len(jobs), unit="batch"):
            pass
    print(f"done in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
