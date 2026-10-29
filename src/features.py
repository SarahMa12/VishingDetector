"""Feature functions.

Each function takes one clip (a float32 array from data.load_clip) and
returns an array whose size depends only on the clip length, so clips can
be stacked together. Frame-based features have one column per frame.

FEATURES at the bottom lists every feature with the type it is saved as.
Large spectrogram-style features are saved as float16 to halve their size.
"""

import numpy as np
import librosa
import torch
import torchaudio
import parselmouth
from parselmouth.praat import call

from config import HOP_LENGTH, N_LFCC, N_MELS, N_MFCC, SAMPLE_RATE

N_FFT = 2048

_lfcc = torchaudio.transforms.LFCC(
    sample_rate=SAMPLE_RATE,
    n_lfcc=N_LFCC,
    speckwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH},
)


def mfcc(y):
    """MFCC plus its first and second change over time. Shape (3 * N_MFCC, frames)."""
    m = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    return np.vstack([m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)])


def lfcc(y):
    """Like MFCC but with evenly spaced frequency bands, which keeps more high-frequency detail."""
    return _lfcc(torch.from_numpy(y)).numpy()


def logmel(y):
    """Log-mel spectrogram. Shape (N_MELS, frames)."""
    S = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return librosa.power_to_db(S)


def cqt(y):
    """Constant-Q spectrogram (84 bins), which shows harmonic structure."""
    C = librosa.cqt(y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_bins=84)
    return librosa.amplitude_to_db(np.abs(C))


def spectral(y):
    """Centroid, flatness, rolloff and flux per frame. Shape (4, frames).

    Flux is how much the spectrum changes from the previous frame
    (0 for the first frame).
    """
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    centroid = librosa.feature.spectral_centroid(S=S, sr=SAMPLE_RATE)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=SAMPLE_RATE)[0]
    flux = np.r_[0.0, np.sqrt((np.diff(S, axis=1) ** 2).sum(axis=0))]
    return np.vstack([centroid, flatness, rolloff, flux])


def pitch(y):
    """Pitch in Hz (NaN where there is no pitch) and voiced probability. Shape (2, frames)."""
    f0, _, voiced_prob = librosa.pyin(
        y, fmin=60, fmax=500, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    )
    return np.vstack([f0, voiced_prob])


def jitter_shimmer(y):
    """Local jitter and shimmer for the whole clip (NaN if no pitch pulses are found)."""
    try:
        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=SAMPLE_RATE)
        pulses = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except parselmouth.PraatError:
        jitter = shimmer = np.nan
    return np.array([jitter, shimmer])


def noise(y):
    """Mean and spread (in dB) of the quietest 20% of frames, a clue to the recording noise."""
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    quiet = np.sort(rms)[: max(1, len(rms) // 5)]
    db = 20 * np.log10(quiet + 1e-10)
    return np.array([db.mean(), db.std()])


# name -> (function, type to save as)
FEATURES = {
    "mfcc": (mfcc, np.float32),
    "lfcc": (lfcc, np.float32),
    "logmel": (logmel, np.float16),
    "cqt": (cqt, np.float16),
    "spectral": (spectral, np.float32),
    "pitch": (pitch, np.float32),
    "jitter_shimmer": (jitter_shimmer, np.float32),
    "noise": (noise, np.float32),
}
