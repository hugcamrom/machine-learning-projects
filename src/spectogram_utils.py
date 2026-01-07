from __future__ import annotations # For Python 3.7-3.9 compatibility

from pathlib import Path           
import numpy as np             
from scipy.io import wavfile   
from scipy.signal import spectrogram     


def load_wav_mono_segment(
    wav_path: Path,
    start_t: float,
    end_t: float,
) -> tuple[int, np.ndarray]:
    """
    Load a WAV file and return a mono, float32 audio segment in [-1, 1].

    Notes:
    - Uses scipy.io.wavfile.read (loads full file into memory).
    - Normalises only the extracted segment to minimise memory usage.
    """
    sr, audio_raw = wavfile.read(wav_path)   

    if audio_raw.ndim > 1:
        audio_raw = audio_raw.mean(axis=1).astype(audio_raw.dtype)

    duration = len(audio_raw) / sr
    start_t = max(0.0, start_t)
    end_t = min(duration, end_t)

    start_i = int(start_t * sr)
    end_i = int(end_t * sr)

    seg = audio_raw[start_i:end_i]

    if np.issubdtype(seg.dtype, np.integer):
        seg = seg.astype(np.float32) / np.iinfo(seg.dtype).max
    else:
        seg = seg.astype(np.float32)

    return sr, seg


def compute_spectrogram_db(
    audio: np.ndarray,
    sr: int,
    nperseg: int = 1024,
    noverlap: int = 768,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute magnitude spectrogram and return in dB scale.

    Returns:
    - freqs (Hz)
    - times (s) relative to the input audio segment
    - Sxx_db (freq_bins x time_bins)
    """
    freqs, times, Sxx = spectrogram(
        audio,
        fs=sr,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="magnitude",
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return freqs, times, Sxx_db


def pad_or_crop_2d(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Pad/crop a 2D array to (target_freq_bins, target_time_bins)."""
    target_f, target_t = target_shape
    arr = arr[:target_f, :target_t]

    pad_f = max(0, target_f - arr.shape[0])
    pad_t = max(0, target_t - arr.shape[1])

    out = np.pad(arr, ((0, pad_f), (0, pad_t)), mode="constant", constant_values=0)
    return out
