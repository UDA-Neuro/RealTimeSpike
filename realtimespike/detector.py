"""Reproducible research spike-candidate detector for intracranial EEG."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt


def detect(data: np.ndarray, fs: int, start_s: float, labels: list[str]) -> list[dict]:
    """Detect 10–60 Hz envelope transients using a robust channel threshold.

    This is an envelope detector inspired by Janca et al., not an implementation
    of their fitted log-normal method or a clinical IED classification.
    """
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] != len(labels) or fs < 200:
        raise ValueError("Expected samples by channels, matching labels, fs >= 200 Hz")
    if len(data) < fs:
        return []
    sos = butter(4, (10, 60), btype="bandpass", fs=fs, output="sos")
    events = []
    for channel, label in enumerate(labels):
        raw = data[:, channel].astype(np.float64)
        if not np.isfinite(raw).all() or np.std(raw) == 0:
            continue
        filtered = sosfiltfilt(sos, raw)
        envelope = np.abs(hilbert(filtered))
        median = float(np.median(envelope))
        sigma = float(1.4826 * np.median(np.abs(envelope - median)))
        if sigma <= 0:
            continue
        threshold = median + 6 * sigma
        peaks, properties = find_peaks(envelope, height=threshold,
                                        prominence=3 * sigma,
                                        distance=round(0.2 * fs))
        for peak in peaks:
            half = median + 0.5 * (envelope[peak] - median)
            left = int(peak)
            right = int(peak)
            while left > 0 and envelope[left - 1] >= half:
                left -= 1
            while right + 1 < len(envelope) and envelope[right + 1] >= half:
                right += 1
            width = (right - left + 1) / fs
            if not 0.01 <= width <= 0.2:
                continue
            events.append({"channel_number": channel + 1, "label": label,
                           "start_s": start_s + left / fs,
                           "peak_s": start_s + int(peak) / fs,
                           "end_s": start_s + (right + 1) / fs,
                           "score": float(envelope[peak]), "threshold": threshold})
    return sorted(events, key=lambda event: (event["peak_s"], event["channel_number"]))
