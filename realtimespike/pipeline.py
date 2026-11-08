"""Offline and growing-file BESA detection with incremental audit output."""

from __future__ import annotations

import csv
import json
import re
import struct
import threading
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.signal import firwin, resample_poly

from . import __version__
from .besa_reader import channel_labels, metadata, read_segment
from .detector import detect

NAME = "RealTimeSpike"
CHUNK_S = 5.0
CONTEXT_S = 0.5
OUTPUT_RATE = 1000
FIELDS = ("channel_number", "label", "start_s", "peak_s", "end_s", "score", "threshold")


def analyze_recording(source: str | Path, output_dir: str | Path, *,
                      start_s: float = 0, duration_s: float | None = 300,
                      follow: bool = False, include_dc: bool = False,
                      stop_event: threading.Event | None = None,
                      progress=None, on_events=None) -> dict:
    """Analyze a segment, or keep following committed BESA samples until stopped."""
    source = Path(source)
    output_dir = Path(output_dir)
    if not source.is_file() or source.suffix.lower() not in (".besa", ".npz"):
        raise ValueError("Choose an existing .besa or 1 kHz .npz file")
    if follow and source.suffix.lower() != ".besa":
        raise ValueError("Follow mode requires a growing .besa file")
    if start_s < 0 or (duration_s is not None and duration_s <= 0):
        raise ValueError("Start must be nonnegative; duration must be positive")
    if not follow and duration_s is None:
        raise ValueError("Offline mode requires a duration")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    csv_path = output_dir / f"spike-{run_id}.csv"
    json_path = output_dir / f"spike-{run_id}.json"
    stop_event = stop_event or threading.Event()
    progress = progress or (lambda _: None)
    on_events = on_events or (lambda _: None)
    started = time.perf_counter()
    taps = firwin(301, 350, fs=10_000, window=("kaiser", 8.6))
    npz_data = None
    if source.suffix.lower() == ".besa":
        while True:
            try:
                with source.open("rb") as stream:
                    source_rate, total, channel_count, _ = metadata(stream)
                    labels = channel_labels(stream)
                break
            except (OSError, ValueError, KeyError, struct.error, zlib.error) as exc:
                if not follow:
                    raise
                progress(f"Waiting for BESA header: {exc}")
                if stop_event.wait(0.5):
                    raise RuntimeError("Stopped before a complete BESA header was available") from exc
        if source_rate != 10_000:
            raise ValueError(f"BESA sampling rate must be 10 kHz, got {source_rate:g}")
    else:
        with np.load(source, allow_pickle=False) as archive:
            source_rate = int(archive["sample_rate_hz"])
            npz_data = archive["data"].copy()
            if "channel_labels" in archive:
                labels = [str(x) for x in archive["channel_labels"]]
            else:
                labels = [f"channel_{i+1:03d}" for i in range(npz_data.shape[1])]
        if source_rate != OUTPUT_RATE or npz_data.ndim != 2 or len(labels) != npz_data.shape[1]:
            raise ValueError("NPZ needs samples-by-channels data at 1 kHz and matching labels")
        total = len(npz_data)
        channel_count = len(labels)
    selected = [i for i, label in enumerate(labels)
                if include_dc or re.fullmatch(r"DC\d+", label, re.I) is None]
    if not selected:
        raise ValueError("No electrode channels selected")
    stop_at = start_s + duration_s if duration_s is not None else None
    if not follow and total / source_rate < stop_at - 1e-6:
        raise ValueError("Requested segment extends beyond the recording")
    details = {
        "software": NAME, "version": __version__, "source_type": source.suffix.lower(),
        "source_sampling_hz": source_rate, "output_sampling_hz": OUTPUT_RATE,
        "source_channels": channel_count, "channel_labels": labels,
        "selected_channel_numbers": [i + 1 for i in selected],
        "start_s": start_s, "duration_s": duration_s, "follow": follow,
        "chunk_s": CHUNK_S, "context_s": CONTEXT_S,
        "resampling": "10 kHz to 1 kHz scipy.signal.resample_poly, 301-tap 350 Hz Kaiser FIR" if npz_data is None else "none",
        "detector": {"band_hz": [10, 60], "envelope": "Hilbert magnitude",
                     "threshold": "median + 6 * 1.4826 * MAD",
                     "prominence": "3 * robust sigma", "refractory_s": 0.2,
                     "half_height_width_s": [0.01, 0.2]},
        "events": 0, "processed_until_s": start_s, "elapsed_s": None,
    }
    json_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    next_start = start_s
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            stream.flush()
            while not stop_event.is_set():
                if npz_data is None:
                    try:
                        with source.open("rb") as recording:
                            rate, total, count, _ = metadata(recording)
                        if rate != source_rate or count != channel_count:
                            raise ValueError("BESA rate or channel count changed")
                    except (OSError, ValueError, KeyError, struct.error, zlib.error) as exc:
                        if not follow:
                            raise
                        progress(f"Waiting for committed BESA data: {exc}")
                        stop_event.wait(0.5)
                        continue
                available = total / source_rate
                core_end = min(next_start + CHUNK_S, stop_at or float("inf"))
                if follow and available < core_end + CONTEXT_S:
                    if stop_at is not None and next_start >= stop_at - 1e-6:
                        break
                    stop_event.wait(0.5)
                    continue
                if not follow and core_end > available + 1e-6:
                    raise ValueError("Recording ended before requested segment")
                if core_end <= next_start + 1e-9:
                    break
                read_start = max(0, next_start - CONTEXT_S)
                read_end = min(available, core_end + CONTEXT_S)
                if npz_data is None:
                    try:
                        raw, rate = read_segment(source, read_start, read_end - read_start,
                                                 tuple(selected))
                    except (OSError, ValueError, KeyError, IndexError, struct.error, zlib.error) as exc:
                        if not follow:
                            raise
                        progress(f"Waiting for complete BESA block: {exc}")
                        stop_event.wait(0.5)
                        continue
                    data = resample_poly(raw, 1, 10, axis=0, window=taps).astype(np.float32)
                else:
                    data = npz_data[round(read_start * OUTPUT_RATE):round(read_end * OUTPUT_RATE), selected]
                events = detect(data, OUTPUT_RATE, read_start,
                                [labels[i] for i in selected])
                accepted = []
                for event in events:
                    if next_start <= event["peak_s"] < core_end:
                        event["channel_number"] = selected[event["channel_number"] - 1] + 1
                        accepted.append(event)
                for event in accepted:
                    writer.writerow({key: f"{value:.9g}" if isinstance(value, float) else value
                                     for key, value in event.items()})
                stream.flush()
                details["events"] += len(accepted)
                details["processed_until_s"] = core_end
                details["elapsed_s"] = round(time.perf_counter() - started, 3)
                json_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
                progress(f"Processed to {core_end:g} s; {details['events']} candidates")
                if accepted:
                    on_events(accepted)
                next_start = core_end
                if stop_at is not None and next_start >= stop_at - 1e-6:
                    break
    finally:
        details["elapsed_s"] = round(time.perf_counter() - started, 3)
        json_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    details["csv_path"] = str(csv_path)
    details["metadata_path"] = str(json_path)
    return details
