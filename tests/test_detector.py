"""Synthetic transient and end-to-end NPZ output checks."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from realtimespike.detector import detect
from realtimespike.pipeline import analyze_recording


class SpikeDetectorTest(unittest.TestCase):
    def make_signal(self):
        fs = 1000
        t = np.arange(10 * fs) / fs
        data = np.random.default_rng(12).normal(0, 0.3, (len(t), 3)).astype(np.float32)
        data[:, 1] += (8 * np.exp(-0.5 * ((t - 5.0) / 0.02)**2) *
                       np.sin(2 * np.pi * 25 * (t - 5.0)))
        return data

    def test_spike_is_found_near_chunk_boundary(self):
        data = self.make_signal()
        events = detect(data, 1000, 0, ["DC01", "A1", "A2"])
        near = [event for event in events if event["label"] == "A1" and
                abs(event["peak_s"] - 5) < 0.03]
        self.assertEqual(len(near), 1)

    def test_pipeline_writes_labeled_event_once(self):
        scratch = Path(__file__).resolve().parents[1] / "build"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory) / "demo.npz"
            np.savez(path, data=self.make_signal(), sample_rate_hz=1000,
                     channel_labels=np.array(["DC01", "A1", "A2"]))
            result = analyze_recording(path, Path(directory) / "results", duration_s=10)
            with Path(result["csv_path"]).open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            near = [row for row in rows if row["label"] == "A1" and
                    abs(float(row["peak_s"]) - 5) < 0.03]
            self.assertEqual(len(near), 1)
            self.assertEqual(near[0]["channel_number"], "2")
            self.assertEqual(result["processed_until_s"], 10)


if __name__ == "__main__":
    unittest.main()
