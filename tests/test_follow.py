"""A follower waits for right-hand context from a newly committed BESA block."""

import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from realtimespike.pipeline import analyze_recording


def tag(name, payload):
    return struct.pack("<4sI", name, len(payload)) + payload


def block(samples, value):
    channels = b"".join(b"\x00" + np.full(samples, value + i, dtype="<i2").tobytes()
                        for i in range(2))
    return tag(b"BDAT", tag(b"DATT", struct.pack("<I", 0x11)) +
               tag(b"DATS", struct.pack("<I", samples)) + tag(b"DATA", channels))


def commit(path, offsets, counts):
    with path.open("r+b") as stream:
        stream.seek(0, 2)
        fmi = stream.tell()
        stream.write(tag(b"BFMI", b"\x00" * 8 +
                         tag(b"SAMP", struct.pack("<d", 10000.0)) +
                         tag(b"SAMT", struct.pack("<Q", sum(counts)))))
        table = stream.tell()
        entries = b"".join(tag(b"TAGE", struct.pack("<4sQI", b"BDAT", offset, count))
                           for offset, count in zip(offsets, counts))
        stream.write(tag(b"BTAG", b"\x00" * 8 + entries))
        stream.seek(0)
        stream.write(tag(b"BCF1", tag(b"OFFM", struct.pack("<Q", fmi)) +
                         tag(b"OFBI", struct.pack("<Q", 64)) +
                         tag(b"OFTL", struct.pack("<Q", table))))


class FollowTest(unittest.TestCase):
    def test_waits_for_next_committed_block(self):
        scratch = Path(__file__).resolve().parents[1] / "build"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory) / "growing.besa"
            labels = tag(b"CHNR", struct.pack("<H", 2))
            for index, name in enumerate(("A1", "A2")):
                labels += tag(b"CHLA", struct.pack("<H", index) + name.encode("utf-16le"))
            with path.open("wb") as stream:
                stream.write(b"\x00" * 64)
                stream.write(tag(b"BCAL", b"\x00" * 8 + labels))
                first = stream.tell()
                stream.write(block(50000, 100))
            commit(path, [first], [50000])
            outcome = {}
            def work():
                try:
                    outcome["result"] = analyze_recording(path, Path(directory) / "out",
                        duration_s=5, follow=True)
                except Exception as exc:
                    outcome["error"] = exc
            thread = threading.Thread(target=work)
            thread.start()
            time.sleep(0.6)
            self.assertTrue(thread.is_alive())
            with path.open("ab") as stream:
                second = stream.tell()
                stream.write(block(10000, 200))
            commit(path, [first, second], [50000, 10000])
            thread.join(15)
            self.assertFalse(thread.is_alive())
            self.assertNotIn("error", outcome)
            self.assertEqual(outcome["result"]["processed_until_s"], 5)


if __name__ == "__main__":
    unittest.main()
