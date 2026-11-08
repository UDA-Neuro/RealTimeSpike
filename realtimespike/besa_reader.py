"""Read a bounded segment of a BESA BCF1 file.

Implements the compressed integer BDAT variants documented in the BESA File
Format Specification. No patient metadata is copied into the output.
"""

from __future__ import annotations

import ctypes
import os
import struct
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_native = None


def _native_decoder():
    global _native
    path = os.environ.get("BESA_PACKED_DLL")
    if not path:
        return None
    if _native is None:
        library = ctypes.CDLL(path)
        function = library.besa_decode_packed
        function.argtypes = (ctypes.c_char_p, ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int, ctypes.c_int)
        function.restype = ctypes.c_int
        _native = function
    return _native


def _tagged(buf: bytes, start: int = 0):
    pos = start
    while pos + 8 <= len(buf):
        tag, size = struct.unpack_from("<4sI", buf, pos)
        end = pos + 8 + size
        if end > len(buf):
            raise ValueError("Malformed BESA tag length")
        yield tag, buf[pos + 8 : end]
        pos = end
    if pos != len(buf):
        raise ValueError("Trailing bytes in BESA tag block")


def _block(file, offset: int, expected: bytes):
    file.seek(offset)
    tag, size = struct.unpack("<4sI", file.read(8))
    if tag != expected or size > 100_000_000:
        raise ValueError(f"Expected {expected!r} at {offset}, got {tag!r}")
    return file.read(size)


def metadata(file):
    header = dict(_tagged(_block(file, 0, b"BCF1")))
    fmi = dict(_tagged(_block(file, struct.unpack("<Q", header[b"OFFM"])[0], b"BFMI"), 8))
    cal = dict(_tagged(_block(file, struct.unpack("<Q", header[b"OFBI"])[0], b"BCAL"), 8))
    rate = struct.unpack("<d", fmi[b"SAMP"])[0]
    samples = struct.unpack("<Q", fmi[b"SAMT"])[0]
    channels = int.from_bytes(cal[b"CHNR"], "little")
    table = _block(file, struct.unpack("<Q", header[b"OFTL"])[0], b"BTAG")
    blocks = []
    for tag, value in _tagged(table, 8):
        if tag == b"TAGE":
            kind, offset, count = struct.unpack("<4sQI", value)
            if kind == b"BDAT":
                blocks.append((offset, count))
    if sum(count for _, count in blocks) != samples:
        raise ValueError("BESA data block sample count mismatch")
    return rate, samples, channels, blocks


def channel_labels(file) -> list[str]:
    """Return channel labels in data order from the BESA BCAL/CHLA tags."""
    header = dict(_tagged(_block(file, 0, b"BCF1")))
    cal = _block(file, struct.unpack("<Q", header[b"OFBI"])[0], b"BCAL")
    fields = list(_tagged(cal, 8))
    channel_count = int.from_bytes(next(value for tag, value in fields if tag == b"CHNR"), "little")
    labels: list[str | None] = [None] * channel_count
    for tag, value in fields:
        if tag != b"CHLA":
            continue
        if len(value) < 4 or len(value) % 2:
            raise ValueError("Malformed BESA channel label")
        index = struct.unpack_from("<H", value)[0]
        if index >= channel_count or labels[index] is not None:
            raise ValueError("Duplicate or out-of-range BESA channel label index")
        labels[index] = value[2:].decode("utf-16le").rstrip("\x00")
        if not labels[index]:
            raise ValueError("Empty BESA channel label")
    if any(label is None for label in labels):
        raise ValueError("Missing BESA channel labels")
    return labels


def _decode_packed(payload: bytes, scheme: int, n: int, wide_first: bool) -> np.ndarray:
    out = np.empty(n, dtype=np.int32)
    native = _native_decoder()
    if native is not None:
        status = native(payload, len(payload), out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                        n, scheme, int(wide_first))
        if status != 0:
            raise ValueError(f"Native BESA packed decoder failed with status {status}")
        return out
    first_fmt = "<ii" if wide_first else "<hh"
    out[:2] = struct.unpack_from(first_fmt, payload)
    pos = struct.calcsize(first_fmt)
    k = 2
    view = memoryview(payload)
    while k < n:
        code = view[pos]
        pos += 1
        if scheme == 1 and code <= 224:
            values = (code // 15 - 7, code % 15 - 7)
        elif scheme == 2 and code <= 124:
            values = (code // 25 - 2, code // 5 % 5 - 2, code % 5 - 2)
        elif scheme == 2 and code <= 245:
            x = code - 125
            values = (x // 11 - 5, x % 11 - 5)
        elif scheme == 3 and code <= 80:
            values = (code // 27 - 1, code // 9 % 3 - 1, code // 3 % 3 - 1, code % 3 - 1)
        elif scheme == 3 and code <= 249:
            x = code - 81
            values = (x // 13 - 6, x % 13 - 6)
        else:
            if scheme == 1 and 248 <= code <= 254:
                count, width = 255 - code, 1
            elif scheme == 1 and 242 <= code <= 247:
                count, width = 248 - code, 2
            elif scheme == 1 and 236 <= code <= 241:
                count, width = 242 - code, 4
            elif scheme == 2 and 250 <= code <= 254:
                count, width = 255 - code, 1
            elif scheme == 2 and 246 <= code <= 249:
                count, width = 250 - code, 2
            elif scheme == 3 and 252 <= code <= 254:
                count, width = 255 - code, 1
            elif scheme == 3 and 250 <= code <= 251:
                count, width = 252 - code, 2
            else:
                raise ValueError(f"Unknown BESA packed code {code} in scheme {scheme}")
            fmt = {1: "b", 2: "h", 4: "i"}[width]
            values = struct.unpack_from("<" + fmt * count, view, pos)
            pos += count * width
        if k + len(values) > n:
            raise ValueError("Packed channel contains too many samples")
        out[k : k + len(values)] = values
        k += len(values)
    if pos != len(payload):
        raise ValueError("Packed channel has trailing bytes")
    return out


def _packed_length(payload: bytes, scheme: int, n: int, wide_first: bool) -> int:
    """Find the next channel boundary in an uncompressed packed BDAT payload."""
    pos = 8 if wide_first else 4
    if n < 2 or len(payload) < pos:
        raise ValueError("Truncated packed BESA channel")
    decoded = 2
    while decoded < n:
        if pos >= len(payload):
            raise ValueError("Truncated packed BESA channel")
        code = payload[pos]
        pos += 1
        if scheme == 1 and code <= 224:
            count, width = 2, 0
        elif scheme == 2 and code <= 124:
            count, width = 3, 0
        elif scheme == 2 and code <= 245:
            count, width = 2, 0
        elif scheme == 3 and code <= 80:
            count, width = 4, 0
        elif scheme == 3 and code <= 249:
            count, width = 2, 0
        elif scheme == 1 and 248 <= code <= 254:
            count, width = 255 - code, 1
        elif scheme == 1 and 242 <= code <= 247:
            count, width = 248 - code, 2
        elif scheme == 1 and 236 <= code <= 241:
            count, width = 242 - code, 4
        elif scheme == 2 and 250 <= code <= 254:
            count, width = 255 - code, 1
        elif scheme == 2 and 246 <= code <= 249:
            count, width = 250 - code, 2
        elif scheme == 3 and 252 <= code <= 254:
            count, width = 255 - code, 1
        elif scheme == 3 and 250 <= code <= 251:
            count, width = 252 - code, 2
        else:
            raise ValueError(f"Unknown BESA packed code {code} in scheme {scheme}")
        if decoded + count > n or pos + count * width > len(payload):
            raise ValueError("Malformed packed BESA channel length")
        decoded += count
        pos += count * width
    return pos


def _decode_channel(prefix: int, payload: bytes, n: int) -> np.ndarray:
    mode = prefix
    if prefix in (9, 13, 14, 15, 17, 18, 19, 29):
        payload = zlib.decompress(payload)
        mode = {9: 0, 13: 3, 14: 4, 15: 5, 17: 7, 18: 10, 19: 11, 29: 8}[prefix]
    if mode in (0, 6, 8):
        if mode == 0:
            raw = np.frombuffer(payload, dtype="<i2")
        elif mode == 8:
            raw = np.frombuffer(payload, dtype="<i4")
        else:
            raw = np.concatenate((np.frombuffer(payload[:8], dtype="<i4"), np.frombuffer(payload[8:], dtype="<i2")))
        if raw.size != n:
            raise ValueError("Unpacked channel sample count mismatch")
        if prefix != 9:
            return raw.astype(np.float32)
        dd = raw.astype(np.int64)
    elif mode in (3, 4, 5, 7, 10, 11):
        scheme = {3: 1, 4: 2, 5: 3, 7: 1, 10: 2, 11: 3}[mode]
        dd = _decode_packed(payload, scheme, n, mode in (7, 10, 11)).astype(np.int64)
    else:
        raise ValueError(f"Unsupported BESA compression prefix {prefix}")
    if n > 1:
        dd[1] -= dd[0]
    return np.cumsum(np.cumsum(dd, dtype=np.int64), dtype=np.int64).astype(np.float32)


def read_segment(path: str | Path, start_seconds: float = 0, duration_seconds: float = 300,
                 channel_indices: tuple[int, ...] | None = None):
    """Return samples x selected channels float32 counts and source sampling rate."""
    with open(path, "rb") as file:
        rate, total, channels, blocks = metadata(file)
        start = round(start_seconds * rate)
        stop = start + round(duration_seconds * rate)
        if start < 0 or stop > total or stop <= start:
            raise ValueError("Requested segment is outside the recording")
        selected = tuple(range(channels)) if channel_indices is None else tuple(channel_indices)
        if not selected or len(set(selected)) != len(selected) or any(i < 0 or i >= channels for i in selected):
            raise ValueError("Invalid BESA channel selection")
        output_index = {channel: column for column, channel in enumerate(selected)}
        result = np.empty((stop - start, len(selected)), dtype=np.float32)
        first_sample = 0
        for offset, count in blocks:
            block_end = first_sample + count
            if block_end <= start:
                first_sample = block_end
                continue
            if first_sample >= stop:
                break
            block = _block(file, offset, b"BDAT")
            fields = dict(_tagged(block))
            kind = struct.unpack("<I", fields[b"DATT"])[0]
            block_count = struct.unpack("<I", fields[b"DATS"])[0]
            if kind != 0x11 or block_count != count:
                raise ValueError(f"Unsupported BESA data block type {kind:#x}")
            payload = fields[b"DATA"]
            pos = 0
            left = max(start, first_sample) - first_sample
            right = min(stop, block_end) - first_sample
            dest = max(start, first_sample) - start
            for channel in range(channels):
                prefix = payload[pos]
                if prefix in (9, 13, 14, 15, 17, 18, 19, 29):
                    length = struct.unpack_from("<I", payload, pos + 1)[0]
                    encoded = payload[pos + 5 : pos + 5 + length]
                    advance = 5 + length
                elif prefix in (0, 6, 8):
                    length = {0: 2 * count, 6: 8 + 2 * (count - 2), 8: 4 * count}[prefix]
                    encoded = payload[pos + 1 : pos + 1 + length]
                    advance = 1 + length
                elif prefix in (3, 4, 5, 7, 10, 11):
                    length = _packed_length(payload[pos + 1 :],
                                            {3: 1, 4: 2, 5: 3, 7: 1, 10: 2, 11: 3}[prefix],
                                            count, prefix in (7, 10, 11))
                    encoded = payload[pos + 1 : pos + 1 + length]
                    advance = 1 + length
                else:
                    raise ValueError(f"Unsupported channel prefix {prefix} at channel {channel}, byte {pos}")
                if len(encoded) != length:
                    raise ValueError("Truncated channel")
                if channel in output_index:
                    result[dest : dest + right - left, output_index[channel]] = (
                        _decode_channel(prefix, encoded, count)[left:right])
                pos += advance
            if pos != len(payload):
                raise ValueError("Data block has trailing bytes")
            first_sample = block_end
    return result, rate


def _read_block_job(job):
    path, first, count, rate = job
    values, _ = read_segment(path, first / rate, count / rate)
    return values


def read_segment_parallel(path: str | Path, start_seconds: float = 0,
                          duration_seconds: float = 300, workers: int = 8):
    """Read blocks in separate processes and return samples x channels."""
    with open(path, "rb") as file:
        rate, total, channels, blocks = metadata(file)
    start = round(start_seconds * rate)
    stop = start + round(duration_seconds * rate)
    if start < 0 or stop > total or stop <= start:
        raise ValueError("Requested segment is outside the recording")
    selected = []
    first = 0
    for _, count in blocks:
        end = first + count
        if end > start and first < stop:
            selected.append((str(path), first, count, rate))
        first = end
    result = np.empty((stop - start, channels), dtype=np.float32)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (path_str, first, count, block_rate), values in zip(selected, pool.map(_read_block_job, selected)):
            left = max(start, first)
            right = min(stop, first + count)
            result[left - start:right - start] = values[left - first:right - first]
    return result, rate


def _resample_block_job(job):
    from scipy.signal import resample_poly

    path, first, count, rate, taps, factor = job
    values, _ = read_segment(path, first / rate, count / rate)
    down = resample_poly(values, 1, factor, axis=0, window=taps).astype(np.float32)
    edge = len(taps) - 1
    return down, values[:edge].copy(), values[-edge:].copy()


def read_resampled_parallel(path: str | Path, start_seconds: float,
                            duration_seconds: float, taps: np.ndarray,
                            workers: int = 8, factor: int = 10):
    """Decode and decimate independent BESA blocks, correcting FIR boundaries.

    The requested segment must align with BESA block boundaries. This matches
    the first five minutes of the test recording (5-second blocks). The FIR
    support is reconstructed around each block join, giving the same result as
    one whole-record scipy.signal.resample_poly call.
    """
    from scipy.signal import resample_poly

    with open(path, "rb") as file:
        rate, total, channels, blocks = metadata(file)
    start = round(start_seconds * rate)
    stop = start + round(duration_seconds * rate)
    if start < 0 or stop > total or stop <= start:
        raise ValueError("Requested segment is outside the recording")
    selected = []
    first = 0
    for _, count in blocks:
        if first >= start and first + count <= stop:
            selected.append((str(path), first, count, rate, taps, factor))
        elif first < stop and first + count > start:
            raise ValueError("Requested segment must align to BESA block boundaries")
        first += count
    if any(count % factor for _, _, count, _, _, _ in selected):
        raise ValueError("BESA blocks must be divisible by downsampling factor")
    result = np.empty(((stop - start) // factor, channels), dtype=np.float32)
    edge = len(taps) - 1
    radius_out = (len(taps) - 1) // (2 * factor)
    previous_tail = None
    out_pos = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for job, (down, head, tail) in zip(selected, pool.map(_resample_block_job, selected)):
            n = down.shape[0]
            result[out_pos:out_pos+n] = down
            if previous_tail is not None:
                joined = np.concatenate((previous_tail, head), axis=0)
                center = resample_poly(joined, 1, factor, axis=0, window=taps)
                mid = edge // factor
                result[out_pos-radius_out:out_pos+radius_out] = center[mid-radius_out:mid+radius_out]
            previous_tail = tail
            out_pos += n
    if out_pos != len(result):
        raise ValueError("Resampled BESA sample count mismatch")
    return result, rate / factor
