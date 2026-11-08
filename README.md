# RealTimeSpike

Standalone Python research detector for **spike-like ECoG candidates** in
completed or growing 10 kHz BESA recordings. It can also read 1 kHz NPZ files
for synthetic testing. No MATLAB, BCI2000, GPU, or cloud service is required.

## Run

On Windows, install Python 3.12, then from this repository:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m realtimespike
```

The GUI chooses a BESA/NPZ file, results folder, start, duration, and saved-file
or growing-file mode. Double-click `Start-RealTimeSpike.cmd` after installation.
For command-line analysis of the first five minutes:

```powershell
python -m realtimespike --input 'D:\data\recording.besa' --output-dir results
```

For a recording that is being written, use `--follow`. The detector waits for
committed BESA blocks, processes each five-second span after 0.5 seconds of
right-hand context becomes available, and appends events to a CSV. Press
Ctrl+C to stop the CLI, or **Stop** in the GUI. A finite follow duration can
be set with `--duration 300`; otherwise it continues until stopped.

## Detector and output

The BESA signal is antialias resampled to 1 kHz with a fixed 301-tap Kaiser
FIR, then each channel is bandpassed at 10–60 Hz (fourth-order zero-phase
Butterworth). The Hilbert envelope is screened at median + six times the
robust standard deviation (`1.4826 × MAD`). Peaks also need a prominence of
three robust SD, a 200 ms separation, and a 10–200 ms half-height width.
Five-second processing spans use 0.5 seconds of overlap for filtering; only
peaks in the span's core are recorded, so an event at a span boundary is not
written twice. Default channels exclude `DC01`-style auxiliary inputs; use
`--include-dc` to include them.

Each run writes `spike-*.csv` with original channel number and label,
start/peak/end time in seconds, envelope score, and channel threshold. Its
JSON companion records the settings, selected channels, progress, and elapsed
time. A result is a **candidate**, not an expert-confirmed interictal
epileptiform discharge. Review raw waveforms and artifacts before research
interpretation. Detection sensitivity and false-positive rate have not yet
been measured against expert annotations.

The adaptive envelope approach is inspired by
[Janča et al. (2015)](https://pubmed.ncbi.nlm.nih.gov/24970691/), but this
implementation uses a robust MAD threshold rather than their fitted
log-normal model. It should not be described as the Janča detector.

## Test

```powershell
python -m unittest discover -s tests -p 'test_*.py'
```

No patient recording is distributed in this repository. The BESA reader is
derived from [UDA-Neuro/RealTimeMI](https://github.com/UDA-Neuro/RealTimeMI)
under the MIT license; see [PROVENANCE.md](PROVENANCE.md).

