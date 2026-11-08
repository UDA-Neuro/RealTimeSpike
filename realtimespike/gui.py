"""Small desktop monitor for offline or growing-file detections."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .pipeline import NAME, analyze_recording


class DetectorWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(f"{NAME} — Research candidate detector")
        root.geometry("980x650")
        self.source = tk.StringVar()
        self.output = tk.StringVar(value=str((Path.cwd() / "results").resolve()))
        self.mode = tk.StringVar(value="offline")
        self.start_s = tk.StringVar(value="0")
        self.duration_s = tk.StringVar(value="300")
        self.include_dc = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Choose a BESA recording or 1 kHz NPZ demo.")
        self.events: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.busy = False

        frame = ttk.Frame(root, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Recording (.besa or .npz)").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.source).grid(row=1, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(frame, text="Browse…", command=self.pick_source).grid(row=1, column=1)
        ttk.Label(frame, text="Results folder").grid(row=2, column=0, sticky="w", pady=(12, 0))
        ttk.Entry(frame, textvariable=self.output).grid(row=3, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(frame, text="Browse…", command=self.pick_output).grid(row=3, column=1)
        controls = ttk.Frame(frame)
        controls.grid(row=4, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Radiobutton(controls, text="Analyze saved file", value="offline", variable=self.mode,
                        command=self.mode_changed).pack(side="left")
        ttk.Radiobutton(controls, text="Follow growing BESA", value="follow", variable=self.mode,
                        command=self.mode_changed).pack(side="left", padx=(12, 0))
        ttk.Label(controls, text="Start (s)").pack(side="left", padx=(16, 4))
        ttk.Entry(controls, textvariable=self.start_s, width=7).pack(side="left")
        ttk.Label(controls, text="Duration (s; blank = unlimited in follow)").pack(side="left", padx=(12, 4))
        ttk.Entry(controls, textvariable=self.duration_s, width=7).pack(side="left")
        ttk.Checkbutton(controls, text="Include DC", variable=self.include_dc).pack(side="left", padx=(12, 0))
        self.start_button = ttk.Button(controls, text="Start", command=self.start)
        self.start_button.pack(side="left", padx=(14, 4))
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left")
        ttk.Label(frame, textvariable=self.status, wraplength=940).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.table = ttk.Treeview(frame, columns=("channel", "start", "peak", "end", "score", "threshold"),
                                  show="headings", height=18)
        for key, title in (("channel", "Electrode"), ("start", "Start (s)"),
                           ("peak", "Peak (s)"), ("end", "End (s)"),
                           ("score", "Score"), ("threshold", "Threshold")):
            self.table.heading(key, text=title)
            self.table.column(key, width=130, anchor="center")
        self.table.grid(row=6, column=0, columnspan=2, sticky="nsew")
        ttk.Label(frame, text="Research candidates only. Review waveforms and artifacts before interpreting events.").grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(6, weight=1)
        root.after(100, self.poll)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def pick_source(self):
        path = filedialog.askopenfilename(filetypes=[("Recordings", "*.besa *.npz"), ("All files", "*")])
        if path:
            self.source.set(path)

    def pick_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output.set(path)

    def mode_changed(self):
        self.duration_s.set("" if self.mode.get() == "follow" else "300")

    def start(self):
        if self.busy:
            return
        try:
            source = self.source.get()
            output = self.output.get()
            start = float(self.start_s.get())
            duration = float(self.duration_s.get()) if self.duration_s.get().strip() else None
            follow = self.mode.get() == "follow"
            if not source or not output or start < 0 or (duration is not None and duration <= 0):
                raise ValueError("Choose a recording and valid output, start, and duration")
            if not follow and duration is None:
                raise ValueError("Saved-file analysis requires a duration")
            if follow and Path(source).suffix.lower() != ".besa":
                raise ValueError("Follow mode requires a BESA file")
            include_dc = self.include_dc.get()
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc))
            return
        self.stop_event = threading.Event()
        self.busy = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        for row in self.table.get_children():
            self.table.delete(row)
        self.status.set("Starting detection…")

        def worker():
            try:
                result = analyze_recording(source, output, start_s=start, duration_s=duration,
                    follow=follow, include_dc=include_dc, stop_event=self.stop_event,
                    progress=lambda message: self.events.put(("status", message)),
                    on_events=lambda events: self.events.put(("events", events)))
                self.events.put(("done", result))
            except Exception as exc:
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def stop(self):
        self.stop_event.set()
        self.status.set("Stopping after the current chunk…")

    def close(self):
        self.stop_event.set()
        self.root.destroy()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self.status.set(value)
                elif kind == "events":
                    for event in value:
                        self.table.insert("", "end", values=(event["label"],
                            f'{event["start_s"]:.3f}', f'{event["peak_s"]:.3f}',
                            f'{event["end_s"]:.3f}', f'{event["score"]:.3g}',
                            f'{event["threshold"]:.3g}'))
                    while len(self.table.get_children()) > 500:
                        self.table.delete(self.table.get_children()[0])
                elif kind in ("done", "error"):
                    self.busy = False
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    if kind == "done":
                        self.status.set(f'{value["events"]} candidates; CSV: {value["csv_path"]}')
                    else:
                        self.status.set(value)
                        messagebox.showerror("Detection failed", value)
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self.root.after(100, self.poll)


def launch():
    root = tk.Tk()
    DetectorWindow(root)
    root.mainloop()
