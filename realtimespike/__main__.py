"""Command line and GUI entry point."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Research spike-candidate detector")
    parser.add_argument("--input", help="10 kHz BESA or 1 kHz NPZ recording")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=None,
                        help="Seconds; defaults to 300 offline, unlimited when following")
    parser.add_argument("--follow", action="store_true", help="Watch committed BESA data until interrupted")
    parser.add_argument("--include-dc", action="store_true")
    args = parser.parse_args()
    if not args.input:
        from .gui import launch
        launch()
        return
    from .pipeline import analyze_recording
    try:
        result = analyze_recording(args.input, args.output_dir, start_s=args.start,
            duration_s=args.duration if args.duration is not None else (None if args.follow else 300),
            follow=args.follow, include_dc=args.include_dc, progress=print)
    except KeyboardInterrupt:
        print("Stopped by user")
        return
    print(json.dumps({"events": result["events"], "csv": result["csv_path"],
                      "metadata": result["metadata_path"], "elapsed_s": result["elapsed_s"]}))


if __name__ == "__main__":
    main()
