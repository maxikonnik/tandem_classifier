"""Launch the accelerometer player: python -m tandem.webtool."""
from __future__ import annotations

import argparse
import os
import threading
import webbrowser

from tandem.recon.ffprobe import has_ffmpeg
from tandem.webtool.server import serve


def main() -> None:
    ap = argparse.ArgumentParser(description="GoPro accelerometer player")
    ap.add_argument("--root", action="append", default=[],
                    help="root directory to browse (repeatable)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-open", action="store_true",
                    help="do not open the browser automatically")
    args = ap.parse_args()

    if not has_ffmpeg():
        print("warning: ffmpeg/ffprobe not found on PATH; playback proxy "
              "and metadata will fail.")

    roots = [os.path.abspath(r) for r in args.root] or [os.path.abspath("Samples")]
    roots = [r for r in roots if os.path.isdir(r)] or [os.path.abspath(".")]
    url = f"http://127.0.0.1:{args.port}/"
    print(f"serving {roots} at {url}")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    serve(roots, args.port)


if __name__ == "__main__":
    main()
