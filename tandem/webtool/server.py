"""Local HTTP server for the accelerometer player.

Transport only: it turns a file path into an accelerometer JSON series
(via the recon GPMF helpers and accel.py), streams video with Range
support, browses the filesystem for the picker, and triggers the H.264
proxy on demand. All path access is confined to the active roots.
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from tandem.recon.ffprobe import (
    find_gpmf_stream_index,
    gpmd_packet_times,
    probe_duration,
)
from tandem.recon.telemetry import extract_gpmf_blob
from tandem.webtool.accel import parse_accel
from tandem.webtool.fsbrowse import drive_roots, list_dir, resolve_within
from tandem.webtool.proxy import ensure_h264_proxy

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_CHUNK = 1 << 20


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    if not header or not header.startswith("bytes="):
        return None
    spec = header[len("bytes="):].split(",")[0].strip()
    start_s, _, end_s = spec.partition("-")
    try:
        if start_s == "":
            return None
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    except ValueError:
        return None
    end = min(end, size - 1)
    if start > end:
        return None
    return start, end


def safe_static_path(rel: str) -> str | None:
    base = os.path.normpath(STATIC_DIR)
    full = os.path.normpath(os.path.join(base, rel))
    if full == base or full.startswith(base + os.sep):
        return full
    return None


def make_handler(roots: list[str]) -> type:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console quiet
            pass

        def _json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _safe(self, path: str) -> str | None:
            allowed = roots + drive_roots()
            return resolve_within(path, allowed)

        def do_GET(self):
            parsed = urlparse(self.path)
            route = parsed.path
            query = parse_qs(parsed.query)
            if route == "/" or route == "/index.html":
                return self._static("index.html")
            if route.startswith("/static/"):
                return self._static(route[len("/static/"):])
            if route == "/api/roots":
                return self._json({"roots": roots, "drives": drive_roots()})
            if route == "/api/browse":
                return self._browse(query.get("path", [roots[0]])[0])
            if route == "/api/accel":
                return self._accel(query.get("path", [""])[0])
            if route == "/api/video":
                return self._video(query.get("path", [""])[0])
            self.send_error(404)

        def do_POST(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/api/proxy":
                return self._proxy(query.get("path", [""])[0])
            self.send_error(404)

        def _static(self, rel):
            full = safe_static_path(rel)
            if full is None or not os.path.isfile(full):
                return self.send_error(404)
            ctype = {
                ".html": "text/html", ".js": "text/javascript",
                ".css": "text/css",
            }.get(os.path.splitext(full)[1], "application/octet-stream")
            with open(full, "rb") as fh:
                data = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _browse(self, path):
            safe = self._safe(path)
            if safe is None or not os.path.isdir(safe):
                return self._json({"error": "path not allowed"}, 403)
            entries = [
                {"name": e.name, "path": e.path, "is_dir": e.is_dir}
                for e in list_dir(safe)
            ]
            parent = os.path.dirname(safe.rstrip(os.sep)) or safe
            return self._json({"cwd": safe, "parent": parent, "entries": entries})

        def _accel(self, path):
            safe = self._safe(path)
            if safe is None or not os.path.isfile(safe):
                return self._json({"error": "path not allowed"}, 403)
            index = find_gpmf_stream_index(safe)
            if index is None:
                return self._json({"error": "no GPMF stream"}, 422)
            blob = extract_gpmf_blob(safe)
            times = gpmd_packet_times(safe, index)
            series = parse_accel(blob, times, probe_duration(safe))
            if not series.t:
                return self._json({"error": "no ACCL samples"}, 422)
            return self._json({
                "t": series.t, "ax": series.ax, "ay": series.ay,
                "az": series.az, "amag": series.amag,
                "warnings": series.warnings,
            })

        def _video(self, path):
            safe = self._safe(path)
            if safe is None or not os.path.isfile(safe):
                return self.send_error(403)
            size = os.path.getsize(safe)
            rng = parse_range(self.headers.get("Range"), size)
            with open(safe, "rb") as f:
                if rng is None:
                    self.send_response(200)
                    self.send_header("Content-Length", str(size))
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Type", "video/mp4")
                    self.end_headers()
                    self._copy(f, 0, size - 1)
                else:
                    start, end = rng
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.send_header("Content-Length", str(end - start + 1))
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Type", "video/mp4")
                    self.end_headers()
                    self._copy(f, start, end)

        def _copy(self, f, start, end):
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(_CHUNK, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

        def _proxy(self, path):
            safe = self._safe(path)
            if safe is None or not os.path.isfile(safe):
                return self._json({"error": "path not allowed"}, 403)
            try:
                out = ensure_h264_proxy(safe)
            except Exception as exc:  # surface transcode failure to the UI
                return self._json({"error": f"proxy failed: {exc}"}, 500)
            return self._json({"path": out})

    return Handler


def serve(roots: list[str], port: int) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(roots))
    httpd.serve_forever()
