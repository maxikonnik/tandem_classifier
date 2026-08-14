# GoPro accelerometer player — design

Date: 2026-08-14
Status: approved for planning

## Purpose

A local web tool that opens a GoPro `.MP4`, extracts the accelerometer
readings from its GPMF metadata, and plots them on a clickable timeline
under a video player. The plot doubles as the seek bar: clicking a point
on the accelerometer curve jumps the video to that instant, and a
playhead line tracks the video as it plays. It is an inspection aid for
skydive footage — the operator can see the exit spike and the opening
shock on the curve and jump straight to those moments in the video.

## Scope

- Runs locally on the operator's machine (localhost), single user.
- Reuses the existing dependency-free GPMF parser in
  `tandem/recon/gpmf.py`; adds `ACCL` decoding, which the parser does
  not do yet (it currently reads only `GPS5`/`GPSU`).
- Accelerometer only. No GPS, no gyro, no phase detection — those live
  elsewhere in the project.

## Non-goals

- No cloud deployment, no multi-user, no authentication.
- No editing, trimming, or export of video.
- No reimplementation of GPMF parsing in JavaScript.

## Architecture

New package `tandem/webtool/` with clear module boundaries:

```
tandem/webtool/
  server.py      HTTP transport: routing, Range video streaming, lazy proxy
  accel.py       parse_accel(blob, packet_times) -> AccelSeries  (uses gpmf.py)
  fsbrowse.py    server-side filesystem listing for the Обзор modal
  proxy.py       ensure_h264_proxy(path) -> proxy path  (ffmpeg, disk cache)
  static/
    index.html   layout: video + canvas scrubber + mode toggle + browse
    app.js       player, chart draw, click-seek, playhead tracking
    style.css
```

Boundaries, each independently understandable and testable:

- `accel.py` — pure: bytes + per-packet timings in, arrays out. Knows
  nothing about HTTP. Tested on hand-built GPMF byte strings, mirroring
  the existing `gpmf.py` tests.
- `fsbrowse.py` — pure listing helpers: given a directory path, return
  its subdirectories and `.mp4` files, plus drive roots. Path
  resolution and traversal safety live here and are unit-tested.
- `server.py` — transport only. Wires file path -> `extract_gpmf_blob`
  (existing) -> `accel.py` -> JSON; serves video with Range support;
  calls `proxy.py` lazily. No GPMF knowledge.
- `proxy.py` — isolates the ffmpeg transcode. Invoked only when the
  frontend reports the original will not play.
- `app.js` — all interactivity in the browser; consumes JSON, streams
  video.

## Data flow

1. Frontend requests `GET /accel?path=<file>`.
2. Backend finds the `gpmd` stream index (existing
   `find_gpmf_stream_index`), reads per-packet times, dumps the GPMF
   blob (existing `extract_gpmf_blob`), calls `accel.py`, returns JSON:
   `{t[], ax[], ay[], az[], amag[], meta}`.
3. Frontend draws the curve on the canvas scrubber; `<video>` streams
   from `GET /video?path=<file>` with Range support.

### Accelerometer-to-timeline alignment (the critical part)

`ACCL` arrives in bursts: each top-level `DEVC` record in the GPMF
stream is one metadata packet (~1 s of video), and inside it are ~200
accelerometer samples. To land the curve on the video timeline exactly
rather than assuming a nominal rate:

- `ffprobe -show_packets` on the `gpmd` stream gives each packet's
  `pts_time` and `duration_time` — real positions on the video
  timeline.
- Walk the blob over its top-level `DEVC` records (these are the packet
  boundaries); the i-th `DEVC` pairs with the i-th ffprobe packet.
- Inside each `DEVC`, find the `STRM` containing `ACCL` and its `SCAL`;
  divide the raw int16 values by `SCAL` to get m/s^2.
- Sample j of packet i gets timestamp `pts_i + (j + 0.5)/n_i * dur_i`.
- `amag = sqrt(ax^2 + ay^2 + az^2)`.

Result: `t[]` increases monotonically in the video's own time base, so a
timeline pixel maps 1:1 to `video.currentTime`. Click at pixel x ->
`video.currentTime = t_at_pixel(x)`; the playhead draws at
`pixel_at_time(video.currentTime)` on each `requestAnimationFrame`.

Edge cases:

- Packet count from the blob != packet count from ffprobe -> fall back
  to distributing all samples uniformly across the video duration, and
  flag it in the UI.
- Missing `SCAL` -> divisor 1, flag it.
- No `ACCL` stream -> clear UI error.

## Video playback

Try the original file first. If `<video>` raises a decode error (error
event or zero `videoWidth`), the frontend calls `POST /proxy?path=<file>`;
the backend transcodes to H.264 720p with ffmpeg, caches the result next
to the source (e.g. `.<name>.proxy.mp4`), and returns the proxy URL. The
player switches to it and shows "preparing a compatible copy…". GoPro
`GX*` files are HEVC, which many browsers cannot decode, so this path is
expected to fire often.

## File selection

- **File list** — a dropdown of `.mp4` files in the active root
  directory (defaults to `Samples/`). Selecting one loads it. The
  server knows the path.
- **Обзор (Browse)** — a button opening a modal that browses the
  server's filesystem: folders and `.mp4` files, breadcrumb, up to drive
  roots (`C:\`, `D:\`). Clicking a folder navigates into it; clicking a
  file opens it directly; "Выбрать эту папку" makes the current folder
  the active root and repopulates the list.

Browser constraint that shapes this: a plain web page cannot obtain the
absolute path of a file chosen through the OS picker, nor read a dropped
file's path — only its name and bytes. So all path knowledge comes from
the server-side browser, never from a browser file input. This is why
there is no drag-and-drop and no OS file dialog.

Safety: `/accel`, `/video`, and `/proxy` resolve the requested path and
require it to sit under one of the currently-active roots (the set the
user controls via Обзор), rejecting traversal attempts with 403. On a
local single-user tool this is light defense, but it keeps a crafted
request from reaching arbitrary files.

## UI

- Top: file dropdown + Обзор button.
- Middle: `<video>` element.
- Below: a `<canvas>` scrubber spanning the full video duration. The
  accelerometer curve fills the width; the y-axis is magnitude (or the
  three axes). A red playhead line marks the current time.
- Bottom: play/pause, `mm:ss / mm:ss` time label, and a toggle
  switching the plot between `|a|` and `X / Y / Z`.

Interaction:

- Click or drag on the canvas -> seek the video.
- Playhead follows `video.currentTime` via `requestAnimationFrame`.
- The `|a|` / `X-Y-Z` toggle redraws from data already in memory — no
  refetch.
- Downsampling for the draw: ~200 Hz over minutes is far more points
  than canvas pixels, so draw a min/max envelope per pixel bucket to
  preserve spikes (exit, opening shock).

## Error handling

All surfaced in the UI, not the console: ffmpeg/ffprobe not on PATH; no
`gpmd` or `ACCL` in the file; corrupt GPMF; packet-count mismatch
(fallback + flag); path outside roots (403); proxy transcode failure.

## Testing

- `accel.py` — unit tests on synthetic GPMF bytes: SCAL division, `|a|`
  computation, per-packet time assignment, uniform fallback.
- `fsbrowse.py` — unit tests: listing, path resolution inside a root,
  traversal rejection.
- `server.py` — unit tests of pure helpers: Range header parsing, path
  resolution guard.
- Integration (marked `integration`, per project convention): on a real
  `Samples/` file, assert `t[]` is monotonic and spans the video
  duration.
- Manual acceptance: open a real jump, see the `|a|` spike at exit and
  opening, click the spike, confirm the video jumps to that moment.

## Dependencies

No new Python packages. Backend uses the standard library
(`http.server`, `subprocess`) plus the existing `tandem.recon` helpers.
Frontend is vanilla JS with a hand-drawn `<canvas>` — no chart library,
matching the project's dependency-light style. ffmpeg/ffprobe must be on
PATH (already required by the project).
