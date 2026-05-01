# FaceMeshUdp

Python face-tracking app built on MediaPipe FaceLandmarker. Produces gaze/head-pose output with an optional overlay, capture tooling, a 9-point calibration workflow, and UDP forwarding to OpenTrack. See [eyes.ini](eyes.ini) for an example opentrack profile that consumes the UDP stream.

[![Demo reel](demo.gif)](https://youtu.be/I_M037X3Fb8)

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Calibrate and run

```powershell
python -m facemesh_app.main --calibrate
python -m facemesh_app.main --udp 
```

## Options

At least one mode flag (`--overlay`, `--capture`, `--udp`, `--calibrate`) must be set.

Modes:

- `--overlay` — transparent overlay window with live landmarks
- `--capture` — save frames/mesh data on click (implies overlay)
- `--capture-live` / `--live` — show live camera feed in the capture window
- `--udp` — forward calibrated gaze output over UDP
- `--calibrate` / `--calibration` — run the 9-point calibration workflow
- `--force-recalibrate` — ignore any stored profile and recalibrate
- `--calibration-profile NAME` — named calibration profile (defaults to `default`)
- `--calibration-samples N` — minimum samples per point (default: 5)

Camera:

- `--camera-index N` (`CAMERA_INDEX`, default 0) — which device to use

Resolution, fps, fourcc, and backend are no longer user-configurable: the
mediapipe FaceLandmarker downsamples internally to fixed sizes (128x128
detector, 256x256 landmarks), so high-resolution capture only inflates
per-frame buffer copies without improving accuracy. The app probes a fixed
ladder of (backend, format, size) candidates from cheapest to most expensive
on startup and accepts the first one that the camera actually delivers
without hitting the driver's CPU-scaling slow path.

UDP:

- `--udp-host HOST` (`UDP_HOST`, default `127.0.0.1`)
- `--udp-port PORT` (`UDP_PORT`, default 4242)

Misc:

- `--overlay-fps FPS` — overlay redraw rate (default 60)
- `--log-interval SECONDS` — periodic stats interval (default 2.0)
- `--quiet` — suppress console output

## Capture output

Left-click the overlay in capture mode to write:

- `captures/mesh_capture_*.png` — frame with landmarks drawn
- `captures/mesh_capture_*.json` — 478 landmarks, 52 blendshapes, 4x4 transform matrix

## Build a standalone executable

```powershell
task build-exe
```

Outputs `dist/facemesh.exe` via PyInstaller.

## Profiling

Set `FACEMESH_PROFILE=1` to launch under yappi; stats are written on exit.
