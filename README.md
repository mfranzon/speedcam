# speedcam


https://github.com/user-attachments/assets/1d62c4df-b194-488f-b12d-f000ad854807


Vehicle speed estimation, heatmaps, and traffic flow counting from video — bring your own detector.

```
speedcam traffic.mp4 --heatmap --flow --export-json report.json
```

## Features

- **Speed estimation** — per-vehicle km/h from pixel trajectories, auto-calibrated from known vehicle dimensions or manual scale
- **Speed heatmap** — colour overlay showing where vehicles move fast or slow across the scene
- **Traffic flow counting** — virtual counting line with directional (up/down) counts per class
- **Multi-object tracking** — IoU-based Hungarian-algorithm tracker with trajectory trails
- **SAHI support** — sliced inference for detecting small or distant vehicles
- **Analytics export** — JSON summary with counts, avg speed, dwell time, per-class breakdown
- **Detector-agnostic** — works with RF-DETR (default) or any YOLO model via Ultralytics

---

## Install

```bash
# Core library only (no detector)
pip install speedcam

# With RF-DETR detector (recommended)
pip install "speedcam[cli]"

# With YOLO detector
pip install "speedcam[yolo]"

# Both detectors
pip install "speedcam[cli,yolo]"
```

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org/) on `PATH` for video output.

---

## CLI

```
speedcam VIDEO [options]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `video` | — | Path to input video file |
| `--model` | `rf-detr-base` | Model name (`rf-detr-base`, `rf-detr-large`, or any YOLO model path) |
| `--backend` | `rfdetr` | Detection backend: `rfdetr` or `yolo` |
| `--conf` | `0.25` | Detection confidence threshold |
| `--output` | `output.mp4` | Output video path |
| `--scale` | `0.0` | Meters per pixel (`0` = auto-calibrate from vehicle sizes) |
| `--heatmap` | off | Enable speed heatmap overlay |
| `--flow` | off | Enable traffic flow counting line |
| `--flow-line-y` | `0.5` | Counting line vertical position as fraction of frame height (0–1) |
| `--sahi` | off | Enable sliced inference for small/distant objects |
| `--slice-size` | `640` | SAHI slice size in pixels |
| `--overlap` | `0.25` | SAHI slice overlap ratio |
| `--export-json` | — | Export analytics summary to a JSON file |
| `--clean` | off | Hide bounding boxes, labels, and HUD panel (trails still shown) |
| `--no-trails` | off | Hide trajectory trails |
| `--no-display` | off | Disable live preview window |

### Examples

```bash
# Basic — RF-DETR, live preview, output video
speedcam highway.mp4

# With heatmap and flow counter, export analytics
speedcam highway.mp4 --heatmap --flow --export-json stats.json

# Use YOLO, set known scale, no preview
speedcam parking.mp4 --backend yolo --model yolov8n.pt --scale 0.04 --no-display

# Small vehicles far from camera — use SAHI
speedcam aerial.mp4 --sahi --slice-size 512 --overlap 0.3

# Clean output for presentation
speedcam footage.mp4 --clean --no-display --output clean.mp4
```

---

## Library API

All components are importable directly and can be composed freely around any detector.

```python
import cv2
from speedcam import (
    SpeedEstimator,
    SimpleTracker,
    FlowCounter,
    SpeedHeatmap,
    Analytics,
    draw_tracks,
    VideoWriter,
    Detection,
)

cap = cv2.VideoCapture("traffic.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

tracker = SimpleTracker(iou_threshold=0.3, max_age=30)
speed_estimator = SpeedEstimator(meters_per_pixel=0.05)  # or 0.0 for auto
analytics = Analytics(fps=fps, speed_estimator=speed_estimator)
heatmap = SpeedHeatmap(width, height)
flow_counter = FlowCounter(height, line_y_ratio=0.5)
writer = VideoWriter("output.mp4", fps, width, height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Plug in your own detector here — produce List[Detection]
    detections: list[Detection] = my_detector(frame)

    tracks = tracker.update(detections)
    analytics.update(tracks)

    annotated = draw_tracks(
        frame, tracks, frame_idx=0, fps=fps,
        speed_estimator=speed_estimator,
        heatmap=heatmap,
        flow_counter=flow_counter,
    )
    writer.write(annotated)

cap.release()
writer.release()

analytics.print_summary(tracker.all_tracks)
analytics.export_json(tracker.all_tracks, "report.json")
```

### Detection format

```python
from speedcam import Detection
import numpy as np

det = Detection(
    bbox=np.array([x1, y1, x2, y2], dtype=float),
    score=0.85,
    class_id=2,  # COCO class: 2=car, 3=motorcycle, 5=bus, 7=truck
)
```

### Class IDs

| Class | YOLO (0-indexed) | RF-DETR (1-indexed) |
|---|---|---|
| car | 2 | 3 |
| motorcycle | 3 | 4 |
| bus | 5 | 6 |
| truck | 7 | 8 |

`VEHICLE_CLASSES_YOLO` and `VEHICLE_CLASSES_RFDETR` dicts are exported from the package for convenience.

---

## How scale calibration works

When `--scale 0` (default), `SpeedEstimator` estimates `meters_per_pixel` automatically by observing the pixel width of detected vehicles and comparing to known average lengths (car ≈ 4.5 m, motorcycle ≈ 2.2 m, bus ≈ 12 m, truck ≈ 8 m). The estimate is averaged across all detections seen so far.

For accurate results on perspective or wide-angle cameras, measure a known distance in the scene and pass `--scale <value>` explicitly.

---

## Analytics JSON output

```json
{
  "total_frames": 1800,
  "total_unique_vehicles": 47,
  "avg_vehicles_per_frame": 3.2,
  "per_class_count": { "car": 38, "truck": 6, "motorcycle": 3 },
  "avg_dwell_time_sec": 4.71,
  "avg_speed_kmh": 52.3,
  "meters_per_pixel": 0.0431
}
```

---

## License

MIT
