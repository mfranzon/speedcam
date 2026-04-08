import argparse
import sys

import cv2
import numpy as np
import supervision as sv

from trackers import frames_from_source
from trackers.core.base import BaseTracker

from .models import resolve_class_filter
from .estimator import SpeedEstimator
from .heatmap import SpeedHeatmap
from .flow import FlowCounter
from .analytics import Analytics
from .tracker import Tracker
from .visualizer import draw_tracks
from .video import VideoWriter
from .depth import DepthEstimator
from .exit_counter import ExitCounter, ExitLine


# ---------------------------------------------------------------------------
# Model helpers (mirrors trackers CLI pattern)
# ---------------------------------------------------------------------------

def _load_model(model_id: str):
    """Load a detection model.

    Supports:
    - YOLO .pt files (via ultralytics)
    - rfdetr-base / rfdetr-large (via rfdetr)
    - Anything else via inference_models.AutoModel
    """
    # YOLO .pt file
    if model_id.endswith(".pt"):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            print("Error: ultralytics is required for YOLO models.", file=sys.stderr)
            raise SystemExit(1) from exc
        return YOLO(model_id)

    # rfdetr models
    if model_id.startswith("rfdetr"):
        try:
            import rfdetr
        except ImportError as exc:
            print("Error: rfdetr is required for RF-DETR models.", file=sys.stderr)
            raise SystemExit(1) from exc
        if "large" in model_id:
            return rfdetr.RFDETRLarge()
        return rfdetr.RFDETRBase()

    # Fallback to inference_models
    try:
        from inference_models import AutoModel
    except ImportError as exc:
        print(
            "Error: inference-models is required for detection.\n"
            "Install with: pip install 'speedcam[detection]'",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    return AutoModel.from_pretrained(model_id)


def _run_model(model, frame: np.ndarray, confidence: float) -> sv.Detections:
    """Run model inference and return ``sv.Detections``."""
    # rfdetr returns sv.Detections directly from predict()
    if hasattr(model, 'predict') and hasattr(model, 'model_config'):
        detections = model.predict(frame, threshold=confidence)
        if not isinstance(detections, sv.Detections):
            return sv.Detections.empty()
        # Clear non-per-detection data that breaks boolean indexing
        detections.data = {}
        return detections

    # YOLO models
    if hasattr(model, 'names') and callable(model):
        results = model(frame, conf=confidence, verbose=False)
        if not results:
            return sv.Detections.empty()
        detections = sv.Detections.from_ultralytics(results[0])
        return detections

    # inference_models API
    predictions = model(frame)
    if not predictions:
        return sv.Detections.empty()

    detections = predictions[0].to_supervision()

    if len(detections) > 0 and detections.confidence is not None:
        mask = detections.confidence >= confidence
        detections = detections[mask]

    return detections


# ---------------------------------------------------------------------------
# SAHI (sliced inference)
# ---------------------------------------------------------------------------

def _nms_sv(detections: sv.Detections, iou_threshold: float = 0.5) -> sv.Detections:
    """Non-maximum suppression on ``sv.Detections``."""
    if len(detections) == 0:
        return detections
    bboxes = detections.xyxy
    scores = detections.confidence
    if scores is None:
        return detections

    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(bboxes[i, 0], bboxes[rest, 0])
        yy1 = np.maximum(bboxes[i, 1], bboxes[rest, 1])
        xx2 = np.minimum(bboxes[i, 2], bboxes[rest, 2])
        yy2 = np.minimum(bboxes[i, 3], bboxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (bboxes[i, 2] - bboxes[i, 0]) * (bboxes[i, 3] - bboxes[i, 1])
        area_rest = (bboxes[rest, 2] - bboxes[rest, 0]) * (bboxes[rest, 3] - bboxes[rest, 1])
        iou = inter / (area_i + area_rest - inter + 1e-6)
        order = rest[iou <= iou_threshold]
    return detections[keep]


def _detect_sahi(
    model, frame: np.ndarray, confidence: float, slice_size: int, overlap_ratio: float,
) -> sv.Detections:
    """Run sliced inference + full-frame, then NMS-merge."""
    h, w = frame.shape[:2]
    stride = int(slice_size * (1 - overlap_ratio))

    full_dets = _run_model(model, frame, confidence)
    all_xyxy = [full_dets.xyxy] if len(full_dets) > 0 else []
    all_conf = [full_dets.confidence] if len(full_dets) > 0 and full_dets.confidence is not None else []
    all_cls = [full_dets.class_id] if len(full_dets) > 0 and full_dets.class_id is not None else []

    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1 = min(x0 + slice_size, w)
            y1 = min(y0 + slice_size, h)
            if (x1 - x0) < slice_size // 2 or (y1 - y0) < slice_size // 2:
                continue
            crop = frame[y0:y1, x0:x1]
            dets = _run_model(model, crop, confidence)
            if len(dets) == 0:
                continue
            offset = np.array([x0, y0, x0, y0], dtype=np.float64)
            dets.xyxy = dets.xyxy + offset
            all_xyxy.append(dets.xyxy)
            if dets.confidence is not None:
                all_conf.append(dets.confidence)
            if dets.class_id is not None:
                all_cls.append(dets.class_id)

    if not all_xyxy:
        return sv.Detections.empty()

    merged = sv.Detections(
        xyxy=np.concatenate(all_xyxy),
        confidence=np.concatenate(all_conf) if all_conf else None,
        class_id=np.concatenate(all_cls) if all_cls else None,
    )
    return _nms_sv(merged)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    available_trackers = BaseTracker._registered_trackers()

    parser = argparse.ArgumentParser(
        prog="speedcam",
        description="speedcam -- Vehicle speed estimation, heatmaps, and traffic flow counting",
    )
    parser.add_argument("--source", required=True,
                        help="Video file, webcam index (0, 1, …), or RTSP URL")
    parser.add_argument("--model", default="rfdetr-base",
                        help="Model ID for inference-models AutoModel (default: rfdetr-base)")
    parser.add_argument("--tracker", default="bytetrack", choices=available_trackers,
                        help=f"Tracker algorithm (default: bytetrack)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("--output", default="output.mp4",
                        help="Output video path")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable live preview window")
    parser.add_argument("--scale", type=float, default=0.0,
                        help="Meters per pixel (0 = auto-calibrate from car sizes)")
    parser.add_argument("--heatmap", action="store_true",
                        help="Enable speed heatmap overlay")
    parser.add_argument("--export-json", default=None,
                        help="Export analytics to JSON file")
    parser.add_argument("--sahi", action="store_true",
                        help="Enable sliced inference for small/far objects")
    parser.add_argument("--slice-size", type=int, default=640,
                        help="SAHI slice size in pixels (default: 640)")
    parser.add_argument("--overlap", type=float, default=0.25,
                        help="SAHI slice overlap ratio (default: 0.25)")
    parser.add_argument("--clean", action="store_true",
                        help="Hide bounding boxes, labels, and panel")
    parser.add_argument("--no-trails", action="store_true",
                        help="Hide trajectory trails")
    parser.add_argument("--flow", action="store_true",
                        help="Enable traffic flow counting line")
    parser.add_argument("--flow-line-y", type=float, default=0.5,
                        help="Counting line y-position ratio (0-1, default 0.5)")
    parser.add_argument("--min-area", type=int, default=500,
                        help="Minimum detection area in px² (default: 500)")
    parser.add_argument("--classes", default="vehicles",
                        help="Classes to track: vehicles, person, all, or comma-separated names/IDs (default: vehicles)")
    parser.add_argument("--depth", action="store_true",
                        help="Enable depth estimation for 3D speed (requires depth-anything-v2)")
    parser.add_argument("--depth-model", default="small", choices=["small", "base", "large"],
                        help="Depth model size (default: small)")
    parser.add_argument("--focal-length", type=float, default=500.0,
                        help="Camera focal length in pixels for 3D projection (default: 500)")
    parser.add_argument("--direction", action="store_true",
                        help="Color-code detections by direction of travel")
    parser.add_argument("--exits", default=None,
                        help="Exit counting lines config: 'roundabout' preset or JSON file")
    args = parser.parse_args()

    # ---- Load model ----
    model = _load_model(args.model)
    # YOLO uses .names (dict), rfdetr/inference_models use .class_names (list)
    if hasattr(model, "class_names"):
        class_names: list[str] = model.class_names
    elif hasattr(model, "names"):
        names = model.names
        class_names = [names[i] for i in sorted(names.keys())] if isinstance(names, dict) else list(names)
    else:
        class_names = []
    class_filter = resolve_class_filter(args.classes, class_names)

    # ---- Source ----
    # Try to parse source as int (webcam index)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    # We need fps/dimensions before the loop for writer setup.
    # Peek at the first frame via cv2 to get metadata, then use frames_from_source.
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open source '{args.source}'")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # ---- Pipeline objects ----
    depth_estimator = None
    if args.depth:
        print(f"Loading depth model ({args.depth_model})...")
        depth_estimator = DepthEstimator(
            model_size=args.depth_model,
            metric=True,
            focal_length=args.focal_length,
        )
    tracker = Tracker(
        frame_rate=fps,
        tracker_id=args.tracker,
        depth_estimator=depth_estimator,
    )
    speed_estimator = SpeedEstimator(meters_per_pixel=args.scale)
    analytics = Analytics(fps=fps, speed_estimator=speed_estimator, use_3d=args.depth)
    heatmap = SpeedHeatmap(width, height) if args.heatmap else None
    flow_counter = FlowCounter(height, line_y_ratio=args.flow_line_y) if args.flow else None

    # ---- Exit counter ----
    exit_counter = None
    if args.exits == "roundabout":
        exit_counter = ExitCounter([
            ExitLine("NW", (700, 500), (1300, 50), (0, 255, 0)),
            ExitLine("NE", (2850, 80), (3500, 620), (0, 0, 255)),
            ExitLine("SW", (330, 1540), (1000, 2080), (255, 0, 0)),
            ExitLine("SE", (2850, 2080), (3500, 1540), (255, 255, 0)),
        ])
    elif args.exits:
        import json
        with open(args.exits) as f:
            cfg = json.load(f)
        exit_counter = ExitCounter([
            ExitLine(e["name"], tuple(e["p1"]), tuple(e["p2"]), tuple(e["color"]))
            for e in cfg["exits"]
        ])

    writer = VideoWriter(args.output, fps, width, height)

    print(f"Processing: {args.source} ({width}x{height} @ {fps:.1f} FPS"
          + (f", {total_frames} frames" if total_frames > 0 else "")
          + (", 3D depth enabled" if args.depth else "") + ")")

    frame_idx = 0
    try:
        for _fid, frame in frames_from_source(source):
            # ---- Detection ----
            if args.sahi:
                sv_dets = _detect_sahi(model, frame, args.conf, args.slice_size, args.overlap)
            else:
                sv_dets = _run_model(model, frame, args.conf)

            # ---- Class filter ----
            if class_filter is not None and len(sv_dets) > 0 and sv_dets.class_id is not None:
                mask = np.isin(sv_dets.class_id, class_filter)
                sv_dets = sv_dets[mask]

            # ---- Min-area filter ----
            if len(sv_dets) > 0:
                areas = (sv_dets.xyxy[:, 2] - sv_dets.xyxy[:, 0]) * (sv_dets.xyxy[:, 3] - sv_dets.xyxy[:, 1])
                sv_dets = sv_dets[areas >= args.min_area]

            # ---- Tracking ----
            tracks = tracker.update(sv_dets, frame)
            analytics.update(tracks)

            # ---- Visualisation ----
            annotated = draw_tracks(
                frame, tracks, frame_idx,
                total_unique=tracker.next_id, fps=fps,
                speed_estimator=speed_estimator,
                heatmap=heatmap,
                flow_counter=flow_counter,
                clean=args.clean,
                no_trails=args.no_trails,
                camera_offset=tracker.camera_offset,
                world_to_frame_2x3=tracker.world_to_frame_2x3,
                bg_speed_px=tracker.bg_speed_px,
                use_3d=args.depth,
                color_by_direction=args.direction,
                all_tracks=tracker.all_tracks,
            )

            if exit_counter is not None:
                exit_counter.update(tracks)
                annotated = exit_counter.render(annotated)

            writer.write(annotated)

            if not args.no_display:
                cv2.imshow("speedcam", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break

            frame_idx += 1
            if total_frames > 0 and frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames...")

    finally:
        writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    analytics.print_summary(tracker.all_tracks)

    if exit_counter is not None:
        print("\n=== Exit Counts ===")
        for ex in exit_counter.exits:
            total = ex.count_in + ex.count_out
            print(f"  {ex.name:<6s}  In: {ex.count_in:>3d}  Out: {ex.count_out:>3d}  Total: {total:>3d}")
        print("===================\n")

    if args.export_json:
        analytics.export_json(tracker.all_tracks, args.export_json)

    print(f"Output video saved to {args.output}")


if __name__ == "__main__":
    main()
