import argparse
import sys
import cv2
import numpy as np
from typing import List

from .models import Detection, VEHICLE_CLASSES_YOLO, VEHICLE_CLASSES_RFDETR
from .estimator import SpeedEstimator
from .heatmap import SpeedHeatmap
from .flow import FlowCounter
from .analytics import Analytics
from .tracker import SimpleTracker
from .visualizer import draw_tracks
from .video import VideoWriter


def _nms(detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    if not detections:
        return []
    bboxes = np.array([d.bbox for d in detections])
    scores = np.array([d.score for d in detections])
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
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
    return [detections[i] for i in keep]


# Map RF-DETR 1-indexed IDs to canonical 0-indexed IDs
_RFDETR_TO_CANONICAL = {3: 2, 4: 3, 6: 5, 8: 7}


def _detect_rfdetr(model, frame, confidence):
    from PIL import Image
    pil_image = Image.fromarray(frame[..., ::-1])
    dets = model.predict(pil_image, threshold=confidence)
    detections = []
    for i in range(len(dets)):
        cls = int(dets.class_id[i])
        if cls in VEHICLE_CLASSES_RFDETR:
            xyxy = dets.xyxy[i].astype(np.float64)
            score = float(dets.confidence[i])
            canonical_cls = _RFDETR_TO_CANONICAL[cls]
            detections.append(Detection(bbox=xyxy, score=score, class_id=canonical_cls))
    return detections


def _detect_yolo(model, frame, confidence):
    results = model(frame, conf=confidence, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in VEHICLE_CLASSES_YOLO:
            xyxy = box.xyxy[0].cpu().numpy()
            score = float(box.conf[0])
            detections.append(Detection(bbox=xyxy, score=score, class_id=cls))
    return detections


def _detect(model, frame, confidence, backend, sahi, slice_size, overlap_ratio):
    detect_fn = _detect_rfdetr if backend == "rfdetr" else _detect_yolo

    if not sahi:
        return detect_fn(model, frame, confidence)

    h, w = frame.shape[:2]
    stride = int(slice_size * (1 - overlap_ratio))
    all_detections = list(detect_fn(model, frame, confidence))

    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1 = min(x0 + slice_size, w)
            y1 = min(y0 + slice_size, h)
            if (x1 - x0) < slice_size // 2 or (y1 - y0) < slice_size // 2:
                continue
            crop = frame[y0:y1, x0:x1]
            dets = detect_fn(model, crop, confidence)
            for d in dets:
                d.bbox = d.bbox + np.array([x0, y0, x0, y0], dtype=np.float64)
            all_detections.extend(dets)

    return _nms(all_detections, iou_threshold=0.5)



def main():
    parser = argparse.ArgumentParser(
        prog="speedcam",
        description="speedcam — Vehicle speed estimation, heatmaps, and traffic flow counting",
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--model", default="rf-detr-base", help="Model name (default: rf-detr-base)")
    parser.add_argument("--backend", default="rfdetr", choices=["rfdetr", "yolo"], help="Detection backend (default: rfdetr)")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--no-display", action="store_true", help="Disable live preview window")
    parser.add_argument("--scale", type=float, default=0.0, help="Meters per pixel (0 = auto-calibrate from car sizes)")
    parser.add_argument("--heatmap", action="store_true", help="Enable speed heatmap overlay")
    parser.add_argument("--export-json", default=None, help="Export analytics to JSON file")
    parser.add_argument("--sahi", action="store_true", help="Enable sliced inference for small/far objects")
    parser.add_argument("--slice-size", type=int, default=640, help="SAHI slice size in pixels (default: 640)")
    parser.add_argument("--overlap", type=float, default=0.25, help="SAHI slice overlap ratio (default: 0.25)")
    parser.add_argument("--clean", action="store_true", help="Hide bounding boxes, labels, and panel")
    parser.add_argument("--no-trails", action="store_true", help="Hide trajectory trails")
    parser.add_argument("--flow", action="store_true", help="Enable traffic flow counting line")
    parser.add_argument("--flow-line-y", type=float, default=0.5, help="Counting line y-position ratio (0-1, default 0.5)")
    args = parser.parse_args()

    # Load model
    if args.backend == "rfdetr":
        from rfdetr import RFDETRBase, RFDETRLarge
        model_map = {"rf-detr-base": RFDETRBase, "rf-detr-large": RFDETRLarge}
        model_cls = model_map.get(args.model, RFDETRBase)
        model = model_cls()
    else:
        from ultralytics import YOLO
        model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: cannot open video '{args.video}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = SimpleTracker()
    speed_estimator = SpeedEstimator(meters_per_pixel=args.scale)
    analytics = Analytics(fps=fps, speed_estimator=speed_estimator)
    heatmap = SpeedHeatmap(width, height) if args.heatmap else None
    flow_counter = FlowCounter(height, line_y_ratio=args.flow_line_y) if args.flow else None
    writer = VideoWriter(args.output, fps, width, height)

    print(f"Processing: {args.video} ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = _detect(model, frame, args.conf, args.backend,
                                 args.sahi, args.slice_size, args.overlap)
            tracks = tracker.update(detections)
            analytics.update(tracks)

            annotated = draw_tracks(frame, tracks, frame_idx,
                                     total_unique=tracker.next_id, fps=fps,
                                     speed_estimator=speed_estimator,
                                     heatmap=heatmap,
                                     flow_counter=flow_counter,
                                     clean=args.clean,
                                     no_trails=args.no_trails)
            writer.write(annotated)

            if not args.no_display:
                cv2.imshow("speedcam", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames...")

    finally:
        cap.release()
        writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    analytics.print_summary(tracker.all_tracks)

    if args.export_json:
        analytics.export_json(tracker.all_tracks, args.export_json)

    print(f"Output video saved to {args.output}")


if __name__ == "__main__":
    main()
