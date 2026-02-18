import cv2
import numpy as np
from typing import List
from .models import Track, VEHICLE_CLASSES
from .estimator import SpeedEstimator
from .heatmap import SpeedHeatmap
from .flow import FlowCounter


# Colors per class (BGR)
CLASS_COLORS = {
    2: (0, 255, 0),    # car - green
    3: (255, 0, 0),    # motorcycle - blue
    5: (0, 165, 255),  # bus - orange
    7: (0, 0, 255),    # truck - red
}


def draw_tracks(frame: np.ndarray, tracks: List[Track], frame_idx: int,
                total_unique: int = 0, fps: float = 30.0,
                speed_estimator: SpeedEstimator | None = None,
                heatmap: SpeedHeatmap | None = None,
                flow_counter: FlowCounter | None = None,
                clean: bool = False,
                no_trails: bool = False) -> np.ndarray:
    if speed_estimator is None:
        speed_estimator = SpeedEstimator()
    for track in tracks:
        if track.time_since_update > 0:
            continue
        x1, y1, x2, y2 = track.bbox.astype(int)
        color = CLASS_COLORS.get(track.class_id, (255, 255, 255))
        label = VEHICLE_CLASSES.get(track.class_id, "?")

        # Draw trajectory trail
        if not no_trails and len(track.trajectory) >= 2:
            pts = [(int(p[0]), int(p[1])) for p in track.trajectory[-30:]]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, 2)

        if not clean:
            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label with ID, class, and dwell time
            dwell = track.hits / fps
            text = f"ID:{track.track_id} {label} {dwell:.1f}s"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Speed under the box (km/h)
        if len(track.trajectory) >= 2:
            dx = track.trajectory[-1][0] - track.trajectory[-2][0]
            dy = track.trajectory[-1][1] - track.trajectory[-2][1]
            speed_px = (dx**2 + dy**2) ** 0.5
            speed_kmh = speed_estimator.to_kmh(speed_px, fps)
            if not clean:
                cv2.putText(frame, f"{speed_kmh:.0f} km/h", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if heatmap is not None:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                heatmap.record(cx, cy, speed_kmh)

    if heatmap is not None:
        frame = heatmap.render(frame)

    if flow_counter is not None:
        flow_counter.update(tracks, fps)
        frame = flow_counter.render(frame)

    if not clean:
        # Analytics panel (top-left)
        active = sum(1 for t in tracks if t.time_since_update == 0)
        panel_h = 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (260, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"Frame: {frame_idx}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"Active: {active}", (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(frame, f"Total tracked: {total_unique}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # Per-class count for active tracks
        class_counts: dict[int, int] = {}
        for t in tracks:
            if t.time_since_update == 0:
                class_counts[t.class_id] = class_counts.get(t.class_id, 0) + 1
        class_str = " | ".join(f"{VEHICLE_CLASSES.get(c, '?')}:{n}" for c, n in sorted(class_counts.items()))
        cv2.putText(frame, class_str, (10, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return frame
