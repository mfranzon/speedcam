import cv2
import numpy as np
from typing import List
from .models import Track, ALL_CLASSES
from .estimator import SpeedEstimator
from .heatmap import SpeedHeatmap
from .flow import FlowCounter


CLASS_COLORS = {
    0: (255, 255, 0),
    2: (0, 255, 0),
    3: (255, 0, 0),
    5: (0, 165, 255),
    7: (0, 0, 255),
}


def draw_tracks(frame: np.ndarray, tracks: List[Track], frame_idx: int,
                total_unique: int = 0, fps: float = 30.0,
                speed_estimator: SpeedEstimator | None = None,
                heatmap: SpeedHeatmap | None = None,
                flow_counter: FlowCounter | None = None,
                clean: bool = False,
                no_trails: bool = False,
                camera_offset: tuple[float, float] = (0.0, 0.0),
                world_to_frame_2x3: np.ndarray | None = None,
                bg_speed_px: float = 0.0,
                use_3d: bool = False) -> np.ndarray:
    if speed_estimator is None:
        speed_estimator = SpeedEstimator()

    for track in tracks:
        if track.time_since_update > 0:
            continue
        x1, y1, x2, y2 = track.bbox.astype(int)
        color = CLASS_COLORS.get(track.class_id, (255, 255, 255))
        label = ALL_CLASSES.get(track.class_id, "?")

        if not no_trails and len(track.trajectory) >= 2:
            raw = np.array(track.trajectory, dtype=np.float64)
            if world_to_frame_2x3 is not None:
                pts = raw.reshape(-1, 1, 2)
                frame_pts = cv2.transform(pts, world_to_frame_2x3).reshape(-1, 2)
            else:
                off_x, off_y = camera_offset
                frame_pts = raw + np.array([off_x, off_y])
            if len(frame_pts) >= 7:
                k = 7
                pad = k // 2
                sx = np.convolve(frame_pts[:, 0], np.ones(k) / k, mode='valid')
                sy = np.convolve(frame_pts[:, 1], np.ones(k) / k, mode='valid')
                smooth = np.column_stack([sx, sy]).astype(np.int32)
            else:
                smooth = frame_pts.astype(np.int32)
            cv2.polylines(frame, [smooth], isClosed=False, color=color, thickness=2)

        if not clean:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            dwell = track.hits / fps
            depth_str = f" d:{track.last_depth:.1f}m" if use_3d and track.depths else ""
            text = f"ID:{track.track_id} {label} {dwell:.1f}s{depth_str}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        if use_3d and len(track.trajectory_3d) >= 2:
            speed_kmh = SpeedEstimator.speed_from_trajectory_3d(track.trajectory_3d, fps)
            if not clean:
                cv2.putText(frame, f"{speed_kmh:.0f} km/h (3D)", (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if heatmap is not None:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                heatmap.record(cx, cy, speed_kmh)
        elif len(track.trajectory) >= 2:
            dx = track.trajectory[-1][0] - track.trajectory[-2][0]
            dy = track.trajectory[-1][1] - track.trajectory[-2][1]
            frame_speed_px = (dx**2 + dy**2) ** 0.5
            speed_px = max(frame_speed_px, bg_speed_px)
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
        active = sum(1 for t in tracks if t.time_since_update == 0)
        panel_h = 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (260, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"Active: {active}", (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        mode_str = " (3D)" if use_3d else ""
        cv2.putText(frame, f"Total tracked: {total_unique}{mode_str}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        class_counts: dict[int, int] = {}
        for t in tracks:
            if t.time_since_update == 0:
                class_counts[t.class_id] = class_counts.get(t.class_id, 0) + 1
        class_str = " | ".join(f"{ALL_CLASSES.get(c, '?')}:{n}" for c, n in sorted(class_counts.items()))
        cv2.putText(frame, class_str, (10, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return frame
