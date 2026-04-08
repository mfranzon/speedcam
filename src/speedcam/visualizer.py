import cv2
import numpy as np
from typing import List
from .models import Track, ALL_CLASSES, DIRECTION_COLORS
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
                use_3d: bool = False,
                all_tracks: List[Track] | None = None,
                color_by_direction: bool = False) -> np.ndarray:
    if speed_estimator is None:
        speed_estimator = SpeedEstimator()

    for track in tracks:
        if track.time_since_update > 0:
            continue
        x1, y1, x2, y2 = track.bbox.astype(int)
        if color_by_direction:
            color = track.direction_color
        else:
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
            dir_str = f" [{track.direction}]" if color_by_direction else ""
            text = f"ID:{track.track_id} {label} {dwell:.1f}s{depth_str}{dir_str}"
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

        # Compute avg speed across all tracks that have trajectories
        source_tracks = all_tracks if all_tracks is not None else tracks
        speeds = []
        for t in source_tracks:
            if use_3d and len(t.trajectory_3d) >= 2:
                s = SpeedEstimator.speed_from_trajectory_3d(t.trajectory_3d, fps)
            elif len(t.trajectory) >= 2:
                s = speed_estimator.speed_from_trajectory(t.trajectory, fps)
            else:
                continue
            if s > 0:
                speeds.append(s)
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

        panel_h = 124
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
        cv2.putText(frame, f"Avg speed: {avg_speed:.1f} km/h", (10, 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100), 1)
        class_counts: dict[int, int] = {}
        for t in tracks:
            if t.time_since_update == 0:
                class_counts[t.class_id] = class_counts.get(t.class_id, 0) + 1
        class_str = " | ".join(f"{ALL_CLASSES.get(c, '?')}:{n}" for c, n in sorted(class_counts.items()))
        cv2.putText(frame, class_str, (10, 116),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        if color_by_direction:
            # Total unique vehicles per direction (cumulative)
            total_dir: dict[str, int] = {}
            count_source = all_tracks if all_tracks is not None else tracks
            for t in count_source:
                d = t.direction
                if d != "unknown":
                    total_dir[d] = total_dir.get(d, 0) + 1

            # Active (current frame) per direction
            active_dir: dict[str, int] = {}
            for t in tracks:
                if t.time_since_update == 0 and t.direction != "unknown":
                    active_dir[t.direction] = active_dir.get(t.direction, 0) + 1

            all_dirs = sorted(set(list(total_dir.keys()) + list(active_dir.keys())))
            if all_dirs:
                legend_x = frame.shape[1] - 280
                legend_y = 10
                header_h = 28
                row_h = 26
                legend_h = header_h + row_h * len(all_dirs) + 10
                overlay = frame.copy()
                cv2.rectangle(overlay, (legend_x - 10, legend_y),
                              (legend_x + 270, legend_y + legend_h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                cv2.putText(frame, "Direction      Active  Total",
                            (legend_x + 5, legend_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                for i, d in enumerate(all_dirs):
                    y = legend_y + header_h + 22 + i * row_h
                    c = DIRECTION_COLORS.get(d, (128, 128, 128))
                    cv2.circle(frame, (legend_x + 10, y - 5), 6, c, -1)
                    act = active_dir.get(d, 0)
                    tot = total_dir.get(d, 0)
                    cv2.putText(frame, f"{d:<12s}   {act:>3d}    {tot:>3d}",
                                (legend_x + 24, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

    return frame
