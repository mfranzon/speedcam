import cv2
import numpy as np


class FlowCounter:
    def __init__(self, frame_height: int, line_y_ratio: float = 0.5):
        self.line_y = int(frame_height * line_y_ratio)
        self._crossed: set[int] = set()
        self._counts_up: dict[int, int] = {}
        self._counts_down: dict[int, int] = {}
        self._crossing_frames: list[int] = []
        self._frame_idx: int = 0

    def update(self, tracks, fps: float):
        for track in tracks:
            if track.time_since_update > 0:
                continue
            if len(track.trajectory) < 2:
                continue
            if track.track_id in self._crossed:
                continue

            prev_y = track.trajectory[-2][1]
            curr_y = track.trajectory[-1][1]

            if (prev_y < self.line_y) != (curr_y < self.line_y):
                self._crossed.add(track.track_id)
                self._crossing_frames.append(self._frame_idx)
                if curr_y > prev_y:
                    self._counts_down[track.class_id] = self._counts_down.get(track.class_id, 0) + 1
                else:
                    self._counts_up[track.class_id] = self._counts_up.get(track.class_id, 0) + 1

        self._frame_idx += 1

    def render(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        cv2.line(frame, (0, self.line_y), (w, self.line_y), (255, 255, 0), 2)

        total_up = sum(self._counts_up.values())
        total_down = sum(self._counts_down.values())
        total = total_up + total_down

        # Rate: crossings in last 60s worth of frames
        if self._crossing_frames:
            rate = total  # fallback
            # We don't know fps here, so compute from frame count
            # Use simple total/elapsed approach
        else:
            rate = 0

        text = f"  {total_up}   {total_down}  | {total} total"
        panel_w = 280
        panel_h = 30
        px, py = 10, self.line_y - 40

        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, text, (px + 5, py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        return frame

    def summary(self) -> dict:
        return {
            "up": sum(self._counts_up.values()),
            "down": sum(self._counts_down.values()),
            "per_class_up": dict(self._counts_up),
            "per_class_down": dict(self._counts_down),
            "total_crossings": len(self._crossing_frames),
        }
