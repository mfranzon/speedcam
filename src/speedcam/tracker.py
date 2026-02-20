import numpy as np
import supervision as sv
from typing import List, Optional

from trackers import (  # noqa: F401 — imports register tracker subclasses
    ByteTrackTracker as _ByteTrackTracker,
    SORTTracker as _SORTTracker,
)
from trackers.core.base import BaseTracker

from .models import Track
from .motion import BackgroundMotionEstimator
from .depth import DepthEstimator


class Tracker:
    """Multi-object tracker with background-motion-compensated trajectories
    and speed estimation.

    Trail points are stored in world coordinates (frame-0 reference).
    The accumulated camera displacement is tracked so that trail points
    can be reprojected to the current frame for drawing.
    
    When depth_estimator is provided, 3D trajectories are computed for
    accurate speed estimation.
    """

    def __init__(
        self,
        frame_rate: float = 30.0,
        lost_track_buffer: int = 30,
        tracker_id: str = "bytetrack",
        depth_estimator: Optional[DepthEstimator] = None,
    ):
        tracker_info = BaseTracker._lookup_tracker(tracker_id)
        tracker_cls = tracker_info.tracker_class
        if tracker_id == "bytetrack":
            self._tracker = tracker_cls(
                frame_rate=frame_rate,
                lost_track_buffer=lost_track_buffer,
                minimum_consecutive_frames=1,
            )
        else:
            self._tracker = tracker_cls()

        self._motion = BackgroundMotionEstimator()
        self._depth = depth_estimator
        self._last_depth_map: Optional[np.ndarray] = None
        self._last_tracked: sv.Detections = sv.Detections.empty()
        self._tracks_by_id: dict[int, Track] = {}
        self.all_tracks: List[Track] = []
        self.next_id: int = 0

    @property
    def last_tracked(self) -> sv.Detections:
        return self._last_tracked

    @property
    def last_depth_map(self) -> Optional[np.ndarray]:
        return self._last_depth_map

    @property
    def has_depth(self) -> bool:
        return self._depth is not None

    @property
    def camera_offset(self) -> tuple[float, float]:
        """Camera displacement ``(dx, dy)`` from frame 0."""
        return self._motion.camera_offset

    @property
    def world_to_frame(self) -> np.ndarray:
        """3×3 homogeneous matrix: world coords → current frame coords."""
        return self._motion.world_to_frame

    @property
    def world_to_frame_2x3(self) -> np.ndarray:
        """2×3 affine matrix for ``cv2.transform``."""
        return self._motion.world_to_frame_2x3

    @property
    def bg_speed_px(self) -> float:
        """Background displacement magnitude (px/frame) — real speed proxy."""
        dx, dy = self._motion.last_displacement
        return (dx ** 2 + dy ** 2) ** 0.5

    def update(
        self, sv_detections: sv.Detections, frame: np.ndarray
    ) -> List[Track]:
        # 1. Background motion
        self._motion.update(frame, sv_detections)

        # 2. Depth estimation (if available)
        if self._depth is not None:
            self._last_depth_map = self._depth.estimate(frame)

        # 3. Tracker update
        tracked = self._tracker.update(sv_detections)
        self._last_tracked = tracked

        for t in self._tracks_by_id.values():
            t.time_since_update += 1

        active: List[Track] = []
        if len(tracked) > 0 and tracked.tracker_id is not None:
            for i in range(len(tracked)):
                tid = int(tracked.tracker_id[i])
                if tid < 0:
                    continue

                bbox = tracked.xyxy[i]
                cid = int(tracked.class_id[i]) if tracked.class_id is not None else 0

                if tid not in self._tracks_by_id:
                    track = Track(track_id=tid, bbox=bbox, class_id=cid)
                    self._tracks_by_id[tid] = track
                    self.all_tracks.append(track)
                    if tid >= self.next_id:
                        self.next_id = tid + 1
                else:
                    track = self._tracks_by_id[tid]
                    track.bbox = bbox
                    track.class_id = cid
                    track.hits += 1

                track.time_since_update = 0
                track.age += 1

                # Store in world coords using affine inverse transform
                cx, cy = track.center
                track.trajectory.append(self._motion.transform_point_to_world(cx, cy))

                # Compute 3D position if depth available
                if self._depth is not None and self._last_depth_map is not None:
                    depth = self._depth.get_depth_at(cx, cy, self._last_depth_map)
                    track.depths.append(depth)
                    x3d, y3d, z3d = self._depth.pixel_to_3d(
                        cx, cy, depth=depth, depth_map=self._last_depth_map
                    )
                    track.trajectory_3d.append((x3d, y3d, z3d))

                active.append(track)

        return active
