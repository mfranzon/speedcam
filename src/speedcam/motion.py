"""Background motion estimator using affine motion compensation.

Computes a 2×3 affine warp (rotation + translation + scale) from background
optical flow each frame, accumulates via matrix multiplication in homogeneous
coordinates, and exposes world↔frame transforms for scene-anchored trajectories.
"""

import cv2
import numpy as np
import supervision as sv
from typing import Optional


class BackgroundMotionEstimator:
    """Estimate camera motion via affine warp from background features.

    Each frame we track background keypoints with Lucas-Kanade optical flow,
    then fit an affine transform (similarity: rotation + uniform scale +
    translation) using ``cv2.estimateAffinePartial2D`` with RANSAC.

    Transforms are accumulated multiplicatively (no drift from addition).
    """

    DOWNSCALE = 2  # process at half resolution for speed

    def __init__(self, max_corners: int = 500, quality: float = 0.005,
                 min_distance: int = 7, min_matches: int = 8):
        self._max_corners = max_corners
        self._quality = quality
        self._min_distance = min_distance
        self._min_matches = min_matches

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None

        # Accumulated homogeneous transform: world (frame-0) → current frame
        self._world_to_frame_3x3 = np.eye(3, dtype=np.float64)

        # Per-frame displacement tracking (for bg_speed_px)
        self._prev_offset: tuple[float, float] = (0.0, 0.0)

    @property
    def world_to_frame(self) -> np.ndarray:
        """3×3 homogeneous matrix mapping world coords → current frame coords."""
        return self._world_to_frame_3x3.copy()

    @property
    def world_to_frame_2x3(self) -> np.ndarray:
        """2×3 affine matrix (top two rows) for use with ``cv2.transform``."""
        return self._world_to_frame_3x3[:2, :].copy()

    @property
    def frame_to_world(self) -> np.ndarray:
        """3×3 inverse: current frame coords → world coords."""
        return np.linalg.inv(self._world_to_frame_3x3)

    @property
    def frame_to_world_2x3(self) -> np.ndarray:
        """2×3 inverse affine matrix."""
        return self.frame_to_world[:2, :]

    @property
    def last_displacement(self) -> tuple[float, float]:
        """Per-frame background displacement ``(dx, dy)`` in pixels."""
        cur = self.camera_offset
        return (cur[0] - self._prev_offset[0], cur[1] - self._prev_offset[1])

    @property
    def camera_offset(self) -> tuple[float, float]:
        """Absolute camera displacement (tx, ty) from frame 0."""
        return (self._world_to_frame_3x3[0, 2], self._world_to_frame_3x3[1, 2])

    def transform_point_to_world(self, x: float, y: float) -> tuple[float, float]:
        """Map a point from current frame coords to world coords."""
        inv = self.frame_to_world
        wx = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]
        wy = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2]
        return (wx, wy)

    def _build_mask(self, detections: sv.Detections, h: int, w: int) -> np.ndarray:
        mask = np.full((h, w), 255, dtype=np.uint8)
        if len(detections) > 0:
            pad = 50 // self.DOWNSCALE
            for x1, y1, x2, y2 in (detections.xyxy / self.DOWNSCALE).astype(int):
                mask[max(0, y1 - pad):min(h, y2 + pad),
                     max(0, x1 - pad):min(w, x2 + pad)] = 0
        return mask

    def _find_features(self, gray: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=self._max_corners,
            qualityLevel=self._quality,
            minDistance=self._min_distance // self.DOWNSCALE or 1,
            mask=mask,
        )

    def update(self, frame: np.ndarray, detections: sv.Detections) -> None:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray_full.shape
        gray = cv2.resize(gray_full, (w // self.DOWNSCALE, h // self.DOWNSCALE),
                          interpolation=cv2.INTER_AREA)
        sh, sw = gray.shape
        mask = self._build_mask(detections, sh, sw)

        self._prev_offset = self.camera_offset

        # First frame
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_pts = self._find_features(gray, mask)
            return

        # Track previous features into current frame
        if self._prev_pts is not None and len(self._prev_pts) >= self._min_matches:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._prev_pts, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if new_pts is not None and status is not None:
                good = status.flatten() == 1
                if np.sum(good) >= self._min_matches:
                    old = self._prev_pts[good].reshape(-1, 2)
                    new = new_pts[good].reshape(-1, 2)

                    # Filter: keep features still in background mask
                    bg = np.array([
                        mask[int(np.clip(pt[1], 0, sh - 1)),
                             int(np.clip(pt[0], 0, sw - 1))] > 0
                        for pt in new
                    ])
                    if np.sum(bg) >= self._min_matches:
                        warp, inliers = cv2.estimateAffinePartial2D(
                            old[bg], new[bg], method=cv2.RANSAC,
                            ransacReprojThreshold=3.0,
                        )
                        if warp is not None:
                            # Scale warp back to full resolution
                            # The translation components need scaling, rotation/scale stay
                            warp_full = warp.copy()
                            warp_full[0, 2] *= self.DOWNSCALE
                            warp_full[1, 2] *= self.DOWNSCALE

                            # Convert to 3×3 homogeneous
                            H = np.eye(3, dtype=np.float64)
                            H[:2, :] = warp_full

                            # Accumulate: world_to_frame = current_warp @ prev_world_to_frame
                            self._world_to_frame_3x3 = H @ self._world_to_frame_3x3

        # Refresh features every frame for consecutive tracking
        self._prev_gray = gray
        self._prev_pts = self._find_features(gray, mask)
