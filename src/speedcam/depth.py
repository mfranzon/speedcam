"""Monocular depth estimation using Depth Anything V2.

Provides metric depth estimation for 3D trajectory computation.
"""

import cv2
import numpy as np
import torch
from typing import Optional, Tuple


class DepthEstimator:
    """Depth estimation using Depth Anything V2.

    Supports both relative and metric depth variants.
    Metric depth provides absolute distances in meters.
    """

    def __init__(
        self,
        model_size: str = "small",
        metric: bool = True,
        device: str = "auto",
        focal_length: float = 500.0,
    ):
        """
        Args:
            model_size: 'small', 'base', or 'large'
            metric: Use metric depth model (absolute distances)
            device: 'auto', 'cuda', 'mps', or 'cpu'
            focal_length: Camera focal length in pixels (for metric conversion)
        """
        self.model_size = model_size
        self.metric = metric
        self.focal_length = focal_length
        self._model = None
        self._device = self._resolve_device(device)
        self._depth_cache: Optional[np.ndarray] = None
        self._frame_shape: Optional[Tuple[int, int]] = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError:
            raise ImportError(
                "depth-anything-v2 is required for depth estimation.\n"
                "Install with: pip install depth-anything-v2\n"
                "Or: pip install speedcam[depth]"
            )

        model_configs = {
            "small": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "base": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "large": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }

        config = model_configs.get(self.model_size, model_configs["small"])
        self._model = DepthAnythingV2(**config)

        encoder = config["encoder"]
        if self.metric:
            checkpoint_name = f"depth_anything_v2_metric_vkitti_{encoder}.pth"
            repo_id = f"depth-anything/Depth-Anything-V2-Metric-VKITTI-{encoder.replace('vits', 'Small').replace('vitb', 'Base').replace('vitl', 'Large')}"
        else:
            checkpoint_name = f"depth_anything_v2_{encoder}.pth"
            repo_id = f"depth-anything/Depth-Anything-V2-{encoder.replace('vits', 'Small').replace('vitb', 'Base').replace('vitl', 'Large')}"

        checkpoint_url = f"https://huggingface.co/{repo_id}/resolve/main/{checkpoint_name}"

        try:
            print(f"Loading depth model from {checkpoint_url}...")
            state_dict = torch.hub.load_state_dict_from_url(
                checkpoint_url,
                map_location=self._device,
            )
            self._model.load_state_dict(state_dict)
            print("Depth model loaded successfully.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Attempting to load from local checkpoints directory...")
            import os
            local_path = os.path.join("checkpoints", checkpoint_name)
            if os.path.exists(local_path):
                state_dict = torch.load(local_path, map_location=self._device)
                self._model.load_state_dict(state_dict)
                print(f"Loaded from {local_path}")
            else:
                raise RuntimeError(
                    f"Could not load depth model checkpoint.\n"
                    f"Download from: {checkpoint_url}\n"
                    f"And place in: checkpoints/{checkpoint_name}"
                )

        self._model = self._model.to(self._device).eval()

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth map for a single frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            Depth map (H, W) in meters if metric=True, else relative depth
        """
        self._load_model()

        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth = self._model.infer_image(rgb, self._device)

        self._depth_cache = depth
        self._frame_shape = frame.shape[:2]

        return depth

    def get_depth_at(self, x: float, y: float, depth_map: Optional[np.ndarray] = None) -> float:
        """Get depth value at a 2D point with bilinear interpolation.

        Args:
            x, y: Pixel coordinates
            depth_map: Optional depth map, uses cache if None

        Returns:
            Depth value at (x, y)
        """
        if depth_map is None:
            depth_map = self._depth_cache
        if depth_map is None:
            return 0.0

        h, w = depth_map.shape
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)
        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))

        fx, fy = x - x0, y - y0

        d00 = depth_map[y0, x0]
        d01 = depth_map[y0, x1]
        d10 = depth_map[y1, x0]
        d11 = depth_map[y1, x1]

        return float(
            d00 * (1 - fx) * (1 - fy)
            + d01 * fx * (1 - fy)
            + d10 * (1 - fx) * fy
            + d11 * fx * fy
        )

    def pixel_to_3d(
        self,
        x: float,
        y: float,
        depth: Optional[float] = None,
        depth_map: Optional[np.ndarray] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Unproject a 2D pixel to 3D camera coordinates.

        Args:
            x, y: Pixel coordinates
            depth: Known depth value, estimated from depth_map if None
            depth_map: Depth map for estimation
            cx, cy: Principal point (defaults to image center)

        Returns:
            (X, Y, Z) in camera coordinates (meters if metric depth)
        """
        if depth is None:
            depth = self.get_depth_at(x, y, depth_map)

        if depth_map is not None:
            h, w = depth_map.shape
        elif self._frame_shape is not None:
            h, w = self._frame_shape
        else:
            h, w = 480, 640

        cx_f = w / 2.0 if cx is None else cx
        cy_f = h / 2.0 if cy is None else cy

        f = self.focal_length

        X = (x - cx_f) * depth / f
        Y = (y - cy_f) * depth / f
        Z = depth

        return (X, Y, Z)

    def compute_3d_trajectory(
        self,
        trajectory_2d: list[Tuple[float, float]],
        depth_maps: list[np.ndarray],
        cx: Optional[float] = None,
        cy: Optional[float] = None,
    ) -> list[Tuple[float, float, float]]:
        """Convert a 2D trajectory to 3D using depth maps.

        Args:
            trajectory_2d: List of (x, y) pixel coordinates
            depth_maps: Corresponding depth maps for each frame
            cx, cy: Principal point

        Returns:
            List of (X, Y, Z) 3D points
        """
        if len(trajectory_2d) != len(depth_maps):
            raise ValueError("trajectory_2d and depth_maps must have same length")

        trajectory_3d = []
        for (x, y), depth_map in zip(trajectory_2d, depth_maps):
            pt3d = self.pixel_to_3d(x, y, depth_map=depth_map, cx=cx, cy=cy)
            trajectory_3d.append(pt3d)

        return trajectory_3d
