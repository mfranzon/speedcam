import cv2
import numpy as np


class SpeedHeatmap:
    def __init__(self, width: int, height: int, cell_size: int = 16):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid_w = (width + cell_size - 1) // cell_size
        self.grid_h = (height + cell_size - 1) // cell_size
        self._speed_sum = np.zeros((self.grid_h, self.grid_w), dtype=np.float64)
        self._speed_count = np.zeros((self.grid_h, self.grid_w), dtype=np.float64)

    def record(self, x: float, y: float, speed_kmh: float):
        gx = int(x) // self.cell_size
        gy = int(y) // self.cell_size
        if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
            self._speed_sum[gy, gx] += speed_kmh
            self._speed_count[gy, gx] += 1

    def render(self, frame: np.ndarray, opacity: float = 0.4) -> np.ndarray:
        mask = self._speed_count > 0
        if not np.any(mask):
            return frame

        avg = np.zeros_like(self._speed_sum)
        avg[mask] = self._speed_sum[mask] / self._speed_count[mask]

        max_speed = avg.max()
        if max_speed <= 0:
            return frame

        normalized = np.zeros_like(avg, dtype=np.uint8)
        normalized[mask] = (avg[mask] / max_speed * 255).astype(np.uint8)

        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        colored_full = cv2.resize(colored, (self.width, self.height),
                                  interpolation=cv2.INTER_LINEAR)

        alpha_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        alpha_grid[mask] = opacity
        alpha_full = cv2.resize(alpha_grid, (self.width, self.height),
                                interpolation=cv2.INTER_LINEAR)
        alpha_3ch = alpha_full[:, :, np.newaxis]

        blended = (frame.astype(np.float32) * (1 - alpha_3ch)
                   + colored_full.astype(np.float32) * alpha_3ch)
        return blended.astype(np.uint8)

    def reset(self):
        self._speed_sum[:] = 0
        self._speed_count[:] = 0
