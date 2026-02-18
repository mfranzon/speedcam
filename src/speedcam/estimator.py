from typing import List, Tuple

# Typical vehicle lengths in meters for scale estimation
_VEHICLE_LENGTHS_M = {2: 4.5, 3: 2.2, 5: 12.0, 7: 8.0}  # car, motorcycle, bus, truck


class SpeedEstimator:
    def __init__(self, meters_per_pixel: float = 0.0):
        self._manual_mpp = meters_per_pixel
        self._calibration_widths: List[float] = []

    def calibrate(self, bbox_width_px: float, class_id: int) -> None:
        if class_id in _VEHICLE_LENGTHS_M and bbox_width_px > 10:
            self._calibration_widths.append(
                _VEHICLE_LENGTHS_M[class_id] / bbox_width_px
            )

    def get_meters_per_pixel(self) -> float:
        if self._manual_mpp > 0:
            return self._manual_mpp
        if self._calibration_widths:
            return sum(self._calibration_widths) / len(self._calibration_widths)
        return 0.05  # fallback: 5cm/px

    def to_kmh(self, px_per_frame: float, fps: float) -> float:
        mpp = self.get_meters_per_pixel()
        return px_per_frame * mpp * fps * 3.6

    def speed_from_trajectory(self, trajectory: List[Tuple[float, float]], fps: float) -> float:
        if len(trajectory) < 2:
            return 0.0
        total_dist = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i - 1][0]
            dy = trajectory[i][1] - trajectory[i - 1][1]
            total_dist += (dx**2 + dy**2) ** 0.5
        avg_px_per_frame = total_dist / len(trajectory)
        return self.to_kmh(avg_px_per_frame, fps)
