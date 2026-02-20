from typing import List, Tuple


_VEHICLE_LENGTHS_M = {2: 4.5, 3: 2.2, 5: 12.0, 7: 8.0}


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
        return 0.05

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

    @staticmethod
    def speed_from_trajectory_3d(
        trajectory_3d: List[Tuple[float, float, float]], fps: float
    ) -> float:
        """Compute speed in km/h from a 3D trajectory (meters).

        Args:
            trajectory_3d: List of (X, Y, Z) points in meters
            fps: Frame rate

        Returns:
            Speed in km/h
        """
        if len(trajectory_3d) < 2:
            return 0.0

        total_dist = 0.0
        for i in range(1, len(trajectory_3d)):
            dx = trajectory_3d[i][0] - trajectory_3d[i - 1][0]
            dy = trajectory_3d[i][1] - trajectory_3d[i - 1][1]
            dz = trajectory_3d[i][2] - trajectory_3d[i - 1][2]
            total_dist += (dx**2 + dy**2 + dz**2) ** 0.5

        avg_m_per_frame = total_dist / len(trajectory_3d)
        avg_m_per_sec = avg_m_per_frame * fps
        return avg_m_per_sec * 3.6

    @staticmethod
    def distance_3d(
        p1: Tuple[float, float, float], p2: Tuple[float, float, float]
    ) -> float:
        """Euclidean distance between two 3D points."""
        return ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2) ** 0.5

    @staticmethod
    def velocity_3d(
        trajectory_3d: List[Tuple[float, float, float]], fps: float
    ) -> Tuple[float, float, float]:
        """Compute average velocity vector in m/s from 3D trajectory.

        Returns:
            (vx, vy, vz) velocity components in m/s
        """
        if len(trajectory_3d) < 2:
            return (0.0, 0.0, 0.0)

        p0 = trajectory_3d[0]
        p1 = trajectory_3d[-1]
        n_frames = len(trajectory_3d) - 1
        dt = n_frames / fps

        if dt <= 0:
            return (0.0, 0.0, 0.0)

        vx = (p1[0] - p0[0]) / dt
        vy = (p1[1] - p0[1]) / dt
        vz = (p1[2] - p0[2]) / dt

        return (vx, vy, vz)
