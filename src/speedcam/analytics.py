import json
from typing import List
from .models import Track, ALL_CLASSES
from .estimator import SpeedEstimator


class Analytics:
    def __init__(
        self,
        fps: float = 30.0,
        speed_estimator: SpeedEstimator | None = None,
        use_3d: bool = False,
    ):
        self.fps = fps
        self.speed_estimator = speed_estimator or SpeedEstimator()
        self.use_3d = use_3d
        self.frame_count = 0
        self.frame_vehicle_counts: List[int] = []

    def update(self, active_tracks: List[Track]):
        self.frame_count += 1
        self.frame_vehicle_counts.append(len(active_tracks))
        for t in active_tracks:
            if t.time_since_update == 0:
                w = float(t.bbox[2] - t.bbox[0])
                self.speed_estimator.calibrate(w, t.class_id)

    def summary(self, all_tracks: List[Track]) -> dict:
        class_counts: dict[str, int] = {}
        dwell_times: List[float] = []
        speeds_kmh: List[float] = []
        avg_depths: List[float] = []

        for track in all_tracks:
            label = ALL_CLASSES.get(track.class_id, f"class_{track.class_id}")
            class_counts[label] = class_counts.get(label, 0) + 1

            dwell = track.hits / self.fps
            dwell_times.append(dwell)

            if self.use_3d and len(track.trajectory_3d) >= 2:
                speeds_kmh.append(SpeedEstimator.speed_from_trajectory_3d(track.trajectory_3d, self.fps))
            elif len(track.trajectory) >= 2:
                speeds_kmh.append(self.speed_estimator.speed_from_trajectory(track.trajectory, self.fps))

            if track.depths:
                avg_depths.append(sum(track.depths) / len(track.depths))

        total_vehicles = len(all_tracks)
        avg_dwell = sum(dwell_times) / len(dwell_times) if dwell_times else 0.0
        avg_speed = sum(speeds_kmh) / len(speeds_kmh) if speeds_kmh else 0.0
        avg_per_frame = sum(self.frame_vehicle_counts) / len(self.frame_vehicle_counts) if self.frame_vehicle_counts else 0.0
        avg_depth = sum(avg_depths) / len(avg_depths) if avg_depths else 0.0

        result = {
            "total_frames": self.frame_count,
            "total_unique_vehicles": total_vehicles,
            "avg_vehicles_per_frame": round(avg_per_frame, 1),
            "per_class_count": class_counts,
            "avg_dwell_time_sec": round(avg_dwell, 2),
            "avg_speed_kmh": round(float(avg_speed), 1),
            "meters_per_pixel": round(self.speed_estimator.get_meters_per_pixel(), 4),
        }

        if self.use_3d:
            result["mode"] = "3d"
            result["avg_depth_m"] = round(avg_depth, 2)
        else:
            result["mode"] = "2d"

        return result

    def print_summary(self, all_tracks: List[Track]):
        s = self.summary(all_tracks)
        print("\n=== Analytics Summary ===")
        print(f"Total frames processed: {s['total_frames']}")
        print(f"Total unique vehicles:  {s['total_unique_vehicles']}")
        print(f"Avg vehicles per frame: {s['avg_vehicles_per_frame']}")
        print(f"Avg dwell time:         {s['avg_dwell_time_sec']}s")
        print(f"Avg speed:              {s['avg_speed_kmh']} km/h ({s['mode']})")
        if s['mode'] == '3d':
            print(f"Avg depth:              {s['avg_depth_m']} m")
        else:
            print(f"Scale:                  {s['meters_per_pixel']} m/px")
        print("Per-class breakdown:")
        for cls, count in s["per_class_count"].items():
            print(f"  {cls}: {count}")
        print("=========================\n")

    def export_json(self, all_tracks: List[Track], path: str = "analytics.json"):
        s = self.summary(all_tracks)
        with open(path, "w") as f:
            json.dump(s, f, indent=2)
        print(f"Analytics exported to {path}")
