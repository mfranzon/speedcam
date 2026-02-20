import numpy as np
import pytest

from speedcam.models import Detection, Track, resolve_class_filter
from speedcam.estimator import SpeedEstimator
from speedcam.flow import FlowCounter
from speedcam.analytics import Analytics
from speedcam.heatmap import SpeedHeatmap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_det(x1, y1, x2, y2, class_id=2, score=0.9):
    return Detection(bbox=np.array([x1, y1, x2, y2], dtype=float), score=score, class_id=class_id)


def make_track(track_id=0, x1=0, y1=0, x2=50, y2=30, class_id=2, hits=1):
    t = Track(track_id=track_id, bbox=np.array([x1, y1, x2, y2], dtype=float), class_id=class_id, hits=hits)
    t.trajectory.append(t.center)
    return t


# ---------------------------------------------------------------------------
# Detection / Track models
# ---------------------------------------------------------------------------

class TestModels:
    def test_detection_fields(self):
        det = make_det(0, 0, 100, 50)
        assert det.score == 0.9
        assert det.class_id == 2
        np.testing.assert_array_equal(det.bbox, [0, 0, 100, 50])

    def test_track_center(self):
        t = make_track(x1=0, y1=0, x2=100, y2=50)
        assert t.center == (50.0, 25.0)

    def test_track_center_non_origin(self):
        t = make_track(x1=10, y1=20, x2=30, y2=40)
        assert t.center == (20.0, 30.0)


# ---------------------------------------------------------------------------
# resolve_class_filter
# ---------------------------------------------------------------------------

class TestResolveClassFilter:
    COCO_NAMES = ["person", "bicycle", "car", "motorcycle", "airplane",
                  "bus", "train", "truck"]

    def test_vehicles_preset(self):
        ids = resolve_class_filter("vehicles", self.COCO_NAMES)
        assert sorted(ids) == [2, 3, 5, 7]

    def test_person_preset(self):
        ids = resolve_class_filter("person", self.COCO_NAMES)
        assert ids == [0]

    def test_all_preset(self):
        ids = resolve_class_filter("all", self.COCO_NAMES)
        assert sorted(ids) == [0, 2, 3, 5, 7]

    def test_comma_separated_names(self):
        ids = resolve_class_filter("car,truck", self.COCO_NAMES)
        assert ids == [2, 7]

    def test_numeric_ids(self):
        ids = resolve_class_filter("0,2", self.COCO_NAMES)
        assert ids == [0, 2]

    def test_none_returns_none(self):
        assert resolve_class_filter(None, self.COCO_NAMES) is None

    def test_unknown_name_skipped(self):
        ids = resolve_class_filter("car,spaceship", self.COCO_NAMES)
        assert ids == [2]


# ---------------------------------------------------------------------------
# SpeedEstimator
# ---------------------------------------------------------------------------

class TestSpeedEstimator:
    def test_manual_mpp(self):
        est = SpeedEstimator(meters_per_pixel=0.1)
        assert est.get_meters_per_pixel() == pytest.approx(0.1)

    def test_fallback_mpp(self):
        est = SpeedEstimator()
        assert est.get_meters_per_pixel() == pytest.approx(0.05)

    def test_calibrate_updates_estimate(self):
        est = SpeedEstimator()
        # car (class 2) = 4.5 m; 90 px wide → 0.05 m/px
        est.calibrate(bbox_width_px=90.0, class_id=2)
        assert est.get_meters_per_pixel() == pytest.approx(4.5 / 90, rel=1e-5)

    def test_calibrate_ignores_small_boxes(self):
        est = SpeedEstimator()
        est.calibrate(bbox_width_px=5.0, class_id=2)
        assert est.get_meters_per_pixel() == pytest.approx(0.05)  # fallback unchanged

    def test_calibrate_unknown_class_ignored(self):
        est = SpeedEstimator()
        est.calibrate(bbox_width_px=100.0, class_id=99)
        assert est.get_meters_per_pixel() == pytest.approx(0.05)

    def test_to_kmh(self):
        est = SpeedEstimator(meters_per_pixel=0.05)
        # 10 px/frame * 0.05 m/px * 30 fps * 3.6 = 54 km/h
        assert est.to_kmh(10.0, fps=30.0) == pytest.approx(54.0)

    def test_speed_from_trajectory_single_point(self):
        est = SpeedEstimator(meters_per_pixel=0.05)
        assert est.speed_from_trajectory([(0, 0)], fps=30) == pytest.approx(0.0)

    def test_speed_from_trajectory_two_points(self):
        est = SpeedEstimator(meters_per_pixel=0.05)
        # 10 px step in x, 0 in y → 10 px/step; avg_px_per_frame = 10/2 = 5
        speed = est.speed_from_trajectory([(0, 0), (10, 0)], fps=30)
        assert speed == pytest.approx(5 * 0.05 * 30 * 3.6)


# ---------------------------------------------------------------------------
# FlowCounter
# ---------------------------------------------------------------------------

class TestFlowCounter:
    def _make_crossing_track(self, track_id, y_before, y_after, class_id=2):
        """Track whose last two trajectory points straddle the line."""
        t = Track(
            track_id=track_id,
            bbox=np.array([0, y_after - 15, 50, y_after + 15], dtype=float),
            class_id=class_id,
            hits=2,
            time_since_update=0,
        )
        t.trajectory = [(25, y_before), (25, y_after)]
        return t

    def test_downward_crossing_counted(self):
        fc = FlowCounter(frame_height=200, line_y_ratio=0.5)  # line at y=100
        track = self._make_crossing_track(0, y_before=90, y_after=110)
        fc.update([track], fps=30)
        s = fc.summary()
        assert s["down"] == 1
        assert s["up"] == 0

    def test_upward_crossing_counted(self):
        fc = FlowCounter(frame_height=200, line_y_ratio=0.5)
        track = self._make_crossing_track(0, y_before=110, y_after=90)
        fc.update([track], fps=30)
        s = fc.summary()
        assert s["up"] == 1
        assert s["down"] == 0

    def test_no_double_counting(self):
        fc = FlowCounter(frame_height=200, line_y_ratio=0.5)
        track = self._make_crossing_track(0, y_before=90, y_after=110)
        fc.update([track], fps=30)
        # same track crosses again — should be ignored
        fc.update([track], fps=30)
        assert fc.summary()["down"] == 1

    def test_stale_track_not_counted(self):
        fc = FlowCounter(frame_height=200, line_y_ratio=0.5)
        track = self._make_crossing_track(0, y_before=90, y_after=110)
        track.time_since_update = 1  # stale
        fc.update([track], fps=30)
        assert fc.summary()["total_crossings"] == 0

    def test_summary_keys(self):
        fc = FlowCounter(frame_height=100)
        s = fc.summary()
        assert set(s.keys()) == {"up", "down", "per_class_up", "per_class_down", "total_crossings"}


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

class TestAnalytics:
    def test_frame_count(self):
        a = Analytics(fps=30)
        t = make_track()
        a.update([t])
        a.update([t])
        assert a.frame_count == 2

    def test_summary_structure(self):
        a = Analytics(fps=30)
        t = make_track(hits=30)
        t.trajectory = [(0, 0), (10, 0)]
        a.update([t])
        s = a.summary([t])
        assert "total_frames" in s
        assert "total_unique_vehicles" in s
        assert "avg_speed_kmh" in s
        assert "per_class_count" in s

    def test_summary_vehicle_count(self):
        a = Analytics(fps=30)
        tracks = [make_track(track_id=i) for i in range(3)]
        a.update(tracks)
        s = a.summary(tracks)
        assert s["total_unique_vehicles"] == 3

    def test_avg_speed_zero_without_trajectory(self):
        a = Analytics(fps=30)
        t = make_track()
        a.update([t])
        s = a.summary([t])
        assert s["avg_speed_kmh"] == pytest.approx(0.0)

    def test_per_class_count(self):
        a = Analytics(fps=30)
        car = make_track(track_id=0, class_id=2)
        truck = make_track(track_id=1, class_id=7)
        a.update([car, truck])
        s = a.summary([car, truck])
        assert s["per_class_count"]["car"] == 1
        assert s["per_class_count"]["truck"] == 1


# ---------------------------------------------------------------------------
# SpeedHeatmap
# ---------------------------------------------------------------------------

class TestSpeedHeatmap:
    def test_record_and_render(self):
        hm = SpeedHeatmap(width=320, height=240, cell_size=16)
        hm.record(100, 80, speed_kmh=60.0)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out = hm.render(frame)
        assert out.shape == frame.shape

    def test_render_empty_returns_frame_unchanged(self):
        hm = SpeedHeatmap(width=64, height=64)
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        out = hm.render(frame)
        np.testing.assert_array_equal(out, frame)

    def test_reset_clears_data(self):
        hm = SpeedHeatmap(width=64, height=64, cell_size=8)
        hm.record(10, 10, speed_kmh=50.0)
        hm.reset()
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        out = hm.render(frame)
        np.testing.assert_array_equal(out, frame)

    def test_out_of_bounds_record_ignored(self):
        hm = SpeedHeatmap(width=64, height=64)
        hm.record(1000, 1000, speed_kmh=100.0)  # should not raise
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        out = hm.render(frame)
        np.testing.assert_array_equal(out, frame)
