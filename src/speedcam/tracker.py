import numpy as np
from typing import List

from .models import Detection, Track


def _iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = boxA_area + boxB_area - inter_area
    if union == 0:
        return 0.0
    return inter_area / union


class SimpleTracker:
    """IoU-based multi-object tracker using the Hungarian algorithm.

    Matches detections to existing tracks frame-by-frame via IoU cost.
    Unmatched detections spawn new tracks; tracks missing for more than
    ``max_age`` frames are pruned.

    Parameters
    ----------
    iou_threshold:
        Minimum IoU required to match a detection to an existing track.
    max_age:
        Number of frames a track survives without a matching detection.

    Example
    -------
    >>> tracker = SimpleTracker()
    >>> tracks = tracker.update(detections)          # List[Track]
    >>> all_ever_seen = tracker.all_tracks           # includes lost tracks
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: List[Track] = []
        self.all_tracks: List[Track] = []
        self.next_id: int = 0

    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracker with new detections and return active tracks."""
        from scipy.optimize import linear_sum_assignment

        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1

        if not self.tracks or not detections:
            matched = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self.tracks)))
        else:
            cost = np.zeros((len(self.tracks), len(detections)))
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    cost[t_idx, d_idx] = 1.0 - _iou(track.bbox, det.bbox)

            row_indices, col_indices = linear_sum_assignment(cost)

            matched = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self.tracks)))

            for r, c in zip(row_indices, col_indices):
                if cost[r, c] > (1.0 - self.iou_threshold):
                    continue
                matched.append((r, c))
                unmatched_dets.remove(c)
                unmatched_trks.remove(r)

        for t_idx, d_idx in matched:
            track = self.tracks[t_idx]
            track.bbox = detections[d_idx].bbox
            track.class_id = detections[d_idx].class_id
            track.hits += 1
            track.time_since_update = 0
            track.trajectory.append(track.center)

        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_track = Track(
                track_id=self.next_id,
                bbox=det.bbox,
                class_id=det.class_id,
            )
            new_track.trajectory.append(new_track.center)
            self.tracks.append(new_track)
            self.all_tracks.append(new_track)
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return self.tracks
