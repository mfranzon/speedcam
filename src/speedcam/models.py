import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    class_id: int


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    class_id: int
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    trajectory: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def center(self) -> Tuple[float, float]:
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        return (cx, cy)


# COCO vehicle class IDs (0-indexed, used by YOLO)
VEHICLE_CLASSES_YOLO = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# COCO vehicle class IDs (1-indexed, used by RF-DETR)
VEHICLE_CLASSES_RFDETR = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
# Canonical labels for display
VEHICLE_CLASSES = {**VEHICLE_CLASSES_YOLO}
