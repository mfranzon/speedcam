import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


# Direction labels and their assigned colors (BGR for OpenCV)
DIRECTION_LABELS = ["right", "down-right", "down", "down-left",
                    "left", "up-left", "up", "up-right"]

DIRECTION_COLORS = {
    "right":      (0, 255, 0),      # green
    "down-right": (0, 200, 255),    # orange
    "down":       (0, 0, 255),      # red
    "down-left":  (255, 0, 255),    # magenta
    "left":       (255, 0, 0),      # blue
    "up-left":    (255, 255, 0),    # cyan
    "up":         (255, 255, 255),  # white
    "up-right":   (0, 255, 255),    # yellow
    "unknown":    (128, 128, 128),  # gray
}


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
    trajectory_3d: List[Tuple[float, float, float]] = field(default_factory=list)
    depths: List[float] = field(default_factory=list)

    @property
    def center(self) -> Tuple[float, float]:
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        return (cx, cy)

    @property
    def last_depth(self) -> float:
        return self.depths[-1] if self.depths else 0.0

    @property
    def direction(self) -> str:
        """Dominant direction of travel based on trajectory displacement."""
        if len(self.trajectory) < 5:
            return "unknown"
        x0, y0 = self.trajectory[0]
        x1, y1 = self.trajectory[-1]
        dx, dy = x1 - x0, y1 - y0
        if abs(dx) < 3 and abs(dy) < 3:
            return "unknown"
        angle = math.degrees(math.atan2(dy, dx))  # -180..180, 0=right
        # Quantize to 8 directions (each 45 degrees)
        idx = round(angle / 45) % 8
        return DIRECTION_LABELS[idx]

    @property
    def direction_color(self) -> Tuple[int, int, int]:
        """BGR color based on direction of travel."""
        return DIRECTION_COLORS.get(self.direction, (128, 128, 128))


# Canonical class names used for display and filtering.
# Keys are 0-indexed COCO IDs used throughout speedcam internals.
ALL_CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

VEHICLE_CLASSES = {k: v for k, v in ALL_CLASSES.items() if v != "person"}
PERSON_CLASSES = {0: "person"}


def resolve_class_filter(
    classes_arg: str | None,
    class_names: list[str],
) -> list[int] | None:
    """Resolve ``--classes`` value to a list of integer class IDs.

    Each comma-separated token is either an integer ID or a name looked up in
    *class_names* (where the index equals the class ID, as provided by the
    model).

    The special tokens ``vehicles``, ``person``, and ``all`` expand to the
    well-known COCO names so that existing speedcam presets still work.
    """
    if not classes_arg:
        return None

    _PRESETS = {
        "vehicles": ["car", "motorcycle", "bus", "truck"],
        "person": ["person"],
        "all": ["person", "car", "motorcycle", "bus", "truck"],
    }

    # Expand presets
    tokens: list[str] = []
    for raw in classes_arg.split(","):
        token = raw.strip()
        if token in _PRESETS:
            tokens.extend(_PRESETS[token])
        else:
            tokens.append(token)

    name_to_id = {name: i for i, name in enumerate(class_names)}
    class_filter: list[int] = []
    for token in tokens:
        try:
            class_filter.append(int(token))
        except ValueError:
            if token in name_to_id:
                class_filter.append(name_to_id[token])
            else:
                print(f"Warning: class '{token}' not found in model class list, skipping.")

    return class_filter if class_filter else None
