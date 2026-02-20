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
