from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Detection:
    """Detector output expressed in original-frame pixel coordinates."""
    class_id: int
    label: str
    confidence: float
    xyxy: Tuple[float, float, float, float]

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
